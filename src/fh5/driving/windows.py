"""UDP and Windows adapters. Importing this module never loads the controller driver."""

from __future__ import annotations

import ctypes
import importlib
import select
import socket
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from fh5.driving.control import Command, ControlInput
from fh5.telemetry.packet import Packet

NEUTRAL = Command(0, 0, 0)
LEASE_NS = 250_000_000
MENU_BUTTONS = {
    "A": "XUSB_GAMEPAD_A",
    "B": "XUSB_GAMEPAD_B",
    "X": "XUSB_GAMEPAD_X",
    "Y": "XUSB_GAMEPAD_Y",
    "START": "XUSB_GAMEPAD_START",
    "UP": "XUSB_GAMEPAD_DPAD_UP",
    "DOWN": "XUSB_GAMEPAD_DPAD_DOWN",
    "LEFT": "XUSB_GAMEPAD_DPAD_LEFT",
    "RIGHT": "XUSB_GAMEPAD_DPAD_RIGHT",
}


@dataclass(frozen=True)
class MenuButton:
    button: str

    def __post_init__(self) -> None:
        if self.button not in MENU_BUTTONS:
            raise ValueError("Unsupported menu button")


class Controller(Protocol):
    def send(self, command: Command | MenuButton) -> None: ...
    def close(self) -> None: ...


class DesktopState(Protocol):
    def focused(self) -> bool: ...
    def stop_requested(self) -> bool: ...


class LiveEnvironment:
    """A short command lease; the watchdog can neutralize while the runner is stalled.

    This is a Python thread, not a hard real-time or process-crash safety guarantee.
    The caller owns the UDP socket. close() is idempotent and detaches the controller.
    """

    source_kind: Literal["udp", "synthetic"] = "udp"

    def __init__(
        self,
        receiver: socket.socket | None,
        controller: Controller,
        desktop: DesktopState,
        *,
        monitor_idle: bool = False,
    ) -> None:
        self.receiver = receiver
        self.controller = controller
        self.desktop = desktop
        self.monitor_idle = monitor_idle
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._fault: str | None = None
        self._active = False
        self._last_send = self.now_ns()
        self._closed = False
        self._release_failures = 0
        self._detached = False
        self._detach_error: str | None = None
        self._worker = threading.Thread(target=self._watch, daemon=True, name="fh5-input-watchdog")
        self._worker.start()

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    @property
    def fault(self) -> str | None:
        return self._fault

    def read(self, period_s: float) -> ControlInput:
        if self.receiver is None:
            raise OSError("Input-only controller lease has no UDP receiver")
        deadline = time.monotonic() + period_s
        packets = []
        while (remaining := deadline - time.monotonic()) > 0:
            ready, _, _ = select.select([self.receiver], [], [], remaining)
            if not ready:
                break
            payload, _ = self.receiver.recvfrom(65535)
            packets.append(Packet(self.now_ns(), datetime.now(UTC).isoformat(), payload))
            if len(packets) >= 256:
                self._fault = "telemetry_overflow"
                break
        return ControlInput(
            tuple(packets), self.desktop.focused(), self.desktop.stop_requested(), self._fault
        )

    def send(self, command: Command | MenuButton) -> None:
        with self._lock:
            if self._closed or self._detached:
                raise OSError("Controller is closed")
            if command != NEUTRAL:
                reason = self._fault
                if self.desktop.stop_requested():
                    reason = "user_stop"
                elif not self.desktop.focused():
                    reason = "focus_lost"
                if reason:
                    self._trip(reason)
                    raise OSError(f"Input inhibited: {reason}")
            try:
                self.controller.send(command)
            except Exception:
                self._active = True  # A failed call may have partially applied its report.
                self._trip("interface_error")
                raise
            self._active = command != NEUTRAL
            self._last_send = self.now_ns()

    def _trip(self, reason: str) -> None:
        # Called with the lock held; latch even if the neutral write fails.
        self._fault = reason
        event: dict[str, Any] = {
            "reason": reason,
            "issued_ns": self.now_ns(),
            "owner": "watchdog",
            "status": "failed",
        }
        try:
            self.controller.send(NEUTRAL)
            self._active = False
            event["status"] = "sent"
        except Exception as error:
            event["error"] = str(error)
            self._active = True
            self._release_failures += 1
            if self._release_failures >= 3:
                # A returning driver error is recoverable enough to attempt detach.
                try:
                    self.controller.close()
                    event["detach_status"] = "closed"
                except Exception as close_error:
                    event["detach_status"] = "failed"
                    event["detach_error"] = str(close_error)
                    self._detach_error = str(close_error)
                finally:
                    self._detached = True
                    self._active = False
        finally:
            event["returned_ns"] = self.now_ns()
            self.events.append(event)

    def _watch(self) -> None:
        while not self._done.wait(0.02):
            with self._lock:
                if not self._active and (not self.monitor_idle or self._fault):
                    continue
                try:
                    if self._fault:
                        self._trip(self._fault)
                    elif self.desktop.stop_requested():
                        self._trip("user_stop")
                    elif not self.desktop.focused():
                        self._trip("focus_lost")
                    elif self._active and self.now_ns() - self._last_send >= LEASE_NS:
                        self._trip("watchdog_timeout")
                except Exception:
                    self._trip("desktop_error")

    def close(self) -> None:
        if self._closed:
            return
        self._done.set()
        self._worker.join()
        with self._lock:
            self._closed = True
            if not self._detached:
                self.controller.close()
            elif self._detach_error is not None:
                raise OSError("Controller detach failed: " + self._detach_error)


class WindowsDesktop:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("Live control requires Windows")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = ctypes.c_short
        self.kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel32.OpenProcess.restype = wintypes.HANDLE
        self.kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    def focused(self) -> bool:
        window = self.user32.GetForegroundWindow()
        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
        process = self.kernel32.OpenProcess(0x1000, False, pid.value)
        if not process:
            return False
        try:
            buffer = ctypes.create_unicode_buffer(32768)
            length = wintypes.DWORD(len(buffer))
            if not self.kernel32.QueryFullProcessImageNameW(
                process, 0, buffer, ctypes.byref(length)
            ):
                return False
            return Path(buffer.value).name.lower() == "forzahorizon5.exe"
        finally:
            self.kernel32.CloseHandle(process)

    def stop_requested(self) -> bool:
        return bool(self.user32.GetAsyncKeyState(0x77) & 0x8000)  # F8, independent of focus.


class XboxController:
    def __init__(self) -> None:
        try:
            vg = importlib.import_module("vgamepad")
            self.buttons = vg.XUSB_BUTTON
            self.pad: Any = vg.VX360Gamepad()
        except Exception as error:
            raise OSError(
                "Virtual Xbox unavailable; install the control extra and ViGEmBus. See docs/guides/runtime/control.md."
            ) from error

    def send(self, command: Command | MenuButton) -> None:
        try:
            self.pad.reset()
            if isinstance(command, MenuButton):
                self.pad.press_button(button=getattr(self.buttons, MENU_BUTTONS[command.button]))
            else:
                self.pad.left_joystick(x_value=command.steer_i16, y_value=0)
                self.pad.right_trigger(value=command.throttle_u8)
                self.pad.left_trigger(value=command.brake_u8)
            self.pad.update()
        except Exception as error:
            raise OSError(f"Virtual Xbox write failed: {error}") from error

    def close(self) -> None:
        if self.pad is None:
            return
        try:
            self.send(NEUTRAL)
        finally:
            # vgamepad owns target removal; releasing the final reference detaches it.
            self.pad = None
