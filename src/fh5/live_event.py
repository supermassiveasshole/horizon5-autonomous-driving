"""Live menu-only lifecycle adapter; desktop pixels and UDP stay outside the core."""

from __future__ import annotations

import ctypes
import importlib
import queue
import socket
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from typing import Any, Literal, Protocol

from fh5.events import EventInput, ScreenFrame
from fh5.live import NEUTRAL, Controller, DesktopState, LiveEnvironment, MenuButton, WindowsDesktop


class FrameSource(Protocol):
    def capture(self) -> ScreenFrame: ...
    def close(self) -> None: ...


class BoundedFrames:
    """Own a capture source on one daemon thread, with bounded caller waits.

    A timed-out native grab is quarantined, never reused. It may stay blocked until
    process exit; this worker owns no controller and cannot send game input.
    Sequential resource reuse requires confirm_release, which reports an unfinished
    or failed close instead of treating quarantine as successful release.
    """

    def __init__(
        self, factory: Callable[[], FrameSource], *, confirm_release: bool = False
    ) -> None:
        self._confirm_release = confirm_release
        self._close_error: Exception | None = None
        self._request = threading.Event()
        self._done = threading.Event()
        self._results: queue.Queue[ScreenFrame | Exception | None] = queue.Queue(maxsize=1)

        def work() -> None:
            source = None
            try:
                source = factory()
                self._results.put(None)
                while not self._done.is_set():
                    self._request.wait()
                    self._request.clear()
                    if self._done.is_set():
                        break
                    self._results.put(source.capture())
            except Exception as error:
                self._results.put(error)
            finally:
                if source is not None:
                    try:
                        source.close()
                    except Exception as error:
                        self._close_error = error

        self._worker = threading.Thread(target=work, daemon=True, name="fh5-frame-capture")
        self._worker.start()
        self._receive(2.0)  # Initialization also stays bounded, before controller creation.

    def _receive(self, timeout: float) -> ScreenFrame | None:
        try:
            result = self._results.get(timeout=timeout)
        except queue.Empty as error:
            self.close()
            raise TimeoutError("Frame capture worker timed out") from error
        if isinstance(result, Exception):
            self.close()
            raise OSError(f"Frame capture worker failed: {result}") from result
        return result

    def capture(self) -> ScreenFrame:
        if self._done.is_set():
            raise OSError("Frame capture is closed")
        self._request.set()
        result = self._receive(0.4)
        if result is None:
            raise OSError("Frame capture worker returned no frame")
        return result

    def close(self) -> None:
        self._done.set()
        self._request.set()
        self._worker.join(timeout=0.1)
        if self._confirm_release:
            if self._worker.is_alive():
                raise TimeoutError("Capture resource release is unconfirmed")
            if self._close_error is not None:
                raise OSError(f"Capture resource release failed: {self._close_error}")


class LiveEventEnvironment:
    """Reuse the input lease, F8/focus checks and device release from calibration."""

    source_kind: Literal["udp", "synthetic"] = "udp"

    def __init__(
        self,
        receiver: socket.socket,
        controller: Controller,
        desktop: DesktopState,
        frames: FrameSource,
        *,
        owns_receiver: bool = False,
    ) -> None:
        self.desktop = desktop
        self.frames = frames
        self.owns_receiver = owns_receiver
        self.input = LiveEnvironment(receiver, controller, desktop, monitor_idle=True)

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.input.events

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def read(self, period_s: float) -> EventInput:
        observation = self.input.read(period_s)
        frame = self.frames.capture() if observation.focused else None
        return EventInput(
            packets=observation.packets,
            frame=frame,
            focused=self.desktop.focused(),
            stop_requested=observation.stop_requested or self.desktop.stop_requested(),
            fault=observation.fault or self.input.fault,
        )

    def pulse(self, button: str) -> None:
        try:
            self.input.send(MenuButton(button))
            time.sleep(0.08)
        finally:
            self.release()

    def release(self) -> None:
        self.input.send(NEUTRAL)

    def close(self) -> None:
        try:
            self.input.close()
        finally:
            try:
                self.frames.close()
            finally:
                if self.owns_receiver and self.input.receiver is not None:
                    self.input.receiver.close()


class WindowsFrames:
    """Capture only the foreground FH5 client area at a bounded output resolution.

    Timestamp precedes capture so a slow grab is stale instead of freshly relabelled.
    MSS/Pillow are optional; replay never imports them.
    """

    def __init__(self, desktop: WindowsDesktop, size: tuple[int, int]) -> None:
        if not 1 <= size[0] <= 960 or not 1 <= size[1] <= 540:
            raise ValueError("Live frames require at most 960x540 pixels")
        self.desktop = desktop
        self.size = size
        try:
            self.image = importlib.import_module("PIL.Image")
            self.capture_backend = importlib.import_module("mss").mss()
        except ImportError as error:
            raise OSError("Install the events extra for MSS and Pillow") from error
        desktop.user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        desktop.user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]

    def capture(self) -> ScreenFrame:
        started = time.perf_counter_ns()
        user32 = self.desktop.user32
        window = user32.GetForegroundWindow()
        if not self.desktop.focused():
            raise OSError("FH5 is not foreground")
        rect, origin = wintypes.RECT(), wintypes.POINT()
        if not user32.GetClientRect(window, ctypes.byref(rect)) or not user32.ClientToScreen(
            window, ctypes.byref(origin)
        ):
            raise OSError("Cannot locate FH5 client area")
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if (
            not 320 <= width <= 7680
            or not 180 <= height <= 4320
            or abs(width / height - self.size[0] / self.size[1]) > 0.02
        ):
            raise OSError("FH5 client dimensions do not match the calibrated aspect ratio")
        shot = self.capture_backend.grab(
            {"left": origin.x, "top": origin.y, "width": width, "height": height}
        )
        pixels = self.image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        pixels = pixels.convert("L").resize(self.size, self.image.Resampling.BILINEAR)
        if window != user32.GetForegroundWindow() or not self.desktop.focused():
            raise OSError("Foreground changed during capture")
        return ScreenFrame(started, *self.size, pixels.tobytes())

    def close(self) -> None:
        self.capture_backend.close()
