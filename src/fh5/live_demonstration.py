"""Read-only XInput polling, with explicit ownership and interruption evidence."""

from __future__ import annotations

import ctypes
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from fh5.demonstrations import _profile
from fh5.live import WindowsDesktop
from fh5.vision import VisionEnvironment, VisionInput


class HumanInputSource(Protocol):
    def read(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class HumanInputEnvironment:
    def __init__(
        self, vision: VisionEnvironment, inputs: HumanInputSource, profile_file: Path
    ) -> None:
        self.vision = vision
        self.inputs = inputs
        self.source_kind = vision.source_kind
        self.profile = _profile(profile_file.read_bytes())
        if isinstance(inputs, XInputReader) and inputs.identity() != self.profile["device"]:
            raise ValueError("Connected XInput capabilities differ from the frozen input profile")
        self._closed = False

    def now_ns(self) -> int:
        return self.vision.now_ns()

    def read(self, period_s: float) -> VisionInput:
        batch = self.vision.read(period_s)
        if batch.stop_requested or batch.fault:
            return batch
        raw = self.inputs.read()
        events = [*batch.events, raw]
        return replace(
            batch,
            events=tuple(events),
            fault="input_disconnected" if not raw["connected"] else None,
        )

    def close(self) -> bool:
        try:
            if not self._closed:
                self._closed = True
                self.inputs.close()
        finally:
            released = self.vision.close()
        return released


class _Gamepad(ctypes.Structure):
    _fields_ = [
        ("buttons", ctypes.c_uint16),
        ("left_trigger", ctypes.c_uint8),
        ("right_trigger", ctypes.c_uint8),
        ("thumb_lx", ctypes.c_int16),
        ("thumb_ly", ctypes.c_int16),
        ("thumb_rx", ctypes.c_int16),
        ("thumb_ry", ctypes.c_int16),
    ]


class _State(ctypes.Structure):
    _fields_ = [("packet_number", ctypes.c_uint32), ("gamepad", _Gamepad)]


class _Vibration(ctypes.Structure):
    _fields_ = [("left", ctypes.c_uint16), ("right", ctypes.c_uint16)]


class _Capabilities(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint8),
        ("subtype", ctypes.c_uint8),
        ("flags", ctypes.c_uint16),
        ("gamepad", _Gamepad),
        ("vibration", _Vibration),
    ]


class XInputReader:
    """Poll a fixed logical slot. No XInputSetState, driver or virtual device."""

    def __init__(self, index: int, desktop: WindowsDesktop | None = None) -> None:
        if type(index) is not int or not 0 <= index <= 3:
            raise ValueError("XInput index must be 0..3")
        self.index = index
        self.desktop = desktop
        self.api = ctypes.WinDLL("xinput1_4.dll")
        self.api.XInputGetState.argtypes = [ctypes.c_uint32, ctypes.POINTER(_State)]
        self.api.XInputGetState.restype = ctypes.c_uint32
        self.api.XInputGetCapabilities.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(_Capabilities),
        ]
        self.api.XInputGetCapabilities.restype = ctypes.c_uint32

    def identity(self) -> dict[str, Any]:
        caps = _Capabilities()
        status = self.api.XInputGetCapabilities(self.index, 0, ctypes.byref(caps))
        if status:
            raise OSError(f"XInput slot {self.index}: status {status}")
        return {
            "api": "xinput1_4",
            "index": self.index,
            "capabilities": {
                "type": caps.type,
                "subtype": caps.subtype,
                "flags": caps.flags,
                "gamepad": {
                    field[0]: getattr(caps.gamepad, field[0]) for field in _Gamepad._fields_
                },
            },
            "identity_scope": "logical_slot_and_capabilities_not_hardware_serial",
        }

    def read(self) -> dict[str, Any]:
        started = time.perf_counter_ns()
        state = _State()
        status = self.api.XInputGetState(self.index, ctypes.byref(state))
        focused = bool(self.desktop and self.desktop.focused())
        other = False
        if focused and self.desktop:
            # Only a boolean for competing driving keys; never record typed text.
            other = any(
                self.desktop.user32.GetAsyncKeyState(key) & 0x8000
                for key in (0x57, 0x41, 0x53, 0x44, 0x20, 0x25, 0x26, 0x27, 0x28)
            )
            for index in range(4):
                if index == self.index:
                    continue
                peer = _State()
                if self.api.XInputGetState(index, ctypes.byref(peer)) == 0:
                    other = other or bool(
                        peer.gamepad.buttons
                        or peer.gamepad.left_trigger
                        or peer.gamepad.right_trigger
                        or any(
                            abs(getattr(peer.gamepad, axis)) > 8000
                            for axis in ("thumb_lx", "thumb_ly", "thumb_rx", "thumb_ry")
                        )
                    )
        return {
            "kind": "human_input",
            "observed_ns": started,
            "available_ns": time.perf_counter_ns(),
            "device_index": self.index,
            "connected": status == 0,
            "status_code": status,
            "focused": focused,
            "other_input": other,
            "raw": {
                "packet_number": state.packet_number,
                **{field[0]: getattr(state.gamepad, field[0]) for field in _Gamepad._fields_},
            }
            if status == 0
            else None,
        }

    def close(self) -> None:
        pass


def input_devices() -> list[dict[str, Any]]:
    devices = []
    for index in range(4):
        reader = XInputReader(index)
        try:
            devices.append(reader.identity())
        except OSError:
            continue
        finally:
            reader.close()
    return devices
