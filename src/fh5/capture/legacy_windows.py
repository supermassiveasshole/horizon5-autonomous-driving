"""Bounded, read-only Windows color capture and concurrent UDP reception."""

from __future__ import annotations

import ctypes
import importlib
import socket
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from datetime import UTC, datetime
from io import BytesIO
from queue import Empty, Full, Queue
from typing import Any, Literal, Protocol

from fh5.capture.legacy import ColorFrame, VisionInput
from fh5.driving.windows import DesktopState, WindowsDesktop
from fh5.telemetry.packet import Packet


class ColorSource(Protocol):
    def capture(self) -> ColorFrame: ...
    def close(self) -> None: ...


class CaptureDiscarded(OSError):
    """The foreground window changed during capture; pixels must not be saved."""


class LiveVisionEnvironment:
    """Own capture on one worker and receive UDP on another; never create game inputs.

    Queues are bounded. Overflows are explicit faults/drops, not hidden backlog.
    A stuck native capture is quarantined until process exit, reported as unreleased.
    The caller owns the socket. close() is idempotent.
    """

    source_kind: Literal["udp", "synthetic"] = "udp"

    def __init__(
        self,
        receiver: socket.socket,
        desktop: DesktopState,
        period_s: float,
        *,
        frame_factory: Callable[[], ColorSource],
    ) -> None:
        self.desktop = desktop
        self.period_s = period_s
        self._packets: Queue[Packet] = Queue(maxsize=4096)
        self._frames: Queue[ColorFrame] = Queue(maxsize=2)
        self._events: Queue[dict[str, Any]] = Queue(maxsize=128)
        self._done = threading.Event()
        self._fault: str | None = None
        self._release_failed = False
        self._released: bool | None = None
        self._focused: bool | None = None
        self._capture_started: int | None = self.now_ns()
        self._workers = [
            threading.Thread(
                target=self._receive, args=(receiver,), daemon=True, name="fh5-vision-udp"
            ),
            threading.Thread(
                target=self._capture, args=(frame_factory,), daemon=True, name="fh5-vision-rgb"
            ),
        ]
        receiver.settimeout(0.05)
        for worker in self._workers:
            worker.start()

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def _event(self, kind: str, **detail: Any) -> None:
        try:
            self._events.put_nowait({"kind": kind, "observed_ns": self.now_ns(), **detail})
        except Full:
            self._fault = "event_queue_overflow"
            self._done.set()

    def _receive(self, receiver: socket.socket) -> None:
        try:
            while not self._done.is_set():
                try:
                    data, _ = receiver.recvfrom(65535)
                except TimeoutError:
                    continue
                stamp = self.now_ns()
                try:
                    self._packets.put_nowait(Packet(stamp, datetime.now(UTC).isoformat(), data))
                except Full:
                    self._event("telemetry_overflow")
                    self._fault = "telemetry_overflow"
                    self._done.set()
        except OSError as error:
            self._event("udp_error", detail=str(error))
            self._fault = "udp_error"
            self._done.set()

    def _capture(self, factory: Callable[[], ColorSource]) -> None:
        source = None
        try:
            source = factory()
            self._capture_started = None
            next_capture = self.now_ns()
            while not self._done.is_set():
                tick = self.now_ns()
                if self.desktop.focused():
                    self._capture_started = tick
                    try:
                        frame = source.capture()
                    except CaptureDiscarded as error:
                        self._event("capture_discarded", detail=str(error))
                    else:
                        if not self._done.is_set():
                            try:
                                self._frames.put_nowait(frame)
                            except Full:
                                self._event("frame_queue_drop")
                    finally:
                        self._capture_started = None
                next_capture += int(self.period_s * 1e9)
                now = self.now_ns()
                if now > next_capture:
                    missed = (now - next_capture) // int(self.period_s * 1e9) + 1
                    next_capture += missed * int(self.period_s * 1e9)
                    self._event("frame_schedule_missed", count=missed)
                # Retain phase after a late OS wakeup, rather than accumulating timer drift.
                self._done.wait(max(0, (next_capture - self.now_ns()) / 1e9))
        except Exception as error:
            self._event("capture_error", detail=str(error))
            self._fault = "capture_error"
            self._done.set()
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as error:
                    self._event("capture_close_error", detail=str(error))
                    self._fault = "capture_close_error"
                    self._release_failed = True

    def read(self, period_s: float) -> VisionInput:
        self._done.wait(period_s)
        focused = self.desktop.focused()
        if focused != self._focused:
            self._event("focus_restored" if focused else "focus_lost")
            self._focused = focused
        capture_started = self._capture_started
        if capture_started is not None and self.now_ns() - capture_started > 2e9:
            self._fault = "capture_timeout"
            self._done.set()
        packets = []
        for _ in range(4096):
            try:
                packets.append(self._packets.get_nowait())
            except Empty:
                break
        events = []
        for _ in range(128):
            try:
                events.append(self._events.get_nowait())
            except Empty:
                break
        try:
            frame = self._frames.get_nowait()
        except Empty:
            frame = None
        return VisionInput(
            tuple(packets), frame, tuple(events), self.desktop.stop_requested(), self._fault
        )

    def close(self) -> bool:
        if self._released is not None:
            return self._released
        self._done.set()
        for worker in self._workers:
            worker.join(timeout=0.5)
        self._released = not self._release_failed and all(
            not worker.is_alive() for worker in self._workers
        )
        return self._released


class WindowsColorFrames:
    """MSS client capture, RGB resize, JPEG quality 90 with no chroma subsampling.

    Encoding is explicit and lossy. The compressed image is available before disk I/O;
    neither this timestamp nor Data Out receipt proves which physics tick was rendered.
    """

    def __init__(self, desktop: WindowsDesktop) -> None:
        self.desktop = desktop
        try:
            self.image = importlib.import_module("PIL.Image")
            self.backend = importlib.import_module("mss").mss()
        except ImportError as error:
            raise OSError("Install the events extra for MSS and Pillow") from error
        api = desktop.user32
        api.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]

    def capture(self) -> ColorFrame:
        api = self.desktop.user32
        window = api.GetForegroundWindow()
        if not self.desktop.focused():
            raise CaptureDiscarded("FH5 is not foreground")
        rect, origin = wintypes.RECT(), wintypes.POINT()
        if not api.GetClientRect(window, ctypes.byref(rect)) or not api.ClientToScreen(
            window, ctypes.byref(origin)
        ):
            raise OSError("Cannot locate FH5 client area")
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if (
            not 320 <= width <= 7680
            or not 180 <= height <= 4320
            or abs(width / height - 16 / 9) > 0.02
        ):
            raise OSError("Unsupported FH5 client geometry; expected 16:9")
        utc = datetime.now(UTC).isoformat()
        started = time.perf_counter_ns()
        shot = self.backend.grab(
            {"left": origin.x, "top": origin.y, "width": width, "height": height}
        )
        ended = time.perf_counter_ns()
        if window != api.GetForegroundWindow() or not self.desktop.focused():
            raise CaptureDiscarded("Foreground changed during capture")
        pixels = self.image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        factor = max(1, min(width // 960, height // 540))
        # Integer area reduction avoids a costly full-resolution resampling pass at 4K.
        if factor > 1:
            pixels = pixels.reduce(factor)
        resized = pixels.size != (960, 540)
        if resized:
            pixels = pixels.resize((960, 540), self.image.Resampling.BILINEAR)
        buffer = BytesIO()
        pixels.save(buffer, format="JPEG", quality=90, subsampling=0)
        encoded = buffer.getvalue()
        return ColorFrame(
            started,
            ended,
            time.perf_counter_ns(),
            encoded,
            "jpeg",
            (960, 540),
            (width, height),
            captured_utc=utc,
            resize_method=f"box_reduce_{factor}" + ("_then_bilinear" if resized else ""),
            jpeg_quality=90,
            chroma_subsampling=0,
        )

    def close(self) -> None:
        self.backend.close()
