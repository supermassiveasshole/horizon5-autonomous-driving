"""MSS numerical diagnostic; never selected as an automatic DXGI fallback."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fh5.capture import CaptureEvent, QpcMapping, RawCapture
from fh5.dxgi_capture import ClientArea, DXGISettings
from fh5.live_vision import ColorSource


class MSSFrames:
    source_kind = "mss_numeric_diagnostic"

    def __init__(
        self,
        settings: DXGISettings,
        *,
        target: Callable[[], ClientArea | None],
        grabber_factory: Callable[[], Any],
    ) -> None:
        self.settings, self.target, self.factory = settings, target, grabber_factory
        self.backend: Any = None
        self.previous: ClientArea | None = None

    def capture(self) -> CaptureEvent:
        target = self.target()
        if target is None:
            boundary = "focus_lost" if self.previous is not None else None
            self.previous = None
            return CaptureEvent(
                time.perf_counter_ns(), boundary=boundary, reason="target_unavailable"
            )
        left, top, right, bottom = target.rect
        size = (right - left, bottom - top)
        if size != self.settings.expected_client_size:
            self.previous = None
            return CaptureEvent(
                time.perf_counter_ns(),
                boundary="client_size_mismatch",
                reason="client_size_mismatch",
            )
        if self.backend is None:
            self.backend = self.factory()
        started = time.perf_counter_ns()
        shot = self.backend.grab({"left": left, "top": top, "width": size[0], "height": size[1]})
        pixels = bytes(shot.bgra)
        ended = time.perf_counter_ns()
        if self.target() != target:
            self.previous = None
            return CaptureEvent(
                ended, boundary="window_changed_during_capture", reason="capture_discarded"
            )
        if tuple(shot.size) != size:
            raise OSError("MSS returned pixels that differ from physical client crop")
        boundary = (
            "window_changed" if self.previous is not None and self.previous != target else None
        )
        self.previous = target
        layout = {
            **asdict(target),
            "backend": "mss",
            "client_size": list(size),
            "condition_id": self.settings.condition_id,
            "color_space": "MSS BGRA; HDR state unverified",
            "clock": "monotonic capture-start proxy; unknown presentation uncertainty",
        }
        return CaptureEvent(
            ended,
            boundary=boundary,
            frame=RawCapture(
                started,
                QpcMapping(0, 0, 1_000_000_000, 0),
                size,
                pixels,
                layout,
                time_quality="capture_start_proxy",
            ),
        )

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
            self.backend = None


class WindowsMSSFrames(MSSFrames):
    def __init__(self, settings: DXGISettings) -> None:
        from fh5.dxgi_windows import WindowsClient

        self.client = WindowsClient()
        super().__init__(
            settings,
            target=self.client,
            grabber_factory=lambda: importlib.import_module("mss").mss(),
        )

    def capture(self) -> CaptureEvent:
        if self.client.desktop.stop_requested():
            return CaptureEvent(time.perf_counter_ns(), reason="user_stop")
        return super().capture()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.client.close()


class LegacyJPEGFrames:
    """Cost diagnostic using the old source, one bounded JPEG file and decode.

    Deliberately retains the old lossy path for comparison. No model/controller
    consumes this source; production numerical capture never enters this class.
    """

    source_kind = "mss_jpeg_diagnostic"

    def __init__(
        self,
        settings: DXGISettings,
        directory: Path,
        *,
        target: Callable[[], ClientArea | None],
        source_factory: Callable[[], ColorSource],
    ) -> None:
        self.settings, self.directory = settings, directory
        self.target, self.factory = target, source_factory
        self.source: ColorSource | None = None
        self.previous: ClientArea | None = None

    def capture(self) -> CaptureEvent:
        target = self.target()
        if target is None:
            boundary = "focus_lost" if self.previous is not None else None
            self.previous = None
            return CaptureEvent(
                time.perf_counter_ns(), boundary=boundary, reason="target_unavailable"
            )
        x0, y0, x1, y1 = target.rect
        if (x1 - x0, y1 - y0) != self.settings.expected_client_size:
            self.previous = None
            return CaptureEvent(
                time.perf_counter_ns(),
                boundary="client_size_mismatch",
                reason="client_size_mismatch",
            )
        if self.source is None:
            self.source = self.factory()
        frame = self.source.capture()
        io_start = time.perf_counter_ns()
        path = self.directory / "diagnostic-latest.jpg"
        path.write_bytes(frame.encoded)
        payload = path.read_bytes()
        decode_start = time.perf_counter_ns()
        from io import BytesIO

        image = importlib.import_module("PIL.Image")
        with image.open(BytesIO(payload)) as encoded:
            pixels = encoded.convert("RGBA").tobytes("raw", "BGRA")
            size = encoded.size
        ended = time.perf_counter_ns()
        if self.target() != target:
            self.previous = None
            return CaptureEvent(
                ended, boundary="window_changed_during_capture", reason="capture_discarded"
            )
        if frame.client_size != self.settings.expected_client_size:
            raise OSError("Legacy MSS source differs from physical comparison crop")
        boundary = (
            "window_changed" if self.previous is not None and self.previous != target else None
        )
        self.previous = target
        return CaptureEvent(
            ended,
            boundary=boundary,
            frame=RawCapture(
                frame.capture_start_ns,
                QpcMapping(0, 0, 1_000_000_000, 0),
                size,
                pixels,
                {
                    **asdict(target),
                    "backend": "mss_legacy_jpeg",
                    "lossy_jpeg": True,
                    "condition_id": self.settings.condition_id,
                    "physical_client_size": list(frame.client_size),
                    "legacy_image_size": list(frame.size),
                    "legacy_resize": frame.resize_method,
                    "jpeg_quality": frame.jpeg_quality,
                    "chroma_subsampling": frame.chroma_subsampling,
                },
                time_quality="capture_start_proxy",
            ),
            stage_ms={
                "legacy_capture_encode_ms": (io_start - frame.capture_start_ns) / 1e6,
                "legacy_disk_write_read_ms": (decode_start - io_start) / 1e6,
                "legacy_decode_ms": (ended - decode_start) / 1e6,
            },
        )

    def close(self) -> None:
        if self.source is not None:
            self.source.close()
            self.source = None


class WindowsLegacyJPEGFrames(LegacyJPEGFrames):
    def __init__(self, settings: DXGISettings, directory: Path) -> None:
        from fh5.dxgi_windows import WindowsClient
        from fh5.live_vision import WindowsColorFrames

        self.client = WindowsClient()
        super().__init__(
            settings,
            directory,
            target=self.client,
            source_factory=lambda: WindowsColorFrames(self.client.desktop),
        )

    def capture(self) -> CaptureEvent:
        if self.client.desktop.stop_requested():
            return CaptureEvent(time.perf_counter_ns(), reason="user_stop")
        return super().capture()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.client.close()
