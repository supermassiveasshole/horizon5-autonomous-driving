"""Version-locked DXGI adapter. No virtual controller or encoded-image path."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fh5.capture import CaptureEvent, RawCapture


@dataclass(frozen=True)
class DXGISettings:
    device_idx: int = 0
    output_idx: int = 0
    expected_client_size: tuple[int, int] = (3840, 2160)
    condition_id: str = "fh5-chase-far-4k-unverified-v1"

    def __post_init__(self) -> None:
        if (
            any(type(v) is not int or v < 0 for v in (self.device_idx, self.output_idx))
            or len(self.expected_client_size) != 2
            or any(type(v) is not int or not 1 <= v <= 7680 for v in self.expected_client_size)
            or self.expected_client_size[0] * self.expected_client_size[1] * 4 > 64 * 1024**2
            or not isinstance(self.condition_id, str)
            or not self.condition_id
        ):
            raise ValueError("Invalid DXGI target settings")


@dataclass(frozen=True)
class ClientArea:
    window: int
    monitor: int
    rect: tuple[int, int, int, int]
    dpi: int


@dataclass(frozen=True)
class DesktopImage:
    present_ticks: int
    accumulated_frames: int
    protected: bool
    size: tuple[int, int]
    bgra: bytes
    source_texture_size: tuple[int, int]


class DXGIFrames:
    source_kind = "dxgi"

    def __init__(
        self,
        settings: DXGISettings,
        *,
        target: Callable[[], ClientArea | None],
        camera_factory: Callable[[DXGISettings], Any],
    ) -> None:
        self.settings, self.target, self.camera_factory = settings, target, camera_factory
        self.camera: Any = None
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
        if self.camera is None:
            self.camera = self.camera_factory(self.settings)
        camera = self.camera
        x0, y0, x1, y1 = camera.output_rect
        if target.monitor != camera.monitor or not (
            x0 <= left < right <= x1 and y0 <= top < bottom <= y1
        ):
            raise OSError("FH5 physical client must fit the selected DXGI adapter/output")
        region = (left - x0, top - y0, right - x0, bottom - y0)
        image: DesktopImage | None = camera.grab(region)
        received = time.perf_counter_ns()
        after = self.target()
        if target != after:
            self.previous = after
            return CaptureEvent(
                received, boundary="window_changed_during_capture", reason="capture_discarded"
            )
        boundary = (
            "window_changed" if self.previous is not None and self.previous != target else None
        )
        self.previous = target
        if image is None:
            return CaptureEvent(received, boundary=boundary, reason="no_new_frame")
        if image.size != size:
            raise OSError("DXGI returned pixels that differ from the physical client crop")
        if image.protected:
            return CaptureEvent(received, boundary="protected_content", reason="protected_content")
        layout = {
            **camera.identity,
            "condition_id": self.settings.condition_id,
            "client_size": list(size),
            "client_rect": list(target.rect),
            "crop": list(region),
            "window": target.window,
            "monitor": target.monitor,
            "dpi": target.dpi,
            "source_texture_size": list(image.source_texture_size),
            "color_space": "DXGI_FORMAT_B8G8R8A8_UNORM; HDR state unverified",
        }
        return CaptureEvent(
            received,
            frame=RawCapture(
                image.present_ticks,
                camera.mapping,
                size,
                image.bgra,
                layout,
                time_quality="dxgi_qpc",
                accumulated_frames=image.accumulated_frames,
            ),
            boundary=boundary,
        )

    def close(self) -> None:
        if self.camera is not None:
            self.camera.release()
            self.camera = None
