"""Lazy Windows factories for sequential menu and numerical driving leases."""

from __future__ import annotations

import importlib
import socket
import time
from contextlib import ExitStack
from pathlib import Path

from fh5.capture_runtime import CaptureSource
from fh5.events import EventEnvironment, ScreenFrame, validate_event_file
from fh5.numeric_drive_config import NumericDriveConfiguration


class DXGIEventFrames:
    """Menu templates use fresh DXGI RGB converted in memory, never JPEG."""

    def __init__(self, source: CaptureSource, size: tuple[int, int]) -> None:
        self.source, self.size = source, size
        self.last_ns = -1

    def capture(self) -> ScreenFrame:
        image = importlib.import_module("PIL.Image")
        deadline = time.perf_counter_ns() + 300_000_000
        while time.perf_counter_ns() < deadline:
            event = self.source.capture()
            if event.boundary or event.reason not in (None, "no_new_frame"):
                raise OSError("Menu capture invalidated: " + str(event.reason or event.boundary))
            if event.frame is not None:
                frame = event.frame
                stamp = frame.mapping.convert(frame.present_ticks)
                if stamp > self.last_ns:
                    pixels = image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX")
                    pixels = pixels.convert("L").resize(self.size, image.Resampling.BILINEAR)
                    self.last_ns = stamp
                    return ScreenFrame(stamp, *self.size, pixels.tobytes())
            time.sleep(0.005)
        raise TimeoutError("No fresh DXGI menu frame")

    def close(self) -> None:
        self.source.close()


def native_event_environment(path: Path, plan: NumericDriveConfiguration) -> EventEnvironment:
    from fh5.dxgi_windows import WindowsDXGIFrames
    from fh5.live import WindowsDesktop, XboxController
    from fh5.live_event import BoundedFrames, LiveEventEnvironment

    if not plan.request.live:
        raise ValueError("Native menu requires explicit live opt-in")
    # Menu recovery establishes the parked state needed by a fresh shadow run.
    # Only driving requires that shadow to have completed already.
    if any(reason != "missing_shadow_evidence" for reason in plan.qualification["reasons"]):
        plan.require_eligible()
    config = validate_event_file(path)["event_run"]
    with ExitStack() as cleanup:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cleanup.callback(receiver.close)
        receiver.bind(("127.0.0.1", plan.telemetry.port))
        desktop = WindowsDesktop()
        frames = BoundedFrames(
            lambda: DXGIEventFrames(WindowsDXGIFrames(plan.target), tuple(config["screen_size"])),
            confirm_release=True,
        )
        cleanup.callback(frames.close)
        controller = XboxController()
        cleanup.callback(controller.close)
        environment = LiveEventEnvironment(
            receiver, controller, desktop, frames, owns_receiver=True
        )
        cleanup.pop_all()
        return environment
