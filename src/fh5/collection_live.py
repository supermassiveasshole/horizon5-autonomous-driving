"""Passive DXGI/UDP/XInput composition; never constructs a virtual controller."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

from fh5.capture import CaptureConfig, CaptureRun
from fh5.capture_resources import ResourceMonitor, WindowsResources
from fh5.capture_runtime import CaptureSource, LiveCapture
from fh5.capture_samples import RawSamples
from fh5.collection import CollectionInput, CollectionRun
from fh5.collection_lease import CollectionLease
from fh5.collection_store import read_bounded
from fh5.demonstrations import _profile
from fh5.dxgi_capture import DXGISettings
from fh5.realtime_shadow import DesktopSignals, TelemetrySource


class CalibratedInputs(Protocol):
    def identity(self) -> dict[str, Any]: ...
    def read(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class PassiveCollectionEnvironment:
    def __init__(
        self,
        request: CollectionRun,
        capture_config: CaptureConfig,
        capture_factory: Callable[[], CaptureSource],
        telemetry: TelemetrySource,
        desktop: DesktopSignals,
        inputs: CalibratedInputs,
        *,
        source_kind: Literal["synthetic", "live_passive"],
        lease_path: Path,
        resources: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        if (
            capture_config.pixels != request.config.pixels
            or capture_config.observation_hz != request.config.observation_hz
            or capture_config.max_age_ms != request.config.max_age_ms
        ):
            raise ValueError("Collection and capture input contracts differ")
        self.request, self.config = request, capture_config
        self.capture_factory, self.telemetry = capture_factory, telemetry
        self.desktop, self.inputs, self.source_kind = desktop, inputs, source_kind
        self.lease = CollectionLease(lease_path)
        self.resources = resources
        self.pipeline: LiveCapture | None = None
        self.monitor: ResourceMonitor | None = None
        self.samples: RawSamples | None = None
        self.profile = _profile(read_bounded(request.input_profile, 1024**2))
        self.next_tick = 0.0
        self.closed: dict[str, Any] | None = None

    def read(self, period_s: float) -> CollectionInput:
        if self.closed is not None:
            raise OSError("Passive collection environment is closed")
        if self.pipeline is None:
            self.lease.acquire()
            if self.inputs.identity() != self.profile["device"]:
                raise ValueError("Input device identity differs from the calibrated profile")
            self.samples = RawSamples(CaptureRun(self.request.output_dir, self.config))
            self.pipeline = LiveCapture(self.config, self.capture_factory, self.samples)
            self.monitor = ResourceMonitor(self.resources)
        time.sleep(max(0, self.next_tick - time.perf_counter()))
        self.next_tick = time.perf_counter() + period_s
        batch = self.telemetry.read(0)
        raw = self.inputs.read()
        if raw.get("connected") is True:
            try:
                raw["device_profile_matches"] = self.inputs.identity() == self.profile["device"]
            except OSError as error:
                raw["device_profile_matches"] = False
                raw["identity_error"] = str(error)[:512]
        focused, stop = self.desktop.focused(), self.desktop.stop_requested()
        state, frames = self.pipeline.snapshot()
        now = time.perf_counter_ns()
        fault = batch.fault or self.pipeline.fault
        if self.pipeline.capture_started_ns and now - self.pipeline.capture_started_ns > 2e9:
            fault = "capture_timeout"
        return CollectionInput(
            now,
            packets=batch.packets,
            human_input=raw,
            frames=frames,
            capture_epoch=state["epoch"],
            focused=focused,
            stop_requested=stop or fault == "user_stop",
            fault=None if fault == "user_stop" else fault,
        )

    def close(self) -> dict[str, Any]:
        if self.closed is not None:
            return self.closed
        result: dict[str, Any] = {"resources_released": True, "commands_sent": False}
        for name, resource in (
            ("capture", self.pipeline),
            ("source_samples", self.samples),
            ("resources", self.monitor),
            ("inputs", self.inputs),
            ("telemetry", self.telemetry),
        ):
            if resource is None:
                continue
            try:
                value = resource.close()
                result[name] = value
                if isinstance(value, dict) and value.get("resources_released") is False:
                    result["resources_released"] = False
            except Exception as error:
                result[name] = {"error": f"{type(error).__name__}: {error}"}
                result["resources_released"] = False
        # Keep ownership if a native capture worker remains alive. The process must
        # exit before another collector can take over those resources.
        if result["resources_released"]:
            self.lease.close()
        result["resource_lease_released"] = result["resources_released"]
        self.closed = result
        return result


def native_collection_environment(
    request: CollectionRun, capture: CaptureConfig, target: DXGISettings, port: int
) -> PassiveCollectionEnvironment:
    from fh5.dxgi_windows import WindowsDXGIFrames
    from fh5.live import WindowsDesktop
    from fh5.live_demonstration import XInputReader
    from fh5.realtime_udp import UDPTelemetry

    if os.name != "nt":
        raise OSError("Native passive collection requires Windows")
    profile = _profile(read_bounded(request.input_profile, 1024**2))
    desktop = WindowsDesktop()
    return PassiveCollectionEnvironment(
        request,
        capture,
        lambda: WindowsDXGIFrames(target),
        UDPTelemetry(port),
        desktop,
        XInputReader(profile["device"]["index"], desktop),
        source_kind="live_passive",
        lease_path=Path(os.environ["LOCALAPPDATA"]) / "FH5VisionAI" / "passive-collection.lock",
        resources=WindowsResources(),
    )
