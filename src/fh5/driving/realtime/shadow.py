"""Read-only numerical shadow adapter; no controller or keyboard output capability."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from fh5.capture.pipeline import CaptureConfig, CaptureRun
from fh5.capture.resources import ResourceMonitor
from fh5.capture.runtime import CaptureSource, LiveCapture
from fh5.capture.samples import RawSamples
from fh5.driving.realtime.model import RealtimeObservation, RealtimeRun, SafetyState, TimelineInput
from fh5.observation.routes import load_route, locate_route

if TYPE_CHECKING:
    from fh5.driving.config import NumericDriveConfiguration
    from fh5.driving.control import Command
    from fh5.telemetry.packet import Packet


@dataclass(frozen=True)
class TelemetryBatch:
    packets: tuple[Packet, ...] = ()
    fault: str | None = None


class TelemetrySource(Protocol):
    def read(self, period_s: float) -> TelemetryBatch: ...
    def close(self) -> None: ...


class DesktopSignals(Protocol):
    def focused(self) -> bool: ...
    def stop_requested(self) -> bool: ...


@dataclass(frozen=True)
class LocalTask:
    route_file: Path
    expected_route_sha256: str
    start_station_m: float = 0
    start_tolerance_m: float = 0.5
    end_margin_m: float = 2

    def load(self) -> dict[str, Any]:
        digest = hashlib.sha256(self.route_file.read_bytes()).hexdigest()
        if digest != self.expected_route_sha256:
            raise ValueError("Shadow evaluation route changed")
        route = load_route(self.route_file)
        if not route["low_speed_ready"] or not 1 <= route["length_m"] <= 60:
            raise ValueError("Shadow task requires a verified local route, at most 60 metres")
        for value, lo, hi in (
            (self.end_margin_m, 0, route["length_m"] / 2),
            (self.start_tolerance_m, 0, 1),
            (self.start_station_m, 0, route["length_m"] - self.end_margin_m - 0.5),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError("Invalid shadow task bounds")
        return route


class ShadowEnvironment:
    source_kind: Literal["shadow"] = "shadow"

    @classmethod
    def from_native(cls, configuration: NumericDriveConfiguration) -> ShadowEnvironment:
        """Share native observations; defer capture until read and never create a controller."""
        from fh5.capture.resources import WindowsResources
        from fh5.capture.windows import WindowsDXGIFrames
        from fh5.driving.windows import WindowsDesktop

        return cls(
            configuration.request,
            configuration.capture,
            lambda: WindowsDXGIFrames(configuration.target),
            configuration.telemetry,
            WindowsDesktop(),
            configuration.task,
            input_conditions=configuration.bindings,
            resources=WindowsResources(),
        )

    def __init__(
        self,
        request: RealtimeRun,
        capture_config: CaptureConfig,
        capture_factory: Callable[[], CaptureSource],
        telemetry: TelemetrySource,
        desktop: DesktopSignals,
        task: LocalTask,
        *,
        input_conditions: dict[str, Any] | None = None,
        resources: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        if capture_config.pixels != request.config.pixels:
            raise ValueError("Shadow capture and actor pixel contracts differ")
        self.request, self.capture_config = request, capture_config
        self.capture_factory, self.telemetry, self.desktop = capture_factory, telemetry, desktop
        self.task, self.route = task, task.load()
        self.input_conditions = json.loads(json.dumps(input_conditions or {}, allow_nan=False))
        self.resources = resources
        self.monitor: ResourceMonitor | None = None
        self.pipeline: LiveCapture | None = None
        self.samples: RawSamples | None = None
        self.latest: dict[str, Any] | None = None
        self.geometry: dict[str, Any] = {}
        self.route_state: dict[str, Any] = {"anchor_s_m": task.start_station_m}
        self.packet_count = self.clock_advances = 0
        self.game_advanced_ns: int | None = None
        self.invalid_packets = 0
        self.ready = False
        self.active_segment = False
        self.fault: str | None = None
        self.epoch = 0

    def signals(self) -> tuple[bool, bool]:
        return self.desktop.focused(), self.desktop.stop_requested()

    def send(self, command: Command) -> None:
        # Predictions are diagnostic. They are never forwarded to a game device.
        pass

    def _sample_fault(self, sample: dict[str, Any], now: int) -> str | None:
        cfg, previous = self.request.config, self.latest
        if not 0 <= now - sample["received_monotonic_ns"] <= cfg.max_telemetry_age_ms * 1_000_000:
            return "stale_telemetry"
        if previous:
            if sample["received_monotonic_ns"] <= previous["received_monotonic_ns"]:
                return "telemetry_clock_discontinuity"
            if sample["game_timestamp_ms"] < previous["game_timestamp_ms"]:
                return "game_clock_discontinuity"
            if (
                self.game_advanced_ns is not None
                and sample["game_timestamp_ms"] == previous["game_timestamp_ms"]
                and now - self.game_advanced_ns > 100_000_000
            ):
                return "stalled_game_clock"
        if not sample["is_race_on"]:
            return "inactive"
        if (
            sample["car_ordinal"] != cfg.expected_car_ordinal
            or sample["car_performance_index"] != cfg.expected_pi
        ):
            return "unexpected_vehicle"
        if sample["speed_mps"] < 0:
            return "invalid_telemetry"
        if sample["speed_kmh"] >= cfg.max_speed_kmh:
            return "speed_limit"
        if not sample["motion"]:
            return "invalid_motion"
        return None

    def _segment(self, valid: bool, now: int, reason: str) -> None:
        if valid != self.active_segment:
            assert self.pipeline is not None
            self.epoch += 1
            self.clock_advances = 0
            self.route_state = {"anchor_s_m": self.task.start_station_m}
            self.pipeline.invalidate(reason, now)
            self.active_segment = valid

    def read(self, period_s: float) -> TimelineInput:
        from fh5.telemetry.packet import decode_packet as _decode

        if self.pipeline is None:
            # Model loading/warmup and output creation happen before the first read.
            self.samples = RawSamples(CaptureRun(self.request.output_dir, self.capture_config))
            self.pipeline = LiveCapture(self.capture_config, self.capture_factory, self.samples)
            self.monitor = ResourceMonitor(self.resources)
        batch = self.telemetry.read(period_s)
        packets = batch.packets
        if batch.fault:
            self.fault = self.fault or batch.fault
        focused, stop = self.signals()
        for packet in packets:
            index = self.packet_count
            self.packet_count += 1
            try:
                sample = _decode(packet)
            except ValueError:
                self.invalid_packets += 1
                if self.ready:
                    self.fault = self.fault or "invalid_telemetry"
                self.clock_advances = 0
                self._segment(False, packet.received_monotonic_ns, "invalid_telemetry")
                continue
            sample.update(packet_index=index, segment=self.epoch)
            reason = self._sample_fault(sample, time.perf_counter_ns())
            self._segment(
                reason is None and focused and not stop,
                sample["received_monotonic_ns"],
                reason or ("focus_lost" if not focused else "active_segment"),
            )
            sample["segment"] = self.epoch
            if self.latest and sample["game_timestamp_ms"] > self.latest["game_timestamp_ms"]:
                self.clock_advances += 1
                self.game_advanced_ns = sample["received_monotonic_ns"]
            elif self.latest is None:
                self.game_advanced_ns = sample["received_monotonic_ns"]
            self.latest = sample
            locate_route(
                [sample],
                self.route,
                state=self.route_state if self.ready else {"anchor_s_m": self.task.start_station_m},
            )
            self.geometry = dict(sample["route"])
            if reason is None and self.geometry["status"] not in ("matched", "awaiting_checkpoint"):
                reason = "task_location_untrusted"
            if self.ready and reason:
                self.fault = self.fault or reason
            if reason:
                self.clock_advances = 0
        now = time.perf_counter_ns()
        if (
            not focused
            or stop
            or (
                self.latest is not None
                and now - self.latest["received_monotonic_ns"]
                > self.request.config.max_telemetry_age_ms * 1_000_000
            )
        ):
            self._segment(False, now, "focus_or_telemetry_gap")
        row, frames = self.pipeline.snapshot()
        now, s = time.perf_counter_ns(), self.latest
        task_fault: str | None = None
        if s is not None:
            if not s["motion"]:
                task_fault = "invalid_motion"
            elif self.geometry.get("status") not in ("matched", "awaiting_checkpoint"):
                task_fault = "task_location_untrusted"
            elif (
                self.geometry["confirmed_progress_m"]
                >= self.route["length_m"] - self.task.end_margin_m
            ):
                task_fault = "local_end"
            elif not self.ready:
                pairs = list(zip(self.route["points"], self.route["points"][1:]))
                a, b = next(
                    (
                        (a, b)
                        for a, b in pairs
                        if a["s_m"] <= self.geometry["reference_s_m"] < b["s_m"]
                    ),
                    pairs[-1],
                )
                heading = math.atan2(
                    b["position_m"][0] - a["position_m"][0],
                    b["position_m"][2] - a["position_m"][2],
                )
                difference = s["motion"]["yaw_rad"] - heading
                if (
                    abs(self.geometry["reference_s_m"] - self.task.start_station_m)
                    > self.task.start_tolerance_m
                    or abs(math.atan2(math.sin(difference), math.cos(difference))) > 0.35
                    or self.clock_advances < 2
                ):
                    task_fault = "task_start_not_ready"
                elif (
                    frames
                    and focused
                    and not stop
                    and s["is_race_on"]
                    and s["car_ordinal"] == self.request.config.expected_car_ordinal
                    and s["car_performance_index"] == self.request.config.expected_pi
                    and s["speed_kmh"] <= self.request.config.start_speed_kmh
                    and 0
                    <= now - s["received_monotonic_ns"]
                    <= self.request.config.max_telemetry_age_ms * 1_000_000
                ):
                    self.ready = True
                    locate_route([s], self.route, state=self.route_state)
        safety = SafetyState(
            str(self.epoch),
            s["received_monotonic_ns"] if s else now,
            s["game_timestamp_ms"] if s else 0,
            focused,
            bool(s and s["is_race_on"]),
            stop,
            s["car_ordinal"] if s else 0,
            s["car_performance_index"] if s else 0,
            s["speed_kmh"] if s else 0,
            task_fault,
            self.fault or self.pipeline.fault,
            dict(self.geometry),
            s["packet_index"] if s else None,
        )
        observation = None
        if frames and s and s["motion"] and self.ready and not task_fault:
            observation = RealtimeObservation(
                row["epoch"],
                frames,
                {
                    "speed_mps": s["speed_mps"],
                    "velocity_car_mps": s["motion"]["velocity_car_mps"],
                    "angular_velocity_car_radps": s["motion"]["angular_velocity_car_radps"],
                },
                s["received_monotonic_ns"],
            )
        return TimelineInput(now, safety, observation, packets, capture_epoch=row["epoch"])

    def close(self) -> dict[str, Any]:
        capture = self.pipeline.close() if self.pipeline else {"resources_released": True}
        samples = self.samples.close() if self.samples else {"resources_released": True}
        resources = self.monitor.close() if self.monitor else {"resources_released": True}
        telemetry: dict[str, Any] = {
            "packets": self.packet_count,
            "invalid_packets": self.invalid_packets,
            "resources_released": True,
            "time_basis": "host_receive_time; not game physics or kernel arrival time",
        }
        try:
            self.telemetry.close()
        except Exception as error:
            telemetry.update(resources_released=False, close_error=str(error))
        return {
            "mode": "read_only_shadow",
            "fault": self.fault,
            "controller_created": False,
            "input_conditions": self.input_conditions,
            "task": {
                "route_sha256": self.task.expected_route_sha256,
                "start_station_m": self.task.start_station_m,
                "start_tolerance_m": self.task.start_tolerance_m,
                "end_margin_m": self.task.end_margin_m,
                "last_location": self.geometry,
            },
            "telemetry": telemetry,
            "capture": capture,
            "source_samples": samples,
            "resource_monitor": resources,
            "resources_released": capture["resources_released"]
            and samples["resources_released"]
            and resources["resources_released"]
            and telemetry["resources_released"],
        }
