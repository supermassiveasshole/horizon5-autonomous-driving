"""Revalidate a ready handoff on the policy receiver before any driving command."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from typing import Any

from fh5.driving.control import Command
from fh5.driving.realtime.model import (
    RealtimeConfig,
    RealtimeEnvironment,
    RealtimeRun,
    TimelineInput,
)


class ReadyHandoff:
    def __init__(
        self,
        environment: RealtimeEnvironment,
        ready: dict[str, Any],
        event: dict[str, Any],
        config: RealtimeConfig,
        deadline_ns: int | None = None,
    ) -> None:
        self.environment, self.ready, self.event, self.config = environment, ready, event, config
        self.source_kind = environment.source_kind
        self.confirmed = False
        self.deadline_ns = deadline_ns
        self.error: str | None = None
        self.started = False
        self.lock = threading.Lock()

    def authorize(
        self, request: RealtimeRun, manifest: dict[str, Any], inference_device: str | None
    ) -> None:
        authorize = getattr(self.environment, "authorize", None)
        if not callable(authorize):
            raise ValueError("Native handoff requires qualified driving authorization")
        authorize(request, manifest, inference_device)

    def read(self, period_s: float) -> TimelineInput:
        from fh5.telemetry.packet import decode_packet as _decode

        value = self.environment.read(period_s)
        with self.lock:
            if not self.started and self.error is None:
                if self.deadline_ns is not None and value.at_ns > self.deadline_ns:
                    self.error = "handoff_expired"
                elif not value.raw_packets:
                    if not self.confirmed:
                        self.error = "handoff_missing_telemetry"
                elif (
                    any(
                        sample["received_monotonic_ns"] <= self.ready["received_monotonic_ns"]
                        or sample["received_monotonic_ns"] > value.at_ns
                        or not 0
                        < (sample["game_timestamp_ms"] - self.ready["game_timestamp_ms"]) % 2**32
                        <= 60_000
                        or not sample["is_race_on"]
                        or sample["car_ordinal"] != self.config.expected_car_ordinal
                        or sample["car_performance_index"] != self.config.expected_pi
                        or not math.isfinite(sample["speed_kmh"])
                        or not 0 <= sample["speed_kmh"] <= self.config.start_speed_kmh
                        or any(sample["telemetry_controls"].values())
                        or not math.dist(sample["position_m"], self.event["start_position_m"])
                        <= self.event["start_radius_m"]
                        for sample in (_decode(packet) for packet in value.raw_packets)
                    )
                    or value.raw_packets[-1].received_monotonic_ns != value.safety.received_ns
                ):
                    self.error = "handoff_state_changed"
                else:
                    self.confirmed = True
            if self.error:
                return replace(
                    value, safety=replace(value.safety, fault=self.error), observation=None
                )
        return value

    def signals(self) -> tuple[bool, bool]:
        return self.environment.signals()

    def prepare_control(self) -> None:
        with self.lock:
            if not self.started:
                if self.deadline_ns is not None and time.perf_counter_ns() > self.deadline_ns:
                    self.error = "handoff_expired"
                if self.error:
                    raise ValueError(self.error)
                if not self.confirmed:
                    return
        prepare = getattr(self.environment, "prepare_control", None)
        if callable(prepare):
            prepare()
        with self.lock:
            if not self.started:
                if self.deadline_ns is not None and time.perf_counter_ns() > self.deadline_ns:
                    self.error = "handoff_expired"
                if self.error:
                    raise ValueError(self.error)

    def send(self, command: Command) -> None:
        with self.lock:
            nonzero = command != Command(0, 0, 0)
            if not self.started and nonzero:
                if self.deadline_ns is not None and time.perf_counter_ns() > self.deadline_ns:
                    self.error = "handoff_expired"
                if not self.confirmed or self.error:
                    raise ValueError("Unconfirmed ready handoff cannot send policy commands")
            self.environment.send(command)
            self.started = self.started or nonzero

    def close(self) -> dict[str, Any]:
        return {
            **self.environment.close(),
            "handoff_confirmed": self.confirmed,
            "handoff_error": self.error,
        }
