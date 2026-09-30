"""Revalidate a ready handoff on the policy receiver before any driving command."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Literal

from fh5.control import Command
from fh5.realtime import RealtimeConfig, RealtimeEnvironment, TimelineInput


class ReadyHandoff:
    source_kind: Literal["synthetic"] = "synthetic"

    def __init__(
        self,
        environment: RealtimeEnvironment,
        ready: dict[str, Any],
        event: dict[str, Any],
        config: RealtimeConfig,
        deadline_ns: int | None = None,
    ) -> None:
        self.environment, self.ready, self.event, self.config = environment, ready, event, config
        self.confirmed = False
        self.deadline_ns = deadline_ns
        self.error: str | None = None

    def read(self, period_s: float) -> TimelineInput:
        from fh5.experiment import _decode

        value = self.environment.read(period_s)
        if not self.confirmed and self.error is None:
            if self.deadline_ns is not None and value.at_ns > self.deadline_ns:
                self.error = "handoff_expired"
            elif not value.raw_packets:
                self.error = "handoff_missing_telemetry"
            else:
                sample = _decode(value.raw_packets[-1])
                if (
                    sample["received_monotonic_ns"] <= self.ready["received_monotonic_ns"]
                    or sample["received_monotonic_ns"] != value.safety.received_ns
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
                ):
                    self.error = "handoff_state_changed"
                else:
                    self.confirmed = True
        if self.error:
            return replace(value, safety=replace(value.safety, fault=self.error), observation=None)
        return value

    def signals(self) -> tuple[bool, bool]:
        return self.environment.signals()

    def send(self, command: Command) -> None:
        if not self.confirmed and command != Command(0, 0, 0):
            raise ValueError("Unconfirmed ready handoff cannot send policy commands")
        self.environment.send(command)

    def close(self) -> dict[str, Any]:
        return {
            **self.environment.close(),
            "handoff_confirmed": self.confirmed,
            "handoff_error": self.error,
        }
