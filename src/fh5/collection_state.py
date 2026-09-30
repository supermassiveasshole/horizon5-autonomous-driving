"""Causal segment and coverage accounting for passive collection."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from fh5.collection import CollectionConfig, CollectionInput
from fh5.demonstrations import _mapped
from fh5.numeric_images import NumericFrame, validate_frame_history


class CollectionState:
    def __init__(self, config: CollectionConfig, profile: dict[str, Any]) -> None:
        self.config, self.profile = config, profile
        self.latest: dict[str, Any] | None = None
        self.previous_ns: int | None = None
        self.input_available_ns: int | None = None
        self.previous_ready = False
        self.conditions: tuple[str, ...] | None = None
        self.capture_epoch: str | None = None
        self.segment = self.segment_start = self.next_observation = 0
        self.last_advance_ns = 0
        self.last_source: int | None = None
        self.counts: Counter[str] = Counter()
        self.coverage: Counter[str] = Counter()
        self.active_ns = 0

    def _telemetry_reasons(self, sample: dict[str, Any]) -> list[str]:
        reasons = []
        if not sample["is_race_on"]:
            reasons.append("inactive")
        if (
            sample["car_ordinal"] != self.config.expected_car_ordinal
            or sample["car_performance_index"] != self.config.expected_pi
        ):
            reasons.append("unexpected_vehicle")
        if sample["motion"] is None or sample["speed_mps"] < 0:
            reasons.append("invalid_motion")
        return reasons

    def sample(
        self, value: CollectionInput, sequence: int
    ) -> tuple[dict[str, Any], tuple[NumericFrame, ...]]:
        from fh5.experiment import DIAGNOSTICS, _decode

        cfg, now = self.config, value.at_ns
        if self.previous_ns is not None and now <= self.previous_ns:
            raise ValueError("Collection source clock must advance")
        reasons: list[str] = []
        if not value.focused:
            reasons.append("focus_lost")
        if value.boundary:
            reasons.append(value.boundary)
        if self.capture_epoch is not None and value.capture_epoch != self.capture_epoch:
            reasons.append("capture_boundary")
        self.capture_epoch = value.capture_epoch
        decoded = []
        for packet in value.packets:
            try:
                sample = _decode(packet)
            except ValueError:
                reasons.append("invalid_telemetry")
                self.counts["invalid_packets"] += 1
                continue
            old = self.latest
            if old:
                dt = (sample["received_monotonic_ns"] - old["received_monotonic_ns"]) / 1e9
                if dt <= 0 or dt > cfg.max_age_ms / 1000:
                    reasons.append("telemetry_gap")
                if sample["game_timestamp_ms"] < old["game_timestamp_ms"]:
                    reasons.append("game_time_discontinuity_unverified")
                if sample["is_race_on"] != old["is_race_on"]:
                    reasons.append("activity_boundary")
                if math.dist(sample["position_m"], old["position_m"]) > DIAGNOSTICS[
                    "jump_slack_metres"
                ] + DIAGNOSTICS["jump_speed_metres_per_second"] * max(0, dt):
                    reasons.append("position_jump_unverified")
            if old is None or sample["game_timestamp_ms"] != old["game_timestamp_ms"]:
                self.last_advance_ns = sample["received_monotonic_ns"]
            reasons += self._telemetry_reasons(sample)
            if not 0 <= now - packet.received_monotonic_ns <= cfg.max_age_ms * 1_000_000:
                reasons.append("stale_telemetry")
            self.latest = sample
            decoded.append(sample)
        self.counts["packets"] += len(value.packets)
        latest = self.latest
        if (
            latest is None
            or not 0 <= now - latest["received_monotonic_ns"] <= cfg.max_age_ms * 1_000_000
        ):
            reasons.append("telemetry_missing")
        if latest:
            reasons += self._telemetry_reasons(latest)
        if latest and now - self.last_advance_ns > cfg.max_age_ms * 1_000_000:
            reasons.append("stalled_game_clock")
        mapped = None
        if value.human_input is not None:
            try:
                mapped = _mapped(value.human_input, self.profile)
                poll, available = mapped["poll_ns"], mapped["available_ns"]
                if not 0 <= now - poll <= cfg.max_age_ms * 1_000_000 or available > now:
                    reasons.append("stale_human_input")
                if self.input_available_ns is not None and poll <= self.input_available_ns:
                    reasons.append("input_clock_overlap")
                elif available <= now:
                    if (
                        self.input_available_ns is not None
                        and poll - self.input_available_ns > cfg.max_age_ms * 1_000_000
                    ):
                        reasons.append("input_clock_gap")
                    self.input_available_ns = available
                reasons += mapped["reasons"]
            except (ValueError, KeyError, TypeError):
                reasons.append("invalid_human_input")
        else:
            reasons.append("missing_human_input")
        signature = tuple(sorted(set(reasons)))
        boundary = self.conditions != signature or bool(value.boundary)
        if boundary:
            self.segment += 1
            self.segment_start = now
            self.last_source = None
            self.counts["segment_boundaries"] += 1
            for reason in signature:
                self.counts["boundary:" + reason] += 1
        self.conditions = signature
        ready = not reasons and mapped is not None and mapped["mapping_valid"]
        if ready:
            assert mapped is not None
            self.counts["valid_input_polls"] += 1
            raw = mapped["raw"]
            for name, yes in (
                ("left", raw["thumb_lx"] < -2000),
                ("right", raw["thumb_lx"] > 2000),
                ("rt", raw["right_trigger"] > 0),
                ("lt", raw["left_trigger"] > 0),
                ("coast", raw["left_trigger"] == raw["right_trigger"] == 0),
            ):
                if yes:
                    self.coverage[name] += 1
            if self.previous_ns is not None and self.previous_ready and not boundary:
                self.active_ns += min(now - self.previous_ns, cfg.max_age_ms * 1_000_000)
        frames: tuple[NumericFrame, ...] = ()
        image_reason: str | None = "not_due"
        due = now >= self.next_observation
        if due:
            self.next_observation = now + 1_000_000_000 // cfg.observation_hz
            self.counts["observation_ticks"] += 1
            candidates = value.frames
            image_reason = validate_frame_history(
                value.capture_epoch or "missing", now, candidates, cfg.pixels
            )
            if image_reason is None and candidates:
                if any(f.source_time_ns < self.segment_start for f in candidates):
                    image_reason = "history_before_segment"
                elif now - candidates[-1].source_time_ns > cfg.max_age_ms * 1_000_000:
                    image_reason = "stale_image"
                elif (
                    self.last_source is not None
                    and candidates[-1].source_time_ns <= self.last_source
                ):
                    image_reason = "repeated_image"
            if ready and image_reason is None:
                frames = candidates
                self.last_source = frames[-1].source_time_ns
                self.counts["synchronized_observations"] += 1
            else:
                image_reason = image_reason or "input_unusable"
                self.counts["observation_skip:" + image_reason] += 1
        row = {
            "sequence": sequence,
            "at_ns": now,
            "segment": self.segment,
            "segment_start_ns": self.segment_start,
            "capture_epoch": value.capture_epoch,
            "boundary": boundary,
            "reasons": list(signature),
            "focused": value.focused,
            "packets": value.packets,
            "telemetry": decoded,
            "latest_telemetry_received_ns": latest["received_monotonic_ns"] if latest else None,
            "human_input": value.human_input,
            "mapped_input": mapped,
            "input_usable": ready,
            "observation_due": due,
            "image_reason": image_reason,
            "synchronized": bool(frames),
            "training_eligible": False,
            "unverified": [
                "collision",
                "offroad",
                "navigation",
                "attempt_boundaries",
                "game_input_adoption",
            ],
        }
        self.previous_ns, self.previous_ready = now, ready
        return row, frames

    def progress(self) -> dict[str, Any]:
        return {
            "coverage_polls": dict(self.coverage),
            "counts": dict(self.counts),
            "active_driving_seconds": self.active_ns / 1e9,
            "segments": self.segment,
            "confirmed_attempts": 0,
            "attempts_status": "requires_independent_review",
            "training_eligible": False,
        }
