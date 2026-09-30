"""Deadline/lease state machine, shared by fault replay and independently scheduled workers."""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from fh5.control import Command
from fh5.numeric_images import validate_frame_history
from fh5.realtime import RealtimeConfig, RealtimeObservation, SafetyState

NEUTRAL = Command(0, 0, 0)
MS = 1_000_000


@dataclass(frozen=True)
class Work:
    row: dict[str, Any]
    observation: RealtimeObservation
    actor: dict[str, Any]


class DecisionState:
    """Caller serializes short state transitions. No codecs, hashes or filesystem I/O here."""

    def __init__(
        self,
        config: RealtimeConfig,
        send: Callable[[Command], None],
        *,
        clock: Callable[[], int] | None = None,
        simulated_history: bool = True,
        notify: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.config, self.send = config, send
        self.clock, self.simulated_history = clock, simulated_history
        self.notify = notify or (lambda kind, row: None)
        self.safety: SafetyState | None = None
        self.decisions: list[dict[str, Any]] = []
        self.commands: list[dict[str, Any]] = []
        self.pending: Work | None = None
        self.last_action: dict[str, Any] | None = None
        self.last_accepted_ns: int | None = None
        self.lease_ns: int | None = None
        self.stop_reason: str | None = None
        self.safety_fault: str | None = None
        self.clock_advanced_ns: int | None = None
        self.capture_epoch: str | None = None
        self.submitted_source: tuple[str, int] | None = None

    def update(self, safety: SafetyState) -> None:
        old = self.safety
        if old is None or safety.game_timestamp_ms > old.game_timestamp_ms:
            self.clock_advanced_ns = safety.received_ns
        if old and (self.last_accepted_ns is not None or self.pending is not None):
            if old.epoch != safety.epoch:
                self.safety_fault = "session_boundary"
            elif safety.received_ns < old.received_ns:
                self.safety_fault = "telemetry_clock_discontinuity"
            elif safety.game_timestamp_ms < old.game_timestamp_ms:
                self.safety_fault = "game_clock_discontinuity"
        self.safety = safety
        # An observed hard fault must survive a newer healthy sample and a
        # later signals() poll. Only the supervisor sends the resulting stop.
        if self.last_accepted_ns is not None or self.pending is not None or safety.stop_requested:
            reason = self.safety_reason(self.clock() if self.clock else safety.received_ns)
            if reason is not None:
                self.safety_fault = reason

    def safety_reason(self, now: int) -> str | None:
        s, cfg = self.safety, self.config
        if self.safety_fault:
            return self.safety_fault
        if s is None:
            return "missing_telemetry"
        if s.stop_requested:
            return "user_stop"
        if s.fault:
            return s.fault
        if not s.focused:
            return "focus_lost"
        if not s.active:
            return "inactive"
        if s.car_ordinal != cfg.expected_car_ordinal or s.pi != cfg.expected_pi:
            return "unexpected_vehicle"
        if not math.isfinite(s.speed_kmh) or s.speed_kmh < 0 or s.received_ns > now:
            return "invalid_telemetry"
        if now - s.received_ns > cfg.max_telemetry_age_ms * MS:
            return "stale_telemetry"
        if self.clock_advanced_ns is not None and now - self.clock_advanced_ns > 100 * MS:
            return "stalled_game_clock"
        if s.speed_kmh >= cfg.max_speed_kmh:
            return "speed_limit"
        return s.task_fault

    def _send(self, now: int, command: Command, owner: str, work: Work | None = None) -> None:
        row: dict[str, Any] = {
            "issued_ns": now,
            "returned_ns": now,
            "owner": owner,
            "decision_id": work.row["decision_id"] if work else None,
            "valid_until_ns": work.row["valid_until_ns"] if work else now,
            "target": asdict(command),
            "sent": None,
            "status": "failed",
            "previous_hold_ns": now - self.last_action["returned_ns"] if self.last_action else None,
        }
        try:
            self.send(command)
            row.update(sent=asdict(command), status="sent")
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            row["returned_ns"] = self.clock() if self.clock else now
            self.commands.append(row)
            self.last_action = row
            self.notify("command", dict(row))

    def stop(self, now: int, reason: str) -> None:
        if self.stop_reason is not None:
            return
        self.stop_reason = reason
        if self.last_action is not None:
            for _ in range(3):
                try:
                    self._send(self.clock() if self.clock else now, NEUTRAL, "hard_stop")
                    break
                except Exception:
                    continue
        self.lease_ns = None
        self.notify("stop", {"at_ns": now, "reason": reason})

    def supervise(self, now: int) -> None:
        if self.stop_reason is not None:
            return
        reason = self.safety_reason(now)
        if reason and (
            self.last_accepted_ns is not None or self.pending is not None or reason == "user_stop"
        ):
            self.stop(now, reason)
            return
        if self.lease_ns is not None and now >= self.lease_ns:
            try:
                self._send(now, NEUTRAL, "lease_expiry")
            except Exception:
                self.stop(now, "send_failed")
                return
            self.lease_ns = None
        if (
            self.last_accepted_ns is not None
            and now - self.last_accepted_ns >= self.config.watchdog_ms * MS
        ):
            self.stop(now, "inference_watchdog" if self.pending else "decision_watchdog")
        elif self.pending and now - self.pending.row["decision_ns"] >= self.config.watchdog_ms * MS:
            self.stop(now, "inference_watchdog")

    def _actor(self, now: int, observation: RealtimeObservation) -> dict[str, Any]:
        actions: list[list[float] | None] = []
        ages: list[float | None] = []
        for offset in self.config.action_offsets_ms:
            prior = next(
                (
                    c
                    for c in reversed(self.commands)
                    if c["status"] == "sent" and c["returned_ns"] < now - offset * MS
                ),
                None,
            )
            if not self.simulated_history:
                prior = None
            if prior and now - offset * MS - prior["returned_ns"] <= 200 * MS:
                sent = prior["sent"]
                actions.append(
                    [sent["steer_i16"] / 32767, (sent["throttle_u8"] - sent["brake_u8"]) / 255]
                )
                ages.append((now - prior["returned_ns"]) / MS)
            else:
                actions.append(None)
                ages.append(None)
        return {
            "ego": deepcopy(observation.ego),
            "ego_mask": True,
            "ego_age_ms": (now - observation.telemetry_received_ns) / MS,
            "images": [f.frame_id for f in observation.frames],
            "image_mask": [True] * len(observation.frames),
            "image_age_ms": [(now - f.source_time_ns) / MS for f in observation.frames],
            "actions": actions,
            "action_mask": [a is not None for a in actions],
            "action_age_ms": ages,
            "reference": {
                "waypoints_m": [None] * self.config.reference_count,
                "mask": [False] * self.config.reference_count,
            },
        }

    def observation_reason(self, now: int, observation: RealtimeObservation) -> str | None:
        cfg, frames = self.config, observation.frames
        structural = validate_frame_history(observation.epoch, now, frames, cfg.pixels)
        if structural:
            return {
                "incomplete_history": "history",
                "history_crosses_epoch": "capture_epoch",
                "pixel_contract_mismatch": "pixel_contract",
                "history_layout_changed": "layout_changed",
                "image_not_available": "unavailable_image",
            }.get(structural, structural)
        layout_keys = (
            "size",
            "client_size",
            "format",
            "stride_bytes",
            "color_space",
            "crop",
            "resize_method",
            "window",
            "output",
            "adapter",
        )
        layouts = [tuple(f.source_layout.get(k) for k in layout_keys) for f in frames]
        if any(layout != layouts[0] for layout in layouts):
            return "layout_changed"
        if not 0 <= now - observation.telemetry_received_ns <= cfg.max_telemetry_age_ms * MS:
            return "stale_input"
        if any(
            now - f.source_time_ns > (offset + cfg.max_image_age_ms) * MS
            for f, offset in zip(frames, cfg.pixels.history_offsets_ms)
        ):
            return "stale_input"
        if any(
            abs(frames[-1].source_time_ns - f.source_time_ns - offset * MS) > 40 * MS
            for f, offset in zip(frames, cfg.pixels.history_offsets_ms)
        ):
            return "history_spacing"
        try:
            ego = observation.ego
            if (
                set(ego) != {"speed_mps", "velocity_car_mps", "angular_velocity_car_radps"}
                or len(ego["velocity_car_mps"]) != 3
                or len(ego["angular_velocity_car_radps"]) != 3
            ):
                return "invalid_motion"
            if not all(
                type(v) in (int, float) and math.isfinite(v)
                for v in [
                    ego["speed_mps"],
                    *ego["velocity_car_mps"],
                    *ego["angular_velocity_car_radps"],
                ]
            ):
                return "invalid_motion"
        except (TypeError, KeyError):
            return "invalid_motion"
        return None

    def begin(self, now: int, observation: RealtimeObservation | None) -> Work | None:
        if observation:
            self.capture_epoch = observation.epoch
        row: dict[str, Any] = {
            "index": len(self.decisions),
            "decision_id": f"d{len(self.decisions)}",
            "decision_ns": now,
            "status": "pending",
            "prediction": None,
            "deadline_ns": now + self.config.inference_deadline_ms * MS,
            "valid_until_ns": now + self.config.action_lease_ms * MS,
        }
        self.decisions.append(row)
        if self.stop_reason:
            row["status"] = "skip_stopped"
        elif self.safety_reason(now):
            row.update(status="skip_not_ready", reason=self.safety_reason(now))
        elif (
            self.last_accepted_ns is None
            and self.safety
            and self.safety.speed_kmh > self.config.start_speed_kmh
        ):
            row.update(status="skip_not_ready", reason="start_speed")
        elif self.pending is not None:
            row["status"] = "skip_busy"
        elif observation is None:
            row["status"] = "skip_history"
        elif reason := self.observation_reason(now, observation):
            row["status"] = "skip_" + reason
        elif (
            self.submitted_source is not None
            and self.submitted_source[0] == observation.epoch
            and observation.frames[-1].source_time_ns <= self.submitted_source[1]
        ):
            row["status"] = "skip_repeated_source"
        else:
            actor = self._actor(now, observation)
            row.update(
                epoch=observation.epoch,
                frames=[f.metadata() for f in observation.frames],
                actor=actor,
                safety_at_decision=asdict(self.safety) if self.safety else None,
                telemetry_received_ns=observation.telemetry_received_ns,
            )
            work = Work(row, observation, actor)
            self.pending = work
            self.submitted_source = observation.epoch, observation.frames[-1].source_time_ns
            self.notify("decision_started", dict(row))
            return work
        self.notify("decision_skipped", dict(row))
        return None

    def complete(
        self, now: int, work: Work, prediction: list[float], error: str | None = None
    ) -> None:
        if self.pending is not work:
            return
        self.pending = None
        row = work.row
        valid = len(prediction) == 2 and all(
            type(v) in (int, float) and math.isfinite(v) and abs(v) <= 1 for v in prediction
        )
        row.update(inference_returned_ns=now, prediction=prediction if valid else None)
        self.supervise(now)
        if self.stop_reason:
            row["status"] = "discard_stopped"
        elif now >= row["deadline_ns"]:
            row["status"] = "discard_deadline"
        elif now >= row["valid_until_ns"]:
            row["status"] = "discard_expired_lease"
        elif error:
            row.update(status="discard_inference_error", error=error)
        elif self.safety_reason(now):
            row.update(status="discard_safety", reason=self.safety_reason(now))
        elif not valid:
            row["status"] = "discard_invalid_prediction"
        elif self.capture_epoch != work.observation.epoch:
            row["status"] = "discard_capture_epoch"
        elif reason := self.observation_reason(now, work.observation):
            row["status"] = "discard_" + reason
        else:
            steer, longitudinal = prediction
            cfg = self.config
            command = Command(
                round(max(-cfg.max_steer, min(cfg.max_steer, steer)) * 32767),
                round(max(0, min(cfg.max_throttle, longitudinal)) * 255),
                round(max(0, min(cfg.max_brake, -longitudinal)) * 255),
            )
            try:
                self._send(now, command, "policy", work)
            except Exception:
                row["status"] = "send_failed"
                self.stop(now, "send_failed")
            else:
                row["status"] = "accepted"
                self.lease_ns = row["valid_until_ns"]
                self.last_accepted_ns = now
        self.notify("decision_result", dict(row))
