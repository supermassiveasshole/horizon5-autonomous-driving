"""Offline recovery supervision over explicit synthetic task/UI signals.

Directives are intentions, never controller calls or evidence of game response.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fh5.result import RunResult


@dataclass(frozen=True)
class RecoveryReplay:
    config_file: Path
    trace_file: Path
    output_dir: Path


def _number(value: Any, lower: float, upper: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper


def _validate_config(config: Any) -> None:
    limits = {
        "phase_timeout_s": (0.1, 120),
        "no_progress_timeout_s": (0.1, 120),
        "session_timeout_s": (0.1, 1800),
        "max_frame_age_s": (0.001, 1),
    }
    if not isinstance(config, dict) or set(config) != {
        "version",
        "rewind_available",
        "max_rewinds",
        "warmup_frames",
        *limits,
    }:
        raise ValueError("Recovery config has missing or unknown fields")
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("Unsupported recovery config version")
    if type(config["rewind_available"]) is not bool:
        raise ValueError("Synthetic rewind availability must be boolean")
    for key, bounds in limits.items():
        if not _number(config[key], *bounds):
            raise ValueError(f"Invalid recovery limit: {key}")
    for key, upper in (("max_rewinds", 10), ("warmup_frames", 30)):
        lower = 0 if key == "max_rewinds" else 2
        if type(config[key]) is not int or not lower <= config[key] <= upper:
            raise ValueError(f"Invalid recovery count: {key}")


def _validate_trace(trace: Any) -> None:
    if (
        not isinstance(trace, dict)
        or set(trace) != {"version", "source_kind", "inputs"}
        or type(trace["version"]) is not int
        or trace["version"] != 1
        or trace["source_kind"] != "synthetic"
        or not isinstance(trace["inputs"], list)
        or not trace["inputs"]
    ):
        raise ValueError("Recovery replay only accepts a nonempty v1 synthetic signal trace")
    extras = {
        "sample": {
            "observed_s",
            "packet_index",
            "game_time_ms",
            "progress_m",
            "conditions_valid",
            "route_valid",
            "neutral",
        },
        "failure": {"reason"},
        "released": {"request_id"},
        "rewind_ready": {"request_id", "observed_s"},
        "resumed": {"request_id", "observed_s"},
        "history_ready": {"request_id", "generation"},
        "tick": set(),
        "finish": set(),
        "stop": set(),
        "fault": set(),
    }
    previous_time = 0.0
    for row in trace["inputs"]:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("kind"), str)
            or row["kind"] not in extras
        ):
            raise ValueError("Unknown recovery signal")
        if set(row) != {"at_s", "kind", *extras[row["kind"]]}:
            raise ValueError("Missing or unknown recovery signal fields")
        if not _number(row["at_s"], previous_time, 1e12):
            raise ValueError("Trace time must be finite, nonnegative and ordered")
        previous_time = row["at_s"]
        if "observed_s" in row and not _number(row["observed_s"], 0, row["at_s"]):
            raise ValueError("Observation time must be causal")
        for key in ("request_id", "generation", "packet_index", "game_time_ms"):
            if key in row and (type(row[key]) is not int or not 0 <= row[key] <= 2**63 - 1):
                raise ValueError(f"Invalid signal integer: {key}")
        if row["kind"] == "sample":
            if not _number(row["progress_m"], 0, 1e9) or any(
                type(row[k]) is not bool for k in ("conditions_valid", "route_valid", "neutral")
            ):
                raise ValueError("Invalid task-manager sample")
        if row["kind"] == "failure" and row["reason"] not in (
            "off_road",
            "missed_checkpoint",
            "unrecoverable_heading",
            "stalled",
        ):
            raise ValueError("Unknown confirmed driving failure")


class _Supervisor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.phase = "driving"
        self.now = 0.0
        self.generation = 0
        self.directives: list[dict[str, Any]] = []
        self.recoveries: list[dict[str, Any]] = []
        self.fragments: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.warmup: list[dict[str, Any]] = []
        self.history_ready = False
        self.finish_observed = False
        self.reason = "trace_ended"
        self.previous: dict[str, Any] | None = None
        self.reset_at = 0.0
        self.started_at: float | None = None
        self.progress_at = 0.0
        self.frontier = 0.0

    def command(self, name: str, **fields: Any) -> int:
        identity = len(self.directives) + 1
        self.directives.append(
            {"request_id": identity, "at_s": self.now, "command": name, **fields}
        )
        return identity

    def open_fragment(self, row: dict[str, Any]) -> None:
        parent = self.fragments[-1]["fragment_id"] if self.fragments else None
        self.current = {
            "fragment_id": f"segment-{len(self.fragments)}",
            "parent_fragment_id": parent,
            "generation": self.generation,
            "start_s": self.now,
            "outcome": "open",
            "packet_indices": [row["packet_index"]],
        }
        self.fragments.append(self.current)
        self.progress_at = self.now
        self.frontier = row["progress_m"]

    def close_fragment(self, outcome: str) -> None:
        if self.current is not None:
            self.current.update(outcome=outcome, end_s=self.now)
        self.current = None

    def stop(self, reason: str) -> None:
        if self.phase == "stopped":
            return
        self.close_fragment("truncated")
        if self.recoveries and self.recoveries[-1]["status"] == "pending":
            self.recoveries[-1].update(
                status="failed",
                reason=reason,
                end_s=self.now,
                wall_seconds=self.now - self.recoveries[-1]["start_s"],
            )
        self.reason = reason
        self.command("release")
        self.phase = "stopped"

    def fresh(self, row: dict[str, Any], since: float = 0.0) -> bool:
        observed = row.get("observed_s", -1)
        return bool(
            since <= observed <= self.now and self.now - observed <= self.config["max_frame_age_s"]
        )

    def acknowledgement_valid(self, row: dict[str, Any]) -> bool:
        expected = {
            "released": "releasing",
            "rewind_ready": "rewind_pending",
            "resumed": "resume_pending",
            "history_ready": "warmup",
        }
        if row["kind"] not in expected:
            return True
        if (
            self.phase != expected[row["kind"]]
            or not self.directives
            or row.get("request_id") != self.directives[-1]["request_id"]
        ):
            return False
        if row["kind"] in ("rewind_ready", "resumed"):
            return self.fresh(row, self.directives[-1]["at_s"])
        return row["kind"] != "history_ready" or row.get("generation") == self.generation

    def accept_sample(self, row: dict[str, Any]) -> None:
        usable = self.fresh(row, self.reset_at) and row["conditions_valid"] and row["route_valid"]
        if self.phase == "warmup":
            if not usable or not row["neutral"] or not self.history_ready:
                self.warmup.clear()
                return
            if self.warmup:
                previous = self.warmup[-1]
                if row["game_time_ms"] == previous["game_time_ms"]:
                    return
                if (
                    row["game_time_ms"] < previous["game_time_ms"]
                    or row["observed_s"] <= previous["observed_s"]
                    or row["packet_index"] <= previous["packet_index"]
                ):
                    self.warmup.clear()
                    return
            self.warmup.append(row)
            if len(self.warmup) >= self.config["warmup_frames"]:
                self.recoveries[-1].update(
                    status="recovered",
                    end_s=self.now,
                    wall_seconds=self.now - self.recoveries[-1]["start_s"],
                )
                self.open_fragment(row)
                self.command(
                    "allow_driving",
                    generation=self.generation,
                    packet_index=row["packet_index"],
                    history_start_s=self.reset_at,
                )
                self.previous = row
                self.phase = "driving"
        elif self.phase == "driving":
            if not usable or (
                self.previous is not None
                and (
                    row["game_time_ms"] < self.previous["game_time_ms"]
                    or row["observed_s"] <= self.previous["observed_s"]
                    or row["packet_index"] <= self.previous["packet_index"]
                )
            ):
                self.stop("invalid_forward_sample")
                return
            if self.previous and row["game_time_ms"] == self.previous["game_time_ms"]:
                return
            if self.current is None:
                self.open_fragment(row)
            else:
                self.current["packet_indices"].append(row["packet_index"])
            self.previous = row
            if row["progress_m"] >= self.frontier + 0.01:
                self.frontier = row["progress_m"]
                self.progress_at = self.now

    def fail(self, reason: str) -> None:
        self.failures.append({"at_s": self.now, "reason": reason})
        self.close_fragment("failed")
        self.recoveries.append({"start_s": self.now, "status": "pending"})
        if not self.config["rewind_available"]:
            self.stop("rewind_unavailable")
        elif sum(d["command"] == "rewind" for d in self.directives) >= self.config["max_rewinds"]:
            self.stop("rewind_limit")
        else:
            self.command("release")
            self.phase = "releasing"

    def advance(self, at: float) -> None:
        if self.started_at is None:
            self.started_at = at
        while self.phase != "stopped":
            limits = [(self.started_at + self.config["session_timeout_s"], "session_timeout")]
            if self.phase != "driving":
                limits.append(
                    (self.directives[-1]["at_s"] + self.config["phase_timeout_s"], "phase_timeout")
                )
            elif self.current is not None:
                limits.append((self.progress_at + self.config["no_progress_timeout_s"], "stalled"))
                assert self.previous is not None
                limits.append(
                    (
                        self.previous["observed_s"] + self.config["max_frame_age_s"],
                        "observation_timeout",
                    )
                )
            deadline, reason = min(limits)
            if at < deadline:
                self.now = at
                return
            self.now = deadline
            if reason == "stalled":
                self.fail(reason)
            else:
                self.stop(reason)

    def accept(self, row: dict[str, Any]) -> None:
        kind = row["kind"]
        if self.phase == "stopped":
            return
        self.advance(row["at_s"])
        if self.phase == "stopped":
            return
        if kind in ("stop", "fault"):
            self.stop("user_stop" if kind == "stop" else "interface_fault")
            return
        if not self.acknowledgement_valid(row):
            self.stop("invalid_acknowledgement")
            return
        if kind == "failure":
            if self.phase == "driving":
                self.fail(row["reason"])
            else:
                self.failures.append({"at_s": self.now, "reason": row["reason"]})
                self.stop("failure_during_recovery")
        elif kind == "released" and self.phase == "releasing":
            self.command("rewind")
            self.phase = "rewind_pending"
        elif kind == "rewind_ready" and self.phase == "rewind_pending":
            self.command("resume")
            self.phase = "resume_pending"
        elif kind == "resumed" and self.phase == "resume_pending":
            self.generation += 1
            self.warmup.clear()
            self.history_ready = False
            self.reset_at = self.now
            self.previous = None
            self.command(
                "reset_history",
                generation=self.generation,
                discard_before_s=self.now,
                components=["time", "route", "actions", "frames", "recurrent", "controller"],
            )
            self.phase = "warmup"
        elif kind == "history_ready" and self.phase == "warmup":
            self.history_ready = True
        elif kind == "sample":
            self.accept_sample(row)
        elif kind == "finish" and self.phase == "driving":
            self.finish_observed = True
            self.close_fragment("finish_observed")
            self.reason = "finish_observed"
            self.command("release")
            self.phase = "stopped"

    def result(self) -> dict[str, Any]:
        recovered = sum(r["status"] == "recovered" for r in self.recoveries)
        return {
            "version": 1,
            "source_kind": "synthetic",
            "commands_sent": False,
            "real_rewind_verified": False,
            "learning_transitions_exported": False,
            "phase": self.phase,
            "reason": self.reason,
            "directives": self.directives,
            "recoveries": self.recoveries,
            "fragments": self.fragments,
            "metrics": {
                "recovery_request_count": len(self.recoveries),
                "rewind_directive_count": sum(d["command"] == "rewind" for d in self.directives),
                "recovered_count": recovered,
                "recovery_success_fraction": recovered / len(self.recoveries)
                if self.recoveries
                else None,
                "recovery_wall_seconds": sum(r.get("wall_seconds", 0) for r in self.recoveries),
            },
            "attempt": {
                "attempt_id": "attempt-0",
                "outcome": "failed" if self.failures else "unverified",
                "failures": self.failures,
                "finish_observed": self.finish_observed,
                "no_rewind_completion": False,
            },
        }


def replay_recovery(request: RecoveryReplay) -> RunResult:
    from fh5.result import RunResult

    config_bytes, trace_bytes = request.config_file.read_bytes(), request.trace_file.read_bytes()
    config, trace = json.loads(config_bytes), json.loads(trace_bytes)
    _validate_config(config)
    _validate_trace(trace)
    supervisor = _Supervisor(config)
    for row in trace["inputs"]:
        supervisor.accept(row)
    supervisor.stop("trace_ended")
    recovery = supervisor.result()
    recovery["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    recovery["trace_sha256"] = hashlib.sha256(trace_bytes).hexdigest()
    output = request.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_bytes(config_bytes)
    (output / "trace.json").write_bytes(trace_bytes)
    content = json.dumps(recovery, ensure_ascii=False, indent=2, allow_nan=False)
    (output / "recovery.json").write_text(content + "\n", encoding="utf-8")
    report = output / "report.html"
    report.write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        "<title>恢复监督回放</title><h1>恢复监督回放</h1>"
        "<p>合成任务与 UI 信号；没有发送游戏输入，也不证明实机倒带可用。</p>"
        "<pre>" + html.escape(content) + "</pre></html>",
        encoding="utf-8",
    )
    return RunResult({"source_kind": "synthetic"}, [], [], {"recovery": recovery}, report)
