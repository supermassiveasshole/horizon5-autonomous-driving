"""Duration-weighted execution diagnostics from verified recorded timelines."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from fh5.capture.metrics import percentiles


def execution_metrics(report: dict[str, Any]) -> dict[str, Any]:
    started, ended = report["started_ns"], report["ended_ns"]
    decisions, commands = report["decisions"], report["commands"]
    if (
        type(started) is not int
        or type(ended) is not int
        or not 0 <= started < ended
        or any(
            type(d["decision_ns"]) is not int or not started <= d["decision_ns"] < ended
            for d in decisions
        )
        or any(a["decision_ns"] >= b["decision_ns"] for a, b in zip(decisions, decisions[1:]))
    ):
        raise ValueError("Invalid execution timeline")
    accepted = sum(d["status"] == "accepted" for d in decisions)
    available = sum("actor" in d and all(d["actor"]["image_mask"]) for d in decisions)
    skip_start = None
    longest_skip = 0
    for row in decisions:
        if row["status"] != "accepted" and skip_start is None:
            skip_start = row["decision_ns"]
        elif row["status"] == "accepted" and skip_start is not None:
            longest_skip = max(longest_skip, row["decision_ns"] - skip_start)
            skip_start = None
    if skip_start is not None:
        longest_skip = max(longest_skip, ended - skip_start)
    durations: dict[str, int] = defaultdict(int)
    owner, cursor = "unobserved", started
    previous = None
    holds = []
    variation = 0.0
    previous_direction = 0
    reversals = 0
    for command in commands:
        at = command["returned_ns"]
        if (
            command["status"] != "sent"
            or type(at) is not int
            or not cursor <= command["issued_ns"] <= at <= ended
        ):
            raise ValueError("Incomplete or unordered execution command timeline")
        durations[owner] += at - cursor
        if previous is not None:
            holds.append((at - cursor) / 1e6)
            if owner == command["owner"] == "policy":
                delta = command["sent"]["steer_i16"] - previous["sent"]["steer_i16"]
                variation += abs(delta) / 32767
                direction = (delta > 0) - (delta < 0)
                if direction and previous_direction and direction != previous_direction:
                    reversals += 1
                if direction:
                    previous_direction = direction
            else:
                previous_direction = 0
        owner, cursor, previous = command["owner"], at, command
    durations[owner] += ended - cursor
    elapsed_s = (ended - started) / 1e9
    return {
        "evidence_kind": report["evidence_kind"],
        "scope": "recorded synthetic/shadow timing; not actual game response",
        "duration_s": elapsed_s,
        "decision_count": len(decisions),
        "decision_statuses": dict(Counter(d["status"] for d in decisions)),
        "accepted_decisions": accepted,
        "effective_hz": accepted / elapsed_s,
        "maximum_consecutive_skip_ms": longest_skip / 1e6,
        "visual_known_decisions": available,
        "visual_available_fraction": available / len(decisions) if decisions else None,
        "visual_fraction_basis": "complete recorded actor inputs / all scheduled decisions; skipped visual availability unknown",
        "newest_image_age_ms": percentiles(
            [
                (d["decision_ns"] - d["frames"][-1]["source_time_ns"]) / 1e6
                for d in decisions
                if "actor" in d
            ]
        ),
        "owner_duration_s": {key: value / 1e9 for key, value in durations.items()},
        "owner_fraction": {key: value / (ended - started) for key, value in durations.items()},
        "longest_recorded_hold_ms": max(holds) if holds else None,
        "hold_time_basis": "successful_send_return_proxy",
        "steering_total_variation": variation,
        "steering_change_reversals": reversals,
        "smoothness_interpretation": "consecutive policy commands only; reversals do not establish needless oscillation",
    }


def combine_execution_metrics(executions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [e["metrics"] for e in executions if e["status"] == "bound_diagnostic"]
    seconds = sum(m["duration_s"] for m in metrics)
    decisions = sum(m["decision_count"] for m in metrics)
    accepted = sum(m["accepted_decisions"] for m in metrics)
    available = sum(m["visual_known_decisions"] for m in metrics)
    owners: dict[str, float] = defaultdict(float)
    for m in metrics:
        for name, duration in m["owner_duration_s"].items():
            owners[name] += duration
    return {
        "bound_runs": len(metrics),
        "unbound_runs": len(executions) - len(metrics),
        "duration_s": seconds,
        "decision_count": decisions,
        "accepted_decisions": accepted,
        "effective_hz": accepted / seconds if seconds else None,
        "visual_available_fraction": available / decisions if decisions else None,
        "owner_duration_s": dict(owners),
        "owner_fraction": {name: duration / seconds for name, duration in owners.items()}
        if seconds
        else {},
        "maximum_consecutive_skip_ms": max(
            (m["maximum_consecutive_skip_ms"] for m in metrics), default=None
        ),
        "scope": "verified diagnostic execution durations only; unbound runs excluded from timing, never from attempt counts",
    }
