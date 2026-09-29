"""Read declared, hash-bound prior inputs; never infer commands from telemetry."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any


def read_actions(directory: Path, packets_hash: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "absent",
        "errors": [],
        "rows": [],
        "times": [],
        "sha256": None,
    }
    manifest_path = directory / "action-history.json"
    journal = directory / "actions.jsonl"
    if not manifest_path.exists() and not journal.exists():
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        content = journal.read_bytes()
        result["sha256"] = hashlib.sha256(content).hexdigest()
        if (
            manifest["version"] != 1
            or type(manifest["version"]) is not int
            or manifest["mapping_version"] != "dual-axis-v1"
            or manifest["packets_sha256"] != packets_hash
            or manifest["actions_sha256"] != result["sha256"]
        ):
            raise ValueError("Action history identity, mapping or hash mismatch")
        rows = [json.loads(line) for line in content.splitlines()]
        if len(rows) > 100_000:
            raise ValueError("Action history exceeds 100000 records")
        seen = set()
        for row in rows:
            if set(row) != {
                "occurred_ns",
                "available_ns",
                "telemetry_segment",
                "source",
                "steer",
                "longitudinal",
            }:
                raise ValueError("Invalid action history fields")
            if any(
                type(row[k]) is not int or row[k] < 0
                for k in ("occurred_ns", "available_ns", "telemetry_segment")
            ):
                raise ValueError("Invalid action history clock or segment")
            if row["occurred_ns"] > row["available_ns"] or row["occurred_ns"] in seen:
                raise ValueError("Ambiguous or noncausal action time")
            seen.add(row["occurred_ns"])
            if row["source"] not in ("human_input", "controller_sent"):
                raise ValueError("Unknown action source")
            if any(
                type(row[k]) not in (int, float)
                or not math.isfinite(row[k])
                or not -1 <= row[k] <= 1
                for k in ("steer", "longitudinal")
            ):
                raise ValueError("Invalid dual-axis action")
        arrivals = sorted(rows, key=lambda row: row["available_ns"])
        known = []
        latest: dict[str, Any] | None = None
        for row in arrivals:
            if latest is None or row["occurred_ns"] > latest["occurred_ns"]:
                latest = row
            known.append(latest)
        result.update(
            status="available",
            rows=known,
            times=[row["available_ns"] for row in arrivals],
            mapping_version=manifest["mapping_version"],
            source_validation="declared_source_not_live_calibration_evidence",
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status="invalid_asset", errors=[str(error)])
    return result


def action_slots(
    history: dict[str, Any],
    tick: int,
    boundaries: list[int],
    segment: int | None,
    offsets: list[float],
    max_age: float,
) -> list[dict[str, Any] | None]:
    rows = history["rows"]
    times = history["times"]
    boundary_index = bisect_right(boundaries, tick) - 1
    boundary = boundaries[boundary_index] if boundary_index >= 0 else -1
    slots: list[dict[str, Any] | None] = []
    for offset in offsets:
        cutoff = tick - int(offset * 1e6)
        index = bisect_left(times, cutoff) - 1
        # The arrival prefix caches the latest occurred action known at that time.
        # Strict cutoff excludes current labels; late older arrivals cannot overwrite it.
        if index < 0:
            slots.append(None)
            continue
        row = rows[index]
        age = (tick - row["occurred_ns"]) / 1e6
        reasons = []
        if row["occurred_ns"] < boundary or row["telemetry_segment"] != segment:
            reasons.append("history_discontinuity")
        if age > offset + max_age:
            reasons.append("stale_action")
        slots.append({**row, "age_ms": age, "valid": not reasons, "reasons": reasons})
    return slots
