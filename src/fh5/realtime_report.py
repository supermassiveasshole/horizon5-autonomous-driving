"""Offline deadline timeline. Display previews never feed the actor."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.capture_metrics import percentiles
from fh5.numeric_images import asset
from fh5.realtime import MAX_REALTIME_REPORT_BYTES, RealtimeConfig

if TYPE_CHECKING:
    from fh5.experiment import RunResult


def write_realtime_result(
    directory: Path, result: dict[str, Any], config: RealtimeConfig
) -> RunResult:
    from fh5.experiment import RunResult

    decisions = result["decisions"]
    result["configuration"] = {**asdict(config), "pixels": config.pixels.metadata()}
    accepted = [d for d in decisions if d["status"] == "accepted"]
    sent = {
        c["decision_id"]: c["returned_ns"]
        for c in result["commands"]
        if c["status"] == "sent" and c["decision_id"] is not None
    }
    end = result["ended_ns"]
    longest = 0
    skip_start = None
    for row in decisions:
        if row["status"] != "accepted" and skip_start is None:
            skip_start = row["decision_ns"]
        elif row["status"] == "accepted" and skip_start is not None:
            longest = max(longest, row["decision_ns"] - skip_start)
            skip_start = None
        frames = row.get("frames", [])
        row["adjacent_delta_ms"] = [
            (b["source_time_ns"] - a["source_time_ns"]) / 1e6 for a, b in zip(frames, frames[1:])
        ]
    if skip_start is not None:
        longest = max(longest, end - skip_start)
    result["metrics"] = {
        "interpretation": "inference and send-return timing; game application latency unverified",
        "decision_counts": dict(Counter(d["status"] for d in decisions)),
        "accepted_fraction": len(accepted) / len(decisions) if decisions else None,
        "effective_hz": len(accepted) / max(1e-9, (end - result["started_ns"]) / 1e9),
        "maximum_consecutive_skip_ms": longest / 1e6,
        "source_to_sendable_ms": percentiles(
            [
                (d["inference_returned_ns"] - d["frames"][-1]["source_time_ns"]) / 1e6
                for d in accepted
            ]
        ),
        "source_to_send_return_ms": percentiles(
            [
                (sent[d["decision_id"]] - d["frames"][-1]["source_time_ns"]) / 1e6
                for d in accepted
                if d["decision_id"] in sent
            ]
        ),
        "inference_to_result_ms": percentiles(
            [
                (d.get("worker_returned_ns", d["inference_returned_ns"]) - d["decision_ns"]) / 1e6
                for d in decisions
                if "inference_returned_ns" in d
            ]
        ),
        "decision_interval_ms": percentiles(
            [(b["decision_ns"] - a["decision_ns"]) / 1e6 for a, b in zip(decisions, decisions[1:])]
        ),
    }
    for name, start, finish in (
        ("worker_queue_ms", "decision_ns", "worker_started_ns"),
        ("worker_features_and_inference_ms", "worker_started_ns", "worker_returned_ns"),
        ("result_supervision_ms", "worker_returned_ns", "inference_returned_ns"),
    ):
        result["metrics"][name] = percentiles(
            [
                (row[finish] - row[start]) / 1e6
                for row in decisions
                if start in row and finish in row
            ]
        )
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if result["version"] == 2 and len(payload) > MAX_REALTIME_REPORT_BYTES:
        result["evidence"].update(exact_replay_eligible=False, reason="replay_report_size_limit")
        payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
    (directory / "report.json").write_bytes(payload)
    if result["version"] == 2:
        (directory / "realtime-manifest.json").write_text(
            json.dumps(
                {"version": 1, "report_sha256": hashlib.sha256(payload).hexdigest()}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    display = json.loads(json.dumps(result))
    for row in display["decisions"]:
        row["preview_urls"] = [
            asset(directory, p).as_uri() if p else None
            for p in (row.get("archive") or {}).get("previews", [])
        ]
    data = json.dumps(display, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("realtime-report.html").read_text(encoding="utf-8")
    path = directory / "report.html"
    path.write_text(template.replace("/*REALTIME_DATA*/null", data), encoding="utf-8")
    return RunResult({"source_kind": result["evidence_kind"]}, [], [], {"realtime": result}, path)
