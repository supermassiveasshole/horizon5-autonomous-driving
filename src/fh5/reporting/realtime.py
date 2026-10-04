"""Offline deadline timeline. Display previews never feed the actor."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.artifacts.document import write_replay_document
from fh5.artifacts.io import asset, encode, sha256_file, write_file
from fh5.artifacts.json_view import JsonArray, write_json
from fh5.capture.metrics import percentiles
from fh5.driving.realtime.model import RealtimeConfig
from fh5.learning.sac.context import PROPOSAL_CONTEXT
from fh5.reporting.presentation import optional_report

if TYPE_CHECKING:
    from fh5.result import RunResult


def write_realtime_result(
    directory: Path, result: dict[str, Any], config: RealtimeConfig
) -> RunResult:
    from fh5.result import RunResult

    result["configuration"] = {**asdict(config), "pixels": config.pixels.metadata()}
    evidence = directory / "report.json"
    write_replay_document(evidence, result)
    if result["version"] == 2:
        write_file(
            directory / "realtime-manifest.json",
            encode({"version": 1, "report_sha256": sha256_file(evidence)}),
        )
    try:
        _add_realtime_metrics(result)
    except (OSError, MemoryError) as error:
        result["metrics"] = {
            "status": "unavailable",
            "error": f"{type(error).__name__}: {error}",
        }
    path = optional_report(
        directory / "report.html",
        "数值决策与动作有效期",
        result,
        fallback=evidence,
        render=lambda path, summary: _write_realtime_html(path, summary, directory),
    )
    return RunResult({"source_kind": result["evidence_kind"]}, [], [], {"realtime": result}, path)


def _add_realtime_metrics(result: dict[str, Any]) -> None:
    decisions = result["decisions"]
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
    if result.get("model", {}).get("command_context") == PROPOSAL_CONTEXT:
        proposed = {
            p["decision_id"]: p["proposed_ns"]
            for p in result["proposals"]
            if p["owner"] == "policy"
        }
        result["metrics"]["interpretation"] = (
            "counterfactual proposal timing only; no actuator sends or executed action history"
        )
        result["metrics"]["source_to_proposal_ms"] = percentiles(
            [
                (proposed[d["decision_id"]] - d["frames"][-1]["source_time_ns"]) / 1e6
                for d in accepted
            ]
        )
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


def _write_realtime_html(path: Path, result: dict[str, Any], directory: Path) -> None:
    def display_decision(original: dict[str, Any]) -> dict[str, Any]:
        row = dict(original)
        frames = row.get("frames", [])
        row["adjacent_delta_ms"] = [
            (b["source_time_ns"] - a["source_time_ns"]) / 1e6 for a, b in zip(frames, frames[1:])
        ]
        row["preview_urls"] = [
            asset(directory, p).as_uri() if p else None
            for p in (row.get("archive") or {}).get("previews", [])
        ]
        return row

    display = dict(result, decisions=JsonArray(map(display_decision, result["decisions"])))
    template = Path(__file__).with_name("realtime-report.html").read_text(encoding="utf-8")
    before, after = template.split("/*REALTIME_DATA*/null")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(before)
        write_json(stream, display, script_safe=True)
        stream.write(after)
