"""Independent, evidence-linked verdicts for complete local driving attempts."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.observation.routes import load_route, locate_route
from fh5.reporting.telemetry import write_report

if TYPE_CHECKING:
    from fh5.result import RunResult

REVIEW_CHECKS = frozenset(
    {"wall_riding", "reset_boost", "grass_shortcut", "interventions", "conditions"}
)
EVENT_KINDS = frozenset(
    {
        "wall_riding",
        "reset_boost",
        "grass_shortcut",
        "incidental_contact",
        "reasonable_cut",
        "pause",
        "rewind",
        "restart",
        "stop",
        "human_takeover",
        "human_placement",
        "interface_fault",
        "driving_failure",
        "conditions_changed",
        "race_start",
        "game_finish",
        "navigation_recomputed",
        "navigation_hidden",
        "destination_changed",
    }
)
INVALID_EVENTS = frozenset(
    {
        "wall_riding",
        "reset_boost",
        "grass_shortcut",
        "human_takeover",
        "rewind",
        "conditions_changed",
    }
)


@dataclass(frozen=True)
class AttemptReplay:
    recording_dir: Path
    output_dir: Path
    task_file: Path
    evidence_file: Path | None = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(request: AttemptReplay, digest: str, count: int) -> dict[str, Any]:
    if request.evidence_file is None:
        return {"version": 1, "items": [], "events": [], "coverage": []}
    path = request.evidence_file
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Invalid attempt evidence version")
    if data.get("recording_sha256") != digest:
        raise ValueError("Attempt evidence belongs to a different recording")
    for name in ("items", "events", "coverage"):
        if not isinstance(data.get(name), list):
            raise ValueError(f"Evidence {name} must be an array")
    ids = set()
    for item in data["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("Evidence requires an id")
        if item["id"] in ids or not isinstance(item.get("path"), str):
            raise ValueError("Evidence ids must be unique and paths must be text")
        ids.add(item["id"])
        source = (path.parent / item["path"]).resolve()
        if not source.is_relative_to(path.parent.resolve()) or _sha(source) != item.get("sha256"):
            raise ValueError("Evidence path or hash is invalid")
    for row in [*data["coverage"], *data["events"]]:
        if not isinstance(row, dict) or row.get("source") not in (
            "independent_review",
            "policy_prediction",
        ):
            raise ValueError("Evidence needs an explicit independent or policy source")
        if not isinstance(row.get("reviewer"), str) or not row["reviewer"].strip():
            raise ValueError("Evidence reviewer is required")
        refs = row.get("evidence")
        if (
            not isinstance(refs, list)
            or not refs
            or any(not isinstance(r, str) or r not in ids for r in refs)
        ):
            raise ValueError("Review must reference frozen evidence items")
    for row in data["coverage"]:
        span = row.get("packet_range")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(n) is not int for n in span)
            or not 0 <= span[0] <= span[1] < count
        ):
            raise ValueError("Coverage range is outside the complete recording")
        if (
            not isinstance(row.get("checks"), list)
            or not row["checks"]
            or any(not isinstance(c, str) or c not in REVIEW_CHECKS for c in row["checks"])
        ):
            raise ValueError("Unknown coverage check")
    for row in data["events"]:
        index = row.get("packet_index")
        if (
            type(index) is not int
            or not 0 <= index < count
            or not isinstance(row.get("kind"), str)
            or row["kind"] not in EVENT_KINDS
            or row.get("status") not in ("confirmed", "suspected")
        ):
            raise ValueError("Invalid reviewed event")
        resume = row.get("resume_packet_index")
        if resume is not None and (
            row["kind"] not in {"pause", "rewind", "interface_fault"}
            or type(resume) is not int
            or not index < resume <= count
        ):
            raise ValueError("Invalid recovery end")
    return data


def _task(path: Path) -> dict[str, Any]:
    task = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(task, dict)
        or type(task.get("version")) is not int
        or task["version"] not in (1, 2)
        or task.get("scope") != "local"
    ):
        raise ValueError("Attempt review requires a supported local task")
    for name in ("task_id", "route_file", "route_sha256"):
        if not isinstance(task.get(name), str) or not task[name].strip():
            raise ValueError(f"Task requires {name}")
    if task["version"] == 1 and task.get("start_mode") != "manual_placement":
        raise ValueError("Only reviewed manual placement is supported in local validity v1")
    if task["version"] == 2:
        start = task.get("automatic_start")
        if (
            task.get("start_mode") != "automatic_event_ready"
            or task.get("control_owner") != "policy"
            or not isinstance(start, dict)
            or set(start) != {"event_file", "event_sha256", "handoff_timeout_s"}
            or not isinstance(start["event_file"], str)
            or not start["event_file"].strip()
            or not isinstance(start["event_sha256"], str)
            or len(start["event_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in start["event_sha256"])
            or type(start["handoff_timeout_s"]) not in (int, float)
            or not math.isfinite(start["handoff_timeout_s"])
            or not 0 < start["handoff_timeout_s"] <= 30
        ):
            raise ValueError("Automatic local task requires a frozen event and bounded handoff")
    if task.get("control_owner") not in ("human", "calibration", "policy"):
        raise ValueError("Task requires a known control owner")
    for name in ("expected_car_ordinal", "expected_pi"):
        if type(task.get(name)) is not int or task[name] <= 0:
            raise ValueError(f"Task requires positive integer {name}")
    for name in ("max_speed_kmh", "max_duration_s", "no_progress_timeout_s"):
        value = task.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Task requires finite positive {name}")
    hashes = task.get("geometry_source_sha256")
    if not isinstance(hashes, list) or any(
        not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
        for h in hashes
    ):
        raise ValueError("Task must declare SHA256 hashes of additional geometry sources")
    return task


def _missing_coverage(evidence: dict[str, Any], first: int, last: int) -> list[str]:
    missing = []
    for check in sorted(REVIEW_CHECKS):
        end = first - 1
        for a, b in sorted(
            row["packet_range"]
            for row in evidence["coverage"]
            if row["source"] == "independent_review" and check in row["checks"]
        ):
            if a <= end + 1:
                end = max(end, b)
        if end < last:
            missing.append(check)
    return missing


def _confirmed(event: dict[str, Any]) -> bool:
    return bool(event["source"] == "independent_review" and event["status"] == "confirmed")


def _fragment(
    group: list[dict[str, Any]], route: dict[str, Any], task: dict[str, Any], segment_id: str
) -> dict[str, Any]:
    # Locate only from the first local start. A later return cannot erase a failed path.
    start = None
    for i, sample in enumerate(group):
        sample.update(forward_segment_id=segment_id, attempt_phase="approach")
        locate_route([sample], route)
        r = sample["route"]
        if start is None and r["status"] == "matched" and r["reference_s_m"] <= 0.25:
            start = i
    local = group[start:] if start is not None else []
    route_events = locate_route(local, route)
    finish = next(
        (
            i
            for i, s in enumerate(local)
            if s["route"]["confirmed_progress_m"] >= route["length_m"] - 1e-6
        ),
        None,
    )
    if finish is not None:
        for s in local[finish + 1 :]:
            s["attempt_phase"] = "after_local_task"
        local = local[: finish + 1]
    for s in local:
        s["attempt_phase"] = "local_task"
    failures = set()
    pending = set()
    progress = 0.0
    last_progress_ns = local[0]["received_monotonic_ns"] if local else 0
    for s in local:
        r = s["route"]
        if r["status"] in {"missing_checkpoint", "checkpoint_order"}:
            failures.add(r["status"])
        elif r["status"] not in {"matched", "awaiting_checkpoint"}:
            pending.add("unconfirmed_path")
        if s["speed_kmh"] > task["max_speed_kmh"]:
            failures.add("speed_limit_exceeded")
        now = s["received_monotonic_ns"]
        if (now - local[0]["received_monotonic_ns"]) / 1e9 > task["max_duration_s"]:
            failures.add("task_timeout")
        if r["confirmed_progress_m"] > progress + 0.01:
            last_progress_ns = now
            progress = r["confirmed_progress_m"]
        if (now - last_progress_ns) / 1e9 > task["no_progress_timeout_s"]:
            failures.add("no_progress_timeout")
    return {
        "segment_id": segment_id,
        "packet_range": [group[0]["packet_index"], group[-1]["packet_index"]],
        "task_packet_range": [local[0]["packet_index"], local[-1]["packet_index"]]
        if local
        else None,
        "geometry_completed": finish is not None,
        "confirmed_progress_m": max((s["route"]["confirmed_progress_m"] for s in local), default=0),
        "duration_s": (local[-1]["received_monotonic_ns"] - local[0]["received_monotonic_ns"]) / 1e9
        if local
        else 0,
        "failures": sorted(failures),
        "pending": sorted(pending),
        "checkpoint_events": [
            e for e in route_events if local and e["packet_index"] <= local[-1]["packet_index"]
        ],
        "training_export_allowed": False,
    }


def _attempt(
    base: RunResult,
    task: dict[str, Any],
    route: dict[str, Any],
    evidence: dict[str, Any],
    digest: str,
    first: int,
    last: int,
) -> dict[str, Any]:
    attempt_id = f"{digest[:20]}:{first}"
    events = [e for e in evidence["events"] if first <= e["packet_index"] <= last]
    excluded_intervals = [
        (e["packet_index"], e.get("resume_packet_index", last + 1))
        for e in events
        if _confirmed(e) and e["kind"] in {"pause", "rewind", "interface_fault", "stop"}
    ]
    cuts = sorted(
        {
            e["packet_index"]
            for e in events
            if _confirmed(e)
            and e["kind"]
            in {"pause", "rewind", "restart", "interface_fault", "destination_changed"}
        }
        | {boundary for span in excluded_intervals for boundary in span}
    )
    samples = [s for s in base.samples if first <= s["packet_index"] <= last]
    groups: list[list[dict[str, Any]]] = []
    previous = None
    for sample in samples:
        key = (sample["segment"], bisect_right(cuts, sample["packet_index"]))
        if previous != key:
            groups.append([])
        groups[-1].append(sample)
        previous = key
    fragments = []
    excluded = []
    for group in groups:
        start = group[0]["packet_index"]
        segment_id = f"{attempt_id}:f{start}"
        if any(a <= start < b for a, b in excluded_intervals):
            excluded.append(
                {
                    "segment_id": segment_id,
                    "packet_range": [start, group[-1]["packet_index"]],
                    "reason": "recovery_or_stop",
                    "training_export_allowed": False,
                }
            )
            for sample in group:
                sample["attempt_phase"] = "recovery_excluded"
        else:
            fragments.append(_fragment(group, route, task, segment_id))
        for sample in group:
            sample.update(attempt_id=attempt_id, forward_segment_id=segment_id)
    missing = _missing_coverage(evidence, first, last)
    invalid = sorted({e["kind"] for e in events if e["kind"] in INVALID_EVENTS and _confirmed(e)})
    pending = sorted(
        {
            "suspected_" + e["kind"]
            for e in events
            if e["kind"]
            not in {"incidental_contact", "reasonable_cut", "race_start", "game_finish"}
            and not _confirmed(e)
        }
    )
    if missing:
        pending.append("independent_review_missing")
    if task["version"] == 2:
        pending.append("automatic_start_unverified")
    if any(e["kind"] == "destination_changed" and _confirmed(e) for e in events):
        pending.append("task_phase_changed")
    if any(e["kind"] == "pause" and _confirmed(e) for e in events):
        pending.append("pause")
    task_spans = [f["task_packet_range"] for f in fragments if f["task_packet_range"]]
    if any(
        e["kind"] == "human_placement"
        and _confirmed(e)
        and task_spans
        and e["packet_index"] > task_spans[0][0]
        for e in events
    ):
        invalid.append("human_takeover")
    if any(
        s["is_race_on"]
        and (
            s["car_ordinal"] != task["expected_car_ordinal"]
            or s["car_performance_index"] != task["expected_pi"]
        )
        for s in samples
    ):
        invalid.append("vehicle_mismatch")
    diagnostics = [
        e
        for e in base.events
        if e.get("packet_index") is None or first <= e["packet_index"] <= last
    ]
    if task_spans and any(
        e["kind"] in {"paused", "resumed"}
        and e.get("packet_index") is not None
        and e["packet_index"] > task_spans[0][0]
        for e in diagnostics
    ):
        pending.append("activity_interrupted")
    fault_kinds = {
        "receive_gap",
        "receive_clock_discontinuity",
        "unsupported_packet",
        "invalid_value",
        "corrupt_record",
        "incomplete_tail",
        "capture_source_error",
        "capture_interrupted",
        "capture_recording",
        "no_telemetry",
        "event_evidence_incomplete",
        "control_evidence_incomplete",
    }
    faults = {e["kind"] for e in diagnostics if e["kind"] in fault_kinds}
    control = base.summary.get("control")
    if control:
        if (
            control.get("artifact_errors")
            or control.get("release_sent") is not True
            or any(c["status"] == "failed" for c in control["commands"])
        ):
            faults.add("control_evidence_incomplete")
        if control.get("stop_reason") in {
            "interface_error",
            "telemetry_stale",
            "incomplete",
            "control_stalled",
            "receive_time_jump",
            "invalid_telemetry",
            "game_time_stalled",
        }:
            faults.add("control_interface_error")
        elif control.get("stop_reason") != "completed":
            pending.append("control_stopped_" + str(control.get("stop_reason")))
    if any(e["kind"] == "interface_fault" and _confirmed(e) for e in events):
        faults.add("interface_fault")
    clock_start = None
    clock_key = None
    for sample in samples:
        key = (sample["segment"], sample["game_timestamp_ms"])
        if not sample["is_race_on"] or key != clock_key:
            clock_start = sample["received_monotonic_ns"]
            clock_key = key
        elif (
            clock_start is not None and (sample["received_monotonic_ns"] - clock_start) / 1e9 > 0.5
        ):
            faults.add("game_clock_frozen")
    # Telemetry activity/position changes do not identify pause, rewind, or a collision.
    if any(
        e["kind"] in {"game_time_discontinuity", "position_jump", "game_clock_wrap"}
        and not any(
            _confirmed(v) and v["kind"] == "restart" and v["packet_index"] == e["packet_index"]
            for v in events
        )
        for e in diagnostics
    ):
        pending.append("unexplained_discontinuity")
    if base.metadata["control_source"] != task["control_owner"]:
        pending.append("control_owner_mismatch")
    if digest in [route["source"]["packets_sha256"], *task["geometry_source_sha256"]]:
        pending.append("geometry_source_reused")
    failures = {r for f in fragments for r in f["failures"]}
    if any(
        _confirmed(e)
        and (
            e["kind"] == "driving_failure"
            or (
                e["kind"] == "stop"
                and not any(
                    f["geometry_completed"] and f["task_packet_range"][1] < e["packet_index"]
                    for f in fragments
                )
            )
        )
        for e in events
    ):
        failures.add("reviewed_stop_or_failure")
    pending.extend(r for f in fragments for r in f["pending"])
    if not task_spans:
        pending.append("local_start_missing")
    completed = any(f["geometry_completed"] for f in fragments)
    reasons = sorted(
        set(invalid + pending)
        | faults
        | failures
        | (set() if completed else {"task_not_completed"})
    )
    outcome = (
        "interface_error"
        if faults
        else "invalid"
        if invalid
        else "driving_failed"
        if failures
        else "pending_review"
        if pending
        else "valid_complete"
        if completed
        else "driving_failed"
    )
    return {
        "attempt_id": attempt_id,
        "packet_range": [first, last],
        "task_packet_range": task_spans[0] if len(task_spans) == 1 else None,
        "start_mode": task["start_mode"] if first == 0 else "reviewed_restart",
        "control_owner": base.metadata["control_source"],
        "outcome": outcome,
        "reasons": reasons,
        "pending_checks": sorted(set(pending)),
        "interface_faults": sorted(faults),
        "uncovered_checks": missing,
        "task_completed": completed,
        "confirmed_progress_m": max((f["confirmed_progress_m"] for f in fragments), default=0),
        "unattended": False,
        "record_eligible": outcome == "valid_complete",
        "contact_count": sum(e["kind"] == "incidental_contact" and _confirmed(e) for e in events),
        "evidence_mode": "manual_review" if not missing else "incomplete",
        "forward_segments": fragments,
        "excluded_spans": excluded,
        "events": events,
        "diagnostics": diagnostics,
        "control_stop": {
            k: control.get(k) for k in ("stop_reason", "release_sent", "artifact_errors")
        }
        if control
        else None,
        "automatic_promotion_allowed": False,
        "formal_result": {
            "outcome": "invalid" if invalid else "pending_review",
            "required_evidence": [
                "full_race_start",
                "ordered_official_checkpoints",
                "game_finish_signal",
                "whole_attempt_validity",
            ],
            "reasons": invalid
            or ["local_task_only", "full_event_start_and_checkpoints_and_finish_required"],
        },
    }


def review_attempts(request: AttemptReplay) -> RunResult:
    from fh5.experiment import run_experiment
    from fh5.result import RunResult
    from fh5.telemetry.packet import Replay

    task = _task(request.task_file)
    route_file = request.task_file.parent / task["route_file"]
    if _sha(route_file) != task["route_sha256"]:
        raise ValueError("Frozen route hash does not match the task")
    route = load_route(route_file)
    digest = _sha(request.recording_dir / "packets.jsonl")
    evidence = _evidence(
        request, digest, len((request.recording_dir / "packets.jsonl").read_bytes().splitlines())
    )
    request.output_dir.mkdir(parents=True, exist_ok=False)
    base = run_experiment(Replay(request.recording_dir, request.output_dir / "telemetry.html"))
    count = base.summary["packet_count"]
    boundaries = sorted(
        {
            0,
            count,
            *(
                e["packet_index"]
                for e in evidence["events"]
                if e["kind"] == "restart" and _confirmed(e)
            ),
        }
    )
    attempts = [
        _attempt(base, task, route, evidence, digest, first, end - 1)
        for first, end in zip(boundaries, boundaries[1:])
    ] or [_attempt(base, task, route, evidence, digest, 0, -1)]
    events = [*base.events, *({**e, "kind": "reviewed_" + e["kind"]} for e in evidence["events"])]
    review = {
        "version": 1,
        "rules_version": "local-validity-v3" if task["version"] == 2 else "local-validity-v2",
        "task": task,
        "recording_packet_count": base.summary["packet_count"],
        "source_kind": base.metadata["source_kind"],
        "source_hashes": {
            "packets": digest,
            "session": _sha(request.recording_dir / "session.json"),
            "task": _sha(request.task_file),
            "route": _sha(route_file),
            "review": _sha(request.evidence_file) if request.evidence_file else None,
            "control_artifacts": {
                name: _sha(request.recording_dir / name)
                for name in ("control.json", "commands.jsonl")
                if (request.recording_dir / name).is_file()
            },
        },
        "attempts": attempts,
        "evidence": evidence,
    }
    (request.output_dir / "task.json").write_bytes(request.task_file.read_bytes())
    if request.evidence_file is not None:
        (request.output_dir / "review.json").write_bytes(request.evidence_file.read_bytes())
        for item in evidence["items"]:
            source = request.evidence_file.parent / item["path"]
            destination = request.output_dir / "evidence" / (item["sha256"] + source.suffix)
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(source.read_bytes())
            item["frozen_path"] = destination.relative_to(request.output_dir).as_posix()
    (request.output_dir / "route-snapshot.json").write_text(
        json.dumps(route, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (request.output_dir / "attempts.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = {**base.summary, "route": route, "attempt_review": review}
    report_path = request.output_dir / "report.html"
    write_report(
        report_path,
        {"metadata": base.metadata, "samples": base.samples, "events": events, "summary": summary},
    )
    return RunResult(base.metadata, base.samples, events, summary, report_path)
