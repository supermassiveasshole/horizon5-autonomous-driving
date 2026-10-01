"""Offline integrity and frozen-model replay; never re-executes timing or actions."""

from __future__ import annotations

import hashlib
import html
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.artifact_io import read_json, sha256_file
from fh5.numeric_images import (
    DecisionActor,
    NumericDecision,
    PixelContract,
    asset,
    decision_prediction,
    validate_decision,
)
from fh5.numeric_recording import numeric_features, read_numeric_frame
from fh5.realtime import RealtimeNumericReplay
from fh5.sac_actions import ActionBounds, ActionSupportUnavailable
from fh5.sac_replay import command_action

if TYPE_CHECKING:
    from fh5.experiment import RunResult


def read_realtime_recording(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "realtime-manifest.json")
    path = root / "report.json"
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or sha256_file(path) != manifest.get("report_sha256")
    ):
        raise ValueError("Real-time report hash mismatch or unsupported manifest")
    report: dict[str, Any] = read_json(path)
    if (
        not isinstance(report, dict)
        or report["version"] != 2
        or type(report["commands_sent_to_game"]) is not bool
        or (report["evidence_kind"] != "native" and report["commands_sent_to_game"])
        or report["evidence_kind"] not in ("synthetic", "shadow", "native")
    ):
        raise ValueError("Unsupported real-time numerical recording")
    return report


def read_realtime_journal(
    root: Path, report: dict[str, Any], *, time_limit_s: float | None = None
) -> list[dict[str, Any]]:
    """Verify decisions, commands and stop; optionally enforce the requested deadline."""
    reference = report["journal"]
    if (
        reference["dropped"]
        or reference["missing_sequences"]
        or reference["error"]
        or reference["external_sink"]
        or not reference["resources_released"]
        or reference["offered"] != reference["written"]
    ):
        raise ValueError("Incomplete real-time journal")
    path = asset(root, reference["path"])
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != reference["sha256"]:
            raise ValueError("Real-time journal hash mismatch")
    events: dict[str, list[dict[str, Any]]] = {}
    sequences: set[int] = set()
    commands = []
    packets = []
    stops = []
    with path.open("rb") as stream:
        for line in stream:
            event = json.loads(line)
            if (
                not isinstance(event, dict)
                or not isinstance(event.get("kind"), str)
                or not isinstance(event.get("data"), dict)
            ):
                raise ValueError("Malformed real-time journal event")
            sequence = event["sequence"]
            if (
                type(sequence) is not int
                or not 0 <= sequence < reference["offered"]
                or sequence in sequences
            ):
                raise ValueError("Duplicate or invalid journal sequence")
            sequences.add(sequence)
            row = event["data"]
            if event["kind"].startswith("decision_"):
                events.setdefault(row["decision_id"], []).append(event)
            elif event["kind"] == "command":
                commands.append((sequence, row))
            elif event["kind"] == "packet":
                packets.append((sequence, row))
            elif event["kind"] == "stop":
                stops.append(row)
    if len(sequences) != reference["offered"]:
        raise ValueError("Missing journal events")
    if (
        len(stops) != 1
        or type(stops[0].get("at_ns")) is not int
        or not 0 <= stops[0]["at_ns"] <= report["ended_ns"]
        or not isinstance(stops[0].get("reason"), str)
        or not stops[0]["reason"]
        or stops[0]["reason"] != report["stop_reason"]
    ):
        raise ValueError("Original journal stop is missing, ambiguous or differs from report")
    if (
        time_limit_s is not None
        and report["stop_reason"] == "time_limit"
        and stops[0]["at_ns"] - report["started_ns"] < int(time_limit_s * 1e9)
    ):
        raise ValueError("Recorded time-limit duration is shorter than its request")
    if [r for _, r in sorted(commands)] != report["commands"]:
        raise ValueError("Recorded commands differ from journal")
    ids = [d["decision_id"] for d in report["decisions"]]
    if len(ids) != len(set(ids)) or set(ids) != set(events):
        raise ValueError("Missing or duplicated decision outcomes")
    for row in report["decisions"]:
        records = sorted(events[row["decision_id"]], key=lambda event: event["sequence"])
        expected = (
            ["decision_started", "decision_result"] if "actor" in row else ["decision_skipped"]
        )
        if [e["kind"] for e in records] != expected:
            raise ValueError("Incomplete decision outcome journal")
        if "actor" in row:
            started = records[0]["data"]
            if (
                started.get("status") != "pending"
                or started.get("prediction") is not None
                or started.get("command_context") != row.get("command_context")
                or any(
                    started[key] != row[key]
                    for key in (
                        "index",
                        "decision_id",
                        "epoch",
                        "decision_ns",
                        "deadline_ns",
                        "valid_until_ns",
                        "actor",
                        "frames",
                        "safety_at_decision",
                        "telemetry_received_ns",
                    )
                )
            ):
                raise ValueError("Journal scheduling snapshot differs from outcome")
        final = records[-1]["data"]
        if any(row.get(key) != value for key, value in final.items()):
            raise ValueError("Decision outcome differs from journal")
    return [row for _, row in sorted(packets)]


def _valid_prediction(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(type(v) in (float, int) and math.isfinite(v) and abs(v) <= 1 for v in value)
    )


def _verify_action_wait(row: dict[str, Any], report: dict[str, Any]) -> None:
    """Recompute a supervisor wait from the frozen bounds and actual send history."""
    if (
        report["model"].get("command_context") != "successful-send-return-proxy-v1"
        or row.get("prediction") is not None
        or any(
            key in row for key in ("actor", "error", "inference_returned_ns", "worker_started_ns")
        )
        or any(c.get("decision_id") == row["decision_id"] for c in report["commands"])
    ):
        raise ValueError("Invalid action support wait outcome")
    prior = [
        (i, c)
        for i, c in enumerate(report["commands"])
        if c["status"] == "sent" and c["returned_ns"] < row["decision_ns"]
    ]
    if not prior:
        raise ValueError("Action support wait lacks prior command context")
    index, command = prior[-1]
    context = {
        "version": 1,
        "command_index": index,
        **{key: command[key] for key in ("sent", "issued_ns", "returned_ns", "owner")},
    }
    if (
        row.get("command_context") != context
        or command["owner"] not in ("initial_neutral", "policy", "lease_expiry")
        or not 0 <= command["issued_ns"] <= command["returned_ns"]
    ):
        raise ValueError("Action support wait differs from successful command context")
    bounds = ActionBounds(**report["model"]["bounds"])
    if any(
        getattr(bounds, key) != report["configuration"][key]
        for key in ("max_steer", "max_throttle", "max_brake")
    ):
        raise ValueError("Action support wait bounds differ from execution contract")
    try:
        bounds.interval(
            command_action(command["sent"]), (row["decision_ns"] - command["returned_ns"]) / 1e9
        )
    except ActionSupportUnavailable:
        return
    raise ValueError("Action support wait has executable policy support")


def read_realtime_decision(
    root: Path, row: dict[str, Any], contract: PixelContract
) -> NumericDecision:
    """Verify retained input metadata and RGB without loading or running a model."""
    reference = row["archive"]
    if not reference or row["archive_reason"] is not None:
        raise ValueError("Numerical input was not archived")
    path = asset(root, reference["path"])
    if sha256_file(path) != reference["sha256"]:
        raise ValueError("Numerical input metadata hash mismatch")
    saved = read_json(path)
    for key in (
        "index",
        "decision_id",
        "epoch",
        "decision_ns",
        "deadline_ns",
        "valid_until_ns",
        "actor",
        "safety_at_decision",
        "telemetry_received_ns",
    ):
        if saved[key] != row[key]:
            raise ValueError("Decision differs from archived input")
    if saved.get("command_context") != row.get("command_context"):
        raise ValueError("Archived command context differs from decision")
    if len(saved["frames"]) != len(contract.history_offsets_ms):
        raise ValueError("Archived frame count differs from contract")
    frames = tuple(
        read_numeric_frame(root, f, byte_limit=contract.size[0] * contract.size[1] * 3)
        for f in saved["frames"]
    )
    if [f.metadata() for f in frames] != row["frames"]:
        raise ValueError("Frame metadata differs from recorded decision")
    decision = NumericDecision(
        row["decision_id"], row["epoch"], row["decision_ns"], frames, row["actor"]
    )
    reason = validate_decision(decision, contract)
    if reason:
        raise ValueError("Invalid recorded observation: " + reason)
    return decision


def verify_realtime_decision(
    root: Path, row: dict[str, Any], contract: PixelContract, actor: DecisionActor, tolerance: float
) -> dict[str, Any]:
    decision = read_realtime_decision(root, row, contract)
    frames = decision.frames
    if row["prediction"] is None or row.get("error"):
        raise ValueError("Inference failure has no reproducible prediction")
    if not _valid_prediction(row["prediction"]):
        raise ValueError("Invalid recorded prediction")
    features = numeric_features(actor, deepcopy(decision.actor), frames)
    if features != row.get("features"):
        raise ValueError("Numerical time/state features differ from recorded inputs")
    prediction = list(decision_prediction(actor, decision, deepcopy(row.get("command_context"))))
    if not _valid_prediction(prediction):
        raise ValueError("Replay actor returned invalid prediction")
    error = max(abs(a - b) for a, b in zip(prediction, row["prediction"]))
    if error > tolerance:
        raise ValueError("Prediction differs beyond replay tolerance")
    return {
        "decision_id": row["decision_id"],
        "features_match": True,
        "pixels_match": True,
        "prediction": prediction,
        "prediction_max_abs_error": error,
    }


def replay_realtime_numeric(request: RealtimeNumericReplay, actor: DecisionActor) -> RunResult:
    from fh5.experiment import RunResult

    path = request.report_path
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    summary: dict[str, Any] = {
        "version": 1,
        "verified": False,
        "errors": [],
        "checks": [],
        "verified_predictions": 0,
        "verified_action_waits": 0,
        "decisions": [],
        "commands": [],
        "tolerance": request.tolerance,
        "timing_reexecuted": False,
        "commands_sent_to_game": False,
        "training_eligible": False,
        "promotion_eligible": False,
        "scope": "frozen numerical prediction reproduction; recorded timing and outcomes only",
    }
    try:
        report = read_realtime_recording(request.recording_dir)
        summary.update(
            decisions=report["decisions"],
            commands=report["commands"],
            source_evidence_kind=report["evidence_kind"],
            model=report["model"],
        )
        if actor.manifest != report["model"] or actor.kind != report["actor_kind"]:
            raise ValueError("Frozen replay model differs from recorded model")
        if (
            not report["evidence"]["exact_replay_eligible"]
            or not report["evidence"]["recording_complete"]
        ):
            raise ValueError("Recording is incomplete or quarantined")
        read_realtime_journal(request.recording_dir, report)
        contract = PixelContract.from_metadata(report["configuration"]["pixels"])
        if actor.manifest.get("numeric_contract", contract.metadata()) != contract.metadata():
            raise ValueError("Frozen replay pixel contract differs")
        for row in report["decisions"]:
            try:
                if row["status"] == "skip_action_support":
                    _verify_action_wait(row, report)
                    summary["verified_action_waits"] += 1
                    continue
                if "actor" not in row:
                    continue
                summary["checks"].append(
                    verify_realtime_decision(
                        request.recording_dir, row, contract, actor, request.tolerance
                    )
                )
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                summary["errors"].append({"decision_id": row["decision_id"], "error": str(error)})
        summary["verified_predictions"] = len(summary["checks"])
        if not summary["checks"]:
            summary["errors"].append({"error": "No independently reproduced predictions"})
        summary["verified"] = not summary["errors"]
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        summary["errors"].append({"error": str(error)})
    finally:
        clear = getattr(actor, "clear_input_cache", None)
        if clear is not None:
            clear()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
    path.with_suffix(".json").write_text(data + "\n", encoding="utf-8")
    path.write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>实时记录数值重放</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto}pre{white-space:pre-wrap}</style>"
        "<h1>实时记录数值重放</h1><p>仅核对冻结模型的数值输入与预测。原始时间线、跳过和丢弃结果保留；"
        "不重新执行时序，不发送控制，不证明实机驾驶能力。</p><pre>"
        + html.escape(data)
        + "</pre></html>",
        encoding="utf-8",
    )
    return RunResult(
        {"source_kind": "numeric_diagnostic"}, [], [], {"realtime_numeric_replay": summary}, path
    )
