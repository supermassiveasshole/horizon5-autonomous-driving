"""Offline integrity and frozen-model replay; never re-executes timing or actions."""

from __future__ import annotations

import hashlib
import html
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.numeric_images import (
    NumericActor,
    NumericDecision,
    PixelContract,
    asset,
    validate_decision,
)
from fh5.numeric_recording import numeric_features, read_numeric_frame
from fh5.realtime import MAX_REALTIME_REPORT_BYTES, RealtimeNumericReplay

if TYPE_CHECKING:
    from fh5.experiment import RunResult


def _read(path: Path, limit: int = MAX_REALTIME_REPORT_BYTES) -> bytes:
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Replay asset exceeds bounded read limit")
    return payload


def read_realtime_recording(root: Path) -> dict[str, Any]:
    manifest = json.loads(_read(root / "realtime-manifest.json", 4096))
    payload = _read(root / "report.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or hashlib.sha256(payload).hexdigest() != manifest.get("report_sha256")
    ):
        raise ValueError("Real-time report hash mismatch or unsupported manifest")
    report: dict[str, Any] = json.loads(payload)
    if (
        not isinstance(report, dict)
        or report["version"] != 2
        or report["commands_sent_to_game"] is not False
        or report["evidence_kind"] not in ("synthetic", "shadow")
    ):
        raise ValueError("Unsupported real-time numerical recording")
    return report


def _journal(root: Path, report: dict[str, Any]) -> None:
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
    if path.stat().st_size > 256 * 1024**2:
        raise ValueError("Journal exceeds bounded replay limit")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != reference["sha256"]:
            raise ValueError("Real-time journal hash mismatch")
    events: dict[str, list[dict[str, Any]]] = {}
    sequences: set[int] = set()
    commands = []
    with path.open("rb") as stream:
        while line := stream.readline(1024**2 + 1):
            if len(line) > 1024**2 or len(sequences) >= 1_000_000:
                raise ValueError("Journal event exceeds bounded replay limit")
            event = json.loads(line)
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
    if len(sequences) != reference["offered"]:
        raise ValueError("Missing journal events")
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


def _valid_prediction(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(type(v) in (float, int) and math.isfinite(v) and abs(v) <= 1 for v in value)
    )


def _decision(
    root: Path, row: dict[str, Any], contract: PixelContract, actor: NumericActor, tolerance: float
) -> dict[str, Any]:
    reference = row["archive"]
    if not reference or row["archive_reason"] is not None:
        raise ValueError("Numerical input was not archived")
    payload = _read(asset(root, reference["path"]), 1024**2)
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError("Numerical input metadata hash mismatch")
    saved = json.loads(payload)
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
    if row["prediction"] is None or row.get("error"):
        raise ValueError("Inference failure has no reproducible prediction")
    if not _valid_prediction(row["prediction"]):
        raise ValueError("Invalid recorded prediction")
    features = numeric_features(actor, deepcopy(decision.actor), frames)
    if features != row.get("features"):
        raise ValueError("Numerical time/state features differ from recorded inputs")
    prediction = list(actor.predict(deepcopy(decision.actor), frames))
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


def replay_realtime_numeric(request: RealtimeNumericReplay, actor: NumericActor) -> RunResult:
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
        _journal(request.recording_dir, report)
        contract = PixelContract.from_metadata(report["configuration"]["pixels"])
        if actor.manifest.get("numeric_contract", contract.metadata()) != contract.metadata():
            raise ValueError("Frozen replay pixel contract differs")
        for row in report["decisions"]:
            if "actor" not in row:
                continue
            try:
                summary["checks"].append(
                    _decision(request.recording_dir, row, contract, actor, request.tolerance)
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
