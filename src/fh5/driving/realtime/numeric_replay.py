"""Offline integrity and frozen-model replay; never re-executes timing or actions."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from contextlib import ExitStack
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.artifacts.document import ReplayArray, replay_document, write_replay_document
from fh5.artifacts.io import VerifiedFile, asset, read_json
from fh5.driving.realtime.model import RealtimeNumericReplay
from fh5.learning.sac.actions import ActionBounds, ActionSupportUnavailable
from fh5.learning.sac.context import (
    PROPOSAL_CONTEXT,
    SEND_CONTEXT,
    context_action,
    proposal_context,
)
from fh5.learning.sac.replay import command_action
from fh5.observation.numeric import (
    DecisionActor,
    NumericDecision,
    PixelContract,
    decision_prediction,
    validate_decision,
)
from fh5.observation.recording import numeric_features, read_numeric_frame
from fh5.reporting.presentation import optional_report

if TYPE_CHECKING:
    from fh5.result import RunResult


def read_realtime_recording(
    root: Path, *, expected_manifest_sha256: str | None = None, resources: ExitStack | None = None
) -> dict[str, Any]:
    if resources is None:
        # Existing consumers retain their ordinary dict/list result. Decode
        # records one at a time instead of retaining the complete JSON text.
        with ExitStack() as owned:
            report = read_realtime_recording(
                root, expected_manifest_sha256=expected_manifest_sha256, resources=owned
            )
            return {key: _materialize(value) for key, value in report.items()}
    manifest = read_json(root / "realtime-manifest.json", expected_sha256=expected_manifest_sha256)
    path = root / "report.json"
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or not isinstance(manifest.get("report_sha256"), str)
    ):
        raise ValueError("Real-time report hash mismatch or unsupported manifest")
    report = resources.enter_context(
        replay_document(
            VerifiedFile(path, manifest["report_sha256"]),
            nested_arrays={("environment", "source_samples", "records")},
        )
    )
    if (
        not isinstance(report, dict)
        or report["version"] != 2
        or type(report["commands_sent_to_game"]) is not bool
        or (report["evidence_kind"] != "native" and report["commands_sent_to_game"])
        or report["evidence_kind"] not in ("synthetic", "shadow", "native")
    ):
        raise ValueError("Unsupported real-time numerical recording")
    return report


def _materialize(value: Any) -> Any:
    if isinstance(value, (ReplayArray, list)):
        return [_materialize(row) for row in value]
    if isinstance(value, dict):
        return {key: _materialize(item) for key, item in value.items()}
    return value


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
    proposals = []
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
            elif event["kind"] == "proposal":
                proposals.append((sequence, row))
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
    missing = object()
    if any(
        a != b
        for a, b in zip_longest(
            (r for _, r in sorted(commands)), report["commands"], fillvalue=missing
        )
    ):
        raise ValueError("Recorded commands differ from journal")
    if any(
        a != b
        for a, b in zip_longest(
            (r for _, r in sorted(proposals)), report.get("proposals", []), fillvalue=missing
        )
    ):
        raise ValueError("Recorded counterfactual proposals differ from journal")
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


def _verify_shadow_proposals(report: dict[str, Any]) -> None:
    """Bind every hypothetical context to the earlier proposal, never to a send."""
    if report["evidence_kind"] != "shadow" or report["commands"]:
        raise ValueError("Counterfactual proposals are not execution evidence")
    proposals = report["proposals"]
    accepted = {d["decision_id"]: d for d in report["decisions"] if d["status"] == "accepted"}
    neutral = {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0}
    keys = {"proposal_index", "proposed_ns", "owner", "decision_id", "valid_until_ns", "proposal"}
    times: list[int] = []
    seen = set()
    cfg = report["configuration"]
    for index, proposal in enumerate(proposals):
        if (
            set(proposal) != keys
            or type(proposal["proposal_index"]) is not int
            or proposal["proposal_index"] != index
            or type(proposal["proposed_ns"]) is not int
            or not report["started_ns"] <= proposal["proposed_ns"] <= report["ended_ns"]
            or (times and proposal["proposed_ns"] < times[-1])
        ):
            raise ValueError("Invalid counterfactual proposal sequence or time")
        times.append(proposal["proposed_ns"])
        command_action(proposal["proposal"])
        if proposal["owner"] != "policy":
            if (
                proposal["owner"] not in ("initial_neutral", "lease_expiry", "hard_stop")
                or proposal["decision_id"] is not None
                or proposal["proposal"] != neutral
                or proposal["valid_until_ns"] != proposal["proposed_ns"]
            ):
                raise ValueError("Invalid neutral counterfactual proposal")
            continue
        identity = proposal["decision_id"]
        if identity not in accepted or identity in seen:
            raise ValueError("Counterfactual proposal lacks a unique accepted prediction")
        seen.add(identity)
        row = accepted[identity]
        steer, longitudinal = row["prediction"]
        expected = {
            "steer_i16": round(max(-cfg["max_steer"], min(cfg["max_steer"], steer)) * 32767),
            "throttle_u8": round(max(0, min(cfg["max_throttle"], longitudinal)) * 255),
            "brake_u8": round(max(0, min(cfg["max_brake"], -longitudinal)) * 255),
        }
        if (
            proposal["proposal"] != expected
            or proposal["proposed_ns"] != row["inference_returned_ns"]
            or proposal["valid_until_ns"] != row["valid_until_ns"]
        ):
            raise ValueError("Counterfactual proposal differs from its recorded prediction")
    if not proposals or proposals[0]["owner"] != "initial_neutral" or seen != accepted.keys():
        raise ValueError("Incomplete counterfactual proposal history")
    for row in report["decisions"]:
        if "command_context" in row:
            index = bisect_left(times, row["decision_ns"]) - 1
            if index < 0 or row["command_context"] != proposal_context(proposals[index]):
                raise ValueError("Counterfactual context differs from the previous proposal")
            context_action(row["command_context"], PROPOSAL_CONTEXT, row["decision_ns"])
        if "actor" in row and (
            "command_context" not in row
            or any(row["actor"]["action_mask"])
            or any(value is not None for value in row["actor"]["actions"])
            or any(value is not None for value in row["actor"]["action_age_ms"])
        ):
            raise ValueError("Shadow proposals cannot become executed action history")


def _verify_action_wait(row: dict[str, Any], report: dict[str, Any]) -> None:
    """Recompute a wait using the recorded context's explicit time basis."""
    kind = report["model"].get("command_context")
    if (
        kind not in (SEND_CONTEXT, PROPOSAL_CONTEXT)
        or row.get("prediction") is not None
        or any(
            key in row for key in ("actor", "error", "inference_returned_ns", "worker_started_ns")
        )
        or any(c.get("decision_id") == row["decision_id"] for c in report["commands"])
    ):
        raise ValueError("Invalid action support wait outcome")
    if kind == PROPOSAL_CONTEXT:
        earlier = [p for p in report["proposals"] if p["proposed_ns"] < row["decision_ns"]]
        if not earlier:
            raise ValueError("Action support wait lacks a prior counterfactual proposal")
        context = proposal_context(earlier[-1])
    else:
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
    if row.get("command_context") != context or context["owner"] == "warmup":
        raise ValueError("Action support wait differs from successful command context")
    previous, at = context_action(context, kind, row["decision_ns"])
    bounds = ActionBounds(**report["model"]["bounds"])
    if any(
        getattr(bounds, key) != report["configuration"][key]
        for key in ("max_steer", "max_throttle", "max_brake")
    ):
        raise ValueError("Action support wait bounds differ from execution contract")
    try:
        bounds.interval(previous, (row["decision_ns"] - at) / 1e9)
    except ActionSupportUnavailable:
        return
    raise ValueError("Action support wait has executable policy support")


def read_realtime_decision(
    root: Path, row: dict[str, Any], contract: PixelContract
) -> NumericDecision:
    """Verify retained input metadata and RGB without loading or running a model."""
    reference = row["archive"]
    if (
        not reference
        or row["archive_reason"] is not None
        or not isinstance(reference.get("sha256"), str)
    ):
        raise ValueError("Numerical input was not archived")
    path = asset(root, reference["path"])
    saved = read_json(path, expected_sha256=reference["sha256"])
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
    with ExitStack() as resources:
        return _replay_realtime_numeric(request, actor, resources)


def _replay_realtime_numeric(
    request: RealtimeNumericReplay, actor: DecisionActor, resources: ExitStack
) -> RunResult:
    from fh5.result import RunResult

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
        report = read_realtime_recording(request.recording_dir, resources=resources)
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
        if report["model"].get("command_context") == PROPOSAL_CONTEXT:
            _verify_shadow_proposals(report)
            summary["proposals"] = report["proposals"]
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
    evidence = path.with_suffix(".json")
    write_replay_document(evidence, summary)
    # Preserve the public result's list values after its temporary index closes.
    # The verified source text and unneeded environment diagnostics are never
    # copied into this numerical replay result.
    summary = {key: _materialize(value) for key, value in summary.items()}
    path = optional_report(
        path,
        "实时记录数值重放（仅核对冻结输入与预测，不重新执行时序或发送控制）",
        summary,
        fallback=evidence,
    )
    return RunResult(
        {"source_kind": "numeric_diagnostic"}, [], [], {"realtime_numeric_replay": summary}, path
    )
