"""Frozen local evaluation protocols and complete recorded-attempt accounting."""

from __future__ import annotations

import hashlib
import html
import importlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.artifacts.io import encode, read_bounded, write_file
from fh5.artifacts.usage import bind_evaluation_slots, reserve_batch, review_usage, source_keys
from fh5.driving.realtime.model import RealtimeConfig
from fh5.evaluation.attempts import AttemptReplay, _task
from fh5.evaluation.execution import review_execution
from fh5.evaluation.metrics import combine_execution_metrics
from fh5.evaluation.model import asset_limit, model_payloads, validate_model
from fh5.evaluation.start import event_payloads, review_start
from fh5.learning.bc.legacy_training import VIEWS
from fh5.learning.loop.runtime import preserve_torch_state
from fh5.observation.numeric import PixelContract
from fh5.observation.routes import load_route

if TYPE_CHECKING:
    from fh5.result import RunResult


@dataclass(frozen=True)
class EvaluationPrepare:
    config_file: Path
    output_dir: Path
    registry_file: Path | None = None


@dataclass(frozen=True)
class EvaluationReview:
    batch_dir: Path
    ledger_file: Path
    output_dir: Path
    registry_file: Path | None = None


def _read(path: Path, limit: int = 128 * 1024**2) -> tuple[dict[str, Any], str, bytes]:
    raw = read_bounded(path, limit)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Evaluation document must be an object")
    json.dumps(value, allow_nan=False)
    return value, hashlib.sha256(raw).hexdigest(), raw


def read_evaluation_batch(
    directory: Path, expected_sha256: str
) -> tuple[dict[str, Any], str, bytes]:
    batch, digest, raw = _read(directory / "batch.json", 4 * 1024**2)
    if digest != expected_sha256 or (batch.get("kind"), batch.get("version")) not in (
        ("frozen-local-evaluation-v1", 1),
        ("frozen-local-evaluation-v2", 2),
        ("frozen-local-evaluation-v3", 3),
    ):
        raise ValueError("Frozen evaluation batch changed or unsupported")
    for name, expected in batch["files"].items():
        target = (directory / name).resolve()
        if (
            not target.is_relative_to(directory.resolve())
            or hashlib.sha256(read_bounded(target, asset_limit(name))).hexdigest() != expected
        ):
            raise ValueError("Frozen evaluation dependency changed: " + name)
    return batch, digest, raw


def _config(path: Path) -> dict[str, Any]:
    from fh5.telemetry.packet import validate_record_config as _validate_config

    config, _, _ = _read(path, 1024**2)
    if (
        set(config)
        != {"version", "purpose", "model", "task", "conditions", "runtime", "plan", "criteria"}
        or type(config["version"]) is not int
        or config["version"] not in (1, 2, 3)
    ):
        raise ValueError("Unsupported frozen evaluation configuration")
    if config["purpose"] not in ("development", "final"):
        raise ValueError("Evaluation requires development or final purpose")
    for name, required in (
        (
            "model",
            {"directory", "manifest_sha256"}
            | ({"kind"} if config["version"] in (2, 3) else set())
            | ({"device"} if config["version"] == 3 else set()),
        ),
        ("task", {"file", "sha256"}),
    ):
        binding = config[name]
        if (
            not isinstance(binding, dict)
            or set(binding) != required
            or any(not isinstance(v, str) or not v.strip() for v in binding.values())
        ):
            raise ValueError("Evaluation binding is incomplete: " + name)
    if config["version"] == 2 and config["model"]["kind"] != "sac":
        raise ValueError("Evaluation version 2 requires an explicit SAC model")
    if config["version"] == 3 and (config["model"]["kind"], config["model"]["device"]) not in (
        ("bc", "cpu"),
        ("bc", "cuda"),
        ("sac", "cpu"),
    ):
        raise ValueError("Native evaluation requires BC on CPU/CUDA or SAC on CPU")
    conditions = config["conditions"]
    if not isinstance(conditions, dict) or set(conditions) != {
        "snapshot",
        "camera",
        "navigation",
        "task_basis",
    } | ({"numeric_input_conditions"} if config["version"] == 3 else set()):
        raise ValueError("Evaluation conditions are incomplete")
    _validate_config(
        {"schema_version": 1, "control_source": "policy", "snapshot": conditions["snapshot"]}
    )
    for key in ("camera", "navigation", "task_basis"):
        if not isinstance(conditions[key], str) or not conditions[key].strip():
            raise ValueError("Evaluation condition missing: " + key)
    plan = config["plan"]
    if not isinstance(plan, list) or not 1 <= len(plan) <= 1000:
        raise ValueError("Evaluation plan needs 1..1000 explicit runs")
    identifiers = set()
    for row in plan:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "reference_mode"}
            or not isinstance(row["id"], str)
            or not row["id"]
            or row["id"] in identifiers
            or row["reference_mode"] not in VIEWS
        ):
            raise ValueError("Invalid evaluation run or reference mode")
        identifiers.add(row["id"])
    criteria = config["criteria"]
    if (
        not isinstance(criteria, dict)
        or set(criteria)
        != {
            "min_valid_attempts",
            "reliability_tolerance",
            "min_time_improvement_fraction",
            "anomalies",
            "exploration",
            "rewind",
        }
        or type(criteria["min_valid_attempts"]) is not int
        or not 1 <= criteria["min_valid_attempts"] <= len(plan)
        or criteria["anomalies"] != "quarantine_keep_in_denominator"
        or criteria["exploration"] is not False
        or criteria["rewind"] is not False
    ):
        raise ValueError("Invalid frozen evaluation criteria")
    for key in ("reliability_tolerance", "min_time_improvement_fraction"):
        v = criteria[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError("Evaluation threshold must be a finite fraction")
    if not isinstance(config["runtime"], dict) or set(config["runtime"]) != {
        f.name for f in fields(RealtimeConfig)
    }:
        raise ValueError("Evaluation runtime must specify every bound explicitly")
    runtime = dict(config["runtime"])
    runtime["pixels"] = PixelContract.from_metadata(runtime["pixels"])
    runtime["action_offsets_ms"] = tuple(runtime["action_offsets_ms"])
    RealtimeConfig(**runtime)
    return config


def prepare_evaluation(request: EvaluationPrepare) -> RunResult:
    from fh5.result import RunResult

    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    config = _config(request.config_file)
    base = request.config_file.parent
    model_dir = base / config["model"]["directory"]
    if request.output_dir.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError("Evaluation output must be outside the model")
    model, payloads = model_payloads(model_dir, config["model"])
    if config["version"] == 3 and (
        model.get("provenance", {}).get("diagnostic_only") is not False
        or config["conditions"]["numeric_input_conditions"]
        != model.get("provenance", {}).get("input_conditions")
        or not isinstance(config["conditions"]["numeric_input_conditions"], dict)
    ):
        raise ValueError("Native evaluation conditions differ from the non-diagnostic candidate")
    for view in {p["reference_mode"] for p in config["plan"]}:
        if model.get("training", {}).get("train_by_view", {}).get(view, 0) <= 0:
            raise ValueError("Evaluation view was not trained: " + view)
    task_path = base / config["task"]["file"]
    task = _task(task_path)
    if _read(task_path)[1] != config["task"]["sha256"]:
        raise ValueError("Evaluation task changed")
    runtime = config["runtime"]
    shape = model["contract"]["actor_shape"]
    if (
        shape["action_count"] != len(runtime["action_offsets_ms"])
        or shape["reference_count"] != runtime["reference_count"]
        or any(runtime[k] != task[k] for k in ("expected_car_ordinal", "expected_pi"))
    ):
        raise ValueError("Evaluation runtime contradicts model or task")
    route_path = task_path.parent / task["route_file"]
    route_manifest, route_digest, route_raw = _read(route_path)
    if route_digest != task["route_sha256"]:
        raise ValueError("Evaluation route changed")
    load_route(route_path)
    task["route_file"] = "route/route.json"
    if task["version"] == 2:
        start = task["automatic_start"]
        event_file = task_path.parent / start["event_file"]
        if hashlib.sha256(read_bounded(event_file, 1024**2)).hexdigest() != start["event_sha256"]:
            raise ValueError("Automatic start event changed")
        event_files = event_payloads(event_file)
        event_config = json.loads(event_files["start/event.json"])
        if (
            not event_config["event_run"]["conditions_verified"]
            or event_config["event_run"]["purpose"] != "event"
            or event_config["snapshot"] != config["conditions"]["snapshot"]
            or any(
                event_config["event_run"][k] != task[k]
                for k in ("expected_car_ordinal", "expected_pi")
            )
        ):
            raise ValueError("Automatic start event conditions disagree with the task")
        payloads.update(event_files)
        start.update(
            event_file="start/event.json",
            event_sha256=hashlib.sha256(event_files["start/event.json"]).hexdigest(),
        )
    payloads.update(
        {
            "task.json": encode(task),
            "route/route.json": route_raw,
        }
    )
    for asset in [*route_manifest["assets"].values(), *route_manifest["evidence"]]:
        content = read_bounded(route_path.parent / asset["path"], 128 * 1024**2)
        if hashlib.sha256(content).hexdigest() != asset["sha256"]:
            raise ValueError("Evaluation route dependency changed")
        payloads["route/" + asset["path"]] = content
    request.output_dir.mkdir(parents=True)
    (request.output_dir / "model").mkdir()
    for name, raw in payloads.items():
        (request.output_dir / name).parent.mkdir(parents=True, exist_ok=True)
        write_file(request.output_dir / name, raw)
    # Validate the copied bytes, including the checkpoint's embedded input metadata.
    torch = importlib.import_module("torch")
    with preserve_torch_state(torch):
        policy_contract, diagnostic_only = validate_model(
            torch, request.output_dir / "model", config["model"], config["runtime"]
        )
    config["model"]["directory"] = "model"
    config["task"] = {
        "file": "task.json",
        "sha256": hashlib.sha256(payloads["task.json"]).hexdigest(),
    }
    batch = {
        "version": config["version"],
        "kind": f"frozen-local-evaluation-v{config['version']}",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()},
        "model_diagnostic_only": diagnostic_only,
        "policy_contract": policy_contract,
    }
    path = request.output_dir / "batch.json"
    if request.registry_file is not None:
        reserve_batch(request.registry_file, batch)
    write_file(path, encode(batch))
    return RunResult(
        {},
        [],
        [],
        {
            "evaluation": {
                "state": "prepared",
                "batch_sha256": hashlib.sha256(encode(batch)).hexdigest(),
                "planned_runs": len(config["plan"]),
                "purpose": config["purpose"],
                "commands_sent": False,
            }
        },
        path,
    )


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r["outcome"] for r in rows)
    valid = counts["valid_complete"]
    classified = sum(counts[o] for o in ("valid_complete", "driving_failed", "invalid"))
    times = [r["valid_duration_s"] for r in rows if r["valid_duration_s"] is not None]
    return {
        "all_attempts": len(rows),
        "outcomes": {
            k: counts[k]
            for k in (
                "valid_complete",
                "driving_failed",
                "invalid",
                "pending_review",
                "interface_error",
            )
        },
        "valid_fraction_all_attempts": valid / len(rows) if rows else None,
        "classified_driving_attempts": classified,
        "valid_fraction_classified_driving": valid / classified if classified else None,
        "valid_duration_s": {
            "count": len(times),
            "min": min(times) if times else None,
            "median": statistics.median(times) if times else None,
            "max": max(times) if times else None,
        },
        "contact_count": sum(r["contact_count"] for r in rows),
        "evidence_incomplete_attempts": sum(bool(r["evidence_gaps"]) for r in rows),
    }


def _verify_source(root: Path, files: dict[str, str]) -> None:
    actual = {p.name for p in root.iterdir() if p.suffix in (".json", ".jsonl")}
    if (
        not {"session.json", "packets.jsonl"} <= files.keys()
        or set(files) != actual
        or len(files) > 64
    ):
        raise ValueError("Evaluation source inventory changed or incomplete")
    for name, expected in files.items():
        if (
            Path(name).name != name
            or hashlib.sha256(read_bounded(root / name, 128 * 1024**2)).hexdigest() != expected
        ):
            raise ValueError("Evaluation source changed: " + name)


def _protocol_order(batch: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Check recorded wall-clock ordering, without treating it as execution proof."""
    try:
        frozen = datetime.fromisoformat(batch["created_utc"])
        started = datetime.fromisoformat(metadata["created_utc"])
        ended = datetime.fromisoformat(metadata["ended_utc"])
        if any(t.tzinfo is None for t in (frozen, started, ended)):
            return "unknown"
        if ended < started or ended > datetime.now(UTC):
            return "unknown"
        return "recording_predates_protocol" if started < frozen else "metadata_after_protocol"
    except (ValueError, TypeError, KeyError):
        return "unknown"


def review_evaluation(request: EvaluationReview) -> RunResult:
    from fh5.experiment import run_experiment
    from fh5.result import RunResult

    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    ledger, _, ledger_raw = _read(request.ledger_file, 4 * 1024**2)
    if set(ledger) != {"version", "batch_sha256", "entries"} or ledger["version"] != 1:
        raise ValueError("Evaluation ledger belongs to a different frozen batch")
    batch, digest, raw = read_evaluation_batch(request.batch_dir, ledger["batch_sha256"])
    plan = {row["id"]: row for row in batch["config"]["plan"]}
    entries = ledger["entries"]
    if not isinstance(entries, list) or len(entries) > len(plan):
        raise ValueError("Evaluation ledger exceeds its frozen plan")
    seen = set()
    sources = set()
    for row in entries:
        if (
            set(row) - {"execution", "preparation"} != {"slot_id", "recording", "files", "evidence"}
            or row["slot_id"] not in plan
            or row["slot_id"] in seen
        ):
            raise ValueError("Duplicate or unknown evaluation slot")
        seen.add(row["slot_id"])
        packets_digest = row["files"].get("packets.jsonl")
        # Renaming session metadata does not create an independent driving sample.
        # Empty recordings can still represent distinct failed interface attempts.
        identity = (
            row["files"].get("session.json")
            if packets_digest == hashlib.sha256(b"").hexdigest()
            else None,
            packets_digest,
        )
        if identity in sources:
            raise ValueError("One recording cannot count as independent repeated evaluations")
        sources.add(identity)
    binding_error = bind_evaluation_slots(request.registry_file, digest, entries)
    request.output_dir.mkdir(parents=True)
    write_file(request.output_dir / "batch.json", raw)
    write_file(request.output_dir / "ledger.json", ledger_raw)
    rows = []
    executions = []
    starts = []
    usage_sources = []
    for i, entry in enumerate(entries):
        usage_source: dict[str, Any] = {
            "slot_id": entry["slot_id"],
            "keys": [],
            "origin": "unknown",
        }
        usage_sources.append(usage_source)
        source = request.ledger_file.parent / entry["recording"]
        try:
            _verify_source(source, entry["files"])
            proof = entry["evidence"]
            evidence = request.ledger_file.parent / proof["file"] if proof else None
            if evidence is not None and _read(evidence)[1] != proof["sha256"]:
                raise ValueError("Evaluation evidence changed")
            result = run_experiment(
                AttemptReplay(
                    source,
                    request.output_dir / f"run-{i:04d}",
                    request.batch_dir / "task.json",
                    evidence,
                )
            )
            _verify_source(source, entry["files"])
            if evidence is not None and _read(evidence)[1] != proof["sha256"]:
                raise ValueError("Evaluation evidence changed during review")
            usage_source.update(
                keys=source_keys(entry["files"]),
                origin=entry["files"]["packets.jsonl"],
                created_utc=result.metadata["created_utc"],
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            starts.append(
                {
                    "slot_id": entry["slot_id"],
                    "status": "quarantined",
                    "reasons": ["recording_or_evidence_unreadable"],
                    "scope": "automatic local start only; not whole-task validity",
                }
            )
            executions.append(
                {
                    "slot_id": entry["slot_id"],
                    "status": "source_unreadable",
                    "reasons": [str(error)],
                    "metrics": None,
                    "game_control_verified": False,
                    "observed_reference_modes": [],
                }
            )
            rows.append(
                {
                    "slot_id": entry["slot_id"],
                    "reference_mode": plan[entry["slot_id"]]["reference_mode"],
                    "attempt_id": "unresolved:" + entry["slot_id"],
                    "outcome": "interface_error",
                    "valid_duration_s": None,
                    "contact_count": 0,
                    "source_kind": "unknown",
                    "protocol_order": "unknown",
                    "diagnostic_only": True,
                    "evidence_gaps": ["recording_or_evidence_unreadable", "attempt_count_unknown"],
                    "read_error": str(error),
                    "expected_files": entry["files"],
                    "report": None,
                }
            )
            continue
        execution = review_execution(
            entry.get("execution"),
            ledger_dir=request.ledger_file.parent,
            batch_dir=request.batch_dir,
            batch=batch,
            source_dir=source,
            recording=result,
            reference_mode=plan[entry["slot_id"]]["reference_mode"],
            output=request.output_dir / f"execution-{i:04d}.html",
        )
        execution["slot_id"] = entry["slot_id"]
        executions.append(execution)
        start = review_start(
            entry.get("preparation"),
            ledger_dir=request.ledger_file.parent,
            batch_dir=request.batch_dir,
            batch=batch,
            slot=entry["slot_id"],
            execution=execution,
            execution_binding=entry.get("execution"),
            recording=result,
            output=request.output_dir / f"start-{i:04d}.html",
        )
        starts.append(start)
        order = _protocol_order(batch, result.metadata)
        for attempt in result.summary["attempt_review"]["attempts"]:
            times = [
                s["duration_s"] for s in attempt["forward_segments"] if s["geometry_completed"]
            ]
            gaps = [
                "policy_execution_not_verified",
                "visual_timing_not_verified",
                "automatic_restart_not_verified",
            ]
            outcome = attempt["outcome"]
            reasons = list(attempt["reasons"])
            pending = list(attempt["pending_checks"])
            start_verified = start["status"] == "verified" and attempt["packet_range"][0] == 0
            if start_verified:
                pending = [p for p in pending if p != "automatic_start_unverified"]
                reasons = [r for r in reasons if r != "automatic_start_unverified"]
                gaps.remove("automatic_restart_not_verified")
                if outcome == "pending_review" and not pending:
                    outcome = "valid_complete" if attempt["task_completed"] else "driving_failed"
            if execution["status"] == "quarantined":
                gaps.append("execution_quarantined")
                reasons.append("execution_quarantined")
                pending.append("execution_quarantined")
                if outcome == "valid_complete":
                    outcome = "pending_review"
            if order != "metadata_after_protocol":
                gaps.append("protocol_order:" + order)
                reasons.append("protocol_order:" + order)
                pending.append("protocol_order:" + order)
                if outcome == "valid_complete":
                    outcome = "pending_review"
            if result.metadata["snapshot"] != batch["config"]["conditions"]["snapshot"]:
                gaps.append("conditions_mismatch")
                reasons.append("conditions_mismatch")
                if outcome != "interface_error":
                    outcome = "invalid"
            rows.append(
                {
                    **attempt,
                    "local_outcome": attempt["outcome"],
                    "execution_id": entry["slot_id"],
                    "local_record_eligible": attempt["record_eligible"],
                    "protocol_order": order,
                    "diagnostic_only": batch["model_diagnostic_only"]
                    or result.metadata["source_kind"] != "udp"
                    or order != "metadata_after_protocol"
                    or "conditions_mismatch" in gaps
                    or execution["status"] != "missing",
                    "outcome": outcome,
                    "reasons": reasons,
                    "pending_checks": pending,
                    "record_eligible": outcome == "valid_complete",
                    "automatic_start_verified": start_verified,
                    "slot_id": entry["slot_id"],
                    "reference_mode": plan[entry["slot_id"]]["reference_mode"],
                    "source_hashes": result.summary["attempt_review"]["source_hashes"],
                    "valid_duration_s": times[0]
                    if outcome == "valid_complete" and len(times) == 1
                    else None,
                    "evidence_gaps": gaps,
                    "source_kind": result.metadata["source_kind"],
                    "report": f"run-{i:04d}/report.html",
                    "read_error": None,
                }
            )
    summary = {
        "version": 1,
        "state": "reviewed",
        "batch_sha256": digest,
        "purpose": batch["config"]["purpose"],
        "planned_runs": len(plan),
        "recorded_runs": len(entries),
        "unstarted_slots": [key for key in plan if key not in seen],
        "extra_attempts": len(rows) - len(entries),
        "unresolved_recordings": sum(r["read_error"] is not None for r in rows),
        "quarantined_executions": sum(e["status"] == "quarantined" for e in executions),
        "attempts": rows,
        "executions": executions,
        "starts": starts,
        "verified_starts": sum(s["status"] == "verified" for s in starts),
        "execution_metrics": combine_execution_metrics(executions),
        "execution_metrics_by_observed_reference": {
            view: combine_execution_metrics(
                [e for e in executions if e["observed_reference_modes"] == [view]]
            )
            for view in VIEWS
        },
        "metrics": _metrics(rows),
        "by_reference": {
            view: _metrics([r for r in rows if r["reference_mode"] == view]) for view in VIEWS
        },
        "reference_mode_basis": "attempt groups use frozen plan; observed numerical execution modes reported separately",
        "metrics_scope": "local task validity; not autonomous policy performance",
        "commands_sent": False,
        "automatic_promotion_allowed": False,
        "diagnostic_only": batch["model_diagnostic_only"]
        or any(r["diagnostic_only"] for r in rows),
        "closed_loop_validated": False,
        "scope": "local validity accounting; recorded policy execution, timing and autonomous restart still require evidence",
        "independence": review_usage(
            request.registry_file, batch, digest, usage_sources, request.output_dir, binding_error
        ),
    }
    write_file(request.output_dir / "batch-report.json", encode(summary))
    links = ""
    for row in rows:
        link = f'<a href="{row["report"]}">回放</a>' if row["report"] else "记录不可读"
        reason = row["read_error"] or ", ".join(row["reasons"] + row["evidence_gaps"])
        links += (
            f"<tr><td>{html.escape(row['slot_id'])}</td>"
            f"<td>{html.escape(row['reference_mode'])}</td>"
            f"<td>{html.escape(row['outcome'])}</td>"
            f"<td>{html.escape(reason)}</td><td>{link}</td></tr>"
        )
    path = request.output_dir / "report.html"
    path.write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>冻结评估批次</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}"
        "td,th{text-align:left;border-bottom:1px solid #ccc;padding:.6rem}"
        "pre{white-space:pre-wrap;background:#f4f5f7;padding:1rem}</style>"
        "<h1>冻结评估批次</h1><p>局部有效性统计；尚不证明策略自主驾驶或支持自动晋升。</p>"
        f"<pre>{html.escape(json.dumps({k: v for k, v in summary.items() if k != 'attempts'}, ensure_ascii=False, indent=2))}</pre>"
        "<table><tr><th>计划编号</th><th>参考条件（计划）</th><th>局部结果</th><th>原因/缺口</th><th>记录</th></tr>"
        + links
        + "</table></html>",
        encoding="utf-8",
    )
    return RunResult({}, [], [], {"evaluation": summary}, path)
