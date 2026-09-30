"""Frozen local evaluation protocols and complete recorded-attempt accounting."""

from __future__ import annotations

import hashlib
import html
import importlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.attempts import AttemptReplay, _task
from fh5.bc_learning import VIEWS
from fh5.collection_store import encode, read_bounded, write_file
from fh5.learning_runtime import preserve_torch_state
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig
from fh5.routes import load_route

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class EvaluationPrepare:
    config_file: Path
    output_dir: Path


@dataclass(frozen=True)
class EvaluationReview:
    batch_dir: Path
    ledger_file: Path
    output_dir: Path


def _read(path: Path, limit: int = 128 * 1024**2) -> tuple[dict[str, Any], str, bytes]:
    raw = read_bounded(path, limit)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Evaluation document must be an object")
    json.dumps(value, allow_nan=False)
    return value, hashlib.sha256(raw).hexdigest(), raw


def _config(path: Path) -> dict[str, Any]:
    from fh5.experiment import _validate_config

    config, _, _ = _read(path, 1024**2)
    if (
        set(config)
        != {"version", "purpose", "model", "task", "conditions", "runtime", "plan", "criteria"}
        or type(config["version"]) is not int
        or config["version"] != 1
    ):
        raise ValueError("Unsupported frozen evaluation configuration")
    if config["purpose"] not in ("development", "final"):
        raise ValueError("Evaluation requires development or final purpose")
    conditions = config["conditions"]
    if set(conditions) != {"snapshot", "camera", "navigation", "task_basis"}:
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
            set(row) != {"id", "reference_mode"}
            or not isinstance(row["id"], str)
            or not row["id"]
            or row["id"] in identifiers
            or row["reference_mode"] not in VIEWS
        ):
            raise ValueError("Invalid evaluation run or reference mode")
        identifiers.add(row["id"])
    criteria = config["criteria"]
    if (
        set(criteria)
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
    runtime = dict(config["runtime"])
    runtime["pixels"] = PixelContract.from_metadata(runtime["pixels"])
    runtime["action_offsets_ms"] = tuple(runtime["action_offsets_ms"])
    RealtimeConfig(**runtime)
    return config


def prepare_evaluation(request: EvaluationPrepare) -> RunResult:
    from fh5.experiment import RunResult
    from fh5.numeric_actor import FrozenNumericActor

    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    config = _config(request.config_file)
    base = request.config_file.parent
    model_dir = base / config["model"]["directory"]
    if request.output_dir.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError("Evaluation output must be outside the model")
    model, digest, model_raw = _read(model_dir / "model.json")
    if digest != config["model"]["manifest_sha256"]:
        raise ValueError("Evaluation model manifest changed")
    weights = read_bounded(model_dir / "actor.pt", 128 * 1024**2)
    if hashlib.sha256(weights).hexdigest() != model["weights_sha256"]:
        raise ValueError("Evaluation weights changed")
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
    payloads = {
        "model/model.json": model_raw,
        "model/actor.pt": weights,
        "task.json": encode(task),
        "route/route.json": route_raw,
    }
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
    with preserve_torch_state(importlib.import_module("torch")):
        actor = FrozenNumericActor(
            request.output_dir / "model", PixelContract.from_metadata(config["runtime"]["pixels"])
        )
    config["model"]["directory"] = "model"
    config["task"] = {
        "file": "task.json",
        "sha256": hashlib.sha256(payloads["task.json"]).hexdigest(),
    }
    batch = {
        "version": 1,
        "kind": "frozen-local-evaluation-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()},
        "model_diagnostic_only": actor.manifest["diagnostic_only"],
        "policy_contract": actor.original_contract,
    }
    path = request.output_dir / "batch.json"
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


def review_evaluation(request: EvaluationReview) -> RunResult:
    from fh5.experiment import RunResult, run_experiment

    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    batch, digest, raw = _read(request.batch_dir / "batch.json", 4 * 1024**2)
    ledger, _, ledger_raw = _read(request.ledger_file, 4 * 1024**2)
    if (
        batch.get("kind") != "frozen-local-evaluation-v1"
        or batch.get("version") != 1
        or set(ledger) != {"version", "batch_sha256", "entries"}
        or ledger["version"] != 1
        or ledger["batch_sha256"] != digest
    ):
        raise ValueError("Evaluation ledger belongs to a different frozen batch")
    for name, expected in batch["files"].items():
        target = (request.batch_dir / name).resolve()
        if (
            not target.is_relative_to(request.batch_dir.resolve())
            or hashlib.sha256(read_bounded(target, 128 * 1024**2)).hexdigest() != expected
        ):
            raise ValueError("Frozen evaluation dependency changed: " + name)
    plan = {row["id"]: row for row in batch["config"]["plan"]}
    entries = ledger["entries"]
    if not isinstance(entries, list) or len(entries) > len(plan):
        raise ValueError("Evaluation ledger exceeds its frozen plan")
    seen = set()
    sources = set()
    for row in entries:
        if (
            set(row) != {"slot_id", "recording", "files", "evidence"}
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
    request.output_dir.mkdir(parents=True)
    write_file(request.output_dir / "batch.json", raw)
    write_file(request.output_dir / "ledger.json", ledger_raw)
    rows = []
    for i, entry in enumerate(entries):
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
        except (OSError, ValueError, KeyError, TypeError) as error:
            rows.append(
                {
                    "slot_id": entry["slot_id"],
                    "reference_mode": plan[entry["slot_id"]]["reference_mode"],
                    "attempt_id": "unresolved:" + entry["slot_id"],
                    "outcome": "interface_error",
                    "valid_duration_s": None,
                    "contact_count": 0,
                    "source_kind": "unknown",
                    "evidence_gaps": ["recording_or_evidence_unreadable", "attempt_count_unknown"],
                    "read_error": str(error),
                    "expected_files": entry["files"],
                    "report": None,
                }
            )
            continue
        for attempt in result.summary["attempt_review"]["attempts"]:
            times = [
                s["duration_s"] for s in attempt["forward_segments"] if s["geometry_completed"]
            ]
            gaps = [
                "policy_execution_not_verified",
                "visual_timing_not_verified",
                "automatic_restart_not_verified",
            ]
            if result.metadata["snapshot"] != batch["config"]["conditions"]["snapshot"]:
                gaps.append("conditions_mismatch")
            rows.append(
                {
                    **attempt,
                    "slot_id": entry["slot_id"],
                    "reference_mode": plan[entry["slot_id"]]["reference_mode"],
                    "source_hashes": result.summary["attempt_review"]["source_hashes"],
                    "valid_duration_s": times[0]
                    if attempt["outcome"] == "valid_complete" and len(times) == 1
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
        "attempts": rows,
        "metrics": _metrics(rows),
        "by_reference": {
            view: _metrics([r for r in rows if r["reference_mode"] == view]) for view in VIEWS
        },
        "reference_mode_basis": "frozen plan; actual actor input not yet verified",
        "metrics_scope": "local task validity; not autonomous policy performance",
        "commands_sent": False,
        "automatic_promotion_allowed": False,
        "diagnostic_only": batch["model_diagnostic_only"]
        or any(r["source_kind"] != "udp" for r in rows),
        "closed_loop_validated": False,
        "scope": "local validity accounting; recorded policy execution, timing and autonomous restart still require evidence",
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
