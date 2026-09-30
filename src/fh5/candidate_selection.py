"""Auditable candidate comparisons from freshly replayed, frozen batch evidence."""

from __future__ import annotations

import hashlib
import html
import json
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.collection_store import encode, read_bounded, write_file
from fh5.evaluation import EvaluationReview, read_evaluation_batch

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class CandidateCompare:
    config_file: Path
    output_dir: Path
    registry_file: Path | None = None


@dataclass(frozen=True)
class EvaluationInput:
    batch_dir: Path
    batch_sha256: str
    ledger_file: Path
    ledger_sha256: str
    batch: dict[str, Any]

    def verify(self) -> None:
        read_evaluation_batch(self.batch_dir, self.batch_sha256)
        if (
            hashlib.sha256(read_bounded(self.ledger_file, 4 * 1024**2)).hexdigest()
            != self.ledger_sha256
        ):
            raise ValueError("Candidate comparison ledger changed")


def _input(base: Path, binding: Any) -> EvaluationInput:
    if (
        not isinstance(binding, dict)
        or set(binding) != {"batch", "batch_sha256", "ledger", "ledger_sha256"}
        or any(not isinstance(v, str) or not v.strip() for v in binding.values())
    ):
        raise ValueError("Candidate comparison requires complete evaluation bindings")
    batch_dir = (base / binding["batch"]).resolve()
    batch, _, _ = read_evaluation_batch(batch_dir, binding["batch_sha256"])
    result = EvaluationInput(
        batch_dir,
        binding["batch_sha256"],
        (base / binding["ledger"]).resolve(),
        binding["ledger_sha256"],
        batch,
    )
    result.verify()
    ledger = json.loads(read_bounded(result.ledger_file, 4 * 1024**2))
    if ledger.get("batch_sha256") != result.batch_sha256:
        raise ValueError("Candidate comparison ledger belongs to another batch")
    if batch["config"]["purpose"] != "development":
        raise ValueError("Final acceptance batches cannot be consumed for candidate selection")
    return result


def _conditions(batch: dict[str, Any]) -> dict[str, Any]:
    config = batch["config"]
    return {
        "conditions": config["conditions"],
        "runtime": config["runtime"],
        "criteria": config["criteria"],
        "plan_by_reference": dict(Counter(r["reference_mode"] for r in config["plan"])),
        "task_files": {k: v for k, v in batch["files"].items() if not k.startswith("model/")},
    }


def _eligibility(side: str, review: dict[str, Any], criteria: dict[str, Any]) -> list[str]:
    counts = review["metrics"]["outcomes"]
    reasons = []
    for name in ("invalid", "pending_review", "interface_error"):
        if counts[name]:
            reasons.append(f"{side}:{name}_attempts")
    if review["unstarted_slots"] or review["unresolved_recordings"]:
        reasons.append(side + ":incomplete_plan")
    if counts["valid_complete"] < criteria["min_valid_attempts"]:
        reasons.append(side + ":insufficient_valid_attempts")
    if review["independence"]["status"] == "known_overlap":
        reasons.append(side + ":known_evidence_reuse")
    if review["independence"].get("error"):
        reasons.append(side + ":usage_registry_error")
    return reasons


def _compare(
    reviews: dict[str, Any], conditions: dict[str, Any], models: dict[str, str], reasons: list[str]
) -> dict[str, Any]:
    groups = {
        view: {side: review["by_reference"][view] for side, review in reviews.items()}
        for view in conditions["plan_by_reference"]
    }
    aggressive = {}
    recommendation = "retain_incumbent"
    if not reasons:
        criteria = conditions["criteria"]
        tolerance = Fraction(str(criteria["reliability_tolerance"]))
        time_threshold = Fraction(str(criteria["min_time_improvement_fraction"]))
        time_improved = False
        for view, group in groups.items():
            old, new = group["incumbent"], group["candidate"]
            if any(
                not m["valid_duration_s"]["count"] or m["valid_duration_s"]["min"] <= 0
                for m in (old, new)
            ):
                reasons.append("insufficient_valid_reference_group:" + view)
                continue
            reliability_delta = Fraction(
                new["outcomes"]["valid_complete"], new["all_attempts"]
            ) - Fraction(old["outcomes"]["valid_complete"], old["all_attempts"])
            median_improvement = 1 - Fraction(str(new["valid_duration_s"]["median"])) / Fraction(
                str(old["valid_duration_s"]["median"])
            )
            fastest_improvement = 1 - Fraction(str(new["valid_duration_s"]["min"])) / Fraction(
                str(old["valid_duration_s"]["min"])
            )
            group.update(
                reliability_delta=float(reliability_delta),
                median_improvement_fraction=float(median_improvement),
                fastest_improvement_fraction=float(fastest_improvement),
            )
            time_improved |= median_improvement > 0 and median_improvement >= time_threshold
            if reliability_delta < -tolerance:
                reasons.append("candidate:reliability_regressed:" + view)
                if fastest_improvement > 0 and fastest_improvement >= time_threshold:
                    aggressive[view] = models["candidate"]
            if median_improvement < 0:
                reasons.append("candidate:median_time_regressed:" + view)
        if any("median_improvement_fraction" not in g for g in groups.values()):
            aggressive.clear()
        if not reasons:
            if time_improved:
                recommendation = "prefer_candidate_locally"
                reasons.append("local_thresholds_met")
            else:
                reasons.append("candidate:insufficient_time_improvement")
    return {
        "local_recommendation": recommendation,
        "selected_model_sha256": models[
            "candidate" if recommendation == "prefer_candidate_locally" else "incumbent"
        ],
        "by_reference": groups,
        "aggressive_by_reference": aggressive,
    }


def compare_candidates(request: CandidateCompare) -> RunResult:
    from fh5.experiment import RunResult, run_experiment

    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    raw = read_bounded(request.config_file, 1024**2)
    config = json.loads(raw)
    if (
        not isinstance(config, dict)
        or set(config) != {"version", "incumbent", "candidate"}
        or type(config["version"]) is not int
        or config["version"] != 1
    ):
        raise ValueError("Unsupported candidate comparison configuration")
    inputs = {
        side: _input(request.config_file.parent, config[side])
        for side in ("incumbent", "candidate")
    }
    if any(request.output_dir.resolve().is_relative_to(i.batch_dir) for i in inputs.values()):
        raise ValueError("Comparison output must be outside frozen model batches")
    conditions = _conditions(inputs["incumbent"].batch)
    if conditions != _conditions(inputs["candidate"].batch):
        raise ValueError("Candidate comparison conditions, task, plan or criteria differ")
    request.output_dir.mkdir(parents=True)
    write_file(request.output_dir / "comparison.json", raw)
    reviews = {}
    origins: dict[str, set[str]] = {}
    for side, binding in inputs.items():
        binding.verify()
        reviews[side] = run_experiment(
            EvaluationReview(
                binding.batch_dir,
                binding.ledger_file,
                request.output_dir / side,
                request.registry_file,
            )
        ).summary["evaluation"]
        binding.verify()
        used_ledger = read_bounded(request.output_dir / side / "ledger.json", 4 * 1024**2)
        used_batch = read_bounded(request.output_dir / side / "batch.json", 4 * 1024**2)
        if (
            hashlib.sha256(used_ledger).hexdigest() != binding.ledger_sha256
            or hashlib.sha256(used_batch).hexdigest() != binding.batch_sha256
        ):
            raise ValueError("Comparison review consumed different input bytes")
        origins[side] = {
            r["files"]["packets.jsonl"]
            for r in json.loads(used_ledger)["entries"]
            if r["files"].get("packets.jsonl") != hashlib.sha256(b"").hexdigest()
        }
    for binding in inputs.values():
        binding.verify()
    reasons = [
        reason
        for side, review in reviews.items()
        for reason in _eligibility(side, review, conditions["criteria"])
    ]
    if origins["incumbent"] & origins["candidate"]:
        reasons.append("shared_recording_origins")
    models = {
        side: value.batch["config"]["model"]["manifest_sha256"] for side, value in inputs.items()
    }
    summary = {
        "version": 1,
        "comparison_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "local comparison only; not autonomous policy performance or activation",
        "conditions": conditions,
        "models": models,
        "reviews": reviews,
        **_compare(reviews, conditions, models, reasons),
        "reasons": reasons,
        "candidate_discarded": False,
        "training_state_modified": False,
        "promotion_allowed": False,
        "promotion_blockers": [
            reason
            for reason, key in (
                ("closed_loop_not_verified", "closed_loop_validated"),
                ("upstream_promotion_not_allowed", "automatic_promotion_allowed"),
            )
            if any(not r[key] for r in reviews.values())
        ]
        + (["diagnostic_evidence"] if any(r["diagnostic_only"] for r in reviews.values()) else [])
        + (
            ["independence_not_proven"]
            if any(not r["independence"]["independence_proven"] for r in reviews.values())
            else []
        ),
        "default_changed": False,
        "commands_sent": False,
    }
    write_file(request.output_dir / "selection.json", encode(summary))
    path = request.output_dir / "report.html"
    path.write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>候选比较</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:1rem}</style>"
        "<h1>候选比较</h1><p>局部记录规则比较；不更新实际默认驾驶版本。</p>"
        '<p><a href="incumbent/report.html">既有版本全部尝试</a> · '
        '<a href="candidate/report.html">候选全部尝试</a></p><pre>'
        + html.escape(
            json.dumps(
                {k: v for k, v in summary.items() if k != "reviews"}, ensure_ascii=False, indent=2
            )
        )
        + "</pre></html>",
        encoding="utf-8",
    )
    return RunResult({}, [], [], {"candidate_selection": summary}, path)
