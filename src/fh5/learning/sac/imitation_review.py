"""Recompute development evidence before changing a candidate's guidance phase."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from fh5.artifacts.io import encode, read_bounded
from fh5.evaluation.candidate_selection import CandidateCompare


def review_imitation(
    state: dict[str, Any],
    comparison: Path,
    registry: Path | None,
    output: Path,
    *,
    parent_sha256: str,
    bc_manifest: dict[str, Any],
    learning_origins: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    from fh5.experiment import run_experiment

    if state["phase"] == "exited":
        raise ValueError("Imitation already exited; a new evaluation cannot revive it")
    if len(state["transitions"]) >= 64:
        raise ValueError("Imitation evaluation history exceeds 64 reviews")
    reviewed = run_experiment(CandidateCompare(comparison, output, registry)).summary[
        "candidate_selection"
    ]
    if reviewed["models"] != {
        "incumbent": state["teacher_manifest_sha256"],
        "candidate": parent_sha256,
    }:
        raise ValueError(
            "Imitation evaluation must bind the original BC and exact current candidate"
        )
    if hashlib.sha256(encode(reviewed["conditions"])).hexdigest() != state["protocol_sha256"]:
        raise ValueError("Imitation evaluation differs from its frozen development protocol")
    reasons = (
        []
        if reviewed["local_recommendation"] == "prefer_candidate_locally"
        else list(reviewed["reasons"])
    )
    origins: set[str] = set()
    attempts = {}
    for side, review in reviewed["reviews"].items():
        executions = review["executions"]
        expected_actor = (
            "frozen-numeric-sac-v1" if side == "candidate" else "frozen-numeric-temporal-bc-v2"
        )
        actual = sum(
            item["status"] == "bound_diagnostic"
            and item["verified_predictions"] > 0
            and item.get("actor_kind") == expected_actor
            and (item.get("metrics") or {}).get("evidence_kind") == "synthetic"
            for item in executions
        )
        attempts[side] = actual
        if actual != review["planned_runs"]:
            reasons.append(side + ":actual_synthetic_policy_execution_incomplete")
        if review["verified_starts"] != review["planned_runs"] or any(
            item.get("source_kind") != "synthetic" for item in review["starts"]
        ):
            reasons.append(side + ":verified_synthetic_starts_incomplete")
        if review["independence"]["status"] != "no_known_overlap":
            reasons.append(side + ":development_origins_not_separate_in_registry")
        ledger = json.loads(read_bounded(output / side / "ledger.json", 4 * 1024**2))
        for entry in ledger["entries"]:
            origin = entry["files"].get("packets.jsonl")
            if origin is None:
                reasons.append(side + ":recording_origin_missing")
            else:
                origins.add(origin)
    if bc_manifest.get("provenance", {}).get("kind") != "synthetic":
        reasons.append("teacher_training_lineage_not_supported_by_synthetic_gate")
    if origins & learning_origins:
        reasons.append("evaluation_reuses_learning_recording")
    if any(origins & set(item["recording_origins"]) for item in state["transitions"]):
        raise ValueError(
            "Imitation evaluation recording already consumed by this candidate lineage"
        )
    allowed = not reasons
    summary = {
        "scope": "synthetic_development_only",
        "advanced": allowed,
        "from_index": state["index"],
        "to_index": state["index"] + int(allowed),
        "parent_checkpoint_sha256": parent_sha256,
        "protocol_sha256": state["protocol_sha256"],
        "recording_origins": sorted(origins),
        "actual_policy_attempts": attempts,
        "reasons": reasons or ["frozen_local_performance_criteria_met"],
        "independence_limits": "Registered raw origins only; transformed or undeclared training history is not proven independent",
        "default_changed": False,
        "real_driving_validated": False,
        "comparison": reviewed,
    }
    payload = encode(summary)
    digest = hashlib.sha256(payload).hexdigest()
    path = f"imitation-evidence/{digest}.json"
    new = deepcopy(state)
    new["index"] = summary["to_index"]
    new["weight"] = state["weights"][new["index"]]
    new["phase"] = "exited" if new["weight"] == 0 else "guided"
    new["transitions"].append(
        {
            k: summary[k]
            for k in ("from_index", "to_index", "parent_checkpoint_sha256", "recording_origins")
        }
        | {"review": path, "review_sha256": digest}
    )
    return new, summary, {path: payload}
