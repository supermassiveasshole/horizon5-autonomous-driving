"""Authenticate a completed sampling child before its parent acknowledges it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fh5.collection_store import read_bounded
from fh5.sac_cycle import SACCycle
from fh5.sac_learning import validate_sac_candidate
from fh5.sampling_evidence import verify_sampling_sources


def _sha(path: Path, limit: int = 4 * 1024**2) -> str:
    return hashlib.sha256(read_bounded(path, limit)).hexdigest()


def completed_sampling(
    request: SACCycle, parent: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Only a sealed successful single attempt can cross this recovery boundary."""
    root = request.output_dir
    protocol = json.loads(read_bounded(root / "protocol.json", 1024**2))
    summary: dict[str, Any] = json.loads(read_bounded(root / "summary.json", 4 * 1024**2))
    files = (request.recording_config_file, request.task_file, request.reward_file)
    if protocol != {
        "source_kind": "synthetic",
        "cycles": 1,
        "steps_per_attempt": request.steps_per_attempt,
        "seed": request.seed,
        "update_ratio": "at most one critic update per newly accepted transition",
        "protocol_files": {str(path): _sha(path, 1024**2) for path in files},
    } or not (
        summary.get("source_kind") == "synthetic"
        and summary.get("stop_reason") == "budget_completed"
        and summary.get("resources_released") is True
        and summary.get("commands_sent_to_game") is False
        and summary.get("latest_candidate") == "candidate-000"
        and not summary.get("error")
        and not summary.get("release_error")
        and len(summary.get("attempts", [])) == 1
    ):
        raise ValueError("Pending sampling did not seal a matching successful child")
    attempt = summary["attempts"][0]
    if (
        attempt != json.loads(read_bounded(root / "attempt-000/cycle-result.json", 4 * 1024**2))
        or attempt.get("sampling_checkpoint_sha256") != parent["sha256"]
        or attempt.get("sampler_seed") != request.seed
        or attempt.get("candidate") != "candidate-000"
        or attempt.get("replay") != "attempt-000/prepared/replay.json"
        or attempt.get("error") is not None
        or attempt.get("inference_reload_max_error") != 0
    ):
        raise ValueError("Pending sampling result differs from its parent or sealed attempt")
    verify_sampling_sources(attempt.get("source_assets", {}))
    replay_path = root / attempt["replay"]
    if _sha(replay_path, 128 * 1024**2) != attempt["replay_sha256"]:
        raise ValueError("Pending sampling replay changed")
    replay = json.loads(read_bounded(replay_path, 128 * 1024**2))
    source_hashes = replay["source_hashes"]
    originals = {
        "packets": root / "attempt-000/recording/packets.jsonl",
        "session": root / "attempt-000/recording/session.json",
        "trace": root / "attempt-000/trace.json",
        "task": request.task_file,
        "reward": request.reward_file,
    }
    if any(source_hashes[key] != _sha(path, 256 * 1024**2) for key, path in originals.items()):
        raise ValueError("Pending sampling replay differs from its original evidence")
    candidate = root / "candidate-000"
    manifest_sha = _sha(candidate / "policy.json")
    if attempt["candidate_sha256"] != manifest_sha:
        raise ValueError("Pending sampling candidate changed")
    learner = {
        "directory": str(candidate.resolve()),
        "sha256": manifest_sha,
        **validate_sac_candidate(candidate, manifest_sha),
    }
    manifest = json.loads(read_bounded(candidate / "policy.json", 4 * 1024**2))
    report = json.loads(read_bounded(candidate / "training-report.json", 128 * 1024**2))
    count, updates = attempt["eligible_transitions"], attempt["learner_updates"]
    if (
        type(count) is not int
        or not 1 <= count <= request.steps_per_attempt
        or count != len(replay["transitions"])
        or type(updates) is not int
        or updates != count
        or report["stop_reason"] != "budget_completed"
        or report["steps_completed"] != updates
        or report["steps_requested"] != count
        or learner["total_steps"] != parent["total_steps"] + updates
        or attempt["total_steps"] != learner["total_steps"]
        or manifest["continuation"]
        != {
            "parent_checkpoint_sha256": parent["sha256"],
            "parent_step": parent["total_steps"],
            "experience_additions": [attempt["replay_sha256"]],
            "new_transition_credit": count,
        }
    ):
        raise ValueError("Pending sampling has inconsistent learning progress or ancestry")
    return summary, learner
