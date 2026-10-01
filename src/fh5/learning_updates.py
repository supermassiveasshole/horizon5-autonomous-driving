"""Authenticate remaining update credit against sealed sampling and learner ancestry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fh5.collection_store import read_bounded
from fh5.learning_recovery import completed_sampling
from fh5.sac_cycle import SACCycle, SACRealtimeCycle
from fh5.sac_learning import validate_sac_candidate


@dataclass(frozen=True)
class UpdateProgress:
    learner: dict[str, Any]
    earned: int
    completed: int


def retained_update_progress(
    request: SACCycle | SACRealtimeCycle, parent: dict[str, Any], segments: list[dict[str, Any]]
) -> UpdateProgress:
    if len(segments) > 10:
        raise ValueError("Learning round exceeds 10 retained update continuations")
    summary, learner = completed_sampling(request, parent, allow_stopped_updates=True)
    attempt = summary["attempts"][0]
    earned, completed = attempt["eligible_transitions"], attempt["learner_updates"]
    if isinstance(request, SACRealtimeCycle):
        earned = min(earned, request.max_updates_per_attempt)
    for number, segment in enumerate(segments):
        path = request.output_dir.parent / f"updates-{number:03d}"
        if Path(segment["directory"]) != path or completed >= earned:
            raise ValueError("Update continuation differs from its remaining credit")
        previous = json.loads(read_bounded(Path(learner["directory"]) / "policy.json", 1024**2))
        current = {
            "directory": str(path),
            "sha256": segment["sha256"],
            **validate_sac_candidate(path, segment["sha256"]),
        }
        manifest = json.loads(read_bounded(path / "policy.json", 1024**2))
        report = json.loads(read_bounded(path / "training-report.json", 128 * 1024**2))
        updates = report["steps_completed"]
        configuration = dict(previous["configuration"], steps=earned - completed)
        configuration.setdefault("raw_cache_bytes", 512 * 1024**2)
        if (
            type(updates) is not int
            or not 0 <= updates <= earned - completed
            or report["steps_requested"] != earned - completed
            or report["stop_reason"]
            != ("budget_completed" if updates == earned - completed else "stop_requested")
            or report["experience_added_transitions"] != 0
            or manifest["configuration"] != configuration
            or manifest["replay_sha256"] != previous["replay_sha256"]
            or current["total_steps"] != learner["total_steps"] + updates
            or manifest["continuation"]
            != {
                "parent_checkpoint_sha256": learner["sha256"],
                "parent_step": learner["total_steps"],
            }
        ):
            raise ValueError("Update continuation changed its credit, replay or ancestry")
        completed += updates
        learner = current
    return UpdateProgress(learner, earned, completed)
