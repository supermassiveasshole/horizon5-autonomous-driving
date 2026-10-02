"""Authenticate remaining update credit against sealed sampling and learner ancestry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fh5.artifact_io import VerifiedFile, sha256_file
from fh5.learning_recovery import checkpoint_learning_evidence, completed_sampling
from fh5.replay_document import read_document_fields
from fh5.sac_cycle import SACCycle, SACRealtimeCycle, sampling_update_budget
from fh5.sac_learning import validate_sac_candidate


@dataclass(frozen=True)
class UpdateProgress:
    learner: dict[str, Any]
    earned: int
    completed: int
    segments: int


def retained_update_progress(
    request: SACCycle | SACRealtimeCycle,
    parent: dict[str, Any],
    acknowledged_sha256: str,
    acknowledged_updates: int,
) -> UpdateProgress:
    summary, learner = completed_sampling(request, parent, allow_stopped_updates=True)
    attempt = summary["attempts"][0]
    earned = sampling_update_budget(
        attempt["eligible_transitions"],
        request.max_updates_per_attempt if isinstance(request, SACRealtimeCycle) else None,
    )
    completed = attempt["learner_updates"]
    acknowledged = learner["sha256"] == acknowledged_sha256 and completed == acknowledged_updates
    segments = 0
    while (path := request.output_dir.parent / f"updates-{segments:03d}").exists():
        if completed >= earned:
            raise ValueError("Update continuation differs from its remaining credit")
        previous = read_document_fields(
            VerifiedFile(Path(learner["directory"]) / "policy.json", learner["sha256"]),
            {"configuration", "replay_sha256"},
        )
        digest = sha256_file(path / "policy.json")
        current = {
            "directory": str(path),
            "sha256": digest,
            **validate_sac_candidate(path, digest),
        }
        manifest, report = checkpoint_learning_evidence(path, digest)
        updates = report["steps_completed"]
        configuration = dict(previous["configuration"], steps=earned - completed)
        configuration.setdefault("raw_cache_bytes", 512 * 1024**2)
        if (
            type(updates) is not int
            or not 0 <= updates <= earned - completed
            or report["steps_requested"] != earned - completed
            or report["stop_reason"]
            not in (
                ("budget_completed",)
                if updates == earned - completed
                else ("stop_requested", "training_data_unavailable")
            )
            or report["stop_reason"] == "training_data_unavailable"
            and not report.get("training_error")
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
        segments += 1
        acknowledged |= digest == acknowledged_sha256 and completed == acknowledged_updates
    if not acknowledged:
        raise ValueError("Stopped updates lost their acknowledged checkpoint or progress")
    return UpdateProgress(learner, earned, completed, segments)
