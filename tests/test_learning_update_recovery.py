"""Recover complete update results after an actual parent process exit."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_candidate_store import candidates as candidates
from test_evaluation import sha
from test_learning_loop import SharedBackend
from test_learning_loop import seeded_loop as seeded_loop
from test_learning_update_resume import stopped_sampling

from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue


def interrupted_update(tmp_path, seeded_loop, *, update_mode="stop_before_first_update"):
    assert update_mode in {"stop_before_first_update", "finish_remaining_updates"}
    request, _ = stopped_sampling(tmp_path, seeded_loop)
    root = request.output_dir
    (root / "stop.request").unlink()
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository / "src"), str(repository / "tests"))
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, os, sys
from contextlib import nullcontext
from pathlib import Path
from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue
from test_evaluation import sha
from test_learning_loop import SharedBackend
from test_learning_update_resume import stop_on_creation
root, source = map(Path, sys.argv[1:3])
replace = os.replace
def publish(original, target, *args, **kwargs):
    if Path(target) == root / 'state.json':
        state = json.loads(Path(original).read_bytes())
        if state['phase'] == 'learned':
            os._exit(73)
    return replace(original, target, *args, **kwargs)
os.replace = publish
guard = (stop_on_creation(root / 'round-000/updates-000', root / 'stop.request')
         if sys.argv[3] == 'stop_before_first_update' else nullcontext())
with guard:
    run_experiment(LearningContinue(root, sha(root / 'state.json')),
                   learning_environment=SharedBackend(source))
""",
            str(root),
            str(seeded_loop[0]),
            update_mode,
        ],
        env=environment,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert child.returncode == 73, child.stdout + child.stderr
    state = json.loads((root / "state.json").read_bytes())
    assert state["phase"] == "resuming_updates"
    assert not state["rounds"][0].get("update_segments")
    assert (root / "round-000/updates-000/policy.json").exists()
    return root, state


def test_sealed_update_survives_parent_exit_without_duplicate_credit_or_sampling(
    tmp_path, seeded_loop
):
    root, pending = interrupted_update(tmp_path, seeded_loop)
    checkpoint = root / "round-000/updates-000"
    originals = {path: sha(path) for path in checkpoint.rglob("*") if path.is_file()}
    backend = SharedBackend(seeded_loop[0])
    result = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert result["stop_reason"] == "stop_requested", result.get("error")
    assert result["latest_learner"]["directory"] == str(checkpoint)
    assert result["latest_learner"]["sha256"] == sha(checkpoint / "policy.json")
    assert result["eligible_transitions"] == 3 and result["learner_updates"] == 0
    assert result["latest_learner"]["total_steps"] == pending["latest_learner"]["total_steps"] == 30
    assert result["rounds_completed"] == 0 and len(result["rounds"][0]["update_segments"]) == 1
    assert result["recoveries"][-1]["kind"] == "sealed_updates"
    assert not backend.leases and backend.closed and result["resources_released"]
    assert all(sha(path) == digest for path, digest in originals.items())
    next_backend = SharedBackend(seeded_loop[0])
    repeated = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=next_backend
    ).summary["learning_loop"]
    assert repeated["latest_learner"] == result["latest_learner"]
    assert repeated["learner_updates"] == 0 and repeated["eligible_transitions"] == 3
    assert repeated["recoveries"] == result["recoveries"]
    assert len(repeated["rounds"][0]["update_segments"]) == 1
    assert not (root / "round-000/updates-001").exists()
    assert not next_backend.leases and next_backend.closed


def test_incomplete_pending_update_cannot_change_parent_or_open_an_environment(
    tmp_path, seeded_loop
):
    root, _ = interrupted_update(tmp_path, seeded_loop)
    state = root / "state.json"
    before = state.read_bytes()
    manifest = root / "round-000/updates-000/policy.json"
    manifest.rename(manifest.with_suffix(".unfinished"))
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises(FileNotFoundError):
        run_experiment(LearningContinue(root, sha(state)), learning_environment=backend)
    assert state.read_bytes() == before
    assert not manifest.exists() and manifest.with_suffix(".unfinished").exists()
    assert not backend.leases and backend.closed


def test_valid_ancestor_checkpoint_cannot_replace_the_pending_child(tmp_path, seeded_loop):
    root, pending = interrupted_update(tmp_path, seeded_loop)
    state = root / "state.json"
    before = state.read_bytes()
    checkpoint = root / "round-000/updates-000"
    # Substitute another genuine complete checkpoint, not a fabricated model or validator result.
    ancestor = Path(pending["latest_learner"]["directory"])
    shutil.copytree(ancestor, checkpoint, dirs_exist_ok=True)
    assert sha(checkpoint / "policy.json") == pending["latest_learner"]["sha256"]
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises(ValueError, match="credit, replay or ancestry"):
        run_experiment(LearningContinue(root, sha(state)), learning_environment=backend)
    assert state.read_bytes() == before
    assert sha(checkpoint / "policy.json") == pending["latest_learner"]["sha256"]
    assert not backend.leases and backend.closed


def test_completed_pending_updates_are_evaluated_without_retraining(tmp_path, seeded_loop):
    root, _ = interrupted_update(tmp_path, seeded_loop, update_mode="finish_remaining_updates")
    checkpoint = root / "round-000/updates-000"
    report = json.loads((checkpoint / "training-report.json").read_bytes())
    assert report["steps_completed"] == 3 and report["stop_reason"] == "budget_completed"
    originals = {path: sha(path) for path in checkpoint.rglob("*") if path.is_file()}

    class NoSampling(SharedBackend):
        def sampling(self, identity):
            raise AssertionError("Pending learned checkpoint must not sample again")

    backend = NoSampling(seeded_loop[0])
    result = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert result["stop_reason"] == "budget_completed", result.get("error")
    assert result["rounds_completed"] == 1
    assert result["eligible_transitions"] == result["learner_updates"] == 3
    assert result["latest_learner"]["directory"] == str(checkpoint)
    assert result["latest_learner"]["total_steps"] == 33
    assert result["recoveries"][-1]["completed"] == 3
    assert result["recoveries"][-1]["remaining"] == 0
    assert result["rounds"][0]["evaluation"]["execution_metrics"]["bound_runs"] == 2
    assert len(backend.leases) == 1 and backend.closed and result["resources_released"]
    assert all(sha(path) == digest for path, digest in originals.items())
    assert not (root / "round-000/updates-001").exists()


def test_sealed_update_after_audit_read_failure_is_acknowledged_on_continue(tmp_path, seeded_loop):
    from test_learning_update_resume import (
        test_failed_continuation_audit_preserves_acknowledged_progress,
    )

    test_failed_continuation_audit_preserves_acknowledged_progress(tmp_path, seeded_loop)
    root = tmp_path / "loop"
    before = json.loads((root / "state.json").read_bytes())
    assert before["stop_reason"] == "interface_error"
    checkpoint = root / "round-000/updates-001"
    originals = {path: sha(path) for path in checkpoint.rglob("*") if path.is_file()}
    backend = SharedBackend(seeded_loop[0])
    result = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert result["stop_reason"] == "stop_requested", result.get("error")
    assert result["latest_learner"]["directory"] == str(checkpoint)
    assert result["eligible_transitions"] == 3 and result["learner_updates"] == 0
    assert len(result["rounds"][0]["update_segments"]) == 2
    assert result["recoveries"][-1]["kind"] == "sealed_updates"
    assert result["interruptions"][-1]["stop_reason"] == "interface_error"
    assert all(sha(path) == digest for path, digest in originals.items())
    assert not backend.leases and backend.closed and result["resources_released"]
