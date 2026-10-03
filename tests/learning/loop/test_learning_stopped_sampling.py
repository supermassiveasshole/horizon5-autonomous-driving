"""Retain accepted experience when a stopped child exits before parent publication."""

import json

import pytest

from fh5.experiment import run_experiment
from fh5.learning.loop.runner import LearningContinue
from tests.evaluation.test_candidate_store import candidates as candidates
from tests.evaluation.test_evaluation import sha
from tests.learning.loop.test_learning_loop import SharedBackend, loop_request
from tests.learning.loop.test_learning_loop import seeded_loop as seeded_loop
from tests.learning.loop.test_learning_recovery import interrupt_selection
from tests.support.learning_files import update_bindings


def stopped_pending(tmp_path, seeded_loop, *, retry_budget=0):
    request = loop_request(
        tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": retry_budget}
    )
    interrupt_selection(request, seeded_loop[0], "before_learned", scenario="stopped_updates")
    root = request.output_dir
    state = json.loads((root / "state.json").read_bytes())
    assert state["phase"] == "updating" and "learning" not in state["rounds"][0]
    child = root / "round-000/learning"
    result = json.loads((child / "summary.json").read_bytes())
    assert result["stop_reason"] == "stop_requested" and result["resources_released"]
    assert result["latest_candidate"] == "candidate-000"
    assert result["attempts"][0]["eligible_transitions"] == 3
    assert result["attempts"][0]["learner_updates"] == 0
    assert (child / "candidate-000/policy.json").exists()
    return root, child


def test_stopped_sampling_checkpoint_keeps_accepted_credit_without_resampling(
    tmp_path, seeded_loop
):
    root, child = stopped_pending(tmp_path, seeded_loop)
    originals = {path: sha(path) for path in child.rglob("*") if path.is_file()}
    backend = SharedBackend(seeded_loop[0])
    recovered = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert recovered["stop_reason"] == "stop_requested", recovered.get("error")
    assert recovered["latest_learner"]["directory"] == str(child / "candidate-000")
    assert recovered["latest_learner"]["total_steps"] == 30
    assert recovered["eligible_transitions"] == 3 and recovered["learner_updates"] == 0
    assert recovered["rounds_completed"] == 0
    assert recovered["recoveries"][-1]["kind"] == "sealed_sampling"
    assert not backend.leases and backend.closed and recovered["resources_released"]
    assert all(sha(path) == digest for path, digest in originals.items())
    again = SharedBackend(seeded_loop[0])
    repeated = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=again
    ).summary["learning_loop"]
    assert repeated["latest_learner"] == recovered["latest_learner"]
    assert repeated["recoveries"] == recovered["recoveries"]
    assert repeated["eligible_transitions"] == 3 and repeated["learner_updates"] == 0
    assert not again.leases and again.closed
    assert not (root / "round-000/learning-001").exists()
    assert not (root / "round-000/updates-000").exists()


@pytest.mark.parametrize("fault", ["missing_manifest", "changed_trace"])
def test_stopped_sampling_requires_a_complete_model_and_unchanged_originals(
    tmp_path, seeded_loop, fault
):
    root, child = stopped_pending(tmp_path, seeded_loop, retry_budget=1)
    state_file = root / "state.json"
    before = state_file.read_bytes()
    if fault == "missing_manifest":
        manifest = child / "candidate-000/policy.json"
        manifest.rename(manifest.with_suffix(".unfinished"))
    else:
        (child / "attempt-000/trace.json").write_text('{"changed": true}')
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises((ValueError, FileNotFoundError)):
        run_experiment(LearningContinue(root, sha(state_file)), learning_environment=backend)
    assert state_file.read_bytes() == before
    assert not backend.leases and backend.closed


def test_stopped_sampling_recovery_completes_earned_updates_without_another_sample(
    tmp_path, seeded_loop
):
    root, child = stopped_pending(tmp_path, seeded_loop, retry_budget=1)
    originals = {path: sha(path) for path in child.rglob("*") if path.is_file()}
    (root / "stop.request").unlink()

    class NoSampling(SharedBackend):
        def sampling(self, identity):
            raise AssertionError("Sealed accepted experience must not be sampled again")

    backend = NoSampling(seeded_loop[0])
    recovered = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert recovered["stop_reason"] == "budget_completed", recovered.get("error")
    assert recovered["eligible_transitions"] == recovered["learner_updates"] == 3
    assert recovered["rounds_completed"] == 1
    assert recovered["latest_learner"]["total_steps"] == 33
    assert len(update_bindings(root, recovered["rounds"][0])) == 1
    assert recovered["rounds"][0]["evaluation"]["execution_metrics"]["bound_runs"] == 2
    assert len(backend.leases) == 1 and backend.closed and recovered["resources_released"]
    assert all(sha(path) == digest for path, digest in originals.items())
    assert not (root / "round-000/learning-001").exists()


def test_sampling_retry_setting_does_not_discard_a_stopped_complete_learner(tmp_path, seeded_loop):
    root, child = stopped_pending(tmp_path, seeded_loop, retry_budget=1)
    backend = SharedBackend(seeded_loop[0])
    recovered = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert recovered["stop_reason"] == "stop_requested", recovered.get("error")
    assert recovered["eligible_transitions"] == 3 and recovered["learner_updates"] == 0
    assert recovered["latest_learner"]["sha256"] == sha(child / "candidate-000/policy.json")
    assert not recovered["rounds"][0].get("sampling_history")
    assert recovered["rounds"][0].get("sampling_attempt", 0) == 0
    assert recovered["recoveries"][-1]["kind"] == "sealed_sampling"
    assert not backend.leases and backend.closed and recovered["resources_released"]
