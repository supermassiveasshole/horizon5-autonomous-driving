"""Retain failed sampling attempts while bounded retries keep the same learner."""

import json
from pathlib import Path

import pytest
from test_candidate_store import candidates as candidates
from test_evaluation import sha
from test_learning_loop import SharedBackend, loop_request
from test_learning_loop import seeded_loop as seeded_loop

from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue


def test_failed_sampling_is_retried_in_new_directories_with_a_finite_budget(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})

    class FailingSampler(SharedBackend):
        identities = None

        def sampling(self, identity):
            if self.identities is None:
                self.identities = []
            self.identities.append(identity)
            lease = super().sampling(identity)
            lease.fail_at = 0
            return lease

    backend = FailingSampler(seeded_loop[0])
    store_before = sha(tmp_path / "versions/state.sqlite")
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "sampling_retries_exhausted", result.get("error")
    assert backend.identities == ["round-000", "round-000-sampling-001"]
    assert len(backend.leases) == 2 and all(lease.closed for lease in backend.leases)
    assert result["learner_updates"] == result["rounds_completed"] == 0
    assert result["latest_learner"] == result["explorer"]
    assert sha(tmp_path / "versions/state.sqlite") == store_before
    row = result["rounds"][0]
    bindings = [*row["sampling_history"], row["learning"]]
    assert [Path(item["directory"]).name for item in bindings] == ["learning", "learning-001"]
    for item in bindings:
        directory = Path(item["directory"])
        assert sha(directory / "summary.json") == item["summary_sha256"]
        failed = json.loads((directory / "summary.json").read_bytes())
        assert failed["resources_released"] and failed["stop_reason"] == "sampling_fault"
        assert len(failed["attempts"]) == 1
        assert failed["attempts"][0]["decisions"][0]["status"] == "failed_or_unknown"
        assert (directory / "attempt-000/trace.json").is_file()
    assert backend.closed and result["resources_released"]


def test_stop_requires_explicit_resume_and_does_not_reset_the_sampling_retry_budget(
    tmp_path, seeded_loop
):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})
    stop = request.output_dir / "stop.request"
    first = SharedBackend(seeded_loop[0])
    first.stop_sampling_file = stop
    interrupted = run_experiment(request, learning_environment=first).summary["learning_loop"]
    assert interrupted["stop_reason"] == "stop_requested" and len(first.leases) == 1
    original = {
        path: sha(path)
        for path in (request.output_dir / "round-000/learning").rglob("*")
        if path.is_file()
    }
    state = request.output_dir / "state.json"
    stop.unlink()
    second = SharedBackend(seeded_loop[0])
    second.stop_sampling_file = stop
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(state)), learning_environment=second
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "stop_requested", resumed.get("error")
    assert len(second.leases) == 1 and resumed["rounds"][0]["sampling_attempt"] == 1
    assert resumed["learner_updates"] == resumed["eligible_transitions"] == 0
    assert resumed["latest_learner"] == interrupted["latest_learner"]
    assert all(sha(path) == digest for path, digest in original.items())
    stop.unlink()
    third = SharedBackend(seeded_loop[0])
    exhausted = run_experiment(
        LearningContinue(request.output_dir, sha(state)), learning_environment=third
    ).summary["learning_loop"]
    assert exhausted["stop_reason"] == "sampling_retries_exhausted", exhausted.get("error")
    assert not third.leases and third.closed and exhausted["resources_released"]
    assert exhausted["learner_updates"] == 0 and len(exhausted["recoveries"]) == 1
    assert all(sha(path) == digest for path, digest in original.items())


@pytest.mark.parametrize("changed", ["protocol.json", "attempt-000/trace.json"])
def test_resume_rejects_changes_to_an_archived_failed_attempt(tmp_path, seeded_loop, changed):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})

    class BrokenSampler(SharedBackend):
        def sampling(self, identity):
            lease = super().sampling(identity)
            lease.fail_at = 0
            return lease

    result = run_experiment(request, learning_environment=BrokenSampler(seeded_loop[0])).summary[
        "learning_loop"
    ]
    assert result["stop_reason"] == "sampling_retries_exhausted"
    old = request.output_dir / "round-000/learning" / changed
    old.write_bytes(old.read_bytes() + b"\n")
    state = request.output_dir / "state.json"
    original = state.read_bytes()
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises(ValueError, match="originals|result changed"):
        run_experiment(
            LearningContinue(request.output_dir, sha(state)), learning_environment=backend
        )
    assert state.read_bytes() == original and not backend.leases and backend.closed


def test_a_fresh_sampling_retry_can_learn_and_evaluate_without_erasing_the_failure(
    tmp_path, seeded_loop
):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})

    class RecoveredSampler(SharedBackend):
        def sampling(self, identity):
            lease = super().sampling(identity)
            if len(self.leases) == 1:
                lease.fail_at = 0
            return lease

    backend = RecoveredSampler(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "budget_completed", result.get("error")
    assert result["rounds_completed"] == 1 and result["learner_updates"] == 3
    assert result["eligible_transitions"] == 3 and result["latest_learner"]["total_steps"] == 33
    row = result["rounds"][0]
    assert len(row["sampling_history"]) == 1
    assert row["evaluation"]["metrics"]["all_attempts"] == 2
    assert row["selection"] == "retain_incumbent"
    assert Path(row["learning"]["directory"]).name == "learning-001"
    assert len(backend.leases) == 3 and all(lease.closed for lease in backend.leases)
    failure = Path(row["sampling_history"][0]["directory"])
    failed = json.loads((failure / "summary.json").read_bytes())
    assert failed["stop_reason"] == "sampling_fault" and not failed.get("latest_candidate")
    assert result["resources_released"] and backend.closed


@pytest.mark.parametrize("fault", ["partial_candidate", "unreleased"])
def test_sampling_retry_refuses_partial_updates_or_unreleased_resources(
    tmp_path, seeded_loop, fault
):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 2})

    class UnsafeSampler(SharedBackend):
        def sampling(self, identity):
            lease = super().sampling(identity)
            lease.fail_at = 0
            finish, close = lease.finish, lease.close

            def review(recording):
                proof = finish(recording)
                if fault == "partial_candidate":
                    incomplete = recording.parent.parent / "candidate-000"
                    incomplete.mkdir()
                    (incomplete / "interrupted-write.bin").write_bytes(b"partial output")
                return proof

            def release():
                result = close()
                return {**result, "resources_released": fault != "unreleased"}

            lease.finish, lease.close = review, release
            return lease

    backend = UnsafeSampler(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == (
        "release_fault" if fault == "unreleased" else "interface_error"
    )
    assert len(backend.leases) == 1 and backend.closed
    assert result["learner_updates"] == 0 and not result["rounds"][0].get("sampling_history")
    assert not (request.output_dir / "round-000/learning-001").exists()
    if fault == "partial_candidate":
        assert (
            request.output_dir / "round-000/learning/candidate-000/interrupted-write.bin"
        ).read_bytes() == b"partial output"


def test_no_eligible_experience_uses_the_same_finite_retry_budget(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})

    class UnreviewedSampler(SharedBackend):
        def sampling(self, identity):
            lease = super().sampling(identity)
            lease.finish = lambda _: None
            return lease

    backend = UnreviewedSampler(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "sampling_retries_exhausted", result.get("error")
    assert result["eligible_transitions"] == result["learner_updates"] == 0
    assert len(backend.leases) == 2 and all(lease.closed for lease in backend.leases)
    row = result["rounds"][0]
    for binding in [*row["sampling_history"], row["learning"]]:
        child = json.loads((Path(binding["directory"]) / "summary.json").read_bytes())
        assert child["stop_reason"] == "no_eligible_experience"
        assert child["attempts"][0]["eligible_transitions"] == 0
    assert result["resources_released"] and backend.closed
