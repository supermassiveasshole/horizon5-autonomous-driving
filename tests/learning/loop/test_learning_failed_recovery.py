"""Recover sealed sampling failures after a real parent process exit."""

import json

import pytest

from fh5.experiment import run_experiment
from fh5.learning.loop.runner import LearningContinue
from tests.evaluation.test_candidate_store import candidates as candidates
from tests.evaluation.test_evaluation import sha
from tests.learning.loop.test_learning_loop import SharedBackend, loop_request
from tests.learning.loop.test_learning_loop import seeded_loop as seeded_loop
from tests.learning.loop.test_learning_recovery import interrupt_selection


def test_sealed_failed_sampling_survives_exit_before_parent_acknowledgement(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})
    interrupt_selection(request, seeded_loop[0], "before_learned", scenario="failed_sampling")
    state = request.output_dir / "state.json"
    pending = json.loads(state.read_bytes())
    assert pending["phase"] == "updating" and "learning" not in pending["rounds"][0]
    original = request.output_dir / "round-000/learning"
    files = {path: sha(path) for path in original.rglob("*") if path.is_file()}
    failed = json.loads((original / "summary.json").read_bytes())
    assert failed["stop_reason"] == "sampling_fault" and failed["resources_released"]
    assert not failed.get("latest_candidate")

    class StillFailing(SharedBackend):
        def sampling(self, identity):
            lease = super().sampling(identity)
            lease.fail_at = 0
            return lease

    backend = StillFailing(seeded_loop[0])
    recovered = run_experiment(
        LearningContinue(request.output_dir, sha(state)), learning_environment=backend
    ).summary["learning_loop"]
    assert recovered["stop_reason"] == "sampling_retries_exhausted", recovered.get("error")
    assert recovered["latest_learner"] == pending["latest_learner"]
    assert recovered["learner_updates"] == recovered["eligible_transitions"] == 0
    assert len(backend.leases) == 1 and backend.leases[0].closed
    assert backend.closed and recovered["resources_released"]
    assert recovered["recoveries"][0]["kind"] == "sealed_failed_sampling"
    assert recovered["recoveries"][1]["kind"] == "sampling_retry"
    assert recovered["interruptions"][0]["stop_reason"] == "unclean_exit"
    row = recovered["rounds"][0]
    assert row["sampling_attempt"] == 1 and len(row["sampling_history"]) == 1
    assert row["sampling_history"][0]["summary_sha256"] == sha(original / "summary.json")
    assert all(sha(path) == digest for path, digest in files.items())


def test_pending_failure_cannot_omit_the_binding_of_a_corrupted_original(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})
    interrupt_selection(request, seeded_loop[0], "before_learned", scenario="failed_sampling")
    state = request.output_dir / "state.json"
    before = state.read_bytes()
    child = request.output_dir / "round-000/learning"
    summary_file = child / "summary.json"
    summary = json.loads(summary_file.read_bytes())
    trace = child / "attempt-000/trace.json"
    del summary["attempts"][0]["source_assets"][str(trace.resolve())]
    summary_file.write_text(json.dumps(summary))
    trace.write_text('{"lost": "original trace"}')
    backend = SharedBackend(seeded_loop[0])
    # A false acceptance must still remain a zero-update failure during this test.
    backend.stop_sampling_file = request.output_dir / "stop.request"
    with pytest.raises(ValueError, match="inventory|originals"):
        run_experiment(
            LearningContinue(request.output_dir, sha(state)), learning_environment=backend
        )
    assert state.read_bytes() == before and not backend.leases and backend.closed


def test_pending_second_failure_does_not_restore_spent_retry_credit(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})
    interrupt_selection(
        request, seeded_loop[0], "before_retried_failure", scenario="failed_sampling"
    )
    state = request.output_dir / "state.json"
    pending = json.loads(state.read_bytes())
    assert pending["rounds"][0]["sampling_attempt"] == 1
    assert len(pending["rounds"][0]["sampling_history"]) == 1
    backend = SharedBackend(seeded_loop[0])
    recovered = run_experiment(
        LearningContinue(request.output_dir, sha(state)), learning_environment=backend
    ).summary["learning_loop"]
    assert recovered["stop_reason"] == "sampling_retries_exhausted", recovered.get("error")
    assert not backend.leases and backend.closed and recovered["resources_released"]
    assert recovered["latest_learner"] == pending["latest_learner"]
    assert recovered["learner_updates"] == recovered["eligible_transitions"] == 0
    assert recovered["recoveries"][-1]["kind"] == "sealed_failed_sampling"
    assert recovered["rounds"][0]["sampling_attempt"] == 1


@pytest.mark.parametrize("fault", ["false_packet_count", "empty_attempts", "missing_attempt"])
def test_pending_failure_cannot_hide_recorded_input_or_the_original_attempt(
    tmp_path, seeded_loop, fault
):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})
    interrupt_selection(request, seeded_loop[0], "before_learned", scenario="failed_sampling")
    state = request.output_dir / "state.json"
    before = state.read_bytes()
    child = request.output_dir / "round-000/learning"
    summary_file = child / "summary.json"
    summary = json.loads(summary_file.read_bytes())
    if fault == "false_packet_count":
        trace = child / "attempt-000/trace.json"
        summary["attempts"][0]["received_packets"] = 0
        del summary["attempts"][0]["source_assets"][str(trace.resolve())]
        trace.unlink()
    else:
        summary["attempts"] = []
        summary["stop_reason"] = "stop_requested"
        if fault == "missing_attempt":
            original = (child / "attempt-000").resolve()
            unavailable = (tmp_path / "unavailable-attempt").resolve()
            assert original.is_relative_to(tmp_path.resolve())
            assert unavailable.is_relative_to(tmp_path.resolve())
            original.rename(unavailable)
    summary_file.write_text(json.dumps(summary))
    backend = SharedBackend(seeded_loop[0])
    backend.stop_sampling_file = request.output_dir / "stop.request"
    with pytest.raises(ValueError, match="sampling|originals"):
        run_experiment(
            LearningContinue(request.output_dir, sha(state)), learning_environment=backend
        )
    assert state.read_bytes() == before and not backend.leases and backend.closed


@pytest.mark.parametrize("fault", ["missing_summary", "unreleased", "wrong_seed", "partial_model"])
def test_pending_failed_sampling_requires_complete_released_matching_originals(
    tmp_path, seeded_loop, fault
):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})
    interrupt_selection(request, seeded_loop[0], "before_learned", scenario="failed_sampling")
    state = request.output_dir / "state.json"
    before = state.read_bytes()
    child = request.output_dir / "round-000/learning"
    summary = child / "summary.json"
    if fault == "missing_summary":
        summary.unlink()
    elif fault == "unreleased":
        value = json.loads(summary.read_bytes())
        value["resources_released"] = False
        summary.write_text(json.dumps(value))
    elif fault == "wrong_seed":
        protocol = child / "protocol.json"
        value = json.loads(protocol.read_bytes())
        value["seed"] += 1
        protocol.write_text(json.dumps(value))
    else:
        incomplete = child / "candidate-000"
        incomplete.mkdir()
        (incomplete / "partial.bin").write_bytes(b"unsealed learner")
    backend = SharedBackend(seeded_loop[0])
    error = FileNotFoundError if fault == "missing_summary" else ValueError
    with pytest.raises(error):
        run_experiment(
            LearningContinue(request.output_dir, sha(state)), learning_environment=backend
        )
    assert state.read_bytes() == before and not backend.leases and backend.closed


@pytest.mark.parametrize("fault", ["attachment", "review", "declaration"])
def test_pending_failure_cannot_omit_an_independent_review_original(tmp_path, seeded_loop, fault):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})
    interrupt_selection(request, seeded_loop[0], "before_learned", scenario="failed_sampling")
    state = request.output_dir / "state.json"
    before = state.read_bytes()
    child = request.output_dir / "round-000/learning"
    summary_file = child / "summary.json"
    summary = json.loads(summary_file.read_bytes())
    review = child / "attempt-000/evidence.json"
    proof = json.loads(review.read_bytes())
    original = {
        "attachment": review.parent / proof["items"][0]["path"],
        "review": review,
        "declaration": review.parent / "sampling-sources.json",
    }[fault]
    del summary["attempts"][0]["source_assets"][str(original.resolve())]
    original.unlink()
    summary_file.write_text(json.dumps(summary))
    backend = SharedBackend(seeded_loop[0])
    backend.stop_sampling_file = request.output_dir / "stop.request"
    with pytest.raises(ValueError, match="sampling|inventory|originals"):
        run_experiment(
            LearningContinue(request.output_dir, sha(state)), learning_environment=backend
        )
    assert state.read_bytes() == before and not backend.leases and backend.closed
