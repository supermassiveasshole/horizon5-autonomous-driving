"""Bounded environment acquisition retries through the experiment-run seam."""

import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest
from test_candidate_store import candidates as candidates
from test_evaluation import sha
from test_learning_loop import SharedBackend, loop_request
from test_learning_loop import seeded_loop as seeded_loop

from fh5.experiment import run_experiment
from fh5.learning_io import LearningUnavailable
from fh5.learning_loop import LearningContinue, LearningLoop


def test_unavailable_sampler_exhausts_declared_retries_without_starting_an_attempt(
    tmp_path, seeded_loop
):
    request = loop_request(
        tmp_path,
        seeded_loop,
        rounds=1,
        acquisition_retry={"max_retries": 2, "delay_seconds": 0},
    )

    class UnavailableBackend(SharedBackend):
        def __init__(self, source):
            super().__init__(source)
            self.requests = []

        def sampling(self, identity):
            self.requests.append(identity)
            raise LearningUnavailable("Synthetic service loading", resources_released=True)

    backend = UnavailableBackend(seeded_loop[0])
    original = sha(tmp_path / "versions/state.sqlite")
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "acquisition_retries_exhausted"
    assert backend.requests == ["round-000"] * 3
    failures = result["acquisition_failures"]
    assert [item["attempt"] for item in failures] == [1, 2, 3]
    assert all(item["kind"] == "sampling" and item["round"] == 0 for item in failures)
    assert all(item["error"] == "Synthetic service loading" for item in failures)
    assert all(item["resources_released"] is True for item in failures)
    assert result["learner_updates"] == result["rounds_completed"] == 0
    assert result["resources_released"] and backend.closed and not backend.leases
    assert not (request.output_dir / "round-000/learning").exists()
    assert sha(tmp_path / "versions/state.sqlite") == original


@pytest.fixture
def waiting_retry_evaluation(tmp_path, seeded_loop):
    request = loop_request(
        tmp_path,
        seeded_loop,
        rounds=1,
        acquisition_retry={"max_retries": 2, "delay_seconds": 0},
    )

    class OfflineEvaluation(SharedBackend):
        def evaluation(self, identity):
            raise OSError("Evaluation intentionally unavailable for initial handoff")

    result = run_experiment(
        request, learning_environment=OfflineEvaluation(seeded_loop[0])
    ).summary["learning_loop"]
    assert result["stop_reason"] == "interface_error" and result["learner_updates"] == 3
    assert not (request.output_dir / "round-000/evaluation").exists()
    return request, seeded_loop[0]


def test_evaluation_retries_keep_the_trained_candidate_without_resampling(
    tmp_path, waiting_retry_evaluation
):
    request, source = waiting_retry_evaluation
    state_file = request.output_dir / "state.json"
    before = json.loads(state_file.read_bytes())
    learning_file = Path(before["rounds"][0]["learning"]["directory"]) / "summary.json"
    original_learning = sha(learning_file)

    class UnavailableEvaluation(SharedBackend):
        requests = 0

        def sampling(self, identity):
            raise AssertionError("A trained candidate must not be resampled")

        def evaluation(self, identity):
            self.requests += 1
            raise LearningUnavailable("Evaluation service loading", resources_released=True)

    backend = UnavailableEvaluation(source)
    result = run_experiment(
        LearningContinue(request.output_dir, sha(state_file)), learning_environment=backend
    ).summary["learning_loop"]
    assert result["stop_reason"] == "acquisition_retries_exhausted"
    assert backend.requests == 3 and backend.closed and not backend.leases
    assert result["learner_updates"] == 3 and result["rounds_completed"] == 0
    assert result["latest_learner"] == before["latest_learner"]
    assert sha(learning_file) == original_learning
    failures = result["acquisition_failures"]
    assert [failure["attempt"] for failure in failures] == [1, 2, 3]
    assert all(failure["kind"] == "evaluation" for failure in failures)
    assert result["resources_released"]
    assert not (request.output_dir / "round-000/evaluation").exists()


def test_stop_during_sampler_acquisition_releases_without_creating_a_child(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)

    class StopOnOpen(SharedBackend):
        def sampling(self, identity):
            source = super().sampling(identity)
            (request.output_dir / "stop.request").write_text("Operator stopped while connecting")
            return source

    backend = StopOnOpen(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "stop_requested"
    assert len(backend.leases) == 1 and backend.leases[0].closed
    assert not backend.leases[0].commands
    assert not (request.output_dir / "round-000/learning").exists()
    assert result["learner_updates"] == result["rounds_completed"] == 0
    assert result["resources_released"] and backend.closed


@pytest.mark.parametrize("fault", ["release_failed", "ordinary_error", "no_retry_configuration"])
def test_acquisition_never_retries_an_unapproved_or_unreleased_failure(
    tmp_path, seeded_loop, fault
):
    options = (
        {}
        if fault == "no_retry_configuration"
        else {"acquisition_retry": {"max_retries": 3, "delay_seconds": 0}}
    )
    request = loop_request(tmp_path, seeded_loop, rounds=1, **options)

    class FailingBackend(SharedBackend):
        requests = 0

        def sampling(self, identity):
            self.requests += 1
            if fault == "ordinary_error":
                raise OSError("Undiagnosed connection failure")
            raise LearningUnavailable(
                "Unavailable source", resources_released=fault != "release_failed"
            )

    backend = FailingBackend(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert backend.requests == 1 and backend.closed and not backend.leases
    assert (
        result["learner_updates"] == 0 and not (request.output_dir / "round-000/learning").exists()
    )
    if fault == "release_failed":
        assert result["stop_reason"] == "release_fault"
        assert result["environment"]["resources_released"] is True
        assert result["resources_released"] is False
    elif fault == "ordinary_error":
        assert result["stop_reason"] == "interface_error"
        assert "Undiagnosed connection failure" in result["error"]
    else:
        assert result["stop_reason"] == "acquisition_retries_exhausted"
        assert len(result["acquisition_failures"]) == 1


def test_stop_interrupts_the_retry_wait_without_another_acquisition(tmp_path, seeded_loop):
    request = loop_request(
        tmp_path,
        seeded_loop,
        rounds=1,
        acquisition_retry={"max_retries": 2, "delay_seconds": 5},
    )
    operator_done = threading.Event()
    halt = threading.Event()
    stop_times = []

    def operator():
        while not halt.wait(0.01):
            try:
                phase = json.loads((request.output_dir / "state.json").read_bytes())["phase"]
            except (OSError, ValueError):
                continue
            if phase == "waiting_for_interface":
                stop_times.append(time.monotonic())
                (request.output_dir / "stop.request").write_text(
                    "Operator stopped during retry wait"
                )
                operator_done.set()
                return

    class UnavailableBackend(SharedBackend):
        requests = 0

        def sampling(self, identity):
            self.requests += 1
            raise LearningUnavailable("Service loading", resources_released=True)

    backend = UnavailableBackend(seeded_loop[0])
    observer = threading.Thread(target=operator)
    observer.start()
    try:
        result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
        completed_at = time.monotonic()
    finally:
        halt.set()
        observer.join(timeout=1)
    assert operator_done.is_set() and not observer.is_alive()
    assert completed_at - stop_times[0] < 2, "Stop waited for the five-second retry delay"
    assert backend.requests == 1 and backend.closed and not backend.leases
    assert result["stop_reason"] == "stop_requested" and result["resources_released"]
    assert len(result["acquisition_failures"]) == 1 and result["learner_updates"] == 0
    assert not (request.output_dir / "round-000/learning").exists()


def test_release_error_after_a_stop_cannot_be_reclassified_as_a_retryable_open(
    tmp_path, seeded_loop
):
    request = loop_request(
        tmp_path,
        seeded_loop,
        rounds=1,
        acquisition_retry={"max_retries": 2, "delay_seconds": 0},
    )

    class StopAndFailClose(SharedBackend):
        def sampling(self, identity):
            source = super().sampling(identity)
            original_close = source.close
            stop = request.output_dir / "stop.request"
            stop.write_text("Stop during acquisition")

            def close():
                original_close()
                stop.unlink()  # An external operator clears the stop during cleanup.
                raise LearningUnavailable("Close callback failed", resources_released=True)

            source.close = close
            return source

    backend = StopAndFailClose(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert len(backend.leases) == 1
    assert result["stop_reason"] == "release_fault" and not result["resources_released"]
    assert "Close callback failed" in result["error"]
    assert not result.get("acquisition_failures")
    assert not (request.output_dir / "round-000/learning").exists()


def test_transient_open_failures_allow_actual_learning_and_frozen_evaluation(tmp_path, seeded_loop):
    request = loop_request(
        tmp_path,
        seeded_loop,
        rounds=1,
        acquisition_retry={"max_retries": 2, "delay_seconds": 0},
    )

    class BrieflyUnavailable(SharedBackend):
        def __init__(self, source):
            super().__init__(source)
            self.sampling_requests = self.evaluation_requests = 0

        def sampling(self, identity):
            self.sampling_requests += 1
            if self.sampling_requests == 1:
                raise LearningUnavailable("Sampler starting", resources_released=True)
            return super().sampling(identity)

        def evaluation(self, identity):
            self.evaluation_requests += 1
            if self.evaluation_requests == 1:
                raise LearningUnavailable("Evaluator starting", resources_released=True)
            return super().evaluation(identity)

    backend = BrieflyUnavailable(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "budget_completed", result.get("error")
    assert result["rounds_completed"] == 1 and result["learner_updates"] == 3
    assert result["latest_learner"]["total_steps"] == 33
    assert backend.sampling_requests == backend.evaluation_requests == 2
    assert len(backend.leases) == 2 and all(lease.closed for lease in backend.leases)
    assert result["resources_released"] and backend.closed
    assert [item["kind"] for item in result["acquisition_failures"]] == ["sampling", "evaluation"]
    row = result["rounds"][0]
    assert row["evaluation"]["metrics"]["all_attempts"] == 2
    assert row["selection"] == "retain_incumbent"
    assert result["default"]["sha256"] == seeded_loop[2]["default"]["model_sha256"]


def test_capacity_loss_during_retry_prevents_a_second_acquisition(tmp_path, seeded_loop):
    namespace = Path(os.path.commonpath([str(tmp_path), str(seeded_loop[0])]))
    request = loop_request(
        tmp_path,
        seeded_loop,
        rounds=1,
        version=2,
        acquisition_retry={"max_retries": 2, "delay_seconds": 0},
        storage={
            "root": str(namespace),
            "budget_bytes": 2**40,
            "min_free_bytes": 0,
            "phase_reserve_bytes": 16 * 1024**2,
            "stop_reserve_bytes": 4 * 1024**2,
        },
    )

    class UnavailableBackend(SharedBackend):
        requests = 0

        def sampling(self, identity):
            self.requests += 1
            raise LearningUnavailable("Service unavailable", resources_released=True)

    backend = UnavailableBackend(seeded_loop[0])
    usage = shutil.disk_usage(tmp_path)
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(
            shutil, "disk_usage", lambda _: usage._replace(free=0 if backend.requests else 2**40)
        )
        result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "storage_budget_exhausted"
    assert backend.requests == 1 and not backend.leases and backend.closed
    assert result["resources_released"] and result["learner_updates"] == 0
    assert len(result["acquisition_failures"]) == 1
    assert result["storage_checks"][-1]["reasons"] == ["disk_reserve"]
    assert not (request.output_dir / "round-000/learning").exists()


@pytest.mark.parametrize(
    "retry",
    [
        {"max_retries": 1_000_000, "delay_seconds": 0},
        {"max_retries": 1, "delay_seconds": float("inf")},
        {"max_retries": True, "delay_seconds": 0},
    ],
)
def test_unbounded_or_ambiguous_retry_policy_is_rejected_before_opening(tmp_path, retry):
    config = {
        "version": 1,
        "store": {"directory": "unused", "revision": "unused"},
        "registry": "unused",
        "recording": "unused",
        "task": "unused",
        "reward": "unused",
        "rounds": 1,
        "steps_per_attempt": 1,
        "evaluation_seconds": 1,
        "seed": 0,
        "acquisition_retry": retry,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    backend = SharedBackend(tmp_path)
    with pytest.raises(ValueError, match="explicit finite bounds"):
        run_experiment(LearningLoop(path, tmp_path / "loop"), learning_environment=backend)
    assert not (tmp_path / "loop").exists() and not backend.leases


def test_explicit_continue_keeps_exhausted_history_and_can_acquire_again(tmp_path, seeded_loop):
    request = loop_request(
        tmp_path,
        seeded_loop,
        rounds=1,
        acquisition_retry={"max_retries": 1, "delay_seconds": 0},
    )

    class OfflineBackend(SharedBackend):
        def sampling(self, identity):
            raise LearningUnavailable("Service not ready", resources_released=True)

    first = run_experiment(request, learning_environment=OfflineBackend(seeded_loop[0])).summary[
        "learning_loop"
    ]
    assert first["stop_reason"] == "acquisition_retries_exhausted"
    assert len(first["acquisition_failures"]) == 2

    class AvailableBackend(SharedBackend):
        def sampling(self, identity):
            source = super().sampling(identity)
            (request.output_dir / "stop.request").write_text("Stop after successful acquisition")
            return source

    backend = AvailableBackend(seeded_loop[0])
    result = run_experiment(
        LearningContinue(request.output_dir, sha(request.output_dir / "state.json")),
        learning_environment=backend,
    ).summary["learning_loop"]
    assert result["stop_reason"] == "stop_requested"
    assert result["acquisition_failures"] == first["acquisition_failures"]
    assert result["interruptions"][-1]["stop_reason"] == "acquisition_retries_exhausted"
    assert result["latest_learner"] == first["latest_learner"]
    assert (
        result["learner_updates"] == 0 and not (request.output_dir / "round-000/learning").exists()
    )
    assert len(backend.leases) == 1 and backend.leases[0].closed
    assert result["resources_released"] and backend.closed
