"""Capacity admission and preserved continuation through the experiment seam."""

import json
import os
import shutil
from pathlib import Path

import pytest

from fh5.evaluation.candidate_store import CandidateHistory
from fh5.experiment import run_experiment
from fh5.learning.loop.runner import LearningContinue
from tests.evaluation.test_candidate_store import candidates as candidates
from tests.evaluation.test_evaluation import sha
from tests.learning.loop.test_learning_loop import SharedBackend, loop_request
from tests.learning.loop.test_learning_loop import seeded_loop as seeded_loop


def budget_request(tmp_path, seeded_loop, namespace, **changes):
    storage = {
        "root": str(namespace),
        "budget_bytes": 2**40,
        "min_free_bytes": 0,
        "phase_reserve_bytes": 16 * 1024**2,
        "stop_reserve_bytes": 4 * 1024**2,
        **changes,
    }
    return loop_request(tmp_path, seeded_loop, version=2, storage=storage, rounds=1)


def test_initial_capacity_refusal_preserves_source_and_opens_no_lease(
    tmp_path, tmp_path_factory, seeded_loop
):
    request = budget_request(tmp_path, seeded_loop, tmp_path_factory.getbasetemp(), budget_bytes=1)
    backend = SharedBackend(seeded_loop[0])
    original = sha(Path(seeded_loop[1]) / "state.sqlite")
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "storage_budget_exhausted"
    assert result["rounds"] == []
    assert result["learner_updates"] == 0
    assert result["resources_released"] is True
    assert result["initialized"] is False
    assert backend.leases == [] and backend.closed
    assert not (request.output_dir / "initial").exists()
    assert sha(Path(seeded_loop[1]) / "state.sqlite") == original
    decision = result["storage_checks"][-1]
    assert decision["phase"] == "initialize"
    assert "logical_budget" in decision["reasons"]
    assert decision["protected_bytes"] > 1
    state = json.loads((request.output_dir / "state.json").read_bytes())
    assert state["stop_reason"] == "storage_budget_exhausted"


def test_initial_capacity_stop_can_be_rechecked_without_losing_the_reason(
    tmp_path, tmp_path_factory, seeded_loop
):
    request = budget_request(tmp_path, seeded_loop, tmp_path_factory.getbasetemp())
    usage = shutil.disk_usage(tmp_path)
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", lambda _: usage._replace(free=0))
        first = run_experiment(request, learning_environment=SharedBackend(seeded_loop[0]))
        assert first.summary["learning_loop"]["stop_reason"] == "storage_budget_exhausted"
        backend = SharedBackend(seeded_loop[0])
        resumed = run_experiment(
            LearningContinue(request.output_dir, sha(request.output_dir / "state.json")),
            learning_environment=backend,
        ).summary["learning_loop"]
    assert resumed["stop_reason"] == "storage_budget_exhausted"
    assert len(resumed["storage_checks"]) == 2
    assert resumed["interruptions"][-1]["stop_reason"] == "storage_budget_exhausted"
    assert resumed["rounds"] == [] and resumed["initialized"] is False
    assert backend.leases == [] and backend.closed
    assert not (request.output_dir / "initial").exists()


def test_capacity_is_rechecked_after_initialization_before_sampling(
    tmp_path, tmp_path_factory, seeded_loop
):
    request = budget_request(tmp_path, seeded_loop, tmp_path_factory.getbasetemp())
    usage = shutil.disk_usage(tmp_path)
    reads = []

    class OfflineBackend(SharedBackend):
        sampling_requested = False

        def sampling(self, identity):
            self.sampling_requested = True
            raise OSError("The synthetic driving service is deliberately unavailable")

    backend = OfflineBackend(seeded_loop[0])

    def capacity(_):
        reads.append(True)
        return usage._replace(free=2**40 if len(reads) == 1 else 0)

    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", capacity)
        result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "storage_budget_exhausted"
    assert result["initialized"] is True
    assert result["learner_updates"] == 0
    assert not backend.sampling_requested and backend.closed
    assert result["storage_checks"][-1]["phase"] == "sampling"
    assert result["storage_checks"][-1]["reasons"] == ["disk_reserve"]
    assert not (request.output_dir / "round-000/learning").exists()
    assert Path(result["latest_learner"]["directory"]).is_dir()


def test_failed_initial_recheck_remains_stopped_and_can_be_retried(
    tmp_path, tmp_path_factory, seeded_loop
):
    request = budget_request(tmp_path, seeded_loop, tmp_path_factory.getbasetemp())
    usage = shutil.disk_usage(tmp_path)
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", lambda _: usage._replace(free=0))
        first = run_experiment(request, learning_environment=SharedBackend(seeded_loop[0])).summary[
            "learning_loop"
        ]
    assert first["stop_reason"] == "storage_budget_exhausted"

    def unavailable(_):
        raise OSError("Disk capacity service unavailable")

    backend = SharedBackend(seeded_loop[0])
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", unavailable)
        failed = run_experiment(
            LearningContinue(request.output_dir, sha(request.output_dir / "state.json")),
            learning_environment=backend,
        ).summary["learning_loop"]
    assert failed["stop_reason"] == "interface_error"
    assert "Disk capacity service unavailable" in failed["error"]
    assert failed["phase"] == "stopped" and failed["initialized"] is False
    assert backend.leases == [] and backend.closed
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", lambda _: usage._replace(free=0))
        resumed = run_experiment(
            LearningContinue(request.output_dir, sha(request.output_dir / "state.json")),
            learning_environment=SharedBackend(seeded_loop[0]),
        ).summary["learning_loop"]
    assert resumed["stop_reason"] == "storage_budget_exhausted"
    assert resumed["interruptions"][-1]["stop_reason"] == "interface_error"
    assert not (request.output_dir / "initial").exists()


@pytest.fixture
def waiting_for_evaluation(tmp_path, tmp_path_factory, seeded_loop):
    request = budget_request(tmp_path, seeded_loop, tmp_path_factory.getbasetemp())

    class EvaluationUnavailable(SharedBackend):
        def evaluation(self, identity):
            raise OSError("Evaluation service temporarily unavailable")

    result = run_experiment(
        request, learning_environment=EvaluationUnavailable(seeded_loop[0])
    ).summary["learning_loop"]
    assert result["learner_updates"] == 3
    assert result["stop_reason"] == "interface_error"
    assert result["rounds"][0]["candidate_sha256"] == result["latest_learner"]["sha256"]
    return request, seeded_loop[0]


def test_capacity_stop_after_learning_keeps_candidate_for_evaluation(
    tmp_path, waiting_for_evaluation
):
    request, source = waiting_for_evaluation
    before = json.loads((request.output_dir / "state.json").read_bytes())
    saved_learning = Path(before["rounds"][0]["learning"]["directory"]) / "summary.json"
    original_learning = sha(saved_learning)

    class OfflineBackend(SharedBackend):
        evaluation_requested = False

        def evaluation(self, identity):
            self.evaluation_requested = True
            raise OSError("No environment should open under storage pressure")

    backend = OfflineBackend(source)
    usage = shutil.disk_usage(tmp_path)
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", lambda _: usage._replace(free=0))
        result = run_experiment(
            LearningContinue(request.output_dir, sha(request.output_dir / "state.json")),
            learning_environment=backend,
        ).summary["learning_loop"]
    assert result["stop_reason"] == "storage_budget_exhausted"
    assert result["storage_checks"][-1]["phase"] == "evaluation"
    assert result["latest_learner"] == before["latest_learner"]
    assert result["learner_updates"] == before["learner_updates"] == 3
    assert sha(saved_learning) == original_learning
    assert not backend.evaluation_requested and backend.closed
    assert backend.leases == []
    assert not (request.output_dir / "round-000/evaluation").exists()


def test_stop_during_capacity_check_prevents_initial_model_copy(
    tmp_path, tmp_path_factory, seeded_loop
):
    request = budget_request(tmp_path, seeded_loop, tmp_path_factory.getbasetemp())
    usage = shutil.disk_usage(tmp_path)

    def stop_while_checking(_):
        (request.output_dir / "stop.request").write_text("operator stop")
        return usage._replace(free=2**40)

    backend = SharedBackend(seeded_loop[0])
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", stop_while_checking)
        result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "stop_requested"
    assert result["initialized"] is False
    assert not (request.output_dir / "initial").exists()
    assert not backend.leases and backend.closed


@pytest.fixture
def waiting_for_retention(tmp_path, tmp_path_factory, seeded_loop):
    request = budget_request(tmp_path, seeded_loop, tmp_path_factory.getbasetemp())
    replace = os.replace
    injected = False

    def interrupt_commit(source, target, *args, **kwargs):
        nonlocal injected
        if Path(target) == request.output_dir / "state.json" and not injected:
            state = json.loads(Path(source).read_bytes())
            if state["phase"] == "saving_versions":
                injected = True
                raise OSError("Pre-commit filesystem interruption")
        return replace(source, target, *args, **kwargs)

    with pytest.MonkeyPatch.context() as filesystem:
        filesystem.setattr(os, "replace", interrupt_commit)
        result = run_experiment(
            request, learning_environment=SharedBackend(seeded_loop[0])
        ).summary["learning_loop"]
    assert injected and result["stop_reason"] == "interface_error"
    assert result["rounds"][0]["candidate_evaluation"]
    assert result["rounds_completed"] == 0
    return request, seeded_loop[0]


def test_capacity_stop_before_retention_preserves_all_evaluated_attempts(
    tmp_path, waiting_for_retention
):
    request, source = waiting_for_retention
    before = json.loads((request.output_dir / "state.json").read_bytes())
    config = json.loads((request.output_dir / "config.json").read_bytes())
    binding = before["rounds"][0]["candidate_evaluation"]
    ledger = Path(binding["ledger"])
    original_ledger = sha(ledger)
    backend = SharedBackend(source)
    usage = shutil.disk_usage(tmp_path)
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", lambda _: usage._replace(free=0))
        result = run_experiment(
            LearningContinue(request.output_dir, sha(request.output_dir / "state.json")),
            learning_environment=backend,
        ).summary["learning_loop"]
    assert result["stop_reason"] == "storage_budget_exhausted"
    assert result["storage_checks"][-1]["phase"] == "retention"
    assert result["rounds_completed"] == 0
    assert result["latest_learner"] == before["latest_learner"]
    assert result["rounds"][0]["candidate_evaluation"] == binding
    assert sha(ledger) == original_ledger
    history = run_experiment(CandidateHistory(Path(config["store"]["directory"]))).summary[
        "candidate_store"
    ]
    assert history["revision"] == before["store_revision"]
    assert not backend.leases and backend.closed


def test_freeing_capacity_finishes_retention_without_repeating_training_or_evaluation(
    tmp_path, waiting_for_retention
):
    request, source = waiting_for_retention
    usage = shutil.disk_usage(tmp_path)
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", lambda _: usage._replace(free=0))
        stopped = run_experiment(
            LearningContinue(request.output_dir, sha(request.output_dir / "state.json")),
            learning_environment=SharedBackend(source),
        ).summary["learning_loop"]
    assert stopped["stop_reason"] == "storage_budget_exhausted"
    binding = stopped["rounds"][0]["candidate_evaluation"]
    original_ledger = sha(Path(binding["ledger"]))
    backend = SharedBackend(source)
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", lambda _: usage._replace(free=2**40))
        resumed = run_experiment(
            LearningContinue(request.output_dir, sha(request.output_dir / "state.json")),
            learning_environment=backend,
        ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed"
    assert resumed["rounds_completed"] == 1
    assert resumed["learner_updates"] == stopped["learner_updates"] == 3
    assert resumed["latest_learner"] == stopped["latest_learner"]
    assert resumed["rounds"][0]["candidate_evaluation"] == binding
    assert sha(Path(binding["ledger"])) == original_ledger
    assert resumed["interruptions"][-1]["stop_reason"] == "storage_budget_exhausted"
    assert not backend.leases and backend.closed
