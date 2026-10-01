"""Parent review publication recovery through real process exits and experiment I/O."""

import json
import os
from pathlib import Path

from test_candidate_store import candidates as candidates
from test_evaluation import sha
from test_learning_loop import SharedBackend, loop_request
from test_learning_loop import seeded_loop as seeded_loop
from test_learning_recovery import interrupt_selection
from test_learning_storage import storage_request

from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue


def test_parent_review_survives_exit_before_its_result_is_acknowledged(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_parent_review_ack")
    root = request.output_dir
    state = root / "state.json"
    interrupted = json.loads(state.read_bytes())
    assert interrupted["phase"] == "reviewing_evaluation"
    assert "candidate_evaluation" not in interrupted["rounds"][0]
    assert (root / "round-000/reviewed/report.html").is_file()
    originals = {
        path: sha(path)
        for directory in (root / "round-000/evaluation", root / "round-000/reviewed")
        for path in directory.rglob("*")
        if path.is_file()
    }
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(root, sha(state)), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed", resumed.get("error")
    assert resumed["rounds_completed"] == 1
    assert resumed["learner_updates"] == interrupted["learner_updates"] == 3
    assert resumed["latest_learner"] == interrupted["latest_learner"]
    assert resumed["rounds"][0]["evaluation"]["metrics"]["all_attempts"] == 2
    assert resumed["interruptions"][-1]["phase"] == "reviewing_evaluation"
    assert not backend.leases and backend.closed and resumed["resources_released"]
    assert all(sha(path) == digest for path, digest in originals.items())


def test_parent_evidence_is_frozen_without_changing_the_completed_child(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(
        request, seeded_loop[0], "before_parent_review_ack", scenario="parent_evidence"
    )
    root = request.output_dir
    state = root / "state.json"
    child = root / "round-000/evaluation"
    completion = json.loads((child / "completion.json").read_bytes())
    assert sha(child / "ledger.json") == completion["files"]["ledger.json"]
    originals = {path: sha(path) for path in child.rglob("*") if path.is_file()}
    proofs = {
        path: sha(path) for path in (tmp_path / "independent-evidence").rglob("*") if path.is_file()
    }
    assert len(proofs) == 4

    class RetainedReviewBackend(SharedBackend):
        def review(self, recording):
            raise AssertionError("Frozen independent evidence must not be requested again")

    backend = RetainedReviewBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(root, sha(state)), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed", resumed.get("error")
    assert resumed["learner_updates"] == 3 and resumed["rounds_completed"] == 1
    ledger = json.loads(Path(resumed["rounds"][0]["candidate_evaluation"]["ledger"]).read_bytes())
    assert len(ledger["entries"]) == 2
    assert all(entry["evidence"] is not None for entry in ledger["entries"])
    assert not backend.leases and backend.closed
    assert all(sha(path) == digest for path, digest in {**originals, **proofs}.items())


def test_partial_parent_report_is_preserved_while_review_resumes(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_review_report", scenario="parent_evidence")
    root = request.output_dir
    partial = root / "round-000/reviewed"
    assert partial.is_dir() and not (partial / "batch-report.json").exists()
    originals = {path: sha(path) for path in partial.rglob("*") if path.is_file()}
    assert originals
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed", resumed.get("error")
    assert resumed["rounds_completed"] == 1 and resumed["learner_updates"] == 3
    assert resumed["rounds"][0]["evaluation"]["metrics"]["all_attempts"] == 2
    assert not backend.leases and backend.closed
    assert all(sha(path) == digest for path, digest in originals.items())
    assert not (partial / "batch-report.json").exists()


def test_frozen_parent_input_survives_exit_before_ledger_publication(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_parent_ledger", scenario="parent_evidence")
    root = request.output_dir
    state = root / "state.json"
    interrupted = json.loads(state.read_bytes())
    assert interrupted["rounds"][0]["review_input"]
    assert not (root / "round-000/parent-ledger.json").exists()
    scope = Path(os.path.commonpath([str(tmp_path), str(seeded_loop[0])]))
    inventory = run_experiment(storage_request(tmp_path, (root, scope))).summary["storage"]
    retained = {item["path"] for item in inventory["files"]}
    proofs = {
        str(path.resolve())
        for path in (tmp_path / "independent-evidence").rglob("*")
        if path.is_file()
    }
    assert len(proofs) == 4 and proofs <= retained
    backend = SharedBackend(seeded_loop[0])
    result = run_experiment(
        LearningContinue(root, sha(state)), learning_environment=backend
    ).summary["learning_loop"]
    assert result["stop_reason"] == "budget_completed", result.get("error")
    assert result["learner_updates"] == 3 and result["rounds_completed"] == 1
    ledger = json.loads((root / "round-000/parent-ledger.json").read_bytes())
    assert all(entry["evidence"] is not None for entry in ledger["entries"])
    assert not backend.leases and backend.closed


def test_user_stop_survives_parent_review_exit_and_resumes_remaining_rounds(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=2)
    interrupt_selection(
        request, seeded_loop[0], "before_parent_review_ack", scenario="stopped_evaluation"
    )
    root = request.output_dir
    state = root / "state.json"
    interrupted = json.loads(state.read_bytes())
    assert interrupted["rounds"][0]["evaluation_interrupted_by_stop"]
    child = root / "round-000/evaluation"
    originals = {path: sha(path) for path in child.rglob("*") if path.is_file()}
    (root / "stop.request").unlink()
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(root, sha(state)), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed", resumed.get("error")
    assert resumed["rounds_completed"] == 2 and resumed["learner_updates"] == 6
    first = resumed["rounds"][0]
    assert first["evaluation_interrupted_by_stop"]
    assert first["evaluation"]["metrics"]["all_attempts"] == 1
    assert first["evaluation"]["unstarted_slots"] == ["run-1"]
    assert first["selection"] == "retain_incumbent"
    assert all(sha(path) == digest for path, digest in originals.items())
    assert len(backend.leases) == 2 and backend.closed and resumed["resources_released"]
