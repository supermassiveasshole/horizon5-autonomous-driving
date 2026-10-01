"""Real process exits at filesystem publication boundaries, through run_experiment."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_candidate_store import candidates as candidates
from test_evaluation import sha
from test_learning_loop import SharedBackend, loop_request
from test_learning_loop import seeded_loop as seeded_loop

from fh5.candidate_store import CandidateHistory, CandidateRecord
from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue


def interrupt_selection(request, source, boundary="after_commit"):
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository / "src"), str(repository / "tests"))
    )
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, os, sys
from pathlib import Path
from fh5.experiment import run_experiment
from fh5.learning_loop import LearningLoop
from test_learning_loop import SharedBackend
root = Path(sys.argv[2]).resolve()
original_replace = os.replace
def replace(source, destination, *args, **kwargs):
    if Path(destination).resolve() == root / 'state.json':
        value = json.loads(Path(source).read_bytes())
        if sys.argv[4] == 'before_commit' and value['phase'] == 'saving_versions':
            original_replace(source, destination, *args, **kwargs)
            os._exit(73)
        if sys.argv[4] == 'after_commit' and value['phase'] == 'ready' and value['rounds_completed'] == 1:
            os._exit(73)
        if sys.argv[4] == 'before_learned' and value['phase'] == 'learned':
            os._exit(73)
    return original_replace(source, destination, *args, **kwargs)
os.replace = replace
run_experiment(LearningLoop(Path(sys.argv[1]), root),
               learning_environment=SharedBackend(Path(sys.argv[3])))
raise SystemExit('Expected filesystem exit was not reached')
""",
            str(request.config_file),
            str(request.output_dir),
            str(source),
            boundary,
        ],
        env=environment,
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert process.returncode == 73, process.stdout + process.stderr


def test_sealed_sampling_is_adopted_after_exit_before_parent_acknowledgement(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_learned")
    state_file = request.output_dir / "state.json"
    interrupted = json.loads(state_file.read_bytes())
    assert interrupted["phase"] == "updating"
    assert interrupted["learner_updates"] == 0
    assert "candidate_sha256" not in interrupted["rounds"][0]
    learning = request.output_dir / "round-000/learning"
    summary = json.loads((learning / "summary.json").read_bytes())
    candidate = learning / summary["latest_candidate"]
    candidate_sha = sha(candidate / "policy.json")
    originals = {path: sha(path) for path in learning.rglob("*") if path.is_file()}
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(state_file)), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed", resumed.get("error")
    assert resumed["rounds_completed"] == 1
    assert resumed["learner_updates"] == resumed["eligible_transitions"] == 3
    assert resumed["latest_learner"]["sha256"] == candidate_sha
    assert resumed["latest_learner"]["total_steps"] == 33
    assert resumed["recoveries"][0]["kind"] == "sealed_sampling"
    assert resumed["interruptions"][0]["phase"] == "updating"
    assert resumed["interruptions"][0]["stop_reason"] == "unclean_exit"
    assert len(backend.leases) == 1  # Evaluate the saved model; no new sampling lease.
    assert backend.closed and resumed["resources_released"]
    assert all(sha(path) == digest for path, digest in originals.items())
    history = run_experiment(CandidateHistory(tmp_path / "versions")).summary["candidate_store"]
    assert len(history["history"]) == 2
    assert history["explorer"]["model_sha256"] == candidate_sha
    assert history["default"]["model_sha256"] == seeded_loop[2]["default"]["model_sha256"]


def test_committed_candidate_is_reconciled_after_process_exit_without_duplicate_learning(
    tmp_path, seeded_loop
):
    request = loop_request(tmp_path, seeded_loop)
    interrupt_selection(request, seeded_loop[0])
    state_file = request.output_dir / "state.json"
    interrupted = json.loads(state_file.read_bytes())
    assert interrupted["phase"] == "saving_versions"
    assert interrupted["rounds_completed"] == 0
    assert interrupted["learner_updates"] == 3
    history = run_experiment(CandidateHistory(tmp_path / "versions")).summary["candidate_store"]
    assert len(history["history"]) == 2
    assert history["revision"] != interrupted["store_revision"]
    saved_candidate = history["explorer"]["model_sha256"]
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(state_file)), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed"
    assert resumed["rounds_completed"] == 2
    assert resumed["learner_updates"] == resumed["eligible_transitions"] == 6
    assert resumed["latest_learner"]["total_steps"] == 36
    assert resumed["rounds"][1]["sampling_checkpoint_sha256"] == saved_candidate
    assert len(backend.leases) == 2  # Only the unfinished second round samples/evaluates.
    final = run_experiment(CandidateHistory(tmp_path / "versions")).summary["candidate_store"]
    assert len(final["history"]) == 3
    assert final["default"]["model_sha256"] == history["default"]["model_sha256"]
    assert resumed["resources_released"]
    assert resumed["recoveries"][0]["kind"] == "candidate_commit"
    assert resumed["interruptions"][0] == {
        "phase": "saving_versions",
        "stop_reason": "unclean_exit",
        "error": None,
    }


def test_prepared_selection_survives_exit_before_the_store_commit(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_commit")
    state_file = request.output_dir / "state.json"
    interrupted = json.loads(state_file.read_bytes())
    assert interrupted["phase"] == "saving_versions"
    assert interrupted["learner_updates"] == 3
    store = CandidateHistory(tmp_path / "versions")
    before = run_experiment(store).summary["candidate_store"]
    assert before["revision"] == interrupted["store_revision"]
    root = request.output_dir / "round-000"
    originals = {
        path: path.read_bytes() for path in (root / "comparison.json", root / "retain.json")
    }
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(state_file)), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed"
    assert resumed["rounds_completed"] == 1
    assert resumed["learner_updates"] == 3
    assert not backend.leases and backend.closed
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    after = run_experiment(store).summary["candidate_store"]
    assert len(after["history"]) == 2
    assert after["explorer"]["model_sha256"] == interrupted["latest_learner"]["sha256"]
    assert resumed["interruptions"][0] == {
        "phase": "saving_versions",
        "stop_reason": "unclean_exit",
        "error": None,
    }


def test_unacknowledged_sampling_rejects_missing_or_inconsistent_child_evidence(
    tmp_path, seeded_loop
):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_learned")
    state = request.output_dir / "state.json"
    original_state = state.read_bytes()
    root = request.output_dir / "round-000/learning"
    summary_path = root / "summary.json"
    result_path = root / "attempt-000/cycle-result.json"
    summary = json.loads(summary_path.read_bytes())
    wrong_count = json.loads(summary_path.read_bytes())
    wrong_count["attempts"][0]["eligible_transitions"] = 2
    unreleased = dict(summary, resources_released=False)
    cases = (
        {summary_path: None},
        {summary_path: json.dumps(unreleased).encode()},
        {root / "protocol.json": b'{"source_kind": "synthetic", "seed": 192}'},
        {root / "attempt-000/trace.json": b"original action trace changed"},
        {root / "candidate-000/policy.pt": b"weights changed after child finished"},
        {
            summary_path: json.dumps(wrong_count).encode(),
            result_path: json.dumps(wrong_count["attempts"][0]).encode(),
        },
    )
    for changes in cases:
        originals = {path: path.read_bytes() for path in changes}
        backend = SharedBackend(seeded_loop[0])
        try:
            for path, raw in changes.items():
                if raw is None:
                    path.unlink()
                else:
                    path.write_bytes(raw)
            with pytest.raises((OSError, ValueError)):
                run_experiment(
                    LearningContinue(request.output_dir, sha(state)), learning_environment=backend
                )
            assert state.read_bytes() == original_state
            assert not backend.leases and backend.closed
            history = run_experiment(CandidateHistory(tmp_path / "versions")).summary[
                "candidate_store"
            ]
            assert len(history["history"]) == 1
            assert history["revision"] == seeded_loop[2]["revision"]
        finally:
            for path, raw in originals.items():
                path.write_bytes(raw)


def test_commit_recovery_rejects_changed_assets_and_another_store_successor(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0])
    state = request.output_dir / "state.json"
    original_state = state.read_bytes()
    store = tmp_path / "versions"
    history = run_experiment(CandidateHistory(store)).summary["candidate_store"]
    proposal = request.output_dir / "round-000/retain.json"
    archived_weights = store / history["explorer"]["archive"] / "checkpoint/policy.pt"
    for path in (proposal, archived_weights):
        original = path.read_bytes()
        path.write_bytes(original + b"changed externally")
        backend = SharedBackend(seeded_loop[0])
        try:
            with pytest.raises(ValueError):
                run_experiment(
                    LearningContinue(request.output_dir, sha(state)), learning_environment=backend
                )
            assert state.read_bytes() == original_state
            assert not backend.leases and backend.closed
        finally:
            path.write_bytes(original)
    # Even a further legitimate commit with the same candidate is not this pending write.
    other = run_experiment(
        CandidateRecord(proposal, store, history["revision"], seeded_loop[3])
    ).summary["candidate_store"]
    assert other["explorer"]["model_sha256"] == history["explorer"]["model_sha256"]
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises(ValueError, match="outside the pending candidate commit"):
        run_experiment(
            LearningContinue(request.output_dir, sha(state)), learning_environment=backend
        )
    assert state.read_bytes() == original_state
    assert not backend.leases and backend.closed
    assert (
        run_experiment(CandidateHistory(store)).summary["candidate_store"]["revision"]
        == other["revision"]
    )
