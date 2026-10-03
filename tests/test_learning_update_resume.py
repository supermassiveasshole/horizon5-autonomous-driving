"""Keep accepted experience credit across stopped complete learner snapshots."""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from test_candidate_store import candidates as candidates
from test_evaluation import sha
from test_learning_loop import SharedBackend, loop_request
from test_learning_loop import seeded_loop as seeded_loop

from fh5.evaluation import EvaluationPrepare
from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue
from fh5.sac_learning import SACResume


@contextmanager
def stop_on_creation(directory, stop):
    mkdir = Path.mkdir

    def create_directory(path, *args, **kwargs):
        result = mkdir(path, *args, **kwargs)
        if path == directory:
            stop.write_text("external stop at update boundary")
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "mkdir", create_directory)
        yield


def stopped_sampling(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    initial = request.output_dir / "round-000/learning/candidate-000"
    backend = SharedBackend(seeded_loop[0])
    with stop_on_creation(initial, request.output_dir / "stop.request"):
        first = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert first["stop_reason"] == "stop_requested", first.get("error")
    assert first["eligible_transitions"] == 3 and first["learner_updates"] == 0
    assert first["latest_learner"]["total_steps"] == 30
    assert first["resources_released"] and len(backend.leases) == 1
    return request, first


def test_stopped_update_resumes_its_remaining_credit_without_sampling_or_evaluation(
    tmp_path, seeded_loop
):
    request, _ = stopped_sampling(tmp_path, seeded_loop)
    initial = request.output_dir / "round-000/learning/candidate-000"
    resumed = request.output_dir / "round-000/updates-000"
    stop = request.output_dir / "stop.request"
    originals = {path: path.read_bytes() for path in initial.parent.rglob("*") if path.is_file()}
    stop.unlink()
    second_backend = SharedBackend(seeded_loop[0])
    with stop_on_creation(resumed, stop):
        result = run_experiment(
            LearningContinue(request.output_dir, sha(request.output_dir / "state.json")),
            learning_environment=second_backend,
        ).summary["learning_loop"]
    assert result["stop_reason"] == "stop_requested", result.get("error")
    assert result["rounds_completed"] == 0
    assert result["eligible_transitions"] == 3 and result["learner_updates"] == 0
    assert result["latest_learner"]["directory"] == str(resumed)
    assert result["latest_learner"]["total_steps"] == 30
    assert not second_backend.leases and second_backend.closed and result["resources_released"]
    assert all(path.read_bytes() == raw for path, raw in originals.items())


def test_legacy_continuation_keeps_its_original_parent_binding(tmp_path, seeded_loop):
    request, stopped = stopped_sampling(tmp_path, seeded_loop)
    root = request.output_dir
    state_file = root / "state.json"
    legacy = json.loads(state_file.read_bytes())
    legacy["rounds"][0].pop("sampling_parent")
    state_file.write_text(json.dumps(legacy))
    (root / "stop.request").unlink()
    backend = SharedBackend(seeded_loop[0])
    with stop_on_creation(root / "round-000/updates-000", root / "stop.request"):
        result = run_experiment(
            LearningContinue(root, sha(state_file)), learning_environment=backend
        ).summary["learning_loop"]
    assert result["stop_reason"] == "stop_requested", result.get("error")
    assert result["rounds"][0].get("sampling_parent") == stopped["explorer"]
    assert result["learner_updates"] == 0 and result["eligible_transitions"] == 3
    assert not backend.leases and backend.closed and result["resources_released"]


def test_remaining_updates_finish_once_and_match_uninterrupted_learning(tmp_path, seeded_loop):
    test_legacy_continuation_keeps_its_original_parent_binding(tmp_path, seeded_loop)
    root = tmp_path / "loop"
    checkpoint = root / "round-000/updates-000"
    expected = run_experiment(SACResume(checkpoint, tmp_path / "reference", steps=3)).summary[
        "sac_learning"
    ]
    originals = {
        path: path.read_bytes()
        for path in (root / "round-000/learning").rglob("*")
        if path.is_file()
    }
    (root / "stop.request").unlink()
    backend = SharedBackend(seeded_loop[0])
    result = run_experiment(LearningContinue(root), learning_environment=backend).summary[
        "learning_loop"
    ]
    assert result["stop_reason"] == "budget_completed", result.get("error")
    assert result["rounds_completed"] == 1
    assert result["eligible_transitions"] == result["learner_updates"] == 3
    assert result["latest_learner"]["total_steps"] == 33
    assert len(backend.leases) == 1 and not hasattr(backend.leases[0], "commands")
    report = json.loads(
        (Path(result["latest_learner"]["directory"]) / "training-report.json").read_bytes()
    )
    assert report["learner_state_sha256"] == expected["learner_state_sha256"]
    assert report["updates"] == expected["updates"]
    assert report["predictions"] == expected["predictions"]
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    completed_backend = SharedBackend(seeded_loop[0])
    completed = run_experiment(
        LearningContinue(root), learning_environment=completed_backend
    ).summary["learning_loop"]
    assert completed["stop_reason"] == "budget_completed"
    assert completed["rounds_completed"] == 1 and completed["learner_updates"] == 3
    assert not completed_backend.leases and completed_backend.closed


def test_failed_continuation_audit_preserves_acknowledged_progress(tmp_path, seeded_loop):
    test_stopped_update_resumes_its_remaining_credit_without_sampling_or_evaluation(
        tmp_path, seeded_loop
    )
    root = tmp_path / "loop"
    before = json.loads((root / "state.json").read_bytes())
    output = root / "round-000/updates-001"
    source_summary = root / "round-000/learning/summary.json"
    stop = root / "stop.request"
    stop.unlink()
    open_file = Path.open
    failures = []

    def open_source(path, *args, **kwargs):
        if path == source_summary and (output / "policy.json").exists() and not failures:
            failures.append(path)
            raise OSError("external audit read unavailable")
        return open_file(path, *args, **kwargs)

    backend = SharedBackend(seeded_loop[0])
    with stop_on_creation(output, stop), pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", open_source)
        result = run_experiment(
            LearningContinue(root, sha(root / "state.json")), learning_environment=backend
        ).summary["learning_loop"]
    assert failures == [source_summary]
    assert result["stop_reason"] == "interface_error"
    assert "external audit read unavailable" in result["error"]
    assert result["rounds"][0] == before["rounds"][0]
    assert result["latest_learner"] == before["latest_learner"]
    assert result["eligible_transitions"] == 3 and result["learner_updates"] == 0
    assert not backend.leases and backend.closed and result["resources_released"]


def test_legacy_frozen_evaluation_refuses_updates_before_changing_saved_evidence(
    tmp_path, seeded_loop
):
    request, stopped = stopped_sampling(tmp_path, seeded_loop)
    root = request.output_dir
    state_file = root / "state.json"
    basis = Path(stopped["incumbent"]["batch"])
    config = json.loads((basis / "batch.json").read_bytes())["config"]
    config["version"] = 2
    config["model"] = {
        "kind": "sac",
        "directory": stopped["latest_learner"]["directory"],
        "manifest_sha256": stopped["latest_learner"]["sha256"],
    }
    config["task"]["file"] = str(basis / "task.json")
    config_file = root / "round-000/evaluation.json"
    config_file.write_text(json.dumps(config))
    batch = root / "round-000/batch"
    run_experiment(EvaluationPrepare(config_file, batch, seeded_loop[3]))
    # An old parent could bind a genuine frozen batch before using all update credit.
    legacy = json.loads(state_file.read_bytes())
    legacy["rounds"][0].pop("sampling_parent")
    legacy["rounds"][0]["evaluation_prepared"] = {
        "config_sha256": sha(config_file),
        "batch_sha256": sha(batch / "batch.json"),
    }
    state_file.write_text(json.dumps(legacy))
    before = state_file.read_bytes()
    evidence = {path: path.read_bytes() for path in batch.rglob("*") if path.is_file()}
    (root / "stop.request").unlink()
    backend = SharedBackend(seeded_loop[0])
    with stop_on_creation(root / "round-000/updates-000", root / "stop.request"):
        with pytest.raises(ValueError, match="Incomplete updates already have frozen evaluation"):
            run_experiment(LearningContinue(root, sha(state_file)), learning_environment=backend)
    assert state_file.read_bytes() == before
    assert all(path.read_bytes() == raw for path, raw in evidence.items())
    assert not (root / "round-000/updates-000").exists()
    assert not backend.leases and backend.closed
