"""Growing continuation history through the public parent experiment."""

import json
from pathlib import Path

import pytest
from learning_files import update_bindings
from test_candidate_store import candidates as candidates
from test_evaluation import sha
from test_learning_loop import SharedBackend
from test_learning_loop import seeded_loop as seeded_loop
from test_learning_update_resume import stop_on_creation, stopped_sampling

from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue
from fh5.sac_learning import SACResume


def test_update_bindings_are_external_and_legacy_bindings_remain_readable(tmp_path, seeded_loop):
    request, first = stopped_sampling(tmp_path, seeded_loop)
    root = request.output_dir
    stop = root / "stop.request"
    stop.unlink()
    with stop_on_creation(root / "round-000/updates-000", stop):
        stopped = run_experiment(
            LearningContinue(root, sha(root / "state.json")),
            learning_environment=SharedBackend(seeded_loop[0]),
        ).summary["learning_loop"]
    assert stopped["stop_reason"] == "stop_requested"
    assert stopped["learner_updates"] == 0 and stopped["eligible_transitions"] == 3
    index = stopped["rounds"][0]["update_segments"]
    assert isinstance(index, dict), "Parent state must not accumulate update bindings"
    assert index["format"] == "learning-update-history-v1" and index["count"] == 1
    node_path = root / "round-000/update-history/000000.json"
    assert sha(node_path) == index["head_sha256"]
    node = json.loads(node_path.read_bytes())
    assert node["previous_sha256"] is None
    assert node["entry"] == {
        "directory": str(root / "round-000/updates-000"),
        "sha256": stopped["latest_learner"]["sha256"],
    }
    # Reconstruct the old public state representation without changing learner data.
    state = json.loads((root / "state.json").read_bytes())
    state["rounds"][0]["update_segments"] = [node["entry"]]
    (root / "state.json").write_text(json.dumps(state))
    backend = SharedBackend(seeded_loop[0])
    continued = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert continued["stop_reason"] == "stop_requested" and not backend.leases
    assert continued["latest_learner"] == stopped["latest_learner"]
    assert continued["eligible_transitions"] == first["eligible_transitions"]
    original = node_path.read_bytes()
    stop.unlink()
    with stop_on_creation(root / "round-000/updates-001", stop):
        migrated = run_experiment(
            LearningContinue(root, sha(root / "state.json")),
            learning_environment=SharedBackend(seeded_loop[0]),
        ).summary["learning_loop"]
    assert migrated["stop_reason"] == "stop_requested"
    assert migrated["rounds"][0]["update_segments"]["count"] == 2
    assert len(update_bindings(root, migrated["rounds"][0])) == 2
    assert node_path.read_bytes() == original


def test_more_than_ten_stopped_continuations_keep_the_original_credit(tmp_path, seeded_loop):
    request, first = stopped_sampling(tmp_path, seeded_loop)
    root, stop = request.output_dir, request.output_dir / "stop.request"
    for attempt in range(11):
        stop.unlink()
        backend = SharedBackend(seeded_loop[0])
        with stop_on_creation(root / f"round-000/updates-{attempt:03d}", stop):
            result = run_experiment(
                LearningContinue(root, sha(root / "state.json")), learning_environment=backend
            ).summary["learning_loop"]
        assert result["stop_reason"] == "stop_requested", result.get("error")
        assert result["learner_updates"] == 0 and result["eligible_transitions"] == 3
        assert len(update_bindings(root, result["rounds"][0])) == attempt + 1
        assert not backend.leases and result["resources_released"]
    stop.unlink()
    backend = SharedBackend(seeded_loop[0])
    completed = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert completed["stop_reason"] == "budget_completed", completed.get("error")
    assert completed["rounds_completed"] == 1
    assert completed["learner_updates"] == completed["eligible_transitions"] == 3
    assert completed["latest_learner"]["total_steps"] == 33
    assert len(backend.leases) == 1  # Frozen evaluation only; no additional sampling.
    bindings = update_bindings(root, completed["rounds"][0])
    assert len(bindings) == 12
    expected = run_experiment(
        SACResume(root / "round-000/learning/candidate-000", tmp_path / "reference", steps=3)
    ).summary["sac_learning"]
    assert first["learner_updates"] == 0
    assert completed["latest_learner"]["learner_state_sha256"] == expected["learner_state_sha256"]


@pytest.mark.parametrize("damage", ["delete", "change"])
def test_required_update_index_damage_cannot_advance_parent_or_open_environment(
    tmp_path, seeded_loop, damage
):
    request, _ = stopped_sampling(tmp_path, seeded_loop)
    root, stop = request.output_dir, request.output_dir / "stop.request"
    stop.unlink()
    with stop_on_creation(root / "round-000/updates-000", stop):
        run_experiment(
            LearningContinue(root, sha(root / "state.json")),
            learning_environment=SharedBackend(seeded_loop[0]),
        )
    state = root / "state.json"
    prior = state.read_bytes()
    node = root / "round-000/update-history/000000.json"
    if damage == "delete":
        node.unlink()
    else:
        node.write_bytes(node.read_bytes() + b" ")
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises((OSError, ValueError)):
        run_experiment(LearningContinue(root, sha(state)), learning_environment=backend)
    assert state.read_bytes() == prior
    assert not backend.leases and backend.closed
    assert not (root / "round-000/updates-001").exists()


def test_failed_index_publication_keeps_sealed_updates_for_later_acknowledgment(
    tmp_path, seeded_loop, monkeypatch
):
    request, first = stopped_sampling(tmp_path, seeded_loop)
    root, stop = request.output_dir, request.output_dir / "stop.request"
    stop.unlink()
    original_replace = Path.replace
    failed = []

    def unavailable_index(path, target):
        if Path(target).parent.name == "update-history":
            failed.append(Path(target))
            raise OSError("update index publication unavailable")
        return original_replace(path, target)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "replace", unavailable_index)
        interrupted = run_experiment(
            LearningContinue(root, sha(root / "state.json")),
            learning_environment=SharedBackend(seeded_loop[0]),
        ).summary["learning_loop"]
    assert failed and interrupted["stop_reason"] == "interface_error"
    assert "update index publication unavailable" in interrupted["error"]
    assert interrupted["latest_learner"] == first["latest_learner"]
    assert interrupted["learner_updates"] == 0
    checkpoint = root / "round-000/updates-000/policy.json"
    sealed = sha(checkpoint)
    stop.write_text("keep the recovered learner without opening evaluation")
    backend = SharedBackend(seeded_loop[0])
    recovered = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert recovered["stop_reason"] == "stop_requested", recovered.get("error")
    assert recovered["learner_updates"] == recovered["eligible_transitions"] == 3
    assert recovered["latest_learner"]["sha256"] == sealed == sha(checkpoint)
    assert recovered["latest_learner"]["total_steps"] == 33
    assert len(update_bindings(root, recovered["rounds"][0])) == 1
    assert not backend.leases and backend.closed
