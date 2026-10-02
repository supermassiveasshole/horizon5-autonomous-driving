"""Growing continuation history through the public parent experiment."""

import json
from pathlib import Path

import pytest
from learning_files import update_bindings
from test_candidate_store import candidates as candidates
from test_evaluation import sha
from test_learning_loop import SharedBackend, loop_request
from test_learning_loop import seeded_loop as seeded_loop
from test_learning_update_resume import stop_on_creation, stopped_sampling

from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue
from fh5.sac_learning import SACResume


def test_legacy_update_bindings_are_removed_without_changing_old_files(tmp_path, seeded_loop):
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
    assert "update_segments" not in stopped["rounds"][0]
    checkpoint = root / "round-000/updates-000/policy.json"
    sealed = sha(checkpoint)
    entry = {"directory": str(checkpoint.parent), "sha256": sealed}
    # Old metadata is redundant with the genuine checkpoint and its continuation ancestry.
    node_path = root / "round-000/update-history/000000.json"
    node_path.parent.mkdir()
    node_path.write_text(
        json.dumps({"format": "learning-update-node-v1", "previous_sha256": None, "entry": entry})
    )
    original = node_path.read_bytes()
    state = json.loads((root / "state.json").read_bytes())
    state["rounds"][0]["update_segments"] = [entry]
    (root / "state.json").write_text(json.dumps(state))
    stop.unlink()
    backend = SharedBackend(seeded_loop[0])
    continued = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert continued["stop_reason"] == "budget_completed", continued.get("error")
    assert continued["learner_updates"] == 3 and continued["rounds_completed"] == 1
    assert continued["latest_learner"]["total_steps"] == 33
    assert continued["eligible_transitions"] == first["eligible_transitions"]
    assert "update_segments" not in continued["rounds"][0]
    assert "update_segments" not in json.loads((root / "state.json").read_bytes())["rounds"][0]
    assert len(update_bindings(root, continued["rounds"][0])) == 2
    assert len(backend.leases) == 1 and not hasattr(backend.leases[0], "commands")
    assert backend.closed and continued["resources_released"]
    assert sha(checkpoint) == sealed
    assert node_path.read_bytes() == original


def test_legacy_completed_sampling_keeps_its_parent_after_evaluation_and_repeated_continue(
    tmp_path, seeded_loop
):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    root = request.output_dir

    class UnavailableEvaluation(SharedBackend):
        def evaluation(self, identity):
            raise OSError("synthetic evaluation receiver unavailable")

    failed = run_experiment(
        request, learning_environment=UnavailableEvaluation(seeded_loop[0])
    ).summary["learning_loop"]
    assert failed["stop_reason"] == "interface_error"
    assert failed["learner_updates"] == failed["eligible_transitions"] == 3
    assert failed["latest_learner"]["total_steps"] == 33
    parent = failed["explorer"]
    originals = {
        path: sha(path)
        for learner in (parent, failed["latest_learner"])
        for path in Path(learner["directory"]).rglob("*")
        if path.is_file()
    }
    state_file = root / "state.json"
    legacy = json.loads(state_file.read_bytes())
    legacy["rounds"][0].pop("sampling_parent")
    state_file.write_text(json.dumps(legacy))

    backend = SharedBackend(seeded_loop[0])
    completed = run_experiment(
        LearningContinue(root, sha(state_file)), learning_environment=backend
    ).summary["learning_loop"]
    assert completed["stop_reason"] == "budget_completed", completed.get("error")
    assert completed["rounds_completed"] == 1
    assert completed["learner_updates"] == completed["eligible_transitions"] == 3
    assert completed["latest_learner"] == failed["latest_learner"]
    assert json.loads(state_file.read_bytes())["rounds"][0].get("sampling_parent") == parent
    assert len(backend.leases) == 1 and not hasattr(backend.leases[0], "commands")
    assert backend.closed and completed["resources_released"]

    # Also accept an already-completed legacy round without saved sampling ancestry.
    legacy = json.loads(state_file.read_bytes())
    legacy["rounds"][0].pop("sampling_parent")
    state_file.write_text(json.dumps(legacy))
    repeated_backend = SharedBackend(seeded_loop[0])
    repeated = run_experiment(
        LearningContinue(root, sha(state_file)), learning_environment=repeated_backend
    ).summary["learning_loop"]
    assert repeated["stop_reason"] == "budget_completed", repeated.get("error")
    assert repeated["rounds_completed"] == 1
    assert repeated["learner_updates"] == repeated["eligible_transitions"] == 3
    assert repeated["latest_learner"] == completed["latest_learner"]
    assert not repeated_backend.leases and repeated_backend.closed
    assert repeated["resources_released"]
    assert not (root / "round-000/updates-000").exists()
    assert all(sha(path) == digest for path, digest in originals.items())


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
        assert "update_segments" not in result["rounds"][0]
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
def test_acknowledged_checkpoint_damage_cannot_advance_parent_or_open_environment(
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
    checkpoint = root / "round-000/updates-000/policy.json"
    if damage == "delete":
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
    stop.unlink()
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises((OSError, ValueError)):
        run_experiment(LearningContinue(root, sha(state)), learning_environment=backend)
    assert state.read_bytes() == prior
    assert not backend.leases and backend.closed
    assert not (root / "round-000/updates-001").exists()


def test_continuation_completes_original_budget_without_update_history_writes(
    tmp_path, seeded_loop, monkeypatch
):
    request, first = stopped_sampling(tmp_path, seeded_loop)
    root, stop = request.output_dir, request.output_dir / "stop.request"
    stop.unlink()
    original_replace = Path.replace
    sidecar_writes = []

    def unavailable_index(path, target):
        if Path(target).parent.name == "update-history":
            sidecar_writes.append(Path(target))
            raise OSError("update index publication unavailable")
        return original_replace(path, target)

    backend = SharedBackend(seeded_loop[0])
    with monkeypatch.context() as fault:
        fault.setattr(Path, "replace", unavailable_index)
        completed = run_experiment(
            LearningContinue(root, sha(root / "state.json")), learning_environment=backend
        ).summary["learning_loop"]
    assert completed["stop_reason"] == "budget_completed", completed.get("error")
    assert not sidecar_writes and not (root / "round-000/update-history").exists()
    assert "update_segments" not in completed["rounds"][0]
    assert completed["latest_learner"]["total_steps"] == first["latest_learner"]["total_steps"] + 3
    assert completed["learner_updates"] == completed["eligible_transitions"] == 3
    assert completed["rounds_completed"] == 1
    assert len(backend.leases) == 1 and not hasattr(backend.leases[0], "commands")
    assert backend.closed and completed["resources_released"]
    checkpoint = root / "round-000/updates-000/policy.json"
    sealed = sha(checkpoint)
    repeated_backend = SharedBackend(seeded_loop[0])
    repeated = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=repeated_backend
    ).summary["learning_loop"]
    assert repeated["stop_reason"] == "budget_completed", repeated.get("error")
    assert repeated["learner_updates"] == repeated["eligible_transitions"] == 3
    assert repeated["latest_learner"]["sha256"] == sealed == sha(checkpoint)
    assert repeated["latest_learner"]["total_steps"] == 33
    assert len(update_bindings(root, repeated["rounds"][0])) == 1
    assert not (root / "round-000/updates-001").exists()
    assert not repeated_backend.leases and repeated_backend.closed
