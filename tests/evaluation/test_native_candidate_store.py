"""Native development roles use real learners; external game I/O stays simulated."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from fh5.evaluation.candidate_store import CandidateHistory, CandidateRecord, CandidateRollback
from fh5.evaluation.native import NativeEvaluationEnvironment
from fh5.evaluation.prepare import EvaluationPrepare, EvaluationReview
from fh5.experiment import run_experiment
from fh5.learning.sac.training import SACTrain
from tests.driving.test_numeric_drive_cli import eligible_model as eligible_model
from tests.evaluation.test_candidate_store import candidates as candidates
from tests.evaluation.test_candidate_store import record_config
from tests.evaluation.test_evaluation import sha
from tests.learning.sac.test_native_sac_drive import native_candidate as native_candidate
from tests.learning.sac.test_native_sac_drive import qualified_sac_config
from tests.learning.sac.test_native_sac_evaluation import (
    Devices,
    bind_native_validity,
    sac_evaluation,
)
from tests.learning.sac.test_numeric_sac_assembly import ExternalWorld
from tests.support.native_device_timing import native_device_timing as native_device_timing


class SteeringWorld(ExternalWorld):
    """Identical toy drag for both policies, independent of comparison role."""

    def speed_for(self, command):
        return (
            3.5 / (1 + 20 * abs(command.steer_i16 / 32767))
            if command.throttle_u8 and not command.brake_u8
            else 0
        )


@pytest.fixture(scope="module")
def native_pair(tmp_path_factory, native_candidate):
    root = tmp_path_factory.mktemp("native-store-candidates")
    registry = root / "usage.sqlite"
    evaluated = []
    setup = root / "setup"
    setup.mkdir()
    original_operation = sac_evaluation(setup, native_candidate)
    template = json.loads((setup / "evaluation.json").read_bytes())
    for number, steps in enumerate((0, 4)):
        checkpoint = root / f"model-{number}"
        run_experiment(
            SACTrain(
                native_candidate.parent / "warm",
                native_candidate / "experience/replay.json",
                checkpoint,
                steps=steps,
                actor_lr=0.001,
            )
        )
        folder = root / f"evaluation-{number}"
        folder.mkdir()
        options = {
            **template,
            "model": {
                **template["model"],
                "directory": str(checkpoint),
                "manifest_sha256": sha(checkpoint / "policy.json"),
            },
        }
        evaluation_config = folder / "evaluation.json"
        evaluation_config.write_text(json.dumps(options))
        # Each independently frozen batch is registered before collecting raw UDP.
        batch = folder / "registered-batch"
        run_experiment(EvaluationPrepare(evaluation_config, batch, registry))
        operation = replace(
            original_operation,
            batch_dir=batch,
            batch_sha256=sha(batch / "batch.json"),
            registry_file=registry,
            output_dir=folder / "run",
            seconds=20,
        )
        drive = folder / "drive"
        drive.mkdir()
        config = qualified_sac_config(
            drive, checkpoint, route_file=batch / "route/route.json", end_margin_m=0
        )
        devices = Devices(world_factory=SteeringWorld)
        environment = NativeEvaluationEnvironment(
            config, menu_factory=devices.menu, driving_factory=devices.drive
        )
        try:
            result = run_experiment(operation, evaluation_environment=environment)
            assert result.summary["evaluation_run"]["stop_reason"] == "plan_complete"
            ledger = bind_native_validity(operation, devices)
            reviewed = run_experiment(
                EvaluationReview(batch, ledger, folder / "review", registry)
            ).summary["evaluation"]
            assert all(a["outcome"] == "valid_complete" for a in reviewed["attempts"]), reviewed
            assert all(max(o["position_x_m"] for o in w.observations) >= 3 for w in devices.worlds)
            evaluated.append(
                (
                    reviewed["by_reference"]["no_reference"]["valid_duration_s"]["median"],
                    str(checkpoint),
                    {
                        "batch": str(batch),
                        "batch_sha256": sha(batch / "batch.json"),
                        "ledger": str(ledger),
                        "ledger_sha256": sha(ledger),
                    },
                )
            )
        finally:
            devices.cleanup()
    # The store test names the actually faster run candidate; it does not claim
    # these few updates improve FH5, nor edit any recorded score or timestamp.
    slower, faster = sorted(evaluated, reverse=True)
    assert faster[0] < slower[0] * 0.98, evaluated
    return (
        root,
        {"incumbent": slower[2], "candidate": faster[2]},
        {"incumbent": slower[1], "candidate": faster[1]},
        registry,
    )


def native_record_config(root, pair, **changes):
    path = record_config(root, pair, **changes)
    document = json.loads(path.read_bytes())
    document["version"] = 2
    path.write_text(json.dumps(document))
    return path


def test_native_store_cannot_relabel_synthetic_execution(tmp_path, candidates):
    config = native_record_config(tmp_path, candidates)
    with pytest.raises(ValueError, match="Incumbent lacks native qualification"):
        run_experiment(CandidateRecord(config, tmp_path / "native-store", None, candidates[3]))


def test_native_roles_retain_then_select_and_rollback_without_losing_learner(tmp_path, native_pair):
    store = tmp_path / "versions"
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    original = run_experiment(
        CandidateRecord(
            native_record_config(first, native_pair, missing_execution=True),
            store,
            None,
            native_pair[3],
        )
    ).summary["candidate_store"]
    assert original["scope"] == "native_development_only"
    assert original["selection"] == "retain_incumbent"
    assert "candidate:actual_native_policy_execution_incomplete" in original["reasons"]
    assert original["default"]["model_sha256"] == sha(
        Path(native_pair[2]["incumbent"]) / "policy.json"
    )
    promoted = run_experiment(
        CandidateRecord(
            native_record_config(second, native_pair), store, original["revision"], native_pair[3]
        )
    ).summary["candidate_store"]
    assert promoted["selection"] == "prefer_candidate_locally", promoted["reasons"]
    assert promoted["native_default_changed"] is True
    assert promoted["default"]["model_sha256"] == original["explorer"]["model_sha256"]
    rolled = run_experiment(
        CandidateRollback(
            store,
            promoted["revision"],
            original["revision"],
            "Restore retained baseline",
            native_pair[3],
        )
    ).summary["candidate_store"]
    assert rolled["default"] == original["default"]
    assert rolled["explorer"] == promoted["explorer"]
    history = run_experiment(CandidateHistory(store)).summary["candidate_store"]
    assert [event["operation"] for event in history["history"]] == [
        "selection",
        "selection",
        "rollback",
    ]
    assert all(
        not event["real_driving_validated"] and not event["default_changed"]
        for event in history["history"]
    )
    retained = store / history["explorer"]["archive"] / "checkpoint"
    assert (retained / "policy.pt").read_bytes() == (
        Path(native_pair[2]["candidate"]) / "policy.pt"
    ).read_bytes()
    selected_again = run_experiment(
        CandidateRollback(
            store,
            rolled["revision"],
            promoted["revision"],
            "Revalidate the previously selected candidate",
            native_pair[3],
        )
    ).summary["candidate_store"]
    assert selected_again["default"] == promoted["default"]
    assert selected_again["native_default_changed"] is True


@pytest.mark.parametrize("outcome", ["driving_failure", "wall_riding", "unknown"])
def test_native_invalid_or_unknown_candidate_cannot_become_default(tmp_path, native_pair, outcome):
    selected = run_experiment(
        CandidateRecord(
            native_record_config(tmp_path, native_pair, candidate_outcome=outcome),
            tmp_path / "versions",
            None,
            native_pair[3],
        )
    ).summary["candidate_store"]
    assert selected["selection"] == "retain_incumbent"
    assert selected["default"]["model_sha256"] == sha(
        Path(native_pair[2]["incumbent"]) / "policy.json"
    )
    assert selected["explorer"]["model_sha256"] == sha(
        Path(native_pair[2]["candidate"]) / "policy.json"
    )
    if outcome == "driving_failure":
        assert selected["aggressive_by_reference"] == {"no_reference": selected["explorer"]}
    else:
        assert selected["aggressive_by_reference"] == {}


def test_native_store_rejects_scope_changes_without_committing(tmp_path, native_pair):
    config = native_record_config(tmp_path, native_pair, missing_execution=True)
    store = tmp_path / "versions"
    original = run_experiment(CandidateRecord(config, store, None, native_pair[3])).summary[
        "candidate_store"
    ]
    document = json.loads(config.read_bytes())
    document["version"] = 1
    config.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="qualification scope changed"):
        run_experiment(CandidateRecord(config, store, original["revision"], native_pair[3]))
    history = run_experiment(CandidateHistory(store)).summary["candidate_store"]
    assert history["history"] == [original]
