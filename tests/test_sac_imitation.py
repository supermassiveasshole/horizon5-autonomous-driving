"""Temporary BC guidance and its lifecycle at the experiment-run boundary."""

import json

import pytest
from test_sac_learning import warm_start

from fh5.experiment import run_experiment
from fh5.sac_learning import SACPolicyReplay, SACResume, SACTrain


def test_guidance_freezes_its_development_protocol_before_evaluation(tmp_path):
    from test_evaluation_start import automatic_request

    replay = warm_start(tmp_path)
    (tmp_path / "evaluation").mkdir()
    protocol = automatic_request(tmp_path / "evaluation", tmp_path / "warm/actor")
    trained = run_experiment(
        SACTrain(
            tmp_path / "warm",
            replay,
            tmp_path / "guided",
            steps=2,
            imitation_weights=(1.0, 0.0),
            imitation_protocol_batch=protocol.batch_dir,
        )
    ).summary["sac_learning"]
    assert len(trained["imitation"]["protocol_sha256"]) == 64
    resumed = run_experiment(SACResume(tmp_path / "guided", tmp_path / "resumed", steps=0))
    assert (
        resumed.summary["sac_learning"]["imitation"]["protocol_sha256"]
        == trained["imitation"]["protocol_sha256"]
    )


def test_missing_guidance_evaluation_cannot_be_replaced_by_a_passed_flag(tmp_path):
    replay = warm_start(tmp_path)
    run_experiment(
        SACTrain(
            tmp_path / "warm", replay, tmp_path / "guided", steps=2, imitation_weights=(1.0, 0.0)
        )
    )
    forged = tmp_path / "selection.json"
    forged.write_text(
        json.dumps({"passed": True, "local_recommendation": "prefer_candidate_locally"})
    )
    with pytest.raises(ValueError, match="protocol"):
        run_experiment(
            SACResume(
                tmp_path / "guided", tmp_path / "refused", steps=0, imitation_comparison=forged
            )
        )
    assert not (tmp_path / "refused/policy.json").exists()


def test_cli_resolves_a_guidance_protocol_relative_to_the_config(tmp_path, capsys):
    from test_evaluation_start import automatic_request

    from fh5.cli import main

    replay = warm_start(tmp_path)
    (tmp_path / "evaluation").mkdir()
    protocol = automatic_request(tmp_path / "evaluation", tmp_path / "warm/actor")
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "warmup": "warm",
                "replay": str(replay),
                "steps": 0,
                "imitation_weights": [1.0, 0.0],
                "imitation_protocol_batch": str(protocol.batch_dir.relative_to(tmp_path)),
            }
        )
    )
    assert main(["sac-train", "--config", str(config), "--output", str(tmp_path / "trained")]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "sac-resume",
                "--checkpoint",
                str(tmp_path / "trained"),
                "--output",
                str(tmp_path / "resumed"),
                "--steps",
                "0",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["imitation"]["weight"] == 1


def test_protocol_without_an_adopted_guidance_plan_is_not_silently_ignored(tmp_path):
    replay = warm_start(tmp_path)
    with pytest.raises(ValueError, match="weights"):
        run_experiment(
            SACTrain(
                tmp_path / "warm",
                replay,
                tmp_path / "invalid",
                steps=0,
                imitation_protocol_batch=tmp_path / "not-a-batch",
            )
        )
    assert not (tmp_path / "invalid").exists()


def test_explicitly_exited_imitation_is_identical_to_unconstrained_sac(tmp_path):
    replay = warm_start(tmp_path)
    plain = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "plain", steps=6)
    ).summary["sac_learning"]
    exited = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "exited", steps=6, imitation_weights=(0.0,))
    ).summary["sac_learning"]
    assert exited["learner_state_sha256"] == plain["learner_state_sha256"]
    assert exited["predictions"] == plain["predictions"]
    assert exited["imitation"]["weight"] == 0
    assert exited["imitation"]["phase"] == "exited"
    assert exited["imitation"]["teacher_evaluations"] == 0
    manifest = json.loads((tmp_path / "exited/policy.json").read_bytes())
    assert manifest["version"] == 4
    assert manifest["imitation"]["weights"] == [0.0]


def test_guidance_changes_actor_objective_but_never_owns_encoder_or_teacher(tmp_path):
    replay = warm_start(tmp_path)
    teacher_bytes = (tmp_path / "warm/actor/actor.pt").read_bytes()
    plain = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "plain", steps=12)
    ).summary["sac_learning"]
    guided = run_experiment(
        SACTrain(
            tmp_path / "warm", replay, tmp_path / "guided", steps=12, imitation_weights=(5.0, 0.0)
        )
    ).summary["sac_learning"]
    assert guided["imitation"]["phase"] == "guided"
    assert guided["imitation"]["teacher_evaluations"] == 6
    actor_updates = [step for step in guided["updates"] if "actor_loss" in step]
    assert any(step["imitation_loss"] > 0 for step in actor_updates)
    for step in actor_updates:
        assert step["actor_loss"] == pytest.approx(
            step["sac_actor_loss"] + 5 * step["imitation_loss"]
        )
    assert guided["learner_state_sha256"] != plain["learner_state_sha256"]
    assert guided["encoder_change_during_actor_max"] == 0
    assert guided["actor_change_during_critic_max"] == 0
    assert guided["imitation"]["teacher_change_max"] == 0
    assert (tmp_path / "guided/bc/actor.pt").read_bytes() == teacher_bytes


def test_imitation_schedule_and_updates_survive_an_odd_stop_and_reload(tmp_path):
    replay = warm_start(tmp_path)
    whole = run_experiment(
        SACTrain(
            tmp_path / "warm",
            replay,
            tmp_path / "whole",
            steps=8,
            imitation_weights=(2.0, 0.5, 0.0),
        )
    ).summary["sac_learning"]
    first = run_experiment(
        SACTrain(
            tmp_path / "warm",
            replay,
            tmp_path / "first",
            steps=8,
            imitation_weights=(2.0, 0.5, 0.0),
        ),
        sac_stop_requested=lambda step: step == 3,
    ).summary["sac_learning"]
    resumed = run_experiment(SACResume(tmp_path / "first", tmp_path / "resumed", steps=5)).summary[
        "sac_learning"
    ]
    assert resumed["imitation"]["weights"] == [2.0, 0.5, 0.0]
    assert resumed["imitation"]["weight"] == 2
    assert resumed["imitation"]["index"] == 0
    assert first["updates"] + resumed["updates"] == whole["updates"]
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]
    reloaded = run_experiment(
        SACPolicyReplay(
            tmp_path / "resumed",
            tmp_path / "resumed/experience/replay.json",
            tmp_path / "reloaded.html",
        )
    ).summary["sac_policy"]
    assert reloaded["predictions"] == resumed["predictions"]
