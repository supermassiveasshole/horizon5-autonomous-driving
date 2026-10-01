"""Continuation of actual learner state at the experiment-run boundary."""

import json

import pytest
from test_sac_learning import warm_start

from fh5.experiment import run_experiment
from fh5.sac_learning import SACTrain


def test_resumed_learning_matches_uninterrupted_updates_including_odd_actor_phase(tmp_path):
    from fh5.sac_learning import SACResume

    replay = warm_start(tmp_path)
    whole = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "whole", steps=8)
    ).summary["sac_learning"]
    first = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "first", steps=3)
    ).summary["sac_learning"]
    source = (tmp_path / "first/policy.pt").read_bytes()
    resumed = run_experiment(SACResume(tmp_path / "first", tmp_path / "resumed", steps=5)).summary[
        "sac_learning"
    ]
    assert resumed["steps_completed"] == 5
    assert resumed["total_steps"] == 8
    assert resumed["actor_updates"] == 3
    assert resumed["actor_updates_total"] == 4
    assert first["updates"] + resumed["updates"] == whole["updates"]
    assert resumed["predictions"] == whole["predictions"]
    assert resumed["alpha_after"] == whole["alpha_after"]
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]
    assert resumed["bc_transfer_command_error"] is None
    assert resumed["commands_sent"] is False
    assert (tmp_path / "first/policy.pt").read_bytes() == source
    manifest = json.loads((tmp_path / "resumed/policy.json").read_bytes())
    assert manifest["continuation"]["parent_step"] == 3


def test_resume_rejects_output_inside_the_frozen_source(tmp_path):
    from fh5.sac_learning import SACResume

    replay = warm_start(tmp_path)
    candidate = tmp_path / "candidate"
    run_experiment(SACTrain(tmp_path / "warm", replay, candidate, steps=1))
    with pytest.raises(ValueError, match="outside"):
        run_experiment(SACResume(candidate, candidate / "child", steps=1))
    assert not (candidate / "child").exists()


def test_cli_continues_a_moved_snapshot_without_the_original_training_directories(tmp_path, capsys):
    from fh5.cli import main

    replay = warm_start(tmp_path)
    run_experiment(SACTrain(tmp_path / "warm", replay, tmp_path / "first", steps=3))
    (tmp_path / "warm").rename(tmp_path / "old-warm")
    replay.parent.rename(tmp_path / "old-experience")
    moved = tmp_path / "moved"
    (tmp_path / "first").rename(moved)
    assert (
        main(
            [
                "sac-resume",
                "--checkpoint",
                str(moved),
                "--output",
                str(tmp_path / "next"),
                "--steps",
                "2",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["total_steps"] == 5
    assert result["steps_completed"] == 2
    assert result["actor_updates"] == 1
    assert (tmp_path / "next/experience/replay.json").read_bytes() == (
        moved / "experience/replay.json"
    ).read_bytes()


def test_requested_stop_saves_a_completed_update_boundary_and_can_continue(tmp_path):
    from fh5.sac_learning import SACResume

    replay = warm_start(tmp_path)
    whole = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "whole", steps=8)
    ).summary["sac_learning"]
    stopped = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "stopped", steps=8),
        sac_stop_requested=lambda completed: completed == 3,
    ).summary["sac_learning"]
    assert stopped["stop_reason"] == "stop_requested"
    assert stopped["steps_completed"] == 3
    resumed = run_experiment(
        SACResume(tmp_path / "stopped", tmp_path / "continued", steps=5)
    ).summary["sac_learning"]
    assert resumed["stop_reason"] == "budget_completed"
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]


def test_resume_preserves_prior_training_history_and_rejects_changed_reports(tmp_path):
    from fh5.sac_learning import SACResume

    replay = warm_start(tmp_path)
    run_experiment(SACTrain(tmp_path / "warm", replay, tmp_path / "first", steps=3))
    run_experiment(SACResume(tmp_path / "first", tmp_path / "second", steps=2))
    manifest = json.loads((tmp_path / "second/policy.json").read_bytes())
    assert len(manifest["history"]) == 2  # Q warm-up and the first SAC update segment.
    warmup = manifest["history"][0]
    assert (tmp_path / "second" / warmup["report"]).read_bytes() == (
        tmp_path / "warm/training-report.json"
    ).read_bytes()
    entry = manifest["history"][1]
    assert (tmp_path / "second" / entry["report"]).read_bytes() == (
        tmp_path / "first/training-report.json"
    ).read_bytes()
    (tmp_path / "second/training-report.json").write_text("{}")
    with pytest.raises(ValueError, match="training report"):
        run_experiment(SACResume(tmp_path / "second", tmp_path / "third", steps=1))
    assert not (tmp_path / "third").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "missing_pixels",
        "pixel_bytes",
        "frame_time",
        "action_hold",
        "reward",
        "history_offsets",
        "numeric_shape",
        "bounds",
        "weights",
    ],
)
def test_resume_rejects_missing_or_changed_learning_dependencies(tmp_path, fault):
    from fh5.sac_learning import SACResume

    replay = warm_start(tmp_path)
    candidate = tmp_path / "candidate"
    run_experiment(SACTrain(tmp_path / "warm", replay, candidate, steps=1))
    if fault in ("missing_pixels", "pixel_bytes"):
        frame = next((candidate / "experience/frames").glob("*.rgb"))
        if fault == "missing_pixels":
            frame.unlink()
        else:
            frame.write_bytes(b"bad pixels")
    elif fault == "weights":
        (candidate / "policy.pt").write_bytes(b"damaged weights")
    else:
        path = candidate / "experience/replay.json"
        if fault == "numeric_shape":
            path = candidate / "bc/model.json"
        elif fault == "bounds":
            path = candidate / "policy.json"
        document = json.loads(path.read_bytes())
        if fault == "frame_time":
            document["transitions"][0]["current"]["frames"][0]["source_time_ns"] += 1
        elif fault == "action_hold":
            document["transitions"][0]["hold_dt_s"] *= 2
        elif fault == "reward":
            document["transitions"][0]["reward"] += 100
        elif fault == "history_offsets":
            document["pixel_contract"]["history_offsets_ms"] = [400, 200, 0]
        elif fault == "numeric_shape":
            document["contract"]["numeric_size"] += 1
        elif fault == "bounds":
            document["bounds"]["max_steer"] = 0.2
        path.write_text(json.dumps(document))
    with pytest.raises((ValueError, OSError)):
        run_experiment(SACResume(candidate, tmp_path / "rejected", steps=1))
    assert not (tmp_path / "rejected").exists()
