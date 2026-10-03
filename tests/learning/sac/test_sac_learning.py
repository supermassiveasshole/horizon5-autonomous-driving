"""Actual bounded SAC updates through the experiment-run interface."""

import json

import pytest

from fh5.experiment import run_experiment
from fh5.learning.bc.training import TemporalBCTrain
from fh5.learning.sac.critic import SACCriticWarmup
from tests.learning.bc.test_temporal_bc import temporal_fixture
from tests.learning.sac.test_sac import experience
from tests.support.checkpoint_files import prediction_records, update_records


def warm_start(root, *, bounds=None):
    from fh5.learning.sac.actions import ActionBounds

    request = experience(root)
    prepared = run_experiment(request).summary["sac_replay"]
    bc = root / "bc"
    bc.mkdir()
    config, _ = temporal_fixture(bc)
    run_experiment(TemporalBCTrain(config, bc / "model"))
    run_experiment(
        SACCriticWarmup(
            bc / "model",
            request.output_dir / "replay.json",
            prepared["replay_sha256"],
            root / "warm",
            steps=3,
            bounds=bounds or ActionBounds(),
        )
    )
    return request.output_dir / "replay.json"


def test_sac_updates_policy_temperature_and_encoder_with_separate_owners(tmp_path):
    from fh5.learning.sac.training import SACTrain

    replay = warm_start(tmp_path)
    original = (tmp_path / "warm/actor/actor.pt").read_bytes()
    result = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "candidate", steps=4)
    ).summary["sac_learning"]
    assert result["steps_completed"] == 4
    assert result["actor_updates"] == 2
    assert result["encoder_change_max"] > 0
    assert result["actor_change_max"] > 0
    assert result["critic_change_max"] > 0
    assert result["target_encoder_change_max"] > 0
    assert result["alpha_after"] > result["alpha_before"]
    assert result["optimizer_parameters_disjoint"] is True
    assert result["encoder_change_during_actor_max"] == 0
    assert result["actor_change_during_critic_max"] == 0
    assert result["bc_transfer_command_error"] == 0
    assert result["commands_sent"] is False
    assert result["real_driving_validated"] is False
    assert (tmp_path / "warm/actor/actor.pt").read_bytes() == original
    assert (tmp_path / "candidate/policy.pt").is_file()
    assert json.loads((tmp_path / "candidate/policy.json").read_bytes())["stage"] == "sac_updates"


def test_learned_policy_reloads_with_same_commands_and_continuous_density(tmp_path):
    import math

    from fh5.learning.sac.training import SACPolicyReplay, SACTrain

    replay = warm_start(tmp_path)
    trained = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "candidate", steps=4)
    ).summary["sac_learning"]
    restored = run_experiment(
        SACPolicyReplay(tmp_path / "candidate", replay, tmp_path / "replayed.html")
    ).summary["sac_policy"]
    assert prediction_records(tmp_path, restored) == prediction_records(
        tmp_path / "candidate", trained
    )
    assert restored["commands_sent"] is False
    for row in prediction_records(tmp_path, restored):
        # Density at the Gaussian mean = 1/(sigma*sqrt(2*pi)), divided
        # by each transform derivative scale*(1-unit^2).
        density = 1.0
        log_area = 0.0
        for axis in range(2):
            lo, hi = row["context"][3 + axis], row["context"][5 + axis]
            scale, center = (hi - lo) / 2, (hi + lo) / 2
            unit = (row["continuous"][axis] - center) / scale
            density /= math.exp(row["log_std"][axis]) * math.sqrt(2 * math.pi)
            density /= scale * (1 - unit * unit)
            log_area += math.log(scale)
        assert row["log_probability"] == pytest.approx(math.log(density), abs=2e-5)
        assert row["target_entropy"] == pytest.approx(-2 + log_area, abs=2e-6)
    manifest_path = tmp_path / "candidate/policy.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["bounds"]["max_steer"] = 0.3
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metadata"):
        run_experiment(SACPolicyReplay(tmp_path / "candidate", replay, tmp_path / "bad.html"))


def test_sac_terminal_target_keeps_physical_reward_without_bootstrap(tmp_path):
    from fh5.learning.sac.training import SACTrain

    replay = warm_start(tmp_path)
    trained = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "candidate", steps=2)
    ).summary["sac_learning"]
    assert trained["steps_completed"] == 2
    for update in update_records(tmp_path / "candidate"):
        targets = dict(zip(update["transition_ids"], update["targets"]))
        assert targets["transition-1"] == pytest.approx(2.66)


def test_saturated_bc_handoff_and_stochastic_commands_share_quantized_bounds(tmp_path):
    from fh5.learning.sac.actions import ActionBounds
    from fh5.learning.sac.training import SACPolicyReplay, SACTrain

    replay = warm_start(tmp_path, bounds=ActionBounds(max_steer=0.01))
    trained = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "candidate", steps=0)
    ).summary["sac_learning"]
    assert trained["bc_transfer_command_error"] == 0
    assert prediction_records(tmp_path / "candidate", trained)[0]["deterministic"][
        0
    ] * 32767 == pytest.approx(328, abs=2e-5)
    sampled = run_experiment(
        SACPolicyReplay(tmp_path / "candidate", replay, tmp_path / "sampled.html", noise=(-2, 2))
    ).summary["sac_policy"]
    assert prediction_records(tmp_path, sampled) != prediction_records(
        tmp_path / "candidate", trained
    )
    for row in prediction_records(tmp_path, sampled):
        for i, scale in enumerate((32767, 255)):
            value = row["command"][i]
            assert row["context"][3 + i] - 1e-7 <= value <= row["context"][5 + i] + 1e-7
            assert value * scale == pytest.approx(round(value * scale), abs=0.002)


@pytest.mark.parametrize("explicit_replay", [False, True])
def test_cli_trains_and_replays_the_updated_sac_policy(tmp_path, capsys, explicit_replay):
    from fh5.cli import main

    replay = warm_start(tmp_path)
    options = {"version": 1, "warmup": "warm", "steps": 2}
    if explicit_replay:
        options["replay"] = str(replay)
    else:
        # Only the portable warm-up is available; do not reach back into the
        # original BC or experience directories to initialize SAC.
        (tmp_path / "bc").rename(tmp_path / "old-bc")
        replay.parent.rename(tmp_path / "old-experience")
        replay = tmp_path / "warm/experience/replay.json"
    config = tmp_path / "sac.json"
    config.write_text(json.dumps(options))
    assert (
        main(["sac-train", "--config", str(config), "--output", str(tmp_path / "candidate")]) == 0
    )
    trained = json.loads(capsys.readouterr().out)
    assert trained["actor_updates"] == 1
    assert trained["steps_completed"] == 2
    assert trained["commands_sent"] is False
    assert (tmp_path / "candidate/experience/replay.json").read_bytes() == replay.read_bytes()
    if not explicit_replay:
        # The learned candidate retains its own experience and no longer needs
        # the preheating checkpoint or the original dataset to replay.
        (tmp_path / "warm").rename(tmp_path / "old-warm")
    assert (
        main(
            [
                "sac-policy-replay",
                "--checkpoint",
                str(tmp_path / "candidate"),
                *(["--replay", str(replay)] if explicit_replay else []),
                "--report",
                str(tmp_path / "cli.html"),
            ]
        )
        == 0
    )
    restored = json.loads(capsys.readouterr().out)
    assert prediction_records(tmp_path, restored) == prediction_records(
        tmp_path / "candidate", trained
    )


@pytest.mark.parametrize("fault", ["explicit_missing", "explicit_changed", "sealed_changed"])
def test_cli_rejects_unavailable_or_changed_experience_without_fallback(tmp_path, capsys, fault):
    from fh5.cli import main

    replay = warm_start(tmp_path)
    warmup = tmp_path / "warm"
    options = {"version": 1, "warmup": "warm", "steps": 2}
    if fault.startswith("explicit"):
        options["replay"] = str(replay)
    else:
        replay = warmup / "experience/replay.json"
    if fault == "explicit_missing":
        replay.rename(replay.with_suffix(".saved"))
    else:
        replay.write_text("{}")
    preserved = {p: p.read_bytes() for p in warmup.rglob("*") if p.is_file()}
    config = tmp_path / "sac.json"
    config.write_text(json.dumps(options))
    output = tmp_path / "rejected"
    assert main(["sac-train", "--config", str(config), "--output", str(output)]) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["status"] == "error"
    if fault != "explicit_missing":
        assert "hash mismatch" in failure["message"]
    assert not output.exists()
    assert all(path.read_bytes() == payload for path, payload in preserved.items())


@pytest.mark.parametrize("mode", ["sac-critic-replay", "sac-policy-replay"])
@pytest.mark.parametrize("fault", ["explicit_missing", "explicit_changed", "sealed_changed"])
def test_cli_frozen_replay_never_substitutes_other_experience(tmp_path, capsys, mode, fault):
    from fh5.cli import main
    from fh5.learning.sac.training import SACTrain

    replay = warm_start(tmp_path)
    checkpoint = tmp_path / "warm"
    if mode == "sac-policy-replay":
        checkpoint = tmp_path / "candidate"
        run_experiment(SACTrain(tmp_path / "warm", replay, checkpoint, steps=1))
    report = tmp_path / "rejected.html"
    args = [mode, "--checkpoint", str(checkpoint), "--report", str(report)]
    if fault.startswith("explicit"):
        args += ["--replay", str(replay)]
    else:
        replay = checkpoint / "experience/replay.json"
    if fault == "explicit_missing":
        replay.rename(replay.with_suffix(".saved"))
    else:
        replay.write_text("{}")
    preserved = {p: p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}
    assert main(args) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["status"] == "error"
    if fault != "explicit_missing":
        assert "hash mismatch" in failure["message"]
    assert not report.exists()
    assert not report.with_suffix(".json").exists()
    assert all(path.read_bytes() == payload for path, payload in preserved.items())


def test_policy_replay_cannot_overwrite_candidate_experience_or_existing_reports(tmp_path):
    from fh5.learning.sac.training import SACPolicyReplay, SACTrain

    replay = warm_start(tmp_path)
    candidate = tmp_path / "candidate"
    run_experiment(SACTrain(tmp_path / "warm", replay, candidate, steps=0))
    preserved = {
        path: path.read_bytes()
        for path in (
            candidate / "policy.pt",
            candidate / "policy.json",
            candidate / "bc/actor.pt",
            candidate / "report.html",
            replay,
            tmp_path / "frame.rgb",
        )
    }
    for destination, original in preserved.items():
        with pytest.raises(FileExistsError):
            run_experiment(SACPolicyReplay(candidate, replay, destination))
        assert destination.read_bytes() == original
    with pytest.raises(ValueError, match="HTML"):
        run_experiment(SACPolicyReplay(candidate, replay, tmp_path / "wrong.pt"))
    assert not (tmp_path / "wrong.pt").exists()
    report = tmp_path / "reports/fresh.html"
    result = run_experiment(SACPolicyReplay(candidate, replay, report))
    assert result.report_path == report
    assert report.is_file()
