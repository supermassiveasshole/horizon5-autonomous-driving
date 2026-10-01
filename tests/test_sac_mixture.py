"""Demonstration/online replay through the approved experiment-run seam."""

import hashlib
import json
from dataclasses import replace

import pytest
from test_sac import experience

from fh5.experiment import run_experiment
from fh5.sac_learning import SACPolicyReplay, SACResume, SACTrain


def test_complete_demonstration_keeps_human_actions_and_independent_terminal_feedback(tmp_path):
    request = experience(tmp_path, owner="human")
    result = run_experiment(request).summary["sac_replay"]
    replay = json.loads((request.output_dir / "replay.json").read_bytes())
    assert result["eligible_transitions"] == 2
    assert replay["source_role"] == "demonstration"
    assert replay["source_kind"] == "synthetic"
    assert replay["task_contract"]["control_owner"] == "human"
    assert [r["control_owner"] for r in replay["transitions"]] == ["human", "human"]
    assert [r["action"] for r in replay["transitions"]] == [[0, 0.2], [0, 0.2]]
    assert replay["transitions"][-1]["reward"] == pytest.approx(2.66)
    assert replay["transitions"][-1]["terminated"]
    assert result["real_driving_validated"] is False


def mixture_inputs(root):
    from test_sac_learning import warm_start

    replay = warm_start(root)
    initial = root / "initial"
    run_experiment(SACTrain(root / "warm", replay, initial, steps=0, batch_size=4))
    demo = root / "demo"
    demo.mkdir()
    request = experience(
        demo, owner="human", timeline=[(0, 0), (0.5, 100), (1, 200), (2, 300), (2.5, 400)]
    )
    task = json.loads((root / "task.json").read_bytes())
    task["control_owner"] = "human"
    request.task_file.write_text(json.dumps(task))
    request = replace(request, reward_file=root / "reward.json")
    prepared = run_experiment(request).summary["sac_replay"]
    assert prepared["eligible_transitions"] == 4
    return initial, (request.output_dir / "replay.json", prepared["replay_sha256"])


def test_mixed_sac_batches_learn_from_both_complete_sources_and_seal_actual_counts(tmp_path):
    pytest.importorskip("torch")
    initial, demonstration = mixture_inputs(tmp_path)
    output = tmp_path / "mixed"
    result = run_experiment(
        SACResume(initial, output, steps=4, additions=(demonstration,), demonstration_fraction=0.5)
    ).summary["sac_learning"]
    assert result["steps_completed"] == 4
    assert result["actor_updates"] == 2
    assert result["actor_change_max"] > 0
    assert result["encoder_change_during_actor_max"] == 0
    assert result["sampling"]["available"] == {"demonstration": 4, "online": 2}
    assert result["sampling"]["sampled"] == {"demonstration": 8, "online": 8}
    for update in result["updates"]:
        assert update["source_roles"].count("demonstration") == 2
        assert update["source_roles"].count("online") == 2
        assert len(set(update["transition_ids"])) == 4
    manifest = json.loads((output / "policy.json").read_bytes())
    assert manifest["configuration"]["demonstration_fraction"] == 0.5
    assert manifest["version"] == 3


def test_mixture_continuation_matches_uninterrupted_learning_and_zero_quota_exits_demos(tmp_path):
    pytest.importorskip("torch")
    initial, demo = mixture_inputs(tmp_path)
    seed = tmp_path / "seed"
    run_experiment(SACResume(initial, seed, steps=0, additions=(demo,), demonstration_fraction=0.5))
    whole = run_experiment(SACResume(seed, tmp_path / "whole", steps=6)).summary["sac_learning"]
    first = run_experiment(
        SACResume(seed, tmp_path / "first", steps=6), sac_stop_requested=lambda step: step == 3
    ).summary["sac_learning"]
    second = run_experiment(SACResume(tmp_path / "first", tmp_path / "second", steps=3)).summary[
        "sac_learning"
    ]
    assert first["stop_reason"] == "stop_requested"
    assert first["updates"] + second["updates"] == whole["updates"]
    assert second["learner_state_sha256"] == whole["learner_state_sha256"]
    reloaded = run_experiment(
        SACPolicyReplay(
            tmp_path / "second",
            tmp_path / "second/experience/replay.json",
            tmp_path / "reloaded.html",
        )
    ).summary["sac_policy"]
    assert reloaded["predictions"] == second["predictions"]
    exited = run_experiment(
        SACResume(tmp_path / "second", tmp_path / "exit", steps=2, demonstration_fraction=0)
    ).summary["sac_learning"]
    assert exited["sampling"]["sampled"] == {"demonstration": 0, "online": 4}
    assert exited["total_steps"] == 8
    continued = run_experiment(
        SACResume(tmp_path / "exit", tmp_path / "after-exit", steps=1)
    ).summary["sac_learning"]
    assert continued["sampling"]["sampled"] == {"demonstration": 0, "online": 2}


def test_small_pool_shrinks_batch_without_duplicating_or_backfilling(tmp_path):
    pytest.importorskip("torch")
    initial, demo = mixture_inputs(tmp_path)
    result = run_experiment(
        SACResume(
            initial, tmp_path / "mixed", steps=2, additions=(demo,), demonstration_fraction=0.25
        )
    ).summary["sac_learning"]
    assert result["sampling"]["quotas"] == {"demonstration": 1, "online": 3}
    assert result["sampling"]["sampled"] == {"demonstration": 2, "online": 4}
    for update in result["updates"]:
        assert len(update["transition_ids"]) == len(set(update["transition_ids"])) == 3


@pytest.mark.parametrize("error", ["owner", "action_only", "reward", "task", "source_relabel"])
def test_mixture_rejects_unknown_actions_incomplete_feedback_or_changed_sources(tmp_path, error):
    pytest.importorskip("torch")
    initial, (path, _) = mixture_inputs(tmp_path)
    if error == "source_relabel":
        seed = tmp_path / "seed"
        run_experiment(
            SACResume(
                initial,
                seed,
                steps=0,
                additions=((path, hashlib.sha256(path.read_bytes()).hexdigest()),),
                demonstration_fraction=0.5,
            )
        )
        path = seed / "experience/replay.json"
        # Independent edited union must still agree with its sealed original manifests.
    replay = json.loads(path.read_bytes())
    expected = {
        "owner": "control owner",
        "action_only": "complete transitions",
        "reward": "Incompatible SAC experience: reward",
        "task": "Incompatible SAC experience: task",
        "source_relabel": "sealed original source",
    }[error]
    if error == "owner":
        replay["transitions"][0]["control_owner"] = "policy"
    elif error == "action_only":
        del replay["transitions"][0]["reward"]
    elif error == "reward":
        replay["source_hashes"]["reward"] = "f" * 64
    elif error == "task":
        replay["task_contract"]["max_speed_kmh"] = 40
    else:
        replay["transitions"][-1]["control_owner"] = "policy"
    path.write_text(json.dumps(replay))
    with pytest.raises(ValueError, match=expected):
        run_experiment(
            SACResume(
                initial,
                tmp_path / "rejected",
                steps=1,
                additions=((path, hashlib.sha256(path.read_bytes()).hexdigest()),),
                demonstration_fraction=0.5,
            )
        )
    assert not (tmp_path / "rejected/policy.json").exists()


def test_legacy_online_experience_retains_uniform_sampling_and_rejects_missing_demo_pool(tmp_path):
    pytest.importorskip("torch")
    from test_temporal_bc import temporal_fixture

    from fh5.sac import SACCriticWarmup
    from fh5.temporal_bc import TemporalBCTrain

    request = experience(tmp_path)
    run_experiment(request)
    path = request.output_dir / "replay.json"
    legacy = json.loads(path.read_bytes())
    legacy.update(version=1, kind="sac-numeric-replay-v1")
    del legacy["source_role"], legacy["task_contract"]
    for row in legacy["transitions"]:
        del row["control_owner"]
    path.write_text(json.dumps(legacy))
    bc = tmp_path / "bc"
    bc.mkdir()
    config, _ = temporal_fixture(bc)
    run_experiment(TemporalBCTrain(config, bc / "model"))
    run_experiment(
        SACCriticWarmup(
            bc / "model",
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            tmp_path / "warm",
            steps=1,
        )
    )
    result = run_experiment(
        SACTrain(tmp_path / "warm", path, tmp_path / "uniform", steps=2)
    ).summary["sac_learning"]
    assert result["sampling"]["mode"] == "uniform"
    assert result["sampling"]["sampled"] == {"demonstration": 0, "online": 4}
    assert json.loads((tmp_path / "uniform/policy.json").read_bytes())["version"] == 2
    with pytest.raises(ValueError, match="nonempty source pool"):
        run_experiment(
            SACResume(
                tmp_path / "uniform", tmp_path / "invalid", steps=1, demonstration_fraction=0.5
            )
        )


def test_out_of_support_demonstration_is_rejected_without_clipping_its_action(tmp_path):
    pytest.importorskip("torch")
    initial, (path, _) = mixture_inputs(tmp_path)
    replay = json.loads(path.read_bytes())
    replay["transitions"][0]["action"] = [0, 1]
    path.write_text(json.dumps(replay))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="outside executable command support"):
        run_experiment(
            SACResume(
                initial,
                tmp_path / "invalid",
                steps=1,
                additions=((path, hashlib.sha256(original).hexdigest()),),
                demonstration_fraction=0.5,
            )
        )
    assert path.read_bytes() == original


def test_declaring_legacy_format_does_not_relabel_known_human_actions_as_online(tmp_path):
    pytest.importorskip("torch")
    initial, (path, _) = mixture_inputs(tmp_path)
    replay = json.loads(path.read_bytes())
    replay.update(version=1, kind="sac-numeric-replay-v1")
    del replay["source_role"]
    path.write_text(json.dumps(replay))
    with pytest.raises(ValueError, match="Legacy SAC replay"):
        run_experiment(
            SACResume(
                initial,
                tmp_path / "invalid",
                steps=0,
                additions=((path, hashlib.sha256(path.read_bytes()).hexdigest()),),
            )
        )


def test_mixed_replay_cannot_change_the_original_task_context_for_q_warmup(tmp_path):
    pytest.importorskip("torch")
    from fh5.sac import SACCriticWarmup

    initial, demo = mixture_inputs(tmp_path)
    seed = tmp_path / "seed"
    run_experiment(SACResume(initial, seed, steps=0, additions=(demo,), demonstration_fraction=0.5))
    path = seed / "experience/replay.json"
    replay = json.loads(path.read_bytes())
    replay["task_context"]["route_length_m"] = 300
    path.write_text(json.dumps(replay))
    with pytest.raises(ValueError, match="Incompatible SAC experience: task_context"):
        run_experiment(
            SACCriticWarmup(
                tmp_path / "bc/model",
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                tmp_path / "invalid",
                steps=1,
            )
        )
