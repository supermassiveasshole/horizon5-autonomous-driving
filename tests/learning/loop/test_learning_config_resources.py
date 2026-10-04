"""Declared learning batches and update schedules have no extra arbitrary ceilings."""

import json

from fh5.experiment import run_experiment
from fh5.learning.sac.critic import SACCriticResume, SACCriticWarmup
from fh5.learning.sac.training import SACResume, SACTrain
from tests.artifacts.test_replay_provenance_resources import source_pair
from tests.learning.sac.test_critic_resume import warm_inputs
from tests.support.checkpoint_files import update_records


def large_batch_inputs(root):
    model, replay, _ = warm_inputs(root)
    template = json.loads(replay.read_bytes())
    path = replay.with_name("batch.json")
    digest = source_pair(path, template, count=257)
    return model, path, digest


def test_critic_uses_the_declared_batch_above_the_old_ceiling_and_resumes(tmp_path):
    model, replay, digest = large_batch_inputs(tmp_path)
    first, resumed, whole = (tmp_path / name for name in ("first", "resumed", "whole"))
    run_experiment(
        SACCriticWarmup(model, replay, digest, first, steps=2, batch_size=257),
        sac_stop_requested=lambda step: step == 1,
    )
    continued = run_experiment(SACCriticResume(first, resumed)).summary["sac"]
    continuous = run_experiment(
        SACCriticWarmup(model, replay, digest, whole, steps=2, batch_size=257)
    ).summary["sac"]
    assert continued["total_steps"] == 2
    assert continued["learner_state_sha256"] == continuous["learner_state_sha256"]
    assert update_records(first) + update_records(resumed) == update_records(whole)
    for row in update_records(whole):
        assert len(row["transition_ids"]) == len(set(row["transition_ids"])) == 257


def test_sac_uses_declared_batch_and_actor_interval_across_resume(tmp_path):
    model, replay, digest = large_batch_inputs(tmp_path)
    warm = tmp_path / "warm"
    run_experiment(SACCriticWarmup(model, replay, digest, warm, steps=1))
    first, resumed, whole = (tmp_path / name for name in ("first", "resumed", "whole"))
    partial = run_experiment(
        SACTrain(warm, replay, first, steps=12, batch_size=257, actor_interval=11),
        sac_stop_requested=lambda step: step == 10,
    ).summary["sac_learning"]
    continued = run_experiment(SACResume(first, resumed, steps=2)).summary["sac_learning"]
    continuous = run_experiment(
        SACTrain(warm, replay, whole, steps=12, batch_size=257, actor_interval=11)
    ).summary["sac_learning"]
    assert partial["actor_updates"] == 0
    assert continued["total_steps"] == 12 and continued["actor_updates"] == 1
    assert continuous["actor_updates"] == 1
    assert continued["learner_state_sha256"] == continuous["learner_state_sha256"]
    assert update_records(first) + update_records(resumed) == update_records(whole)
    for row in update_records(whole):
        assert len(row["transition_ids"]) == len(set(row["transition_ids"])) == 257
