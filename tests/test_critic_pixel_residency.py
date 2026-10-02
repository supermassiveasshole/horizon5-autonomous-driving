"""Frozen critic warm-up releases pixels through the public experiment seam."""

import hashlib
import json
import tracemalloc
from copy import deepcopy

import pytest
from test_critic_resume import warm_inputs
from test_sac import experience
from test_temporal_bc import temporal_fixture

from fh5.experiment import run_experiment
from fh5.sac import SACCriticReplay, SACCriticResume, SACCriticWarmup
from fh5.temporal_bc import TemporalBCTrain


def critic_corpus(tmp_path, count=16):
    pytest.importorskip("torch")
    request = experience(tmp_path)
    run_experiment(request)
    replay_path = request.output_dir / "replay.json"
    replay = json.loads(replay_path.read_bytes())
    bc = tmp_path / "bc"
    bc.mkdir()
    config, dataset_path = temporal_fixture(bc)
    dataset = json.loads(dataset_path.read_bytes())
    size = [512, 288]
    pixels = bytes([51, 17, 34]) * (size[0] * size[1])
    pixel_sha = hashlib.sha256(pixels).hexdigest()

    for document in (dataset, replay):
        document["pixel_contract"]["size"] = size
    (dataset_path.parent / "frame.rgb").write_bytes(pixels)
    for sample in dataset["decisions"]:
        for frame in sample["frames"]:
            frame.update(size=size, sha256=pixel_sha, source_layout={"size": size, "format": "RGB"})
    dataset_path.write_text(json.dumps(dataset))
    settings = json.loads(config.read_bytes())
    settings["dataset_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    config.write_text(json.dumps(settings))
    run_experiment(TemporalBCTrain(config, bc / "model"))

    # Distinct raw files and observation identities; these are synthetic task
    # inputs, not a claim of independent game driving or a production quota.
    template = replay["transitions"][-1]
    replay["transitions"] = []
    for index in range(count):
        row = deepcopy(template)
        row["id"] = f"terminal-{index}"
        row["current"]["decision_id"] = f"current-{index}"
        row["next"] = None
        for slot, frame in enumerate(row["current"]["frames"]):
            name = f"corpus-{index}-{slot}.rgb"
            (replay_path.parent / name).write_bytes(pixels)
            frame.update(
                frame_id=name,
                path=name,
                size=size,
                sha256=pixel_sha,
                source_layout={"size": size, "format": "RGB"},
            )
        replay["transitions"].append(row)
    replay_path.write_text(json.dumps(replay))
    digest = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    source_bytes = sum(p.stat().st_size for p in replay_path.parent.glob("corpus-*.rgb"))
    return bc / "model", replay_path, digest, source_bytes


def test_critic_pixel_residency_does_not_grow_with_the_frozen_corpus(tmp_path):
    model, replay_path, digest, source_bytes = critic_corpus(tmp_path)
    observed = []

    def stop_after_three(step):
        if step == 0:
            observed.append(tracemalloc.get_traced_memory()[0])
        return step == 3

    tracemalloc.start()
    try:
        first = run_experiment(
            SACCriticWarmup(model, replay_path, digest, tmp_path / "first", steps=5),
            sac_stop_requested=stop_after_three,
        ).summary["sac"]
    finally:
        tracemalloc.stop()
    assert observed and observed[0] < source_bytes, (
        "Warm-up retains the entire raw corpus before its first update",
        observed,
        source_bytes,
    )
    assert first["steps_completed"] == 3
    resumed = run_experiment(SACCriticResume(tmp_path / "first", tmp_path / "resumed")).summary[
        "sac"
    ]
    whole = run_experiment(
        SACCriticWarmup(model, replay_path, digest, tmp_path / "whole", steps=5)
    ).summary["sac"]
    replayed = run_experiment(
        SACCriticReplay(tmp_path / "resumed", replay_path, tmp_path / "reloaded.html")
    ).summary["sac"]
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]
    assert resumed["predictions"] == whole["predictions"] == replayed["predictions"]
    assert resumed["actor_change_max"] == resumed["reload_max_abs_error"] == 0


@pytest.mark.parametrize("name", ["model.json", "actor.pt"])
def test_critic_rechecks_sealed_actor_after_preflight_and_updates(tmp_path, name):
    model, replay, digest = warm_inputs(tmp_path)
    output = tmp_path / "warm"

    def change_sealed_actor(step):
        if step == 1:
            with (output / "actor" / name).open("ab") as stream:
                stream.write(b"changed")
        return False

    with pytest.raises(ValueError, match="Frozen BC changed"):
        run_experiment(
            SACCriticWarmup(model, replay, digest, output, steps=2),
            sac_stop_requested=change_sealed_actor,
        )
    assert not (output / "critic.json").exists()
