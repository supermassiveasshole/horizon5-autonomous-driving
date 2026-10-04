"""Explicit BC batches exercise the full observation pool through public runs."""

import hashlib
import json
from copy import deepcopy

import pytest

from fh5.experiment import run_experiment
from fh5.learning.bc.training import TemporalBCReplay, TemporalBCTrain
from tests.learning.bc.test_bc_loss_resources import loss_values
from tests.learning.bc.test_temporal_bc import temporal_fixture


def full_batch_inputs(tmp_path):
    config, snapshot = temporal_fixture(tmp_path)
    document = json.loads(snapshot.read_bytes())
    template = document["decisions"][0]
    heldout = document["decisions"][2::2]
    training = []
    for index in range(129):
        row = deepcopy(template)
        row["decision_id"] = f"attempt-0:{index}"
        row["decision_ns"] += index * 1_000_000_000
        row["supervision"]["action"] = [0.0, 0.0]
        rgb = bytes((index, 17, (index * 3) % 256)) * (64 * 36)
        image = snapshot.parent / f"observation-{index}.rgb"
        image.write_bytes(rgb)
        for slot, frame in enumerate(row["frames"]):
            frame.update(
                frame_id=f"attempt-0:{index}:{slot}",
                path=image.name,
                sha256=hashlib.sha256(rgb).hexdigest(),
            )
            for field in ("source_time_ns", "capture_received_ns", "preprocess_ready_ns"):
                frame[field] += index * 1_000_000_000
        for actor in row["views"].values():
            actor["ego"]["speed_mps"] = 5 + index / 16
            actor["ego"]["velocity_car_mps"][2] = 5 + index / 16
        training.append(row)
    document["decisions"] = [*training, *heldout]
    options = json.loads(config.read_bytes())
    settings = {}
    for name, tail_target in (("zero", 0.0), ("positive", 1.0), ("negative", -1.0)):
        training[-1]["supervision"]["action"] = [tail_target, 0.0]
        dataset = snapshot.with_name(name + ".json")
        dataset.write_text(json.dumps(document), encoding="utf-8")
        values = dict(
            options,
            dataset=str(dataset),
            dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            steps=1,
            batch_size=258,
        )
        path = tmp_path / (name + "-config.json")
        path.write_text(json.dumps(values), encoding="utf-8")
        settings[name] = path
    larger = json.loads(settings["positive"].read_bytes())
    larger["batch_size"] = 260
    settings["larger"] = tmp_path / "larger-config.json"
    settings["larger"].write_text(json.dumps(larger), encoding="utf-8")
    return settings, snapshot.with_name("positive.json")


def test_explicit_large_bc_batch_consumes_all_129_observations_and_replays_exactly(tmp_path):
    torch = pytest.importorskip("torch")
    configs, positive_dataset = full_batch_inputs(tmp_path)
    summaries = {}
    for name, config in configs.items():
        summaries[name] = run_experiment(TemporalBCTrain(config, tmp_path / name)).summary[
            "temporal_bc"
        ]
        assert summaries[name]["training"]["steps_completed"] == 1
        assert summaries[name]["training"]["train_by_view"] == {
            "no_reference": 129,
            "reference_assisted": 129,
        }

    # Only the final observation's first target coordinate changes between
    # three otherwise identical initial updates. The worked squared-error
    # identity (p-1)^2 + (p+1)^2 - 2*p^2 = 2 cancels unknown initial predictions.
    # Two paired views and two action coordinates leave 1/129 after reduction.
    # Omitting that observation gives zero; consuming only 128 gives 1/128.
    losses = {
        name: loss_values(tmp_path / name, value["training"])[0]
        for name, value in summaries.items()
    }
    assert losses["positive"] + losses["negative"] - 2 * losses["zero"] == pytest.approx(
        1 / 129, rel=0, abs=2e-7
    )
    assert losses["positive"] == losses["larger"]
    assert summaries["positive"]["decisions"] == summaries["larger"]["decisions"]
    saved = {
        name: torch.load(tmp_path / name / "actor.pt", weights_only=True)["actor"]
        for name in ("zero", "positive", "larger")
    }
    assert saved["positive"].keys() == saved["larger"].keys()
    assert all(torch.equal(value, saved["larger"][key]) for key, value in saved["positive"].items())
    assert any(
        not torch.equal(value, saved["zero"][key]) for key, value in saved["positive"].items()
    )
    replayed = run_experiment(
        TemporalBCReplay(tmp_path / "positive", positive_dataset, tmp_path / "replayed.html")
    ).summary["temporal_bc"]
    assert len(replayed["decisions"]) == 262
    assert replayed["decisions"] == summaries["positive"]["decisions"]
    assert replayed["verification"]["status"] == "verified"


@pytest.mark.parametrize("batch_size", [0, True, 259])
def test_bc_rejects_nonpositive_noninteger_and_unpaired_batches(tmp_path, batch_size):
    config, _ = temporal_fixture(tmp_path)
    settings = json.loads(config.read_bytes())
    settings["batch_size"] = batch_size
    config.write_text(json.dumps(settings), encoding="utf-8")
    output = tmp_path / "model"
    with pytest.raises(ValueError, match="[Bb]atch"):
        run_experiment(TemporalBCTrain(config, output))
    assert not output.exists()
