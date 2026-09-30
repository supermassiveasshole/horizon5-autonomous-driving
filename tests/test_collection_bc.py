"""Continuous recordings become causal numerical training inputs at the experiment seam."""

import hashlib
import json
from pathlib import Path

import pytest
from test_collection_dataset import dataset_inputs

from fh5.collection_dataset import CollectionDataset
from fh5.experiment import run_experiment


def prepare_inputs(tmp_path):
    config = dataset_inputs(tmp_path, vary_action=True, size=(64, 36))
    run_experiment(CollectionDataset(config, tmp_path / "selection"))
    selection = tmp_path / "selection/dataset.json"
    prepare = tmp_path / "prepare.json"
    prepare.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": str(selection),
                "dataset_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
                "action_history_offsets_ms": [100, 50, 0],
                "max_action_age_ms": 100,
                "waypoint_distances_m": [5, 10, 20],
            }
        )
    )
    return prepare, selection


def test_collection_export_preserves_pixels_past_inputs_and_separate_holdout(tmp_path):
    from fh5.collection_bc import CollectionBCPrepare

    config, selection = prepare_inputs(tmp_path)
    output = tmp_path / "numeric"
    result = run_experiment(CollectionBCPrepare(config, output))
    train = json.loads((output / "dataset.json").read_bytes())
    final = json.loads((output / "evaluation.json").read_bytes())
    assert {g["split"] for g in train["groups"]} == {"train", "development"}
    assert {g["split"] for g in final["groups"]} == {"evaluation"}
    assert {r["decision_id"] for r in train["decisions"]}.isdisjoint(
        r["decision_id"] for r in final["decisions"]
    )
    row = next(r for r in train["decisions"] if r["source_sequence"] == 5)
    actor = row["views"]["no_reference"]
    assert actor["actions"][-1][0] == -1  # strictly previous input, not next poll's label
    assert all(a > 0 for a in actor["action_age_ms"] if a is not None)
    assert row["supervision"]["action"][0] > 0.5
    assert row["supervision"]["label_poll_ns"] > row["decision_ns"]
    assert actor["image_age_ms"] == [200, 100, 0]
    assert row["views"]["reference_assisted"] == actor  # missing optional reference
    assert not any(actor["reference"]["mask"])
    for frame in row["frames"]:
        assert (output / frame["path"]).read_bytes() == bytes([51, 17, 34] * (64 * 36))
    assert (
        train["provenance"]["selection_sha256"]
        == hashlib.sha256(selection.read_bytes()).hexdigest()
    )
    assert result.summary["collection_bc"]["diagnostic_only"] is True
    assert result.summary["collection_bc"]["evaluation"]["metrics"] == "withheld"
    assert not result.summary["collection_bc"]["commands_sent"]


def test_exported_collection_trains_and_reloads_without_final_holdout_feedback(tmp_path):
    pytest.importorskip("torch")
    from fh5.collection_bc import CollectionBCPrepare
    from fh5.temporal_bc import TemporalBCReplay, TemporalBCTrain

    config, _ = prepare_inputs(tmp_path)
    output = tmp_path / "numeric"
    run_experiment(CollectionBCPrepare(config, output))
    dataset = output / "dataset.json"
    train_config = tmp_path / "train.json"
    train_config.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": str(dataset),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "seed": 9,
                "steps": 2,
                "batch_size": 4,
                "learning_rate": 0.001,
                "device": "cpu",
                "time_mode": "actual",
            }
        )
    )
    trained = run_experiment(TemporalBCTrain(train_config, tmp_path / "model"))
    summary = trained.summary["temporal_bc"]
    assert summary["training"]["steps_completed"] == 2
    assert {r["split"] for r in summary["decisions"]} == {"train", "development"}
    assert all(r["timing"]["adjacent_delta_s"] == [0.1, 0.1] for r in summary["decisions"])
    assert summary["model"]["provenance"]["diagnostic_only"]
    reloaded = run_experiment(
        TemporalBCReplay(tmp_path / "model", dataset, tmp_path / "replay.html")
    )
    assert reloaded.summary["temporal_bc"]["decisions"] == summary["decisions"]
    assert reloaded.summary["temporal_bc"]["verification"]["max_abs_error"] <= 1e-6


def test_optional_independent_reference_only_changes_paired_reference_branch(tmp_path):
    from test_routes import recording

    from fh5.collection_bc import CollectionBCPrepare
    from fh5.routes import BuildRoute

    config, _ = prepare_inputs(tmp_path)
    reference_source = recording(tmp_path, [(x, 0) for x in range(0, 101, 5)])
    run_experiment(BuildRoute(reference_source, tmp_path / "route", 0, 20))
    options = json.loads(config.read_bytes())
    options["reference"] = {
        "route_file": str(tmp_path / "route/route.json"),
        "independence_evidence": ["Separate synthetic recording; different session and timestamps"],
    }
    config.write_text(json.dumps(options))
    output = tmp_path / "numeric"
    run_experiment(CollectionBCPrepare(config, output))
    data = json.loads((output / "dataset.json").read_bytes())
    assert (output / "reference/route.json").read_bytes() == (
        tmp_path / "route/route.json"
    ).read_bytes()
    assert data["provenance"]["reference"]["status"] == "loaded"
    assert any(
        any(r["views"]["reference_assisted"]["reference"]["mask"]) for r in data["decisions"]
    )
    for row in data["decisions"]:
        plain, assisted = row["views"]["no_reference"], row["views"]["reference_assisted"]
        assert not any(plain["reference"]["mask"])
        assert {k: v for k, v in plain.items() if k != "reference"} == {
            k: v for k, v in assisted.items() if k != "reference"
        }


@pytest.mark.parametrize("fault", ["selection", "pixels"])
def test_collection_export_rejects_changed_input_before_publishing(tmp_path, fault):
    from fh5.collection_bc import CollectionBCPrepare

    config, selection = prepare_inputs(tmp_path)
    if fault == "selection":
        selection.write_bytes(selection.read_bytes() + b" ")
    else:
        source = Path(json.loads(selection.read_bytes())["sources"][0]["recording"])
        next((source / "blocks").rglob("*.rgb")).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        run_experiment(CollectionBCPrepare(config, tmp_path / "numeric"))
    assert not (tmp_path / "numeric").exists()
