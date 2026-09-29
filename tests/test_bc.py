"""Offline learned behavior through the agreed experiment-run interface."""

import json
from pathlib import Path

import pytest
from test_demonstrations import dataset_fixture

from fh5.demonstration_dataset import DemonstrationDataset
from fh5.experiment import run_experiment

pytest.importorskip("torch")


def setup_bc(tmp_path):
    from fh5.bc import BCTrain

    run_experiment(DemonstrationDataset(dataset_fixture(tmp_path), tmp_path / "data"))
    config = tmp_path / "bc.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": str(tmp_path / "data/dataset.json"),
                "seed": 7,
                "steps": 3,
                "batch_size": 4,
                "learning_rate": 0.001,
                "image_size": [64, 36],
                "device": "cpu",
            }
        )
    )
    return BCTrain(config, tmp_path / "model")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_offline_training_roundtrip_includes_both_views_and_real_visual_computation(
    tmp_path, device
):
    import torch

    from fh5.bc import BCReplay

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA optional")

    request = setup_bc(tmp_path)
    config = json.loads(request.config_file.read_text())
    config["device"] = device
    request.config_file.write_text(json.dumps(config))
    result = run_experiment(request)
    bc = result.summary["bc"]
    assert bc["commands_sent"] is False
    assert bc["training"]["steps_completed"] == 3
    assert bc["training"]["visual_gradient_l1"] > 0
    assert bc["training"]["visual_parameter_change_l1"] > 0
    assert bc["training"]["reload_max_abs_error"] == 0
    assert set(bc["metrics"]["holdout"]) == {"no_reference", "reference_assisted"}
    assert bc["training"]["train_examples_by_view"]["no_reference"] > 0
    replay = run_experiment(
        BCReplay(
            request.output_dir, tmp_path / "data/dataset.json", tmp_path / "reloaded.html", device
        )
    )
    assert replay.summary["bc"]["predictions"] == bc["predictions"]
    assert all(-1 <= a <= 1 for r in bc["predictions"] if r["prediction"] for a in r["prediction"])
    assert bc["future_supervision"] == {"enabled": False, "weight": 0}
    assert result.report_path.exists()


@pytest.mark.parametrize("change", ["future_input", "label_as_history", "split", "quality"])
def test_export_cannot_be_edited_to_leak_targets_or_promote_failures(tmp_path, change):
    request = setup_bc(tmp_path)
    path = tmp_path / "data/dataset.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    example = next(e for e in data["examples"] if e["bc_eligible"])
    if change == "future_input":
        example["views"]["no_reference"]["future"] = [123, 456]
    elif change == "label_as_history":
        example["views"]["no_reference"]["actions"][-1] = example["supervision"]["action"]
    elif change == "split":
        example["split"] = "holdout"
    else:
        example["supervision"]["quality"] = "failed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="canonical|source"):
        run_experiment(request)
    assert not request.output_dir.exists()


@pytest.mark.parametrize(
    "problem", ["missing_image", "action_version", "observation_version", "weights", "manifest"]
)
def test_incompatible_or_broken_evidence_is_rejected_without_output(tmp_path, problem):
    from fh5.bc import BCReplay

    request = setup_bc(tmp_path)
    run_experiment(request)
    if problem == "missing_image":
        next((tmp_path / "train/frames").glob("*.png")).unlink()
    elif problem == "weights":
        (request.output_dir / "actor.pt").write_bytes(b"not a model")
    elif problem == "manifest":
        path = request.output_dir / "model.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["preprocessing"]["size"] = [640, 360]
        path.write_text(json.dumps(data))
    else:
        path = tmp_path / "data/dataset.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if problem == "action_version":
            data["sources"][0]["profile"]["mapping"] = "other"
        else:
            path = tmp_path / "train/observation-config.json"
            data = json.loads(path.read_text())
            data["version"] = 99
        path.write_text(json.dumps(data))
    destination = tmp_path / "invalid.html"
    with pytest.raises((ValueError, FileNotFoundError)):
        run_experiment(BCReplay(request.output_dir, tmp_path / "data/dataset.json", destination))
    assert not destination.exists()


def test_future_auxiliary_targets_never_change_frozen_action_predictions(tmp_path):
    from fh5.bc import BCReplay

    request = setup_bc(tmp_path)
    original = run_experiment(request).summary["bc"]
    config_path = tmp_path / "dataset.json"
    value = json.loads(config_path.read_text())
    value["future_offsets_ms"] = [100, 1200]
    config_path.write_text(json.dumps(value))
    run_experiment(DemonstrationDataset(config_path, tmp_path / "other-targets"))
    replay = run_experiment(
        BCReplay(
            request.output_dir,
            tmp_path / "other-targets/dataset.json",
            tmp_path / "other-targets.html",
        )
    )
    assert [r["prediction"] for r in original["predictions"]] == [
        r["prediction"] for r in replay.summary["bc"]["predictions"]
    ]


def test_model_rejects_a_new_history_contract_even_when_shapes_match(tmp_path):
    from fh5.bc import BCReplay

    request = setup_bc(tmp_path)
    run_experiment(request)
    path = request.output_dir / "model.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["contract"]["observation"]["period_ms"] = 50
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="contract mismatch"):
        run_experiment(
            BCReplay(
                request.output_dir, tmp_path / "data/dataset.json", tmp_path / "bad-contract.html"
            )
        )


def test_cli_finishes_with_offline_status_and_a_frozen_report(tmp_path, capsys):
    from fh5.cli import main

    request = setup_bc(tmp_path)
    assert (
        main(
            ["bc-train", "--config", str(request.config_file), "--output", str(request.output_dir)]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["game_validation"] == "unverified"
    assert output["commands_sent"] is False
    assert Path(output["report"]).exists()
