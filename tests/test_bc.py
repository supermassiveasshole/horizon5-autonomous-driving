"""Read historical v1 actors without retaining their retired training path."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_demonstrations import dataset_fixture

from fh5.bc import BCReplay
from fh5.bc_network import make_actor
from fh5.demonstration_dataset import DemonstrationDataset
from fh5.experiment import run_experiment

pytest.importorskip("torch")


def legacy_model(tmp_path, *, image_size=(64, 36)):
    """Create genuine v1 state_dict files; this fixture has never been trained."""
    import torch

    run_experiment(DemonstrationDataset(dataset_fixture(tmp_path), tmp_path / "data"))
    dataset = tmp_path / "data/dataset.json"
    data = json.loads(dataset.read_text())
    source = data["sources"][0]
    directory = Path(source["directory"])
    observation = json.loads((directory / "observation-config.json").read_text())
    camera = json.loads((directory / "vision-session.json").read_text())
    example = next(row for row in data["examples"] if row["bc_eligible"])
    actor_input = example["views"]["no_reference"]
    assert observation["history_offsets_ms"] == [0]
    assert len(actor_input["actions"]) == 2
    assert len(actor_input["reference"]["mask"]) == 3
    contract = {
        "observation": observation,
        "action": "xinput-lx-rt-lt-v1",
        "image_count": 1,
        # v1: nine ego fields, one image pair, two action quartets,
        # and three reference triples. No temporal feature suffix exists.
        "numeric_size": 28,
        "normalization": "fixed-v1:speed/100,velocity/100,angular/5,age/1000,waypoints/100",
        "actor_fields": list(actor_input),
        "camera": {key: camera[key] for key in ("camera_mode", "camera_pose")},
        "conditions": source["snapshot"],
        "observed_vehicle": source["observed_vehicle"],
        "image_size": list(image_size),
    }
    manifest = {
        "version": 1,
        "architecture": "rgb-history-conv4-64-state64-fusion128-tanh2-v1",
        "contract": contract,
        "config": {
            "version": 1,
            "dataset": str(dataset),
            "seed": 7,
            "steps": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "image_size": list(image_size),
            "device": "cpu",
        },
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "preprocessing": {
            "version": 1,
            "rgb": "full frame bilinear resize, float32 / 255; no crop",
            "size": list(image_size),
        },
        "future_supervision": {"enabled": False, "weight": 0},
        "training": {"steps_completed": 0, "test_fixture": "untrained v1 compatibility actor"},
        "critic_trained": False,
        "closed_loop_validated": False,
    }
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        actor = make_actor(contract)
        torch.save(
            {
                "actor": actor.state_dict(),
                "metadata": {
                    key: manifest[key]
                    for key in (
                        "version",
                        "architecture",
                        "contract",
                        "config",
                        "preprocessing",
                        "future_supervision",
                    )
                },
            },
            model_dir / "actor.pt",
        )
    manifest["weights_sha256"] = hashlib.sha256((model_dir / "actor.pt").read_bytes()).hexdigest()
    (model_dir / "model.json").write_text(json.dumps(manifest), encoding="utf-8")
    return BCReplay(model_dir, dataset, tmp_path / "replay.html")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_legacy_replay_preserves_predictions_both_views_and_original_files(tmp_path, device):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA optional")

    request = replace(legacy_model(tmp_path), device=device)
    original = {
        path: path.read_bytes() for path in (*request.model_dir.iterdir(), request.dataset_file)
    }
    result = run_experiment(request)
    bc = result.summary["bc"]
    assert bc["commands_sent"] is False
    assert bc["training"]["steps_completed"] == 0
    assert bc["training"]["test_fixture"] == "untrained v1 compatibility actor"
    assert set(bc["metrics"]["holdout"]) == {"no_reference", "reference_assisted"}
    replay = run_experiment(replace(request, report_path=tmp_path / "reloaded.html"))
    assert replay.summary["bc"]["predictions"] == bc["predictions"]
    assert all(-1 <= a <= 1 for r in bc["predictions"] if r["prediction"] for a in r["prediction"])
    assert bc["future_supervision"] == {"enabled": False, "weight": 0}
    assert result.report_path.exists()
    assert any(row["prediction"] is not None for row in bc["predictions"])
    for path, content in original.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize("change", ["future_input", "label_as_history", "split", "quality"])
def test_export_cannot_be_edited_to_leak_targets_or_promote_failures(tmp_path, change):
    request = legacy_model(tmp_path)
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
    assert not request.report_path.exists()


@pytest.mark.parametrize(
    "problem",
    [
        "missing_image",
        "action_version",
        "observation_version",
        "weights",
        "manifest",
        "empty_manifest",
    ],
)
def test_incompatible_or_broken_evidence_is_rejected_without_output(tmp_path, problem):
    request = legacy_model(tmp_path)
    if problem == "missing_image":
        next((tmp_path / "train/frames").glob("*.png")).unlink()
    elif problem == "weights":
        (request.model_dir / "actor.pt").write_bytes(b"not a model")
    elif problem == "empty_manifest":
        (request.model_dir / "model.json").write_text("{}")
    elif problem == "manifest":
        path = request.model_dir / "model.json"
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
        run_experiment(replace(request, report_path=destination))
    assert not destination.exists()


def test_future_auxiliary_targets_never_change_frozen_action_predictions(tmp_path):
    request = legacy_model(tmp_path)
    original = run_experiment(request).summary["bc"]
    config_path = tmp_path / "dataset.json"
    value = json.loads(config_path.read_text())
    value["future_offsets_ms"] = [100, 1200]
    config_path.write_text(json.dumps(value))
    run_experiment(DemonstrationDataset(config_path, tmp_path / "other-targets"))
    replay = run_experiment(
        BCReplay(
            request.model_dir,
            tmp_path / "other-targets/dataset.json",
            tmp_path / "other-targets.html",
        )
    )
    assert [r["prediction"] for r in original["predictions"]] == [
        r["prediction"] for r in replay.summary["bc"]["predictions"]
    ]


def test_model_rejects_a_new_history_contract_even_when_shapes_match(tmp_path):
    request = legacy_model(tmp_path)
    path = request.model_dir / "model.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["contract"]["observation"]["period_ms"] = 50
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="contract mismatch"):
        run_experiment(request)
    assert not request.report_path.exists()


def test_cli_finishes_with_offline_status_and_a_frozen_report(tmp_path, capsys):
    from fh5.cli import main

    request = legacy_model(tmp_path)
    assert (
        main(
            [
                "bc-replay",
                "--model",
                str(request.model_dir),
                "--dataset",
                str(request.dataset_file),
                "--report",
                str(request.report_path),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["game_validation"] == "unverified"
    assert output["commands_sent"] is False
    assert Path(output["report"]).exists()


def test_failed_but_complete_observations_keep_diagnostic_predictions(tmp_path):
    request = legacy_model(tmp_path)
    review_path = tmp_path / "train-review.json"
    review = json.loads(review_path.read_text())
    review["intervals"][0]["quality"] = "failed"
    review_path.write_text(json.dumps(review))
    run_experiment(DemonstrationDataset(tmp_path / "dataset.json", tmp_path / "failed-data"))
    result = run_experiment(
        replace(request, dataset_file=tmp_path / "failed-data/dataset.json")
    ).summary["bc"]
    failed = [row for row in result["predictions"] if row["quality"] == "failed"]
    assert failed and any(row["prediction"] is not None for row in failed)
    assert all(not row["scored"] for row in failed)


@pytest.mark.parametrize("valid", [True, False])
def test_legacy_embedded_loss_history_is_read_without_rewriting_model(tmp_path, valid):
    request = legacy_model(tmp_path)
    path = request.model_dir / "model.json"
    manifest = json.loads(path.read_text())
    manifest["training"]["losses"] = [0.5, 0.25] if valid else [False]
    path.write_text(json.dumps(manifest))
    original = path.read_bytes()
    if valid:
        training = run_experiment(request).summary["bc"]["training"]
        assert "losses" not in training
        assert training["loss_history"]["status"] == "legacy_embedded"
        assert training["loss_history"]["records"] == 2
        assert training["steps_completed"] == 0
    else:
        with pytest.raises(ValueError, match="legacy BC loss history"):
            run_experiment(request)
        assert not request.report_path.exists()
    assert path.read_bytes() == original
