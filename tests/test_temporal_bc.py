"""Time-aware numerical BC through the public experiment entry point."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from test_numeric_images import actor_state

from fh5.experiment import run_experiment
from fh5.numeric_images import PixelContract

pytest.importorskip("torch")


def temporal_fixture(tmp_path):
    directory = tmp_path / "snapshot"
    directory.mkdir()
    pixels = bytes([51, 17, 34] * (64 * 36))
    (directory / "frame.rgb").write_bytes(pixels)
    contract = PixelContract(size=(64, 36))
    groups, rows = [], []
    for number, split in enumerate(("train", "development", "evaluation")):
        group = f"attempt-{number}"
        groups.append({"id": group, "split": split, "evidence_id": f"synthetic-{number}"})
        for sample in range(2):
            tick = 2**60 + sample * 1_000_000_000
            state = actor_state()
            state["image_age_ms"] = [210, 130, 10]
            assisted = deepcopy(state)
            assisted["reference"] = {"waypoints_m": [[0, 10]], "mask": [True]}
            frames = []
            for slot, age in enumerate((210_000_000, 130_000_000, 10_000_000)):
                frames.append(
                    {
                        "epoch": group,
                        "frame_id": f"{group}:{sample}:{slot}",
                        "source_time_ns": tick - age,
                        "capture_received_ns": tick - age + 1_000_000,
                        "preprocess_ready_ns": tick - age + 2_000_000,
                        "time_quality": "synthetic",
                        "uncertainty_ns": 0,
                        "size": [64, 36],
                        "source_layout": {"size": [64, 36], "format": "RGB"},
                        "availability_kind": "numeric_ready",
                        "preprocess_version": contract.resize,
                        "path": "frame.rgb",
                        "sha256": hashlib.sha256(pixels).hexdigest(),
                    }
                )
            rows.append(
                {
                    "decision_id": f"{group}:{sample}",
                    "group": group,
                    "epoch": group,
                    "decision_ns": tick,
                    "frames": frames,
                    "views": {"no_reference": state, "reference_assisted": assisted},
                    "bc_eligible": True,
                    "supervision": {
                        "action": [0.1, 0.3],
                        "quality": "trusted",
                        "action_mask": True,
                        "reasons": [],
                        "label_delay_ms": 5,
                    },
                }
            )
    snapshot = directory / "dataset.json"
    snapshot.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "numeric-bc-snapshot-v1",
                "pixel_contract": contract.metadata(),
                "action_contract": "xinput-lx-rt-lt-v1",
                "groups": groups,
                "decisions": rows,
                "provenance": {"kind": "synthetic", "conditions": "test only"},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": str(snapshot),
                "dataset_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "seed": 7,
                "steps": 2,
                "batch_size": 4,
                "learning_rate": 0.001,
                "device": "cpu",
                "time_mode": "actual",
            }
        ),
        encoding="utf-8",
    )
    return config, snapshot


def test_numeric_training_and_reload_preserve_actual_time_inputs(tmp_path):
    from fh5.temporal_bc import TemporalBCReplay, TemporalBCTrain

    config, snapshot = temporal_fixture(tmp_path)
    result = run_experiment(TemporalBCTrain(config, tmp_path / "model"))
    summary = result.summary["temporal_bc"]
    assert summary["dataset_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    first = summary["decisions"][0]
    assert first["features"][-4:] == pytest.approx([0.08, 1, 0.12, 1])
    assert first["timing"]["image_age_s"] == pytest.approx([0.21, 0.13, 0.01])
    assert summary["training"]["steps_completed"] == 2
    assert summary["training"]["time_gradient_l1"] > 0
    assert summary["commands_sent"] is False
    assert set(summary["metrics"]) == {"train", "development", "evaluation"}
    replay = run_experiment(
        TemporalBCReplay(tmp_path / "model", snapshot, tmp_path / "replay.html")
    )
    assert replay.summary["temporal_bc"]["decisions"] == summary["decisions"]
    assert summary["training"]["reload_max_abs_error"] <= 1e-6


def test_relative_dataset_without_manual_hash_trains_the_same_frozen_model(tmp_path):
    import torch

    from fh5.temporal_bc import TemporalBCReplay, TemporalBCTrain

    config, snapshot = temporal_fixture(tmp_path)
    options = json.loads(config.read_text())
    expected_hash = options.pop("dataset_sha256")
    options["dataset"] = "snapshot/dataset.json"
    config.write_text(json.dumps(options))
    explicit_config = tmp_path / "explicit-train.json"
    explicit_config.write_text(json.dumps({**options, "dataset_sha256": expected_hash}))
    originals = {
        path: path.read_bytes()
        for path in (config, explicit_config, snapshot, snapshot.parent / "frame.rgb")
    }

    inferred_model, explicit_model = tmp_path / "inferred-model", tmp_path / "explicit-model"
    inferred = run_experiment(TemporalBCTrain(config, inferred_model)).summary["temporal_bc"]
    explicit = run_experiment(TemporalBCTrain(explicit_config, explicit_model)).summary[
        "temporal_bc"
    ]
    manifest = json.loads((inferred_model / "model.json").read_text())
    assert manifest["config"]["dataset_sha256"] == expected_hash
    assert inferred["dataset_sha256"] == expected_hash
    inferred_weights = torch.load(
        inferred_model / "actor.pt", map_location="cpu", weights_only=True
    )["actor"]
    explicit_weights = torch.load(
        explicit_model / "actor.pt", map_location="cpu", weights_only=True
    )["actor"]
    assert inferred_weights.keys() == explicit_weights.keys()
    assert all(
        torch.equal(inferred_weights[name], explicit_weights[name]) for name in inferred_weights
    )
    assert inferred["decisions"] == explicit["decisions"]
    replay = run_experiment(
        TemporalBCReplay(inferred_model, snapshot, tmp_path / "inferred-replay.html")
    ).summary["temporal_bc"]
    assert replay["decisions"] == inferred["decisions"]
    assert all(path.read_bytes() == content for path, content in originals.items())


@pytest.fixture
def inference_threads(request):
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("inference_threads", [1, 2], indirect=True)
@pytest.mark.parametrize("mode", ["actual", "fixed"])
def test_temporal_model_uses_same_features_in_live_numeric_seam_and_exact_replay(
    tmp_path, mode, inference_threads
):
    from fh5.numeric_actor import FrozenNumericActor
    from fh5.numeric_images import NumericDecision, NumericInfer, NumericReplay
    from fh5.numeric_recording import read_numeric_frame
    from fh5.temporal_bc import TemporalBCTrain

    config, snapshot = temporal_fixture(tmp_path)
    options = json.loads(config.read_text())
    options["time_mode"] = mode
    config.write_text(json.dumps(options))
    trained = run_experiment(TemporalBCTrain(config, tmp_path / "model"))
    entry = json.loads(snapshot.read_text())["decisions"][0]
    frames = tuple(read_numeric_frame(snapshot.parent, f) for f in entry["frames"])
    original = NumericDecision(
        "original", entry["epoch"], entry["decision_ns"], frames, entry["views"]["no_reference"]
    )
    middle = replace(
        frames[1],
        source_time_ns=frames[1].source_time_ns + 40_000_000,
        capture_received_ns=frames[1].capture_received_ns + 40_000_000,
        preprocess_ready_ns=frames[1].preprocess_ready_ns + 40_000_000,
    )
    changed = replace(
        original,
        decision_id="changed",
        frames=(frames[0], middle, frames[2]),
        actor=dict(original.actor, image_age_ms=[210, 90, 10]),
    )
    pixels = PixelContract(size=(64, 36))
    model = FrozenNumericActor(tmp_path / "model", pixels)
    result = run_experiment(
        NumericInfer(tmp_path / "inference", pixels),
        numeric_actor=model,
        numeric_inputs=[original, changed],
    )
    first, second = result.summary["numeric"]["decisions"]
    # Training uses two CPU threads; inference restores the caller's setting.
    # Compare across kernels at the declared reload tolerance, not bit equality.
    assert first["prediction"] == pytest.approx(
        trained.summary["temporal_bc"]["decisions"][0]["prediction"], abs=1e-6, rel=0
    )
    if mode == "actual":
        assert second["features"][-4:] == pytest.approx([0.12, 1, 0.08, 1])
        assert max(abs(a - b) for a, b in zip(first["prediction"], second["prediction"])) > 1e-7
    else:
        assert second["features"] == first["features"]
        assert second["prediction"] == first["prediction"]
        assert second["features"][-4:] == pytest.approx([0.1, 1, 0.1, 1])
        assert [second["features"][i] for i in (10, 12, 14)] == pytest.approx([0.2, 0.1, 0])
    replay = run_experiment(
        NumericReplay(tmp_path / "inference", tmp_path / "exact.html"),
        numeric_actor=FrozenNumericActor(tmp_path / "model", pixels),
    )
    assert replay.summary["numeric"]["replay_errors"] == []
    assert all(
        r["features_match"] and r["prediction_max_abs_error"] == 0
        for r in replay.summary["numeric"]["decisions"]
    )


def test_frozen_model_cannot_silently_relabel_its_capture_distribution(tmp_path):
    from fh5.numeric_actor import FrozenNumericActor
    from fh5.temporal_bc import TemporalBCTrain

    config, _ = temporal_fixture(tmp_path)
    run_experiment(TemporalBCTrain(config, tmp_path / "model"))
    manifest = tmp_path / "model/model.json"
    data = json.loads(manifest.read_text())
    data["numeric_contract"]["origin"] = "legacy_offline"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="metadata mismatch"):
        FrozenNumericActor(
            tmp_path / "model", PixelContract(size=(64, 36), origin="legacy_offline")
        )


@pytest.mark.parametrize(
    "fault",
    [
        "future_action",
        "action_shape",
        "future_field",
        "shared_group",
        "cross_epoch",
        "changed_hash",
        "broken_pixels",
    ],
)
def test_invalid_snapshot_never_produces_a_candidate(tmp_path, fault):
    from fh5.temporal_bc import TemporalBCTrain

    config, snapshot = temporal_fixture(tmp_path)
    data = json.loads(snapshot.read_text())
    if fault == "future_action":
        for state in data["decisions"][0]["views"].values():
            state["actions"][-1] = [0.1, 0.3]
            state["action_mask"][-1] = True
            state["action_age_ms"][-1] = -1
    elif fault == "action_shape":
        for state in data["decisions"][0]["views"].values():
            state["actions"][-1] = [0.1, 0.3, 0.4]
            state["action_mask"][-1] = True
            state["action_age_ms"][-1] = 10
    elif fault == "future_field":
        for state in data["decisions"][0]["views"].values():
            state["target_action"] = [0.1, 0.3]
    elif fault == "shared_group":
        data["groups"][1]["evidence_id"] = data["groups"][0]["evidence_id"]
    elif fault == "cross_epoch":
        data["decisions"][0]["frames"][0]["epoch"] = "previous-rewind"
    elif fault == "broken_pixels":
        (snapshot.parent / "frame.rgb").write_bytes(b"bad")
    data["provenance"]["revision"] = 2
    snapshot.write_text(json.dumps(data))
    if fault != "changed_hash":
        options = json.loads(config.read_text())
        options["dataset_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        config.write_text(json.dumps(options))
    with pytest.raises(ValueError):
        run_experiment(TemporalBCTrain(config, tmp_path / "model"))
    assert not (tmp_path / "model").exists()


def legacy_preparation(tmp_path):
    from test_demonstrations import dataset_fixture

    from fh5.demonstration_dataset import DemonstrationDataset
    from fh5.temporal_import import TemporalBCPrepare

    run_experiment(
        DemonstrationDataset(
            dataset_fixture(tmp_path, history_offsets_ms=[400, 200, 0], max_image_age_ms=800),
            tmp_path / "legacy",
        )
    )
    config = tmp_path / "prepare.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": str(tmp_path / "legacy/dataset.json"),
                "image_size": [64, 36],
                "history_patterns_ms": [[400, 200, 0], [600, 200, 0]],
                "max_selection_error_ms": 210,
                "max_image_age_ms": 1000,
            }
        )
    )
    return TemporalBCPrepare(config, tmp_path / "prepared")


@pytest.mark.parametrize("filename", ["vision.jsonl", "demonstration-session.json"])
def test_temporal_preparation_rejects_changed_bound_source_evidence(tmp_path, filename):
    request = legacy_preparation(tmp_path)
    with (tmp_path / "train" / filename).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError):
        run_experiment(request)
    assert not request.output_dir.exists()


def test_legacy_preparation_reselects_only_available_frames_and_trains_without_codecs(
    tmp_path, monkeypatch
):
    from PIL import Image

    from fh5.temporal_bc import TemporalBCTrain

    result = run_experiment(legacy_preparation(tmp_path))
    summary = result.summary["temporal_import"]
    assert summary["decoded_unique_frames"] > 0
    assert summary["jitter_variants"] > 0
    snapshot = tmp_path / "prepared/dataset.json"
    data = json.loads(snapshot.read_text())
    assert {g["split"] for g in data["groups"]} == {"train", "development"}
    assert data["provenance"]["final_evaluation_available"] is False
    for row in data["decisions"]:
        assert all(f["preprocess_ready_ns"] <= row["decision_ns"] for f in row["frames"])
        assert all(f["epoch"] == row["epoch"] for f in row["frames"])
        assert {f["time_quality"] for f in row["frames"]} == {"capture_start_proxy"}
    options = tmp_path / "numeric-train.json"
    options.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": str(snapshot),
                "dataset_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "seed": 3,
                "steps": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "device": "cpu",
                "time_mode": "actual",
            }
        )
    )

    def no_codec(*args, **kwargs):
        pytest.fail("Training reopened compressed images")

    monkeypatch.setattr(Image, "open", no_codec)
    trained = run_experiment(TemporalBCTrain(options, tmp_path / "model"))
    assert trained.summary["temporal_bc"]["training"]["steps_completed"] == 1


def test_temporal_cli_exports_a_loadable_model_and_reports_without_game_input(tmp_path, capsys):
    from fh5.cli import main

    config, snapshot = temporal_fixture(tmp_path)
    assert (
        main(["temporal-train", "--config", str(config), "--output", str(tmp_path / "model")]) == 0
    )
    trained = json.loads(capsys.readouterr().out)
    assert trained["commands_sent"] is False
    assert (
        main(
            [
                "temporal-replay",
                "--model",
                str(tmp_path / "model"),
                "--dataset",
                str(snapshot),
                "--report",
                str(tmp_path / "replayed.html"),
            ]
        )
        == 0
    )
    replayed = json.loads(capsys.readouterr().out)
    assert replayed["decisions"] == 12


def test_temporal_replay_detects_prediction_drift_from_frozen_evidence(tmp_path):
    from fh5.temporal_bc import TemporalBCReplay, TemporalBCTrain

    config, snapshot = temporal_fixture(tmp_path)
    run_experiment(TemporalBCTrain(config, tmp_path / "model"))
    evidence = tmp_path / "model/report.json"
    data = json.loads(evidence.read_text())
    data["decisions"][0]["prediction"][0] += 0.01
    evidence.write_text(json.dumps(data))
    manifest = tmp_path / "model/model.json"
    value = json.loads(manifest.read_text())
    value["verification"] = {
        "path": "report.json",
        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "tolerance": 1e-6,
    }
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="prediction drift"):
        run_experiment(TemporalBCReplay(tmp_path / "model", snapshot, tmp_path / "drift.html"))
