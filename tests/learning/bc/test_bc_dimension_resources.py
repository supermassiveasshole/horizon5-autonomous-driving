"""Actual RGB shapes survive preparation, learning and frozen public replay."""

import hashlib
import json
from dataclasses import replace

import pytest

from fh5.collection.bc import CollectionBCPrepare
from fh5.experiment import run_experiment
from fh5.learning.bc.training import TemporalBCReplay, TemporalBCTrain
from fh5.observation.numeric import (
    NumericDecision,
    NumericFrame,
    NumericInfer,
    NumericReplay,
    PixelContract,
)
from tests.collection.test_collection_dataset import dataset_inputs
from tests.learning.bc.test_bc import legacy_model
from tests.learning.bc.test_temporal_bc import temporal_fixture
from tests.observation.test_numeric_images import actor_state

DIMENSIONS = ((641, 32), (4097, 1), (1, 1))


def numerical_inputs(tmp_path, size):
    config, dataset = temporal_fixture(tmp_path)
    document = json.loads(dataset.read_bytes())
    width, height = size
    pixels = bytes(
        (x * 3 + y * 7 + channel * 61 + 17) % 256
        for y in range(height)
        for x in range(width)
        for channel in range(3)
    )
    (dataset.parent / "frame.rgb").write_bytes(pixels)
    document["pixel_contract"]["size"] = list(size)
    for row in document["decisions"]:
        for frame in row["frames"]:
            frame.update(
                size=list(size),
                source_layout={"size": list(size), "format": "RGB"},
                sha256=hashlib.sha256(pixels).hexdigest(),
            )
    bind_dataset(config, dataset, document)
    return config, dataset, pixels


def bind_dataset(config, dataset, document):
    dataset.write_text(json.dumps(document), encoding="utf-8")
    settings = json.loads(config.read_bytes())
    settings.update(
        steps=1, batch_size=2, dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest()
    )
    config.write_text(json.dumps(settings), encoding="utf-8")


def verify_temporal_roundtrip(config, dataset, output, size, expected_ids):
    original_dataset = dataset.read_bytes()
    trained = run_experiment(TemporalBCTrain(config, output)).summary["temporal_bc"]
    assert trained["training"]["steps_completed"] == 1
    assert trained["training"]["time_gradient_l1"] > 0
    assert trained["commands_sent"] is False
    assert trained["contract"]["size"] == list(size)
    assert trained["model"]["model_contract"]["image_size"] == list(size)
    assert trained["dataset_sha256"] == hashlib.sha256(original_dataset).hexdigest()
    assert [row["decision_id"] for row in trained["decisions"]] == expected_ids
    assert all(
        frame["size"] == list(size) for row in trained["decisions"] for frame in row["frames"]
    )
    assert all(row["status"] == "predicted" for row in trained["decisions"])
    original_weights = (output / "actor.pt").read_bytes()
    manifest = json.loads((output / "model.json").read_bytes())
    assert manifest["weights_sha256"] == hashlib.sha256(original_weights).hexdigest()
    assert manifest["numeric_contract"]["size"] == list(size)
    replayed = run_experiment(
        TemporalBCReplay(output, dataset, output.with_suffix(".replay.html"))
    ).summary["temporal_bc"]
    assert replayed["verification"]["status"] == "verified"
    assert replayed["verification"]["compared_decisions"] == len(expected_ids)
    assert replayed["verification"]["max_abs_error"] == 0
    assert replayed["decisions"] == trained["decisions"]
    assert (output / "actor.pt").read_bytes() == original_weights
    assert dataset.read_bytes() == original_dataset


@pytest.mark.parametrize("size", DIMENSIONS)
def test_temporal_bc_learns_and_replays_real_rgb_beyond_old_dimension_bounds(tmp_path, size):
    config, dataset, pixels = numerical_inputs(tmp_path, size)
    document = json.loads(dataset.read_bytes())
    assert len(pixels) == size[0] * size[1] * 3
    expected_ids = [
        f"attempt-{group}:{sample}:{view}"
        for group in range(3)
        for sample in range(2)
        for view in ("no_reference", "reference_assisted")
    ]
    for row in document["decisions"]:
        for frame in row["frames"]:
            assert (dataset.parent / frame["path"]).read_bytes() == pixels
            assert frame["sha256"] == hashlib.sha256(pixels).hexdigest()
    verify_temporal_roundtrip(config, dataset, tmp_path / "model", size, expected_ids)


@pytest.mark.parametrize("size", DIMENSIONS)
def test_legacy_bc_replays_the_saved_resize_shape_without_training(tmp_path, size):
    request = legacy_model(tmp_path, image_size=size)
    original_weights = (request.model_dir / "actor.pt").read_bytes()
    original_manifest = (request.model_dir / "model.json").read_bytes()
    baseline = run_experiment(request).summary["bc"]
    assert baseline["training"]["steps_completed"] == 0
    assert baseline["contract"]["image_size"] == list(size)
    assert baseline["preprocessing"]["size"] == list(size)
    assert baseline["weights_sha256"] == hashlib.sha256(original_weights).hexdigest()
    replayed = run_experiment(replace(request, report_path=tmp_path / "replayed.html")).summary[
        "bc"
    ]
    assert replayed["predictions"] == baseline["predictions"]
    assert replayed["training"]["steps_completed"] == 0
    assert (request.model_dir / "actor.pt").read_bytes() == original_weights
    assert (request.model_dir / "model.json").read_bytes() == original_manifest


@pytest.mark.parametrize("size", ((641, 32), (1, 1)))
def test_collection_prepares_original_rgb_shape_then_learns_and_replays(tmp_path, size):
    prepare = dataset_inputs(tmp_path, vary_action=True, size=size)
    output = tmp_path / "numeric"
    prepared = run_experiment(CollectionBCPrepare(prepare, output)).summary["collection_bc"]
    assert prepared["commands_sent"] is False
    dataset = output / "dataset.json"
    document = json.loads(dataset.read_bytes())
    final = json.loads((output / "evaluation.json").read_bytes())
    assert document["pixel_contract"]["size"] == list(size)
    assert {group["split"] for group in document["groups"]} == {"train", "development"}
    assert {group["split"] for group in final["groups"]} == {"evaluation"}
    assert len(document["decisions"]) == 20
    assert len(final["decisions"]) == 10
    original_pixels = bytes((51, 17, 34)) * (size[0] * size[1])
    expected_sha = hashlib.sha256(original_pixels).hexdigest()
    for partition in (document, final):
        for row in partition["decisions"]:
            for frame in row["frames"]:
                assert frame["size"] == list(size)
                assert frame["sha256"] == expected_sha
                assert (output / frame["path"]).read_bytes() == original_pixels
    assert (
        prepared["snapshot_sha256"]["dataset.json"]
        == hashlib.sha256(dataset.read_bytes()).hexdigest()
    )
    train_config = tmp_path / "train.json"
    train_config.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": str(dataset),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "seed": 9,
                "steps": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "device": "cpu",
                "time_mode": "actual",
            }
        ),
        encoding="utf-8",
    )
    expected_ids = [
        row["decision_id"] + ":" + view
        for row in document["decisions"]
        for view in ("no_reference", "reference_assisted")
    ]
    verify_temporal_roundtrip(train_config, dataset, tmp_path / "model", size, expected_ids)


@pytest.mark.parametrize("size", ([0, 32], [-1, 32], [True, 32], [32], [32, 1.5]))
def test_bc_rejects_nonpositive_noninteger_or_incomplete_dimensions(tmp_path, size):
    config, dataset = temporal_fixture(tmp_path)
    document = json.loads(dataset.read_bytes())
    document["pixel_contract"]["size"] = size
    bind_dataset(config, dataset, document)
    request = TemporalBCTrain(config, tmp_path / "model")
    with pytest.raises(ValueError, match="dimensions|image size"):
        run_experiment(request)
    assert not request.output_dir.exists()


@pytest.mark.parametrize(
    "fault,message",
    (("length", "byte length"), ("shape", "pixel_contract_mismatch"), ("digest", "pixel hash")),
)
def test_larger_shapes_preserve_rgb_length_shape_and_digest_checks(tmp_path, fault, message):
    config, dataset, pixels = numerical_inputs(tmp_path, (641, 32))
    document = json.loads(dataset.read_bytes())
    if fault == "shape":
        for row in document["decisions"]:
            for frame in row["frames"]:
                frame["size"] = [32, 641]
                frame["source_layout"]["size"] = [32, 641]
    else:
        broken = pixels[:-1] if fault == "length" else bytes(len(pixels))
        (dataset.parent / "frame.rgb").write_bytes(broken)
        if fault == "length":
            for row in document["decisions"]:
                for frame in row["frames"]:
                    frame["sha256"] = hashlib.sha256(broken).hexdigest()
    bind_dataset(config, dataset, document)
    with pytest.raises(ValueError, match=message):
        run_experiment(TemporalBCTrain(config, tmp_path / "model"))
    assert not (tmp_path / "model").exists()


class PixelHashProbe:
    """Diagnostic public actor for exact pixel I/O; not a replacement BC learner."""

    kind = "synthetic_pixel_hash_probe"
    manifest = {"weights_sha256": "synthetic", "diagnostic_only": True}

    def __init__(self, size, digest):
        self.size, self.digest = size, digest
        self.calls = 0

    def predict(self, actor, frames):
        (frame,) = frames
        assert frame.pixels.shape == (self.size[1], self.size[0], 3)
        assert hashlib.sha256(frame.pixels).hexdigest() == self.digest
        self.calls += 1
        return [frame.pixels[0, 0, 0] / 255, frame.pixels[-1, -1, 1] / 255]


def pixel_decision(size):
    pixels = bytes((51, 17, 34)) * (size[0] * size[1])
    digest = hashlib.sha256(pixels).hexdigest()
    frame = NumericFrame(
        epoch="pixel-source",
        frame_id="frame-0",
        source_time_ns=1_000_000_000,
        capture_received_ns=1_001_000_000,
        preprocess_ready_ns=1_002_000_000,
        time_quality="synthetic",
        uncertainty_ns=0,
        size=size,
        pixels=memoryview(pixels),
        source_layout={"size": list(size), "format": "RGB"},
    )
    state = dict(actor_state(), images=[None], image_mask=[True], image_age_ms=[10])
    return NumericDecision("pixel-decision", "pixel-source", 1_010_000_000, (frame,), state), digest


def test_numeric_replay_reads_actual_rgb_beyond_old_product_cap(tmp_path):
    size = (4097, 4096)
    decision, digest = pixel_decision(size)
    frame_bytes = decision.frames[0].pixels.nbytes
    assert frame_bytes > 4096 * 4096 * 3
    actor = PixelHashProbe(size, digest)
    recorded = run_experiment(
        NumericInfer(
            tmp_path / "recording",
            PixelContract(size=size, history_offsets_ms=(0,)),
            max_decisions=1,
            archive_capacity=1,
            archive_bytes=frame_bytes,
        ),
        numeric_inputs=[decision],
        numeric_actor=actor,
    ).summary["numeric"]
    assert actor.calls == 1
    assert recorded["archive"]["resources_released"] is True
    (row,) = recorded["decisions"]
    assert row["exact_replay_available"] is True
    replay_actor = PixelHashProbe(size, digest)
    replay = run_experiment(
        NumericReplay(tmp_path / "recording", tmp_path / "replay.html"),
        numeric_actor=replay_actor,
    ).summary["numeric"]
    assert replay_actor.calls == 1
    assert replay["replay_errors"] == []
    (restored,) = replay["decisions"]
    assert restored["pixels_match"] and restored["features_match"]
    assert restored["prediction_max_abs_error"] == 0
    assert restored["replayed_prediction"] == row["prediction"]


def test_numeric_source_preserves_explicit_caller_byte_budget(tmp_path):
    from fh5.learning.bc.legacy_import import PreparedNumericSource

    decision, digest = pixel_decision((1, 1))
    (frame,) = decision.frames
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "frame.rgb").write_bytes(bytes(frame.pixels))
    contract = PixelContract(size=(1, 1), history_offsets_ms=(0,))
    (prepared / "prepared.json").write_text(
        json.dumps(
            {
                "version": 1,
                "contract": contract.metadata(),
                "decisions": [
                    {
                        "decision_id": decision.decision_id,
                        "epoch": decision.epoch,
                        "decision_ns": decision.decision_ns,
                        "actor": decision.actor,
                        "supervision": None,
                        "frames": [dict(frame.metadata(), path="frame.rgb", sha256=digest)],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    probe = PixelHashProbe((1, 1), digest)
    result = run_experiment(
        NumericInfer(tmp_path / "inference", contract),
        numeric_inputs=PreparedNumericSource(prepared, max_cache_bytes=2),
        numeric_actor=probe,
    ).summary["numeric"]
    assert result["stop_reason"] == "execution_error"
    assert "source byte budget" in result["execution_error"]
    assert result["decisions"] == []
    assert result["source_released"] is True
    assert probe.calls == 0
