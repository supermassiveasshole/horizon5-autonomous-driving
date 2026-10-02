"""Training configuration representation sizes do not replace schema validation."""

import hashlib
import json
import tracemalloc

import pytest

from fh5.collection_bc import CollectionBCPrepare
from fh5.experiment import run_experiment
from fh5.learning_schedule import ScheduledBCTrain


def write_config(path, value, *, padding_mib=0, extra=b""):
    with path.open("wb") as stream:
        stream.write(json.dumps(value).encode()[:-1] + extra + b"}")
        for _ in range(padding_mib * 128):
            stream.write(b" " * 8192)
    return path


def preparation_config(tmp_path, **kwargs):
    return write_config(
        tmp_path / "prepare.json",
        {
            "version": 1,
            "dataset": "missing-selection.json",
            "dataset_sha256": "0" * 64,
            "action_history_offsets_ms": [100, 50, 0],
            "max_action_age_ms": 100,
            "waypoint_distances_m": [5, 10, 20],
        },
        **kwargs,
    )


def schedule_config(tmp_path, **kwargs):
    return write_config(
        tmp_path / "schedule.json",
        {
            "version": 1,
            "training_config": "train.json",
            "training_config_sha256": "0" * 64,
            "collector_bundle": "collector",
            "collector_manifest_sha256": "0" * 64,
            "budget": {
                "cpu_threads": 1,
                "poll_interval_s": 0.1,
                "max_wait_s": 1,
                "max_total_s": 120,
                "max_unit_s": 10,
                "max_private_bytes": 4 * 1024**3,
                "min_free_disk_bytes": 1024**3,
                "max_status_age_ms": 3000,
                "max_image_age_ms": 250,
                "max_pending_bytes": 16 * 1024**2,
                "max_gpu_memory_mib": 20000,
                "max_gpu_utilization_percent": 80,
            },
        },
        **kwargs,
    )


def training_config(tmp_path, **kwargs):
    return write_config(
        tmp_path / "train.json",
        {
            "version": 1,
            "dataset": "snapshot/dataset.json",
            "dataset_sha256": "0" * 64,
            "seed": 7,
            "steps": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "device": "cpu",
            "time_mode": "actual",
        },
        **kwargs,
    )


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind_training(config, training):
    options = json.loads(config.read_bytes())
    options["training_config_sha256"] = file_hash(training)
    write_config(config, options)


class InterruptedResources:
    source_kind = "synthetic"
    closed = False

    def now_ns(self):
        return 0

    def sample(self):
        raise KeyboardInterrupt

    def wait(self, seconds):
        raise AssertionError("Admission was interrupted before any waiting")

    def close(self):
        self.closed = True


def configuration_request(kind, tmp_path, **kwargs):
    output = tmp_path / "output"
    if kind == "preparation":
        config = preparation_config(tmp_path, **kwargs)
        return config, CollectionBCPrepare(config, output)
    if kind == "schedule":
        config = schedule_config(tmp_path, **kwargs)
    else:
        training = training_config(tmp_path, **kwargs)
        config = schedule_config(tmp_path)
        bind_training(config, training)
        return training, ScheduledBCTrain(config, output)
    return config, ScheduledBCTrain(config, output)


def test_large_preparation_config_reaches_dataset_validation_without_retention(tmp_path):
    config = preparation_config(tmp_path, padding_mib=2)
    output = tmp_path / "prepared"
    tracemalloc.start()
    try:
        with pytest.raises(FileNotFoundError) as missing:
            run_experiment(CollectionBCPrepare(config, output))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert missing.value.filename == str(tmp_path / "missing-selection.json")
    assert peak < config.stat().st_size
    assert not output.exists()


def test_large_schedule_config_reaches_training_validation_without_retention(tmp_path):
    config = schedule_config(tmp_path, padding_mib=2)
    output = tmp_path / "scheduled"
    tracemalloc.start()
    try:
        with pytest.raises(FileNotFoundError) as missing:
            run_experiment(ScheduledBCTrain(config, output))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert missing.value.filename == str(tmp_path / "train.json")
    assert peak < config.stat().st_size
    assert not output.exists()


def test_large_training_config_is_frozen_exactly_before_interrupted_admission(tmp_path):
    training = training_config(tmp_path, padding_mib=2)
    config = schedule_config(tmp_path)
    bind_training(config, training)
    expected_hash = file_hash(training)
    output = tmp_path / "scheduled"
    resources = InterruptedResources()
    tracemalloc.start()
    try:
        summary = run_experiment(
            ScheduledBCTrain(config, output), learning_resources=resources
        ).summary["learning_schedule"]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert summary["state"] == "stopped"
    assert summary["stop_reason"] == "interrupted"
    assert summary["steps_completed"] == 0
    assert file_hash(output / "requested-training.json") == expected_hash
    assert (output / "requested-training.json").stat().st_size == training.stat().st_size
    assert peak < training.stat().st_size
    assert resources.closed
    assert not (output / "candidate").exists()


@pytest.mark.parametrize("kind", ["preparation", "schedule", "training"])
@pytest.mark.parametrize("extra", [b',"unexpected":true', b',"unexpected":[1,]'])
def test_training_configuration_rejects_unknown_or_malformed_fields(tmp_path, kind, extra):
    _, request = configuration_request(kind, tmp_path, padding_mib=2, extra=extra)
    with pytest.raises(ValueError) as invalid:
        run_experiment(request, learning_resources=InterruptedResources())
    assert "bounded limit" not in str(invalid.value)
    assert not request.output_dir.exists()


@pytest.mark.parametrize("kind", ["preparation", "schedule", "training"])
def test_training_configuration_keeps_last_duplicate_field_value(tmp_path, kind):
    _, request = configuration_request(kind, tmp_path, extra=b',"version":0,"version":1')
    if kind == "training":
        result = run_experiment(request, learning_resources=InterruptedResources())
        assert result.summary["learning_schedule"]["stop_reason"] == "interrupted"
    else:
        with pytest.raises(FileNotFoundError) as missing:
            run_experiment(request)
        filename = "missing-selection.json" if kind == "preparation" else "train.json"
        assert missing.value.filename == str(tmp_path / filename)


@pytest.mark.parametrize("kind", ["preparation", "schedule", "training"])
def test_training_configuration_requires_every_schema_field(tmp_path, kind):
    config, request = configuration_request(kind, tmp_path)
    settings = json.loads(config.read_bytes())
    del settings["version"]
    write_config(config, settings)
    if kind == "training":
        bind_training(request.config_file, config)
    with pytest.raises(ValueError, match="Unsupported"):
        run_experiment(request, learning_resources=InterruptedResources())
    assert not request.output_dir.exists()


@pytest.mark.parametrize("kind", ["preparation", "schedule", "training"])
def test_training_configuration_rejects_changed_private_bytes(
    tmp_path, corrupt_private_reads, kind
):
    config, request = configuration_request(kind, tmp_path)
    original = config.read_bytes()
    with corrupt_private_reads(config, b'"version": 1', b'"version": 9') as changed:
        with pytest.raises(ValueError, match="changed"):
            run_experiment(request, learning_resources=InterruptedResources())
    assert changed
    assert config.read_bytes() == original
    assert not request.output_dir.exists()


def test_training_binding_includes_trailing_representation_bytes(tmp_path):
    training, request = configuration_request("training", tmp_path)
    with training.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="Frozen training configuration changed"):
        run_experiment(request, learning_resources=InterruptedResources())
    assert not request.output_dir.exists()


def test_changed_training_source_during_freeze_releases_resources_and_records_failure(tmp_path):
    training, request = configuration_request("training", tmp_path)

    class ChangedSourceResources(InterruptedResources):
        def now_ns(self):
            with training.open("ab") as stream:
                stream.write(b" ")
            return 0

    resources = ChangedSourceResources()
    with pytest.raises(ValueError, match="changed during copy"):
        run_experiment(request, learning_resources=resources)
    assert resources.closed
    summary = json.loads((request.output_dir / "schedule.json").read_bytes())
    assert summary["state"] == "stopped"
    assert "changed during copy" in summary["stop_reason"]
    assert summary["steps_completed"] == 0
    assert summary["candidate"] is None


def test_large_preparation_configuration_exports_the_same_causal_dataset(tmp_path):
    from test_collection_bc import prepare_inputs

    config = prepare_inputs(tmp_path)
    original = json.loads(config.read_bytes())
    reference = run_experiment(CollectionBCPrepare(config, tmp_path / "reference")).summary[
        "collection_bc"
    ]
    write_config(config, original, padding_mib=2)
    result = run_experiment(CollectionBCPrepare(config, tmp_path / "prepared")).summary[
        "collection_bc"
    ]
    assert result["snapshot_sha256"] == reference["snapshot_sha256"]
    assert result["unique_frames"] == reference["unique_frames"]
    assert result["unique_frames"] > 0
    for name in ("dataset.json", "evaluation.json"):
        assert (tmp_path / "prepared" / name).read_bytes() == (
            tmp_path / "reference" / name
        ).read_bytes()


def test_temporal_configuration_beyond_128_mib_trains_and_replays_exactly(tmp_path):
    pytest.importorskip("torch")
    from test_temporal_bc import temporal_fixture

    from fh5.temporal_bc import TemporalBCReplay, TemporalBCTrain

    config, dataset = temporal_fixture(tmp_path)
    original = json.loads(config.read_bytes())
    write_config(config, original, padding_mib=129)
    assert config.stat().st_size > 128 * 1024**2
    result = run_experiment(TemporalBCTrain(config, tmp_path / "model")).summary["temporal_bc"]
    manifest = json.loads((tmp_path / "model/model.json").read_bytes())
    assert manifest["config"] == original
    assert result["training"]["steps_completed"] == 2
    assert result["training"]["time_gradient_l1"] > 0
    assert result["training"]["reload_max_abs_error"] <= 1e-6
    replay = run_experiment(
        TemporalBCReplay(tmp_path / "model", dataset, tmp_path / "replay.html")
    ).summary["temporal_bc"]
    assert replay["decisions"] == result["decisions"]
    assert replay["verification"]["max_abs_error"] <= 1e-6
