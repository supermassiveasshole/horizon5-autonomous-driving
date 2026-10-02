"""Frozen BC reports remain accessible without retaining the prediction corpus."""

import gc
import hashlib
import json
import sqlite3
import tempfile
import tracemalloc

import pytest
from test_temporal_bc import temporal_fixture
from test_temporal_data_resources import corpus

from fh5.experiment import run_experiment
from fh5.temporal_bc import TemporalBCReplay, TemporalBCTrain


def test_completed_bc_releases_prediction_corpus_and_survives_model_publication(tmp_path):
    warm = tmp_path / "warm"
    warm.mkdir()
    warm_config, _ = temporal_fixture(warm)
    run_experiment(TemporalBCTrain(warm_config, warm / "model"))
    source = tmp_path / "source"
    source.mkdir()
    config, snapshot, _ = corpus(source, size=(64, 36))
    dataset = json.loads(snapshot.read_bytes())
    for row in dataset["decisions"]:
        for frame in row["frames"]:
            frame["time_quality"] = "synthetic" + "x" * 65536
    snapshot.write_text(json.dumps(dataset))
    options = json.loads(config.read_bytes())
    options["dataset_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    config.write_text(json.dumps(options))
    del dataset, row, frame
    gc.collect()
    model = tmp_path / ".candidate"
    tracemalloc.start()
    try:
        result = run_experiment(TemporalBCTrain(config, model))
        resident = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    assert resident < snapshot.stat().st_size, (
        "Returned BC result retains a complete prediction corpus",
        resident,
        snapshot.stat().st_size,
    )
    predictions = result.summary["temporal_bc"]["decisions"]
    assert len(predictions) == 48
    expected = json.loads((model / "report.json").read_bytes())["decisions"]
    assert predictions == expected
    published = model.rename(tmp_path / "published")
    rebuilt = run_experiment(TemporalBCReplay(published, snapshot, tmp_path / "rebuilt.html"))
    assert predictions[0] == expected[0] and predictions[-1] == expected[-1]
    assert list(predictions) == expected
    assert rebuilt.summary["temporal_bc"]["decisions"] == predictions


def test_optional_metric_storage_failure_preserves_verified_predictions(tmp_path, monkeypatch):
    config, snapshot = temporal_fixture(tmp_path)
    original_connect = sqlite3.connect

    def unavailable_metrics(database, *args, **kwargs):
        if str(database).endswith("metrics.sqlite3"):
            raise sqlite3.OperationalError("optional metric storage unavailable")
        return original_connect(database, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(sqlite3, "connect", unavailable_metrics)
        result = run_experiment(TemporalBCTrain(config, tmp_path / "model"))
    summary = result.summary["temporal_bc"]
    assert summary["metrics"]["status"] == "unavailable"
    assert "optional metric storage unavailable" in summary["metrics"]["error"]
    assert summary["training"]["steps_completed"] == 2
    assert len(summary["decisions"]) == 12
    manifest = json.loads((tmp_path / "model/model.json").read_bytes())
    assert (
        manifest["verification"]["sha256"]
        == hashlib.sha256((tmp_path / "model/report.json").read_bytes()).hexdigest()
    )
    rebuilt = run_experiment(
        TemporalBCReplay(tmp_path / "model", snapshot, tmp_path / "rebuilt.html")
    )
    assert (
        rebuilt.summary["temporal_bc"]["metrics"]["train"]["no_reference"]["nominal"]["count"] == 2
    )
    assert rebuilt.summary["temporal_bc"]["decisions"] == summary["decisions"]


@pytest.mark.parametrize("failed_file", [0, 1])
def test_prediction_storage_failure_retains_weights_and_closes_temporary_files(
    tmp_path, monkeypatch, failed_file
):
    config, _ = temporal_fixture(tmp_path)
    original_temporary_file = tempfile.TemporaryFile
    opened = []

    class ExhaustedFile:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, data):
            self.stream.write(data[:4])
            raise OSError("prediction temporary storage exhausted")

    def temporary_file(*args, **kwargs):
        stream = original_temporary_file(*args, **kwargs)
        position = len(opened)
        opened.append(stream)
        return ExhaustedFile(stream) if position == failed_file else stream

    with monkeypatch.context() as fault:
        fault.setattr(tempfile, "TemporaryFile", temporary_file)
        with pytest.raises(OSError, match="prediction temporary storage exhausted"):
            run_experiment(TemporalBCTrain(config, tmp_path / "model"))
    assert len(opened) >= 2 and all(stream.closed for stream in opened)
    assert (tmp_path / "model/actor.pt").is_file()
    manifest = json.loads((tmp_path / "model/model.json").read_bytes())
    assert "verification" not in manifest


def test_returned_prediction_storage_is_released_when_result_is_discarded(tmp_path, monkeypatch):
    config, _ = temporal_fixture(tmp_path)
    original_temporary_file = tempfile.TemporaryFile
    opened = []

    def temporary_file(*args, **kwargs):
        stream = original_temporary_file(*args, **kwargs)
        opened.append(stream)
        return stream

    with monkeypatch.context() as observed:
        observed.setattr(tempfile, "TemporaryFile", temporary_file)
        result = run_experiment(TemporalBCTrain(config, tmp_path / "model"))
    assert len(opened) >= 2 and not opened[0].closed and not opened[1].closed
    assert len(result.summary["temporal_bc"]["decisions"]) == 12
    del result
    gc.collect()
    assert all(stream.closed for stream in opened)


def test_frozen_prediction_report_is_not_rejected_by_legacy_file_size(tmp_path):
    config, snapshot = temporal_fixture(tmp_path)
    model = tmp_path / "model"
    original = run_experiment(TemporalBCTrain(config, model))
    evidence = model / "report.json"
    # JSON whitespace tests representation compatibility, not actual sample growth.
    # The size deliberately crosses the removed 128 MiB reader gate.
    with evidence.open("ab") as output:
        for _ in range(129):
            output.write(b" " * 1024**2)
    with evidence.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    manifest_path = model / "model.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["verification"]["sha256"] = digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rebuilt = run_experiment(TemporalBCReplay(model, snapshot, tmp_path / "rebuilt.html"))
    assert rebuilt.summary["temporal_bc"]["verification"]["status"] == "verified"
    assert (
        rebuilt.summary["temporal_bc"]["decisions"] == original.summary["temporal_bc"]["decisions"]
    )
