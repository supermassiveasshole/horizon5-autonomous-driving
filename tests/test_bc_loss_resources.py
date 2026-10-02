"""BC update budgets and optional loss histories at the public experiment seam."""

import gc
import hashlib
import json
import tracemalloc
from dataclasses import replace
from pathlib import Path

import pytest
from test_bc import setup_bc
from test_temporal_bc import temporal_fixture

from fh5.bc import BCReplay
from fh5.experiment import run_experiment
from fh5.temporal_bc import TemporalBCReplay, TemporalBCTrain


def loss_inputs(tmp_path, kind, *, steps=3):
    if kind == "temporal":
        config, dataset = temporal_fixture(tmp_path)
        request = TemporalBCTrain(config, tmp_path / "model")
        replay = TemporalBCReplay(request.output_dir, dataset, tmp_path / "replay.html")
        section, predictions = "temporal_bc", "decisions"
    else:
        request = setup_bc(tmp_path)
        replay = BCReplay(
            request.output_dir, tmp_path / "data/dataset.json", tmp_path / "replay.html"
        )
        section, predictions = "bc", "predictions"
    settings = json.loads(request.config_file.read_bytes())
    settings.update(steps=steps, batch_size=2)
    request.config_file.write_text(json.dumps(settings), encoding="utf-8")
    return request, replay, section, predictions


def loss_values(root, training):
    if "losses" in training:
        return training["losses"]
    history = training["loss_history"]
    assert history["status"] == "complete"
    raw = (root / history["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == history["sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == history["records"]
    assert [row["step"] for row in rows] == list(range(1, len(rows) + 1))
    return [row["loss"] for row in rows]


@pytest.mark.parametrize("kind", ["temporal", "legacy"])
def test_bc_seals_loss_history_outside_manifest_and_replays_without_it(tmp_path, kind):
    request, replay, section, predictions = loss_inputs(tmp_path, kind)
    trained = run_experiment(request).summary[section]
    stats = trained["training"]
    assert stats["steps_completed"] == 3
    assert "losses" not in stats
    assert stats["loss_history"]["format"] == "bc-loss-history-jsonl-v1"
    values = loss_values(request.output_dir, stats)
    assert len(values) == 3 and all(value >= 0 for value in values)
    manifest_path = request.output_dir / "model.json"
    original = manifest_path.read_bytes()
    assert json.loads(original)["training"] == stats
    (request.output_dir / stats["loss_history"]["path"]).unlink()
    replayed = run_experiment(replay).summary[section]
    assert replayed[predictions] == trained[predictions]
    assert replayed["training"]["steps_completed"] == 3
    assert manifest_path.read_bytes() == original


@pytest.mark.parametrize("kind", ["temporal", "legacy"])
def test_bc_replay_projects_legacy_loss_array_without_changing_frozen_bytes(tmp_path, kind):
    request, replay, section, predictions = loss_inputs(tmp_path, kind)
    trained = run_experiment(request).summary[section]
    path = request.output_dir / "model.json"
    manifest = json.loads(path.read_bytes())
    stats = manifest["training"]
    stats.pop("loss_history", None)
    stats["losses"] = [(index % 31) / 32 for index in range(513)]
    stats["archived_training_note"] = {"source": "old loss-array fixture", "retained": True}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    original = path.read_bytes()
    replayed = run_experiment(replay).summary[section]
    assert "losses" not in replayed["training"]
    history = replayed["training"]["loss_history"]
    assert history["format"] == "bc-loss-history-embedded-v1"
    assert history["status"] == "legacy_embedded"
    assert history["records"] == 513
    assert history["manifest_sha256"] == hashlib.sha256(original).hexdigest()
    assert history["field"] == ["training", "losses"]
    assert {key: value for key, value in replayed["training"].items() if key != "loss_history"} == {
        key: value for key, value in stats.items() if key != "losses"
    }
    assert replayed[predictions] == trained[predictions]
    assert path.read_bytes() == original


@pytest.mark.parametrize("kind", ["temporal", "legacy"])
def test_explicit_bc_budget_performs_10001_updates_and_replays(tmp_path, kind):
    request, replay, section, predictions = loss_inputs(tmp_path, kind, steps=10001)
    trained = run_experiment(request).summary[section]
    stats = trained["training"]
    assert stats["steps_completed"] == 10001
    values = loss_values(request.output_dir, stats)
    assert len(values) == 10001
    assert values[-1] < values[0]
    assert len(set(values)) > 100
    manifest_path = request.output_dir / "model.json"
    assert "losses" not in json.loads(manifest_path.read_bytes())["training"]
    assert (
        manifest_path.stat().st_size
        < (request.output_dir / stats["loss_history"]["path"]).stat().st_size
    )
    replayed = run_experiment(replay).summary[section]
    assert replayed[predictions] == trained[predictions]
    assert replayed["training"]["steps_completed"] == 10001


@pytest.mark.parametrize("exception", [OSError, MemoryError])
def test_legacy_html_failure_keeps_completed_learning_and_numerical_report(
    tmp_path, monkeypatch, exception
):
    request, replay, section, predictions = loss_inputs(tmp_path, "legacy")
    original = Path.read_text

    def unavailable(path, *args, **kwargs):
        if path.name == "bc_report.html":
            raise exception("optional HTML unavailable")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", unavailable)
        result = run_experiment(request)
    trained = result.summary[section]
    assert trained["training"]["steps_completed"] == 3
    assert trained["presentation"]["status"] == "unavailable"
    assert "optional HTML unavailable" in trained["presentation"]["error"]
    assert result.report_path == request.output_dir / "report.json"
    retained = json.loads(result.report_path.read_bytes())
    assert retained["training"] == trained["training"]
    assert retained[predictions] == trained[predictions]
    assert run_experiment(replay).summary[section][predictions] == trained[predictions]


@pytest.mark.parametrize("kind", ["temporal", "legacy"])
@pytest.mark.parametrize(
    "failure", ["allocate", "open", "write", "flush", "publish", "copy_integrity"]
)
@pytest.mark.parametrize("exception", [OSError, MemoryError])
def test_optional_loss_io_failure_preserves_every_completed_update(
    tmp_path, monkeypatch, kind, failure, exception
):
    import tempfile

    request, replay, section, predictions = loss_inputs(tmp_path, kind)
    baseline = run_experiment(replace(request, output_dir=tmp_path / "baseline")).summary[section]
    original_open, original_mkdtemp, original_replace = Path.open, tempfile.mkdtemp, Path.replace
    injected = []

    def fail():
        injected.append(failure)
        raise exception("optional loss " + failure + " unavailable")

    class LossStream:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.wrapped.close()

        def read(self, size=-1):
            raw = self.wrapped.read(size)
            if failure == "copy_integrity" and raw:
                injected.append(failure)
                return raw.replace(b'"loss"', b'"lost"')
            return raw

        def write(self, raw):
            if failure == "write":
                fail()
            return self.wrapped.write(raw)

        def flush(self):
            if failure == "flush":
                fail()
            return self.wrapped.flush()

    def allocate(*args, **kwargs):
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else None)
        if failure == "allocate" and prefix == "fh5-bc-losses-":
            fail()
        return original_mkdtemp(*args, **kwargs)

    def opening(path, mode="r", *args, **kwargs):
        if path.name == "losses.jsonl" and mode == "xb":
            if failure == "open":
                fail()
            return LossStream(original_open(path, mode, *args, **kwargs))
        if path.name == "losses.jsonl" and mode == "rb" and failure == "copy_integrity":
            return LossStream(original_open(path, mode, *args, **kwargs))
        return original_open(path, mode, *args, **kwargs)

    def publishing(path, target):
        if failure == "publish" and path.name == ".pending-losses.jsonl":
            fail()
        return original_replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "mkdtemp", allocate)
        patch.setattr(Path, "open", opening)
        patch.setattr(Path, "replace", publishing)
        trained = run_experiment(request).summary[section]
    assert injected
    assert trained["training"]["steps_completed"] == 3
    history = trained["training"]["loss_history"]
    assert history["status"] == "unavailable" and history["sha256"] is None
    assert (
        "changed during copy" if failure == "copy_integrity" else "optional loss " + failure
    ) in history["error"]
    assert (
        hashlib.sha256((request.output_dir / "actor.pt").read_bytes()).hexdigest()
        == hashlib.sha256((tmp_path / "baseline/actor.pt").read_bytes()).hexdigest()
    )
    assert trained[predictions] == baseline[predictions]
    assert not (request.output_dir / "diagnostics/.pending-losses.jsonl").exists()
    assert run_experiment(replay).summary[section][predictions] == baseline[predictions]


@pytest.mark.parametrize("kind", ["temporal", "legacy"])
def test_bc_replay_streams_a_growing_legacy_loss_history(tmp_path, kind):
    from test_bc_manifest_resources import add_loss_history

    request, replay, section, predictions = loss_inputs(tmp_path, kind)
    reference = run_experiment(request).summary[section]
    history_bytes = add_loss_history(request.output_dir)
    with (request.output_dir / "model.json").open("rb") as stream:
        original_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    gc.collect()
    tracemalloc.start()
    try:
        result = run_experiment(replay).summary[section]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < history_bytes, (peak, history_bytes)
    assert result["training"]["loss_history"]["records"] == 524288
    assert result["training"]["loss_history"]["manifest_sha256"] == original_hash
    assert result[predictions] == reference[predictions]
    with (request.output_dir / "model.json").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == original_hash


@pytest.mark.parametrize("kind", ["temporal", "legacy"])
@pytest.mark.parametrize("bad", ["[true]", "[null]", "[NaN]", "[1e999]", "[[1, 2]]", "{}", "[1,]"])
def test_bc_replay_rejects_invalid_legacy_loss_history(tmp_path, kind, bad):
    request, replay, _, _ = loss_inputs(tmp_path, kind, steps=1)
    run_experiment(request)
    path = request.output_dir / "model.json"
    manifest = json.loads(path.read_bytes())
    stats = manifest.pop("training")
    stats.pop("loss_history")
    path.write_text(
        json.dumps(manifest)[:-1]
        + ',"training":'
        + json.dumps(stats)[:-1]
        + ',"losses":'
        + bad
        + "}}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        run_experiment(replay)
    assert not replay.report_path.exists()


@pytest.mark.parametrize("kind", ["temporal", "legacy"])
@pytest.mark.parametrize("duplicate", ["training", "losses"])
def test_bc_replay_keeps_last_duplicate_legacy_loss_history(tmp_path, kind, duplicate):
    request, replay, section, predictions = loss_inputs(tmp_path, kind, steps=1)
    reference = run_experiment(request).summary[section]
    path = request.output_dir / "model.json"
    manifest = json.loads(path.read_bytes())
    stats = manifest.pop("training")
    stats.pop("loss_history")
    raw = json.dumps(manifest)[:-1]
    if duplicate == "training":
        raw += ',"training":{"losses":[false]}'
    raw += ',"training":' + json.dumps(stats)[:-1]
    if duplicate == "losses":
        raw += ',"losses":[false]'
    path.write_text(raw + ',"losses":[0.5,0.25]}}', encoding="utf-8")
    result = run_experiment(replay).summary[section]
    assert result["training"]["loss_history"]["records"] == 2
    assert result["training"]["steps_completed"] == 1
    assert result[predictions] == reference[predictions]


@pytest.mark.parametrize("kind", ["temporal", "legacy"])
@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_bc_step_budget_remains_a_positive_integer(tmp_path, kind, steps):
    request, _, _, _ = loss_inputs(tmp_path, kind, steps=steps)
    with pytest.raises(ValueError, match="steps"):
        run_experiment(request)
    assert not request.output_dir.exists()


@pytest.mark.parametrize("kind", ["temporal", "legacy"])
@pytest.mark.parametrize("failure", ["finalize", "cleanup"])
def test_optional_loss_finalization_failure_is_recorded_before_model_sealing(
    tmp_path, monkeypatch, kind, failure
):
    import tempfile

    request, replay, section, predictions = loss_inputs(tmp_path, kind)
    original_hash, original_cleanup = hashlib.sha256, tempfile.TemporaryDirectory.cleanup
    injected, temporary = [], []

    class Digest:
        def __init__(self, *args, **kwargs):
            self.wrapped = original_hash(*args, **kwargs)
            self.loss_records = False

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def update(self, raw):
            self.loss_records |= bytes(raw).startswith(b'{"loss":')
            self.wrapped.update(raw)

        def hexdigest(self):
            if failure == "finalize" and self.loss_records:
                injected.append(failure)
                raise MemoryError("optional loss digest unavailable")
            return self.wrapped.hexdigest()

    def cleanup(directory):
        if failure == "cleanup" and Path(directory.name).name.startswith("fh5-bc-losses-"):
            temporary.append(directory)
            injected.append(failure)
            raise OSError("optional loss cleanup unavailable")
        return original_cleanup(directory)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(hashlib, "sha256", Digest)
            patch.setattr(tempfile.TemporaryDirectory, "cleanup", cleanup)
            trained = run_experiment(request).summary[section]
    finally:
        for directory in temporary:
            original_cleanup(directory)
    assert injected
    stats = trained["training"]
    assert stats["steps_completed"] == 3
    history = stats["loss_history"]
    if failure == "finalize":
        assert history["status"] == "unavailable" and history["sha256"] is None
        assert history["records"] == 3
        assert "optional loss digest unavailable" in history["error"]
    else:
        assert history["status"] == "complete"
        assert "optional loss cleanup unavailable" in history["cleanup_error"]
        assert len(loss_values(request.output_dir, stats)) == 3
    assert json.loads((request.output_dir / "model.json").read_bytes())["training"] == stats
    assert run_experiment(replay).summary[section][predictions] == trained[predictions]
