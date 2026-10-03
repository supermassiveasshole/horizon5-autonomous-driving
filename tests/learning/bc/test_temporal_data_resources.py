"""Growing numerical BC inputs at the public experiment seam."""

import gc
import hashlib
import json
import tracemalloc
from copy import deepcopy
from pathlib import Path

import pytest

from fh5.experiment import run_experiment
from fh5.learning.bc.training import TemporalBCReplay, TemporalBCTrain
from tests.learning.bc.test_temporal_bc import temporal_fixture


def corpus(tmp_path, *, per_group=8, size=(512, 288)):
    config, snapshot = temporal_fixture(tmp_path)
    document = json.loads(snapshot.read_bytes())
    templates = document["decisions"][::2]
    document["decisions"] = []
    document["pixel_contract"]["size"] = list(size)
    total_bytes = 0
    for group, template in enumerate(templates):
        for index in range(per_group):
            row = deepcopy(template)
            row["decision_id"] = f"g{group}-d{index}"
            delta = index * 1_000_000_000
            row["decision_ns"] += delta
            for slot, frame in enumerate(row["frames"]):
                name = f"g{group}-d{index}-f{slot}.rgb"
                pixels = bytes((30 + group, 10 + index % 245, 40 + slot)) * (size[0] * size[1])
                (snapshot.parent / name).write_bytes(pixels)
                total_bytes += len(pixels)
                frame.update(
                    frame_id=name,
                    path=name,
                    size=list(size),
                    sha256=hashlib.sha256(pixels).hexdigest(),
                    source_layout={"size": list(size), "format": "RGB"},
                )
                for field in ("source_time_ns", "capture_received_ns", "preprocess_ready_ns"):
                    frame[field] += delta
            document["decisions"].append(row)
    snapshot.write_text(json.dumps(document), encoding="utf-8")
    options = json.loads(config.read_bytes())
    options["dataset_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    config.write_text(json.dumps(options), encoding="utf-8")
    return config, snapshot, total_bytes


def test_bc_releases_full_pixel_corpus_before_publishing_the_trained_actor(tmp_path, monkeypatch):
    # Warm the actual optional tensor runtime before measuring corpus residency.
    initial = tmp_path / "initial"
    initial.mkdir()
    warm_config, _ = temporal_fixture(initial)
    run_experiment(TemporalBCTrain(warm_config, initial / "model"))
    source = tmp_path / "corpus"
    source.mkdir()
    config, snapshot, source_bytes = corpus(source)
    output = tmp_path / "model"
    observed = []
    mkdir = Path.mkdir

    def observe_publication(path, *args, **kwargs):
        if path == output:
            observed.append(tracemalloc.get_traced_memory()[0])
        return mkdir(path, *args, **kwargs)

    gc.collect()
    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", observe_publication)
        tracemalloc.start()
        try:
            trained = run_experiment(TemporalBCTrain(config, output)).summary["temporal_bc"]
        finally:
            tracemalloc.stop()
    assert trained["training"]["steps_completed"] == 2
    assert observed and observed[0] < source_bytes, (
        "Completed BC retains the entire source pixel corpus",
        observed,
        source_bytes,
    )
    replay = run_experiment(TemporalBCReplay(output, snapshot, tmp_path / "rebuilt.html"))
    assert replay.summary["temporal_bc"]["decisions"] == trained["decisions"]
    assert len(trained["decisions"]) == 48


@pytest.mark.parametrize("fault", ["frame_identity", "bad_heldout_pixels", "duplicate_decision"])
def test_unsampled_tail_is_validated_before_any_candidate_is_published(tmp_path, fault):
    config, snapshot = temporal_fixture(tmp_path)
    document = json.loads(snapshot.read_bytes())
    last = document["decisions"][-1]
    last["bc_eligible"] = False
    if fault == "frame_identity":
        last["frames"][0]["frame_id"] = document["decisions"][-2]["frames"][0]["frame_id"]
    elif fault == "bad_heldout_pixels":
        last["frames"][0]["sha256"] = "0" * 64
    else:
        last["decision_id"] = document["decisions"][0]["decision_id"]
    snapshot.write_text(json.dumps(document))
    options = json.loads(config.read_bytes())
    options["dataset_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    config.write_text(json.dumps(options))
    output = tmp_path / "model"
    with pytest.raises(ValueError):
        run_experiment(TemporalBCTrain(config, output))
    assert not output.exists()


def test_pixels_changed_after_preflight_cannot_produce_verified_model(tmp_path, monkeypatch):
    config, snapshot, _ = corpus(tmp_path, per_group=2, size=(64, 36))
    source = snapshot.parent / "g2-d1-f2.rgb"
    original_open = Path.open
    reads = 0

    def replace_pixels_on_later_read(path, mode="r", *args, **kwargs):
        nonlocal reads
        if path == source and mode == "rb":
            reads += 1
            if reads == 2:
                with original_open(path, "wb") as stream:
                    stream.write(bytes(64 * 36 * 3))
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", replace_pixels_on_later_read)
        with pytest.raises(ValueError, match="pixel hash"):
            run_experiment(TemporalBCTrain(config, tmp_path / "model"))
    assert reads >= 2
    manifest = json.loads((tmp_path / "model/model.json").read_bytes())
    assert "verification" not in manifest


def test_bc_trains_and_replays_more_than_the_old_pixel_budget(tmp_path):
    # Actual unique numerical images, not whitespace padding or a raised quota.
    config, snapshot, source_bytes = corpus(tmp_path, per_group=49, size=(640, 640))
    assert source_bytes > 512 * 1024**2
    output = tmp_path / "model"
    trained = run_experiment(TemporalBCTrain(config, output)).summary["temporal_bc"]
    assert trained["training"]["steps_completed"] == 2
    assert len(trained["decisions"]) == 294
    rebuilt = run_experiment(TemporalBCReplay(output, snapshot, tmp_path / "rebuilt.html"))
    assert rebuilt.summary["temporal_bc"]["decisions"] == trained["decisions"]


def test_bc_accepts_large_dataset_representation_without_changing_learning(tmp_path):
    # Separate representation compatibility from the actual corpus growth case.
    config, snapshot = temporal_fixture(tmp_path)
    expected = run_experiment(TemporalBCTrain(config, tmp_path / "reference")).summary[
        "temporal_bc"
    ]
    with snapshot.open("ab") as stream:
        for _ in range(129):
            stream.write(b" " * 1024**2)
    assert snapshot.stat().st_size > 128 * 1024**2
    options = json.loads(config.read_bytes())
    with snapshot.open("rb") as stream:
        options["dataset_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    config.write_text(json.dumps(options))
    trained = run_experiment(TemporalBCTrain(config, tmp_path / "model")).summary["temporal_bc"]
    assert (
        trained["training"]["loss_history"]["sha256"]
        == expected["training"]["loss_history"]["sha256"]
    )
    assert trained["decisions"] == expected["decisions"]
    rebuilt = run_experiment(
        TemporalBCReplay(tmp_path / "model", snapshot, tmp_path / "replay.html")
    )
    assert rebuilt.summary["temporal_bc"]["decisions"] == trained["decisions"]
