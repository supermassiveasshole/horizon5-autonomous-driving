"""Required source manifests grow without retaining every original in memory."""

import hashlib
import json
import shutil
import sqlite3
import tempfile
import tracemalloc
from copy import deepcopy
from pathlib import Path

import pytest
from test_critic_resume import warm_inputs

from fh5.experiment import run_experiment
from fh5.sac import SACCriticResume, SACCriticWarmup


def source_inventory(root, count, padding_mib=0):
    model, replay, _ = warm_inputs(root)
    template = json.loads(replay.read_bytes())
    union = deepcopy(template)
    union.update(source_inventory=[], transitions=[], source_role="mixed")
    union["source_hashes"] = {k: template["source_hashes"][k] for k in ("task", "route", "reward")}
    sources = replay.parent / "sources"
    sources.mkdir()
    for number in range(count):
        source = deepcopy(template)
        # Independent synthetic origins for this input-format fixture. This is
        # not a claim that the repeated numerical example is new game evidence.
        source["source_hashes"]["packets"] = hashlib.sha256(f"origin-{number}".encode()).hexdigest()
        path = sources / f"original-{number}.json"
        with path.open("wb") as stream:
            stream.write(json.dumps(source).encode())
            for _ in range(padding_mib):
                stream.write(b" " * 1024**2)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        union["source_inventory"].append(
            {
                "path": f"sources/{path.name}",
                "replay_sha256": digest,
                "source_hashes": source["source_hashes"],
            }
        )
        for original in source["transitions"]:
            row = deepcopy(original)
            row["provenance"] = {"replay_sha256": digest, "transition_id": row["id"]}
            row["id"] = digest + ":" + row["id"]
            union["transitions"].append(row)
    replay.write_text(json.dumps(union))
    return model, replay, hashlib.sha256(replay.read_bytes()).hexdigest()


def test_source_bytes_are_loaded_one_document_at_a_time_and_resume_exactly(tmp_path):
    model, replay, digest = source_inventory(tmp_path, count=3, padding_mib=45)
    total = sum(p.stat().st_size for p in (replay.parent / "sources").iterdir())
    assert total > 128 * 1024**2
    first = tmp_path / "first"
    tracemalloc.start()
    try:
        result = run_experiment(
            SACCriticWarmup(model, replay, digest, first, steps=2),
            sac_stop_requested=lambda step: step == 1,
        ).summary["sac"]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < total, "Reading/copying sources must not retain the whole original collection"
    assert result["steps_completed"] == 1
    for source in (replay.parent / "sources").iterdir():
        copied = first / "experience/sources" / source.name
        assert copied.stat().st_size == source.stat().st_size
        with source.open("rb") as original, copied.open("rb") as retained:
            assert (
                hashlib.file_digest(original, "sha256").digest()
                == hashlib.file_digest(retained, "sha256").digest()
            )
    resumed = run_experiment(SACCriticResume(first, tmp_path / "resumed")).summary["sac"]
    whole = run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "whole", steps=2)
    ).summary["sac"]
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]
    assert resumed["predictions"] == whole["predictions"]


def test_source_count_is_not_an_additional_training_budget(tmp_path):
    model, replay, digest = source_inventory(tmp_path, count=1001)
    output = tmp_path / "warm"
    result = run_experiment(SACCriticWarmup(model, replay, digest, output, steps=1)).summary["sac"]
    assert result["steps_completed"] == 1 and result["transitions"] == 2002
    sealed = json.loads((output / "experience/replay.json").read_bytes())
    assert len(sealed["source_inventory"]) == 1001
    for entry in sealed["source_inventory"]:
        with (output / "experience" / entry["path"]).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["replay_sha256"]


def test_single_legacy_source_has_no_independent_size_gate(tmp_path):
    model, replay, digest = source_inventory(tmp_path, count=1, padding_mib=129)
    output = tmp_path / "warm"
    result = run_experiment(SACCriticWarmup(model, replay, digest, output, steps=1)).summary["sac"]
    assert result["steps_completed"] == 1
    document = json.loads((output / "experience/replay.json").read_bytes())
    entry = document["source_inventory"][0]
    with (output / "experience" / entry["path"]).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["replay_sha256"]


@pytest.mark.parametrize("fault", ["index", "copy"])
def test_source_io_failure_preserves_parent_and_releases_private_files(
    tmp_path, monkeypatch, fault
):
    model, replay, digest = source_inventory(tmp_path, count=1)
    parent, output = tmp_path / "parent", tmp_path / "failed"
    run_experiment(
        SACCriticWarmup(model, replay, digest, parent, steps=2),
        sac_stop_requested=lambda step: step == 1,
    )
    manifest = (parent / "critic.json").read_bytes()
    original = (parent / "experience/sources/original-0.json").read_bytes()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    opened, connected = Path.open, sqlite3.connect
    observed = []

    def connect(database, *args, **kwargs):
        if fault == "index" and Path(database).parent.name.startswith("fh5-source-index-"):
            observed.append(True)
            raise sqlite3.OperationalError("database or disk is full")
        return connected(database, *args, **kwargs)

    def open_file(path, mode="r", *args, **kwargs):
        if fault == "copy" and path.is_relative_to(output) and path.parent.name == "sources":
            if mode == "xb":
                observed.append(True)
                raise OSError("disk is full while copying source")
        return opened(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "tempdir", str(temporary))
        patch.setattr(sqlite3, "connect", connect)
        patch.setattr(Path, "open", open_file)
        with pytest.raises(OSError, match="disk is full"):
            run_experiment(SACCriticResume(parent, output))
    assert observed == [True]
    assert not list(temporary.iterdir())
    assert not (output / "critic.json").exists()
    assert (parent / "critic.json").read_bytes() == manifest
    assert (parent / "experience/sources/original-0.json").read_bytes() == original
    retry = run_experiment(SACCriticResume(parent, tmp_path / "retry")).summary["sac"]
    assert retry["steps_completed"] == 1 and retry["total_steps"] == 2


def test_changed_original_during_copy_cannot_publish_a_checkpoint(tmp_path, monkeypatch):
    model, replay, digest = source_inventory(tmp_path, count=1)
    output = tmp_path / "failed"
    original = replay.parent / "sources/original-0.json"
    opened = Path.open
    changed = []

    def open_file(path, mode="r", *args, **kwargs):
        if path.is_relative_to(output) and path.parent.name == "sources" and mode == "xb":
            with opened(original, "ab") as stream:
                stream.write(b" ")
            changed.append(True)
        return opened(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_file)
        with pytest.raises(ValueError, match="changed during copy"):
            run_experiment(SACCriticWarmup(model, replay, digest, output, steps=1))
    assert changed == [True]
    assert not (output / "critic.json").exists()


def test_new_experience_can_extend_a_checkpoint_with_large_originals(tmp_path):
    from fh5.sac_learning import SACResume, SACTrain

    model, replay, digest = source_inventory(tmp_path, count=3, padding_mib=45)
    warm, parent = tmp_path / "warm", tmp_path / "parent"
    run_experiment(SACCriticWarmup(model, replay, digest, warm, steps=1))
    run_experiment(SACTrain(warm, warm / "experience/replay.json", parent, steps=0))
    parent_bytes = (parent / "policy.json").read_bytes()
    addition = tmp_path / "addition"
    addition.mkdir()
    # Extend the same synthetic task with another origin; rebuilding the route
    # fixture would introduce a different, correctly incompatible route digest.
    with (replay.parent / "sources/original-0.json").open("rb") as stream:
        leaf = json.load(stream)
    leaf["source_hashes"]["packets"] = hashlib.sha256(b"new-origin").hexdigest()
    for row in leaf["transitions"]:
        for observation in (row["current"], row["next"]):
            if observation is None:
                continue
            for frame in observation["frames"]:
                destination = addition / frame["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copyfile(replay.parent / frame["path"], destination)
    extra = addition / "replay.json"
    extra.write_text(json.dumps(leaf))
    extra_digest = hashlib.sha256(extra.read_bytes()).hexdigest()
    output = tmp_path / "expanded"
    result = run_experiment(
        SACResume(parent, output, steps=1, additions=((extra, extra_digest),))
    ).summary["sac_learning"]
    assert result["steps_completed"] == 1
    sealed = json.loads((output / "experience/replay.json").read_bytes())
    assert len(sealed["source_inventory"]) == 4
    for entry in sealed["source_inventory"]:
        with (output / "experience" / entry["path"]).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["replay_sha256"]
    assert (parent / "policy.json").read_bytes() == parent_bytes


@pytest.mark.parametrize("fault", ["duplicate_path", "identity"])
def test_invalid_source_inventory_still_rejects_training(tmp_path, fault):
    model, replay, _ = source_inventory(tmp_path, count=1)
    document = json.loads(replay.read_bytes())
    if fault == "duplicate_path":
        document["source_inventory"].append(deepcopy(document["source_inventory"][0]))
        expected = "Duplicate SAC experience source manifest"
    else:
        document["source_inventory"][0]["source_hashes"]["packets"] = "0" * 64
        expected = "SAC experience source manifest changed"
    replay.write_text(json.dumps(document))
    digest = hashlib.sha256(replay.read_bytes()).hexdigest()
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=expected):
        run_experiment(SACCriticWarmup(model, replay, digest, output, steps=1))
    assert not (output / "critic.json").exists()
