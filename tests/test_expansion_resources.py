"""Experience additions grow through the public learner continuation boundary."""

import hashlib
import json
import shutil
import sqlite3
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest
from test_critic_pixel_residency import critic_corpus
from test_critic_resume import warm_inputs

from fh5.experiment import run_experiment
from fh5.sac import SACCriticWarmup
from fh5.sac_learning import SACResume, SACTrain


def small_parent(root):
    model, replay, digest = warm_inputs(root)
    warm, parent = root / "warm", root / "parent"
    run_experiment(SACCriticWarmup(model, replay, digest, warm, steps=1))
    run_experiment(SACTrain(warm, warm / "experience/replay.json", parent, steps=0))
    return parent, replay


def extra_source(replay, root, number):
    shutil.copytree(replay.parent, root)
    extra = root / "replay.json"
    document = json.loads(extra.read_bytes())
    document["source_hashes"]["packets"] = hashlib.sha256(f"origin-{number}".encode()).hexdigest()
    extra.write_text(json.dumps(document))
    return extra, hashlib.sha256(extra.read_bytes()).hexdigest()


def test_expansion_accepts_more_than_the_old_unique_pixel_total_and_remains_portable(tmp_path):
    model, replay, digest, _ = critic_corpus(tmp_path, count=1)
    warm, parent = tmp_path / "warm", tmp_path / "parent"
    run_experiment(SACCriticWarmup(model, replay, digest, warm, steps=1))
    run_experiment(SACTrain(warm, warm / "experience/replay.json", parent, steps=0))
    original_parent = (parent / "policy.json").read_bytes()
    addition = tmp_path / "addition"
    addition.mkdir()
    document = json.loads(replay.read_bytes())
    template = document["transitions"][0]
    document["source_hashes"]["packets"] = hashlib.sha256(b"extra-origin").hexdigest()
    document["transitions"] = []
    frame_bytes = 512 * 288 * 3
    count = (512 * 1024**2) // (frame_bytes * 3) + 1
    base_pixels = bytes([51, 17, 34]) * (512 * 288)
    for number in range(count):
        row = deepcopy(template)
        row["id"] = f"extra-{number}"
        row["current"]["decision_id"] = f"extra-{number}"
        for slot, frame in enumerate(row["current"]["frames"]):
            # Real distinct RGB payloads cross the unique-content total. Synthetic
            # numerical compatibility evidence, not hundreds of driving episodes.
            pixels = (number * 3 + slot).to_bytes(3, "little") + base_pixels[3:]
            name = f"extra-{number}-{slot}.rgb"
            (addition / name).write_bytes(pixels)
            frame.update(path=name, sha256=hashlib.sha256(pixels).hexdigest())
        document["transitions"].append(row)
    extra = addition / "replay.json"
    extra.write_text(json.dumps(document))
    extra_digest = hashlib.sha256(extra.read_bytes()).hexdigest()
    output = tmp_path / "expanded"
    result = run_experiment(
        SACResume(parent, output, steps=1, additions=((extra, extra_digest),))
    ).summary["sac_learning"]
    assert result["steps_completed"] == 1
    expansion = result["experience_expansion"]
    assert expansion["frame_bytes"] == (count * 3 + 1) * frame_bytes
    assert expansion["frame_bytes"] > 512 * 1024**2
    assert expansion["copy_peak_frame_bytes"] == frame_bytes
    sealed = json.loads((output / "experience/replay.json").read_bytes())
    paths = {}
    for row in sealed["transitions"]:
        for frame in row["current"]["frames"]:
            paths[frame["path"]] = frame["sha256"]
    assert len(paths) == count * 3 + 1
    for name, expected in paths.items():
        with (output / "experience" / name).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
    assert (parent / "policy.json").read_bytes() == original_parent
    addition.rename(tmp_path / "original-moved-away")
    moved = tmp_path / "portable"
    output.rename(moved)
    resumed = run_experiment(SACResume(moved, tmp_path / "continued", steps=1)).summary[
        "sac_learning"
    ]
    assert resumed["steps_completed"] == 1 and resumed["total_steps"] == 2


def test_number_of_explicit_additions_is_not_an_extra_learning_budget(tmp_path):
    parent, replay = small_parent(tmp_path)
    additions = tuple(extra_source(replay, tmp_path / f"extra-{i}", i) for i in range(11))
    original = (parent / "policy.json").read_bytes()
    output = tmp_path / "expanded"
    result = run_experiment(SACResume(parent, output, steps=1, additions=additions)).summary[
        "sac_learning"
    ]
    assert result["steps_completed"] == 1
    assert len(json.loads((output / "experience/replay.json").read_bytes())["transitions"]) == 24
    assert (
        len(json.loads((output / "experience/replay.json").read_bytes())["source_inventory"]) == 12
    )
    assert (parent / "policy.json").read_bytes() == original


@pytest.mark.parametrize("failure", ["index", "union_index", "copy"])
def test_expansion_io_failure_preserves_parent_and_cleans_disk_index(
    tmp_path, monkeypatch, failure
):
    parent, replay = small_parent(tmp_path)
    extra = extra_source(replay, tmp_path / "addition", 0)
    parent_manifest = (parent / "policy.json").read_bytes()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    opened, connected = Path.open, sqlite3.connect
    observed = []

    def connect(database, *args, **kwargs):
        prefix = "fh5-experience-union-" if failure == "union_index" else "fh5-expansion-assets-"
        if failure != "copy" and Path(database).parent.name.startswith(prefix):
            observed.append(True)
            raise sqlite3.OperationalError("database or disk is full")
        return connected(database, *args, **kwargs)

    def open_file(path, mode="r", *args, **kwargs):
        if (
            failure == "copy"
            and path.parent.name == "frames"
            and path.parent.parent.name == "expanded"
            and mode == "xb"
        ):
            observed.append(True)
            raise OSError("disk is full while copying expanded frame")
        return opened(path, mode, *args, **kwargs)

    output = tmp_path / "failed"
    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "tempdir", str(temporary))
        patch.setattr(sqlite3, "connect", connect)
        patch.setattr(Path, "open", open_file)
        with pytest.raises(OSError, match="disk is full"):
            run_experiment(SACResume(parent, output, steps=1, additions=(extra,)))
    assert observed == [True]
    assert not list(temporary.iterdir())
    assert not output.exists()
    assert (parent / "policy.json").read_bytes() == parent_manifest
    retry = run_experiment(
        SACResume(parent, tmp_path / "retry", steps=1, additions=(extra,))
    ).summary["sac_learning"]
    assert retry["steps_completed"] == 1


def test_distinct_bundles_can_use_the_same_original_basename(tmp_path):
    parent, replay = small_parent(tmp_path)
    additions, original_hashes = [], []
    for number in range(2):
        extra, digest = extra_source(replay, tmp_path / f"extra-{number}", number)
        original = extra.parent / "sources/shared.json"
        original.parent.mkdir()
        shutil.copyfile(extra, original)
        original_hashes.append(digest)
        document = json.loads(extra.read_bytes())
        document["source_inventory"] = [
            {
                "path": "sources/shared.json",
                "replay_sha256": digest,
                "source_hashes": document["source_hashes"],
            }
        ]
        document["source_hashes"] = {
            k: document["source_hashes"][k] for k in ("task", "route", "reward")
        }
        document["source_role"] = "mixed"
        for row in document["transitions"]:
            row["provenance"] = {"replay_sha256": digest, "transition_id": row["id"]}
            row["id"] = digest + ":" + row["id"]
        extra.write_text(json.dumps(document))
        additions.append((extra, hashlib.sha256(extra.read_bytes()).hexdigest()))
    output = tmp_path / "expanded"
    result = run_experiment(SACResume(parent, output, steps=1, additions=tuple(additions))).summary[
        "sac_learning"
    ]
    assert result["steps_completed"] == 1
    sealed = json.loads((output / "experience/replay.json").read_bytes())
    assert len({entry["path"] for entry in sealed["source_inventory"]}) == 3
    assert set(original_hashes) <= {entry["replay_sha256"] for entry in sealed["source_inventory"]}
    for entry in sealed["source_inventory"]:
        with (output / "experience" / entry["path"]).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["replay_sha256"]
