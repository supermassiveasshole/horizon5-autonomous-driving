"""Merged experience grows on disk without discarding source evidence or update credit."""

import gc
import hashlib
import json
import tracemalloc
from copy import deepcopy

import pytest
from test_expansion_resources import small_parent

from fh5.experiment import run_experiment
from fh5.sac_learning import SACResume


def addition_file(path, template, count, padding=0):
    document = deepcopy(template)
    document["source_hashes"]["packets"] = hashlib.sha256(b"extra-origin").hexdigest()
    terminal = document["transitions"][-1]
    assert terminal["terminated"]
    document["transitions"] = []
    for i in range(count):
        row = deepcopy(terminal)
        row["id"] = f"added-{i}"
        if padding:
            row["diagnostic"] = row["id"] + ":" + "x" * padding
        document["transitions"].append(row)
    path.write_text(json.dumps(document), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_experience_union_does_not_accumulate_every_added_transition_in_memory(tmp_path):
    parent, replay = small_parent(tmp_path)
    parent_bytes = (parent / "policy.json").read_bytes()
    template = json.loads(replay.read_bytes())
    peaks, states = [], []
    count, padding = 128, 128 * 1024
    for name, size in (("small", 0), ("large", padding)):
        path = replay.with_name(name + "-addition.json")
        digest = addition_file(path, template, count, size)
        gc.collect()
        tracemalloc.start()
        try:
            result = run_experiment(
                SACResume(parent, tmp_path / name, steps=1, additions=((path, digest),))
            ).summary["sac_learning"]
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert result["steps_completed"] == 1
        assert result["experience_added_transitions"] == count
        assert result["sampling"]["available"] == {"demonstration": 0, "online": count + 2}
        states.append(result["learner_state_sha256"])
    assert states[0] == states[1]
    assert (parent / "policy.json").read_bytes() == parent_bytes
    assert peaks[1] - peaks[0] < count * padding // 2, (
        "Expansion retained most of the growing added-row corpus",
        peaks,
    )


def test_experience_can_grow_past_ten_thousand_transitions_and_keep_training(tmp_path):
    parent, replay = small_parent(tmp_path)
    template = json.loads(replay.read_bytes())
    path = replay.with_name("many-added.json")
    digest = addition_file(path, template, count=9999)
    output = tmp_path / "expanded"
    result = run_experiment(
        SACResume(parent, output, steps=1, additions=((path, digest),))
    ).summary["sac_learning"]
    assert result["steps_completed"] == 1 and result["experience_added_transitions"] == 9999
    assert result["sampling"]["available"] == {"demonstration": 0, "online": 10_001}
    assert result["stop_reason"] == "budget_completed"
    stored = json.loads((output / "experience/replay.json").read_bytes())
    assert len(stored["transitions"]) == 10_001 and len(stored["source_inventory"]) == 2
    for entry in stored["source_inventory"]:
        with (output / "experience" / entry["path"]).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["replay_sha256"]


@pytest.mark.parametrize("origin", ["e" * 64, None])
def test_reformatted_origin_cannot_earn_duplicate_credit_in_a_union(tmp_path, origin):
    parent, replay = small_parent(tmp_path)
    template = json.loads(replay.read_bytes())
    additions = []
    for i in range(2):
        document = deepcopy(template)
        document["source_hashes"]["packets"] = origin
        document["notes"] = [f"review-{i}"]
        path = replay.with_name(f"review-{i}.json")
        path.write_text(json.dumps(document), encoding="utf-8")
        additions.append((path, hashlib.sha256(path.read_bytes()).hexdigest()))
    assert additions[0][1] != additions[1][1]
    with pytest.raises(ValueError, match="Duplicate SAC experience"):
        run_experiment(SACResume(parent, tmp_path / "failed", steps=1, additions=tuple(additions)))
    assert not (tmp_path / "failed/policy.json").exists()
