"""Growing sealed provenance is verified without a resident original-row corpus."""

import gc
import hashlib
import json
import sqlite3
import tempfile
import tracemalloc
from copy import deepcopy
from pathlib import Path

import pytest
from test_critic_resume import warm_inputs

from fh5.experiment import run_experiment
from fh5.sac import SACCriticWarmup


def source_pair(path, template, count, diagnostic_bytes=0):
    source = deepcopy(template)
    terminal = source["transitions"][-1]
    assert terminal["terminated"]
    source["transitions"] = []
    for i in range(count):
        row = deepcopy(terminal)
        row["id"] = f"terminal-{i}"
        if diagnostic_bytes:
            row["diagnostic"] = row["id"] + ":" + "x" * diagnostic_bytes
        source["transitions"].append(row)
    original = path.with_name(path.stem + "-source.json")
    original.write_text(json.dumps(source), encoding="utf-8")
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    union = deepcopy(source)
    union["source_role"] = "mixed"
    union["source_hashes"] = {k: source["source_hashes"][k] for k in ("task", "route", "reward")}
    union["source_inventory"] = [
        {"path": original.name, "replay_sha256": digest, "source_hashes": source["source_hashes"]}
    ]
    for row in union["transitions"]:
        row["provenance"] = {"replay_sha256": digest, "transition_id": row["id"]}
        row["id"] = digest + ":" + row["id"]
    path.write_text(json.dumps(union), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_provenance_verification_does_not_retain_all_original_transition_records(tmp_path):
    model, replay, _ = warm_inputs(tmp_path)
    template = json.loads(replay.read_bytes())
    peaks, states = [], []
    count, diagnostic_bytes = 128, 64 * 1024
    for name, padding in (("small", 0), ("large", diagnostic_bytes)):
        path = replay.with_name(name + ".json")
        digest = source_pair(path, template, count, padding)
        gc.collect()
        tracemalloc.start()
        try:
            result = run_experiment(
                SACCriticWarmup(model, path, digest, tmp_path / name, steps=1)
            ).summary["sac"]
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert result["steps_completed"] == 1 and result["transitions"] == count
        states.append(result["learner_state_sha256"])
    assert states[0] == states[1], "Diagnostic text cannot change the learned numerical state"
    assert peaks[1] - peaks[0] < count * diagnostic_bytes // 2, (
        "Provenance validation retained most of the additional original-row corpus",
        peaks,
    )


def test_prepared_source_and_union_can_exceed_the_old_transition_gate(tmp_path):
    model, replay, _ = warm_inputs(tmp_path)
    template = json.loads(replay.read_bytes())
    path = replay.with_name("many.json")
    digest = source_pair(path, template, count=10_001)
    output = tmp_path / "warm"
    result = run_experiment(
        SACCriticWarmup(model, path, digest, output, steps=1, batch_size=2)
    ).summary["sac"]
    assert result["steps_completed"] == 1 and result["transitions"] == 10_001
    assert result["actor_change_max"] == 0
    assert hashlib.sha256((output / "experience/replay.json").read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("fault", ["duplicate_id", "numeric_duplicate", "reused_source", "changed"])
def test_indexed_provenance_still_rejects_duplicate_or_changed_experience(tmp_path, fault):
    model, replay, _ = warm_inputs(tmp_path)
    template = json.loads(replay.read_bytes())
    path = replay.with_name("union.json")
    source_pair(path, template, count=2)
    union = json.loads(path.read_bytes())
    first, second = union["transitions"]
    if fault == "duplicate_id":
        second["id"] = first["id"]
        expected = "Duplicate SAC transition"
    elif fault == "numeric_duplicate":
        first["id"], second["id"] = 0.0, False
        expected = "Duplicate SAC transition"
    elif fault == "reused_source":
        second["provenance"] = deepcopy(first["provenance"])
        expected = "unique original source"
    else:
        second["reward"] += 1
        expected = "differs from its sealed original"
    path.write_text(json.dumps(union), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match=expected):
        run_experiment(SACCriticWarmup(model, path, digest, output, steps=1))
    assert not (output / "critic.json").exists()


def test_provenance_disk_failure_releases_private_indexes_and_allows_retry(tmp_path, monkeypatch):
    model, replay, _ = warm_inputs(tmp_path)
    template = json.loads(replay.read_bytes())
    path = replay.with_name("union.json")
    digest = source_pair(path, template, count=2)
    original = path.read_bytes()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    connected = sqlite3.connect
    faults = []

    class UnavailableIndex(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.startswith("INSERT OR REPLACE INTO originals"):
                faults.append(True)
                raise sqlite3.OperationalError("provenance volume full")
            return super().execute(sql, *args, **kwargs)

    def connect(database, *args, **kwargs):
        if Path(database).parent.name.startswith("fh5-provenance-"):
            kwargs["factory"] = UnavailableIndex
        return connected(database, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "tempdir", str(temporary))
        patch.setattr(sqlite3, "connect", connect)
        with pytest.raises(OSError, match="provenance volume full"):
            run_experiment(SACCriticWarmup(model, path, digest, tmp_path / "failed", steps=1))
    assert faults and not list(temporary.iterdir())
    assert path.read_bytes() == original
    assert not (tmp_path / "failed/critic.json").exists()
    result = run_experiment(
        SACCriticWarmup(model, path, digest, tmp_path / "retry", steps=1)
    ).summary["sac"]
    assert result["steps_completed"] == 1
