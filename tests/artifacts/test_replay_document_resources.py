"""Growing replay documents remain trainable through the public experiment seam."""

import hashlib
import io
import json
import sqlite3
import tempfile
import tracemalloc
from pathlib import Path

import pytest

from fh5.evaluation.candidate_archive import CandidateArchive
from fh5.experiment import run_experiment
from fh5.learning.sac.critic import SACCriticResume, SACCriticWarmup
from fh5.learning.sac.training import SACResume, SACTrain
from tests.learning.sac.test_critic_resume import warm_inputs


def large_replay(root, section="excluded"):
    model, replay, _ = warm_inputs(root)
    document = json.loads(replay.read_bytes())
    # Metadata stress fixture, not a claim of this many real driving failures.
    item = json.dumps({"action_index": 0, "reason": "excluded fixture " + "x" * 16384}).encode()
    count = (128 * 1024**2) // (len(item) + 1) + 1
    with replay.open("wb") as stream:
        stream.write(b"{" + json.dumps(section).encode() + b":[")
        for number in range(count):
            if number:
                stream.write(b",")
            stream.write(item)
        stream.write(b"]")
        for key, value in document.items():
            if key != section:
                stream.write(b"," + json.dumps(key).encode() + b":" + json.dumps(value).encode())
        stream.write(b"}\n")
    with replay.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return model, replay, digest


def test_large_replay_arrays_do_not_gate_or_dominate_training_and_resume(tmp_path):
    model, replay, digest = large_replay(tmp_path)
    size = replay.stat().st_size
    assert size > 128 * 1024**2
    warm = tmp_path / "warm"
    tracemalloc.start()
    try:
        stopped = run_experiment(
            SACCriticWarmup(model, replay, digest, warm, steps=2),
            sac_stop_requested=lambda step: step == 1,
        ).summary["sac"]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < size, "The replay text and diagnostic arrays must not remain in RAM"
    assert stopped["steps_completed"] == 1
    finished = tmp_path / "finished"
    run_experiment(SACCriticResume(warm, finished))
    sealed = finished / "experience/replay.json"
    with sealed.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
    first = tmp_path / "first"
    run_experiment(SACTrain(finished, sealed, first, steps=1))
    resumed = run_experiment(SACResume(first, tmp_path / "resumed", steps=1)).summary[
        "sac_learning"
    ]
    whole = run_experiment(SACTrain(finished, sealed, tmp_path / "whole", steps=2)).summary[
        "sac_learning"
    ]
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]
    assert resumed["total_steps"] == 2
    checkpoint_sha = hashlib.sha256((first / "policy.json").read_bytes()).hexdigest()
    tracemalloc.start()
    try:
        archived = run_experiment(
            CandidateArchive(first, tmp_path / "archive", checkpoint_sha, "Retain large replay")
        ).summary["candidate_archive"]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < size, "Archiving must not materialize the diagnostic arrays either"
    assert archived["checkpoint_sha256"] == checkpoint_sha
    from tests.learning.sac.test_sac_cycle import ResponsiveEnvironment, cycle_request

    cycled = run_experiment(
        cycle_request(tmp_path, first), sac_environment=ResponsiveEnvironment(tmp_path)
    ).summary["sac_cycle"]
    assert cycled["stop_reason"] == "budget_completed"
    assert cycled["resources_released"] is True


def test_legacy_json_tokens_across_stream_blocks_keep_learning_and_exact_bytes(tmp_path):
    model, replay, _ = warm_inputs(tmp_path)
    document = json.loads(replay.read_bytes())
    document["transitions"][0]["reward"] = 0.125
    document["excluded"] = [{"reason": '道路 😀 \\ " [ ] { }', "nested": [[], {}, None, True]}]
    plain = json.dumps(document, ensure_ascii=False)
    replay.write_text(plain, encoding="utf-8")
    digest = hashlib.sha256(replay.read_bytes()).hexdigest()
    expected = run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "plain", steps=2)
    ).summary["sac"]
    # Duplicate keys retain the same last-value-wins meaning as legacy json.loads.
    adjusted = '{"transitions":[],"excluded":null,' + plain[1:]
    prefix, suffix = adjusted.split('"reward": 0.125', 1)
    prefix += '"reward": '
    # Put the exponent's first digit at the final character of a decoder block.
    spaces = (io.DEFAULT_BUFFER_SIZE - 1 - len(prefix)) % io.DEFAULT_BUFFER_SIZE
    adjusted = prefix + " " * spaces + "1.25e-1" + suffix
    raw = b"\xef\xbb\xbf" + adjusted.encode("utf-8") + b" \t\r\n"
    replay.write_bytes(raw)
    actual = run_experiment(
        SACCriticWarmup(model, replay, hashlib.sha256(raw).hexdigest(), tmp_path / "split", steps=2)
    ).summary["sac"]
    assert actual["learner_state_sha256"] == expected["learner_state_sha256"]
    assert actual["predictions"] == expected["predictions"]
    assert (tmp_path / "split/experience/replay.json").read_bytes() == raw


@pytest.mark.parametrize("fault", ["trailing", "truncated", "array_comma", "object_comma"])
def test_malformed_replay_cannot_publish_learning_state(tmp_path, fault):
    model, replay, _ = warm_inputs(tmp_path)
    raw = replay.read_bytes().rstrip()
    raw = {
        "trailing": raw + b" garbage",
        "truncated": raw[:-1],
        "array_comma": b'{"invalid":[1,],"valid":' + raw + b"}",
        "object_comma": raw[:-1] + b",}",
    }[fault]
    replay.write_bytes(raw)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_experiment(
            SACCriticWarmup(model, replay, hashlib.sha256(raw).hexdigest(), output, steps=1)
        )
    assert not (output / "critic.json").exists()


@pytest.mark.parametrize("fault", ["index", "copy"])
def test_replay_io_failure_keeps_parent_and_releases_temporary_storage(
    tmp_path, monkeypatch, fault
):
    model, replay, digest = warm_inputs(tmp_path)
    parent, output = tmp_path / "parent", tmp_path / "failed"
    run_experiment(
        SACCriticWarmup(model, replay, digest, parent, steps=2),
        sac_stop_requested=lambda step: step == 1,
    )
    original = (parent / "critic.json").read_bytes()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    opened, connected = Path.open, sqlite3.connect
    observed = []

    def connect(database, *args, **kwargs):
        if fault == "index" and Path(database).parent.name.startswith("fh5-replay-document-"):
            observed.append(True)
            raise sqlite3.OperationalError("database or disk is full")
        return connected(database, *args, **kwargs)

    def open_file(path, mode="r", *args, **kwargs):
        if fault == "copy" and path == output / "experience/replay.json" and mode == "xb":
            observed.append(True)
            raise OSError("disk is full while copying replay")
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
    assert (parent / "critic.json").read_bytes() == original
    retried = run_experiment(SACCriticResume(parent, tmp_path / "retry")).summary["sac"]
    assert retried["total_steps"] == 2


def test_changed_replay_during_sealing_does_not_publish_a_checkpoint(tmp_path, monkeypatch):
    model, replay, digest = warm_inputs(tmp_path)
    output = tmp_path / "invalid"
    opened = Path.open
    changed = []

    def open_file(path, mode="r", *args, **kwargs):
        if path == output / "experience/replay.json" and mode == "xb":
            with opened(replay, "ab") as stream:
                stream.write(b" ")
            changed.append(True)
        return opened(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_file)
        with pytest.raises(ValueError, match="changed during copy"):
            run_experiment(SACCriticWarmup(model, replay, digest, output, steps=1))
    assert changed == [True]
    assert not (output / "critic.json").exists()


def test_experience_addition_preserves_legacy_extension_metadata(tmp_path):
    from tests.artifacts.test_expansion_resources import extra_source

    model, replay, _ = warm_inputs(tmp_path)
    document = json.loads(replay.read_bytes())
    document["notes"] = [{"text": "道路 metadata"}, [1, 2, None]]
    replay.write_text(json.dumps(document), encoding="utf-8")
    digest = hashlib.sha256(replay.read_bytes()).hexdigest()
    warm, parent, output = tmp_path / "warm", tmp_path / "parent", tmp_path / "expanded"
    run_experiment(SACCriticWarmup(model, replay, digest, warm, steps=1))
    run_experiment(SACTrain(warm, warm / "experience/replay.json", parent, steps=0))
    extra = extra_source(replay, tmp_path / "addition", 1)
    result = run_experiment(SACResume(parent, output, steps=1, additions=(extra,))).summary[
        "sac_learning"
    ]
    assert result["steps_completed"] == 1
    merged = json.loads((output / "experience/replay.json").read_bytes())
    assert merged["notes"] == [{"text": "道路 metadata"}, [1, 2, None]]
    canonical = (json.dumps(merged, sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert (output / "experience/replay.json").read_bytes() == canonical


def test_expanded_replay_metadata_streams_past_the_old_output_limit(tmp_path):
    from tests.artifacts.test_expansion_resources import extra_source

    model, replay, digest = large_replay(tmp_path, section="notes")
    size = replay.stat().st_size
    warm, parent = tmp_path / "warm", tmp_path / "parent"
    run_experiment(SACCriticWarmup(model, replay, digest, warm, steps=1))
    run_experiment(SACTrain(warm, warm / "experience/replay.json", parent, steps=0))
    extra = extra_source(replay, tmp_path / "addition", 7)
    output = tmp_path / "expanded"
    tracemalloc.start()
    try:
        result = run_experiment(SACResume(parent, output, steps=1, additions=(extra,))).summary[
            "sac_learning"
        ]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result["steps_completed"] == 1
    assert peak < size, "Union writing must not expand indexed metadata or the complete JSON text"
    merged = output / "experience/replay.json"
    assert merged.stat().st_size > 128 * 1024**2
    with replay.open("rb") as stream:
        expected_notes = json.load(stream)["notes"]
    with merged.open("rb") as stream:
        assert json.load(stream)["notes"] == expected_notes
    resumed = run_experiment(SACResume(output, tmp_path / "resumed", steps=1)).summary[
        "sac_learning"
    ]
    assert resumed["total_steps"] == 2
