"""Native collector admission through public scheduling, without model/worker work."""

import json
import sys
import tracemalloc
from types import SimpleNamespace

import pytest

from fh5.artifacts.io import encode, sha256_file
from fh5.experiment import run_experiment
from fh5.learning.bc.schedule import ScheduledBCTrain
from tests.collection.test_collection_control_resources import (
    add_padding,
    write_document,
)
from tests.collection.test_collection_control_resources import (
    bundle as bundle,
)
from tests.learning.bc.test_training_config_resources import (
    bind_training,
    schedule_config,
    training_config,
)


@pytest.fixture
def native_inputs(tmp_path, bundle, monkeypatch):
    import fh5.learning.resources as native

    training = training_config(tmp_path)
    config = schedule_config(tmp_path)
    bind_training(config, training)
    options = json.loads(config.read_bytes())
    options.update(
        collector_bundle=str(bundle),
        collector_manifest_sha256=sha256_file(bundle / "frozen.json"),
    )
    write_document(config, options)
    status_path = bundle / "recording/status.json"
    status = json.loads(status_path.read_bytes())
    # All identity/freshness evidence is valid; only real archive backlog waits.
    status["pending_bytes"] = options["budget"]["max_pending_bytes"] + 1
    write_document(status_path, status)
    output = tmp_path / "scheduled"
    probes, waits = [], []

    def host_metrics(resources):
        probes.append(resources.include_gpu)
        return {"process_private_bytes": 64 * 1024**2, "gpu_status": "not_requested"}

    def stop_during_wait(seconds):
        assert seconds > 0
        waits.append(seconds)
        (output / "stop.request").touch()

    monkeypatch.setattr(native.WindowsResources, "__call__", host_metrics)
    monkeypatch.setattr(native.time, "perf_counter_ns", lambda: 10_000_000_000)
    monkeypatch.setattr(native.time, "sleep", stop_during_wait)
    monkeypatch.setattr(
        native.shutil, "disk_usage", lambda path: SimpleNamespace(free=50 * 1024**3)
    )
    return SimpleNamespace(
        bundle=bundle,
        config=config,
        training=training,
        output=output,
        probes=probes,
        waits=waits,
    )


def rebind_session(root):
    status_path = root / "recording/status.json"
    status = json.loads(status_path.read_bytes())
    status["session_sha256"] = sha256_file(root / "recording/session.json")
    write_document(status_path, status)


def add_nested_runtime(path):
    session = json.loads(path.read_bytes())
    snapshot = session.pop("software_snapshot")
    with path.open("wb") as stream:
        stream.write(encode(session).rstrip()[:-1] + b',"software_snapshot":')
        stream.write(encode(snapshot).rstrip()[:-1] + b',"runtime":{"packages":[')
        row = b'["synthetic-package-' + b"x" * 800 + b'","1.0"]'
        for index in range(6500):
            if index:
                stream.write(b",")
            stream.write(row)
        stream.write(b"]}}}\n")


@pytest.mark.parametrize("representation", ["small", "runtime", "status", "process", "worker"])
def test_native_large_state_reaches_verified_backlog_then_explicit_stop(
    native_inputs, representation, record_property
):
    inputs = native_inputs
    root = inputs.bundle
    target = None
    if representation == "runtime":
        target = root / "recording/session.json"
        add_nested_runtime(target)
        rebind_session(root)
        assert target.stat().st_size > 4 * 1024**2
    elif representation != "small":
        relative, old_limit = {
            "status": ("recording/status.json", 4 * 1024**2),
            "process": ("process.json", 16 * 1024),
            "worker": ("worker-state.json", 64 * 1024),
        }[representation]
        target = root / relative
        add_padding(target, old_limit)
        assert target.stat().st_size > old_limit
    before = {path: sha256_file(path) for path in root.rglob("*.json")}
    imported = set(sys.modules)
    tracemalloc.start()
    try:
        result = run_experiment(ScheduledBCTrain(inputs.config, inputs.output))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    record_property("python_peak_bytes", peak)
    if target is not None:
        record_property("source_payload_bytes", target.stat().st_size)
    if representation == "runtime":
        assert peak < target.stat().st_size
    summary = result.summary["learning_schedule"]
    assert inputs.probes == [False] and len(inputs.waits) == 1
    assert summary["state"] == "stopped" and summary["stop_reason"] == "requested_stop"
    assert summary["source_kind"] == "native_resources"
    assert summary["sample_count"] == 1
    assert summary["pressure_counts"] == {"collection_backlog": 1}
    assert summary["steps_completed"] == summary["durable_steps_completed"] == 0
    assert summary["candidate"] is None and not (inputs.output / "candidate").exists()
    assert summary["commands_sent"] is False
    with (inputs.output / summary["event_history"]["path"]).open(encoding="utf-8") as stream:
        event = json.loads(next(stream))
        assert not stream.read()
    collector = event["sample"]["collector"]
    assert collector["process_liveness"] == "running"
    assert collector["software_snapshot_verified"] is True
    assert collector["session_sha256"] == sha256_file(root / "recording/session.json")
    assert (
        collector["session_manifest_sha256"]
        == collector["worker_manifest_sha256"]
        == sha256_file(root / "frozen.json")
    )
    assert "runtime" not in json.dumps(event)
    assert all(sha256_file(path) == digest for path, digest in before.items())
    assert json.loads((inputs.output / "training.json").read_bytes()) == {
        **json.loads(inputs.training.read_bytes()),
        "dataset": str((inputs.training.parent / "snapshot/dataset.json").resolve()),
    }
    assert not (inputs.training.parent / "snapshot/dataset.json").exists()
    assert not any(
        name == "torch" or name.startswith("torch.") for name in set(sys.modules) - imported
    )


@pytest.mark.parametrize(
    "fault",
    [
        "session_bytes",
        "session_manifest",
        "worker_manifest",
        "null_hash",
        "bad_hash",
    ],
)
def test_native_rejects_session_or_worker_binding_before_host_metrics(native_inputs, fault):
    inputs = native_inputs
    root = inputs.bundle
    session_path = root / "recording/session.json"
    status_path = root / "recording/status.json"
    if fault == "session_bytes":
        with session_path.open("ab") as stream:
            stream.write(b"\n")
    elif fault == "session_manifest":
        session = json.loads(session_path.read_bytes())
        session["software_snapshot"]["manifest_sha256"] = "0" * 64
        write_document(session_path, session)
        rebind_session(root)
    elif fault == "worker_manifest":
        worker_path = root / "worker-state.json"
        worker = json.loads(worker_path.read_bytes())
        worker["manifest_sha256"] = "0" * 64
        write_document(worker_path, worker)
    else:
        status = json.loads(status_path.read_bytes())
        status["session_sha256"] = None if fault == "null_hash" else "invalid-sha256"
        write_document(status_path, status)
    before = {path: sha256_file(path) for path in root.rglob("*.json")}
    imported = set(sys.modules)
    with pytest.raises(ValueError):
        run_experiment(ScheduledBCTrain(inputs.config, inputs.output))
    assert inputs.probes == [] and inputs.waits == []
    summary = json.loads((inputs.output / "schedule.json").read_bytes())
    assert summary["state"] == "stopped" and summary["sample_count"] == 0
    assert summary["steps_completed"] == summary["durable_steps_completed"] == 0
    assert summary["candidate"] is None and not (inputs.output / "candidate").exists()
    assert all(sha256_file(path) == digest for path, digest in before.items())
    assert not any(
        name == "torch" or name.startswith("torch.") for name in set(sys.modules) - imported
    )


def test_native_missing_session_binding_remains_unverified_without_training(native_inputs):
    inputs = native_inputs
    path = inputs.bundle / "recording/status.json"
    status = json.loads(path.read_bytes())
    status.pop("session_sha256")
    write_document(path, status)
    # Absence is unavailable evidence, including the first-heartbeat window;
    # preserve the existing fail-closed waiting behavior instead of inventing a
    # new exception contract. Present-but-wrong bindings remain rejected above.
    imported = set(sys.modules)
    result = run_experiment(ScheduledBCTrain(inputs.config, inputs.output))
    summary = result.summary["learning_schedule"]
    assert inputs.probes == [False] and len(inputs.waits) == 1
    assert summary["stop_reason"] == "requested_stop" and summary["sample_count"] == 1
    assert summary["pressure_counts"] == {
        "collector_state_unverified": 1,
        "collection_backlog": 1,
    }
    assert summary["steps_completed"] == summary["durable_steps_completed"] == 0
    assert summary["candidate"] is None and not (inputs.output / "candidate").exists()
    with (inputs.output / summary["event_history"]["path"]).open(encoding="utf-8") as stream:
        collector = json.loads(next(stream))["sample"]["collector"]
    assert collector["software_snapshot_verified"] is False
    assert "session_sha256" not in collector
    assert not any(
        name == "torch" or name.startswith("torch.") for name in set(sys.modules) - imported
    )


@pytest.mark.parametrize("value", [[], {}])
def test_native_rejects_container_in_verified_scalar_before_host_metrics(native_inputs, value):
    inputs = native_inputs
    path = inputs.bundle / "recording/session.json"
    session = json.loads(path.read_bytes())
    session["software_snapshot"]["verified"] = value
    write_document(path, session)
    rebind_session(inputs.bundle)
    with pytest.raises(ValueError):
        run_experiment(ScheduledBCTrain(inputs.config, inputs.output))
    assert inputs.probes == [] and inputs.waits == []


def test_native_last_duplicate_snapshot_replaces_earlier_invalid_parent(native_inputs):
    inputs = native_inputs
    path = inputs.bundle / "recording/session.json"
    valid = path.read_bytes().strip()
    path.write_bytes(b'{"software_snapshot":[],' + valid[1:] + b"\n")
    rebind_session(inputs.bundle)
    result = run_experiment(ScheduledBCTrain(inputs.config, inputs.output))
    summary = result.summary["learning_schedule"]
    assert summary["pressure_counts"] == {"collection_backlog": 1}
    assert summary["stop_reason"] == "requested_stop" and summary["sample_count"] == 1
    assert summary["steps_completed"] == summary["durable_steps_completed"] == 0
    assert inputs.probes == [False] and len(inputs.waits) == 1
