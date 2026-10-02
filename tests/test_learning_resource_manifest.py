"""Native learning admission verifies large manifests without retaining their bytes."""

import json
import tracemalloc
from types import SimpleNamespace

import pytest
from test_training_config_resources import (
    bind_training,
    file_hash,
    schedule_config,
    training_config,
    write_config,
)

from fh5.experiment import run_experiment
from fh5.learning_schedule import ScheduledBCTrain


def prepared_inputs(tmp_path):
    training = training_config(tmp_path)
    config = schedule_config(tmp_path)
    bind_training(config, training)
    bundle = tmp_path / "collector"
    bundle.mkdir()
    manifest = write_config(
        bundle / "frozen.json",
        {
            "version": 1,
            "kind": "frozen-passive-collection-v1",
            "source": "synthetic",
            "files": {},
            "commands_sent": False,
        },
        padding_mib=5,
    )
    settings = json.loads(config.read_bytes())
    settings["collector_manifest_sha256"] = file_hash(manifest)
    write_config(config, settings)
    return config, training, manifest


@pytest.mark.parametrize("host_extra_bytes", [0, 32768])
def test_large_bound_manifest_reaches_native_sampling_and_explicit_stop(
    tmp_path, monkeypatch, host_extra_bytes
):
    import fh5.learning_resources as native

    config, training, manifest = prepared_inputs(tmp_path)
    manifest_hash, training_hash = file_hash(manifest), file_hash(training)
    output = tmp_path / "scheduled"
    probes = []

    def host_metrics(resources):
        probes.append(resources.include_gpu)
        return {
            "process_private_bytes": 64 * 1024**2,
            "gpu_status": "not_requested",
            "optional_debug": "x" * host_extra_bytes,
        }

    def stop_during_wait(seconds):
        assert seconds > 0
        (output / "stop.request").touch()

    # Replace only host metrics and time. Collector status, manifest binding,
    # scheduling, configuration preservation, and report publication remain real.
    monkeypatch.setattr(native.WindowsResources, "__call__", host_metrics)
    monkeypatch.setattr(native.time, "perf_counter_ns", lambda: 10_000_000_000)
    monkeypatch.setattr(native.time, "sleep", stop_during_wait)
    monkeypatch.setattr(
        native.shutil, "disk_usage", lambda path: SimpleNamespace(free=50 * 1024**3)
    )
    tracemalloc.start()
    try:
        result = run_experiment(ScheduledBCTrain(config, output))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    summary = result.summary["learning_schedule"]
    assert manifest.stat().st_size > 4 * 1024**2
    assert peak < manifest.stat().st_size
    assert probes == [False]
    assert summary["source_kind"] == "native_resources"
    assert summary["state"] == "stopped" and summary["stop_reason"] == "requested_stop"
    assert summary["sample_count"] == 1
    assert summary["pressure_counts"]["collector_state_unverified"] == 1
    with (output / summary["event_history"]["path"]).open(encoding="utf-8") as stream:
        event = json.loads(next(stream))
    assert event["sample"]["collector"]["process_liveness"] == "not_started"
    assert "optional_debug" not in event["sample"]
    assert summary["steps_completed"] == summary["durable_steps_completed"] == 0
    assert summary["candidate"] is None and not (output / "candidate").exists()
    assert summary["commands_sent"] is False
    assert file_hash(manifest) == manifest_hash
    assert file_hash(output / "requested-training.json") == training_hash
    assert json.loads(result.report_path.read_bytes()) == summary


@pytest.mark.parametrize("change_when", ["before_admission", "during_wait"])
def test_changed_large_manifest_rejects_before_the_next_host_query(
    tmp_path, monkeypatch, change_when
):
    import fh5.learning_resources as native

    config, training, manifest = prepared_inputs(tmp_path)
    bound_hash, training_hash = file_hash(manifest), file_hash(training)
    output = tmp_path / "scheduled"
    probes = []
    clock = 10_000_000_000

    def alter_manifest():
        with manifest.open("ab") as stream:
            stream.write(b"\n")

    def host_metrics(resources):
        probes.append(resources.include_gpu)
        return {"process_private_bytes": 64 * 1024**2, "gpu_status": "not_requested"}

    def alter_during_wait(seconds):
        nonlocal clock
        clock += round(seconds * 1e9)
        alter_manifest()

    monkeypatch.setattr(native.WindowsResources, "__call__", host_metrics)
    monkeypatch.setattr(native.time, "perf_counter_ns", lambda: clock)
    monkeypatch.setattr(native.time, "sleep", alter_during_wait)
    monkeypatch.setattr(
        native.shutil, "disk_usage", lambda path: SimpleNamespace(free=50 * 1024**3)
    )
    if change_when == "before_admission":
        alter_manifest()
    with pytest.raises(ValueError, match="Collector manifest differs"):
        run_experiment(ScheduledBCTrain(config, output))
    assert file_hash(manifest) != bound_hash
    if change_when == "before_admission":
        assert probes == []
        assert not output.exists()
    else:
        assert probes == [False]
        summary = json.loads((output / "schedule.json").read_bytes())
        assert summary["state"] == "stopped"
        assert "Collector manifest differs" in summary["stop_reason"]
        assert summary["sample_count"] == 1
        assert summary["steps_completed"] == summary["durable_steps_completed"] == 0
        assert summary["candidate"] is None and not (output / "candidate").exists()
        assert file_hash(output / "requested-training.json") == training_hash
