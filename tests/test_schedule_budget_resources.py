"""Explicit experiment budgets are not narrowed by unrelated admission caps."""

import json
from pathlib import Path

import pytest
from test_training_config_resources import (
    InterruptedResources,
    bind_training,
    schedule_config,
    training_config,
    write_config,
)

from fh5.experiment import run_experiment
from fh5.learning_schedule import ScheduledBCTrain


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("cpu_threads", 3),
        ("poll_interval_s", 0.01),
        ("poll_interval_s", 6),
        ("max_wait_s", 0.01),
        ("max_wait_s", 3601),
        ("max_total_s", 0.1),
        ("max_total_s", 86401),
        ("max_unit_s", 0.001),
        ("max_unit_s", 61),
        ("max_private_bytes", 512 * 1024),
        ("max_private_bytes", 129 * 1024**3),
        ("min_free_disk_bytes", 17 * 1024**4),
        ("max_status_age_ms", 50),
        ("max_status_age_ms", 10001),
        ("max_image_age_ms", 0.5),
        ("max_image_age_ms", 5001),
        ("max_pending_bytes", 2 * 1024**3),
        ("max_gpu_memory_mib", 0.5),
        ("max_gpu_memory_mib", 131073),
        ("max_gpu_utilization_percent", 0.5),
    ],
)
def test_explicit_schedule_budget_reaches_resource_admission_without_extra_caps(
    tmp_path, name, value
):
    training = training_config(tmp_path)
    config = schedule_config(tmp_path)
    bind_training(config, training)
    settings = json.loads(config.read_bytes())
    settings["budget"][name] = value
    write_config(config, settings)
    resources = InterruptedResources()
    output = tmp_path / "scheduled"
    summary = run_experiment(
        ScheduledBCTrain(config, output), learning_resources=resources
    ).summary["learning_schedule"]
    assert summary["stop_reason"] == "interrupted"
    assert summary["config"]["budget"][name] == value
    assert summary["steps_completed"] == summary["durable_steps_completed"] == 0
    assert resources.closed
    retained = json.loads((output / "schedule-config.json").read_bytes())
    assert retained["budget"][name] == value


@pytest.mark.parametrize(
    "name", ["max_total_s", "max_wait_s", "max_status_age_ms", "max_image_age_ms"]
)
def test_integer_time_budget_does_not_require_a_float_conversion(tmp_path, name):
    training = training_config(tmp_path)
    config = schedule_config(tmp_path)
    bind_training(config, training)
    settings = json.loads(config.read_bytes())
    # This is a serialized experiment budget, not a loop or allocation of this size.
    settings["budget"][name] = 10**400
    write_config(config, settings)
    output = tmp_path / "scheduled"
    summary = run_experiment(
        ScheduledBCTrain(config, output), learning_resources=BackloggedThenStop()
    ).summary["learning_schedule"]
    assert summary["stop_reason"] == "interrupted"
    assert summary["config"]["budget"][name] == 10**400


class BackloggedThenStop(InterruptedResources):
    def __init__(self):
        self.reads = 0
        self.time = 0

    def now_ns(self):
        return self.time

    def sample(self):
        self.reads += 1
        if self.reads > 1:
            raise KeyboardInterrupt
        return {
            "observed_ns": self.time,
            "process_private_bytes": 1024,
            "free_disk_bytes": 1024**4,
            "collector": {
                "process_liveness": "running",
                "software_snapshot_verified": True,
                "state": "recording",
                "heartbeat_ns": self.time,
                "last_poll_ns": self.time,
                "latest_image_source_ns": self.time,
                "pending_bytes": 100 * 1024**2,
                "dropped_rows": 0,
                "seen_rows": 1,
            },
        }

    def wait(self, seconds):
        self.time += int(seconds * 1_000_000_000)


@pytest.mark.parametrize(
    ("budget", "seconds", "reason"),
    [("max_wait_s", 1, "resource_wait_timeout"), ("max_total_s", 0.25, "total_time_limit")],
)
def test_resource_wait_cannot_outlast_the_explicit_experiment_budget(
    tmp_path, budget, seconds, reason
):
    training = training_config(tmp_path)
    config = schedule_config(tmp_path)
    bind_training(config, training)
    settings = json.loads(config.read_bytes())
    settings["budget"].update(poll_interval_s=6, **{budget: seconds})
    write_config(config, settings)
    resources = BackloggedThenStop()
    summary = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
    ).summary["learning_schedule"]
    assert summary["stop_reason"] == reason
    assert resources.time == seconds * 1_000_000_000
    assert summary["wait_s"] == seconds
    assert summary["steps_completed"] == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("cpu_threads", 0),
        ("cpu_threads", 1.5),
        ("cpu_threads", True),
        ("poll_interval_s", 0),
        ("max_wait_s", -1),
        ("max_total_s", float("nan")),
        ("max_unit_s", float("inf")),
        ("max_private_bytes", "1024"),
        ("min_free_disk_bytes", -1),
        ("max_status_age_ms", -1),
        ("max_image_age_ms", False),
        ("max_pending_bytes", -1),
        ("max_gpu_memory_mib", -1),
        ("max_gpu_utilization_percent", 101),
    ],
)
def test_schedule_budget_keeps_type_unit_and_percentage_validation(tmp_path, name, value):
    training = training_config(tmp_path)
    config = schedule_config(tmp_path)
    bind_training(config, training)
    settings = json.loads(config.read_bytes())
    settings["budget"][name] = value
    write_config(config, settings)
    output = tmp_path / "scheduled"
    with pytest.raises(ValueError):
        run_experiment(ScheduledBCTrain(config, output), learning_resources=InterruptedResources())
    assert not output.exists()


@pytest.mark.parametrize(
    ("name", "reason"),
    [("max_wait_s", "resource_wait_timeout"), ("max_total_s", "total_time_limit")],
)
def test_zero_time_allowance_stops_without_waiting_or_starting_learning(tmp_path, name, reason):
    training = training_config(tmp_path)
    config = schedule_config(tmp_path)
    bind_training(config, training)
    settings = json.loads(config.read_bytes())
    settings["budget"][name] = 0
    write_config(config, settings)
    resources = BackloggedThenStop()
    result = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
    ).summary["learning_schedule"]
    assert result["stop_reason"] == reason
    assert result["wait_s"] == resources.time == 0
    assert result["steps_completed"] == 0 and result["candidate"] is None


@pytest.mark.parametrize("journal_error", [None, OSError, MemoryError])
def test_explicit_schedule_budget_can_complete_actual_cpu_updates(
    tmp_path, monkeypatch, journal_error
):
    torch = pytest.importorskip("torch")
    from test_learning_schedule import Resources, configuration

    config, _, _ = configuration(tmp_path, cpu_threads=3, max_total_s=86401, poll_interval_s=6)
    output = tmp_path / "scheduled"
    failures = []
    opening = Path.open

    def unavailable(path, mode="r", *args, **kwargs):
        if (
            journal_error is not None
            and path == output / "diagnostics/schedule-events.jsonl"
            and mode == "xb"
        ):
            failures.append(path)
            raise journal_error("optional scheduler history unavailable")
        return opening(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unavailable)
    summary = run_experiment(
        ScheduledBCTrain(config, output), learning_resources=Resources()
    ).summary["learning_schedule"]
    assert summary["state"] == "completed"
    assert summary["steps_completed"] == summary["durable_steps_completed"] == 3
    saved = torch.load(output / "learner/learner.pt", map_location="cpu", weights_only=True)
    assert saved["step"] == 3
    assert all(entry["step"].item() == 3 for entry in saved["optimizer"]["state"].values())
    assert saved["metadata"]["resume_contract"]["cpu_threads"] == 3
    report = json.loads((output / "candidate/report.json").read_bytes())
    assert report["decisions"]
    assert report["verification"]["status"] == "training_reload"
    assert report["verification"]["compared_decisions"] == len(report["decisions"])
    assert report["verification"]["max_abs_error"] == 0.0
    if journal_error is None:
        assert failures == [] and summary["event_history"]["status"] == "complete"
        assert summary["events_omitted"] == 0
    else:
        assert failures == [output / "diagnostics/schedule-events.jsonl"]
        assert summary["event_history"]["status"] == "unavailable"
        assert summary["events_omitted"] == summary["sample_count"] > 0
        assert "optional scheduler history unavailable" in summary["event_history"]["error"]
