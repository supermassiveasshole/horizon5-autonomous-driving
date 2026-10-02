"""Optional scheduler traces retain observations without controlling admission."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_training_config_resources import (
    bind_training,
    schedule_config,
    training_config,
    write_config,
)

from fh5.experiment import run_experiment
from fh5.learning_schedule import ScheduledBCTrain


class AlternatingResources:
    source_kind = "synthetic"

    def __init__(self, output, samples=1103):
        self.output = output
        self.limit = samples
        self.reads = 0
        self.time = 10_000_000_000
        self.closed = False

    def now_ns(self):
        return self.time

    def sample(self):
        self.reads += 1
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
                "pending_bytes": 100 * 1024**2 if self.reads % 2 else 0,
                "dropped_rows": 0,
                "seen_rows": self.reads,
            },
        }

    def wait(self, seconds):
        self.time += round(seconds * 1_000_000_000)
        if self.reads == self.limit:
            (self.output / "stop.request").touch()

    def close(self):
        self.closed = True


def request_for(tmp_path):
    training = training_config(tmp_path)
    config = schedule_config(tmp_path)
    bind_training(config, training)
    settings = json.loads(config.read_bytes())
    settings["budget"].update(max_wait_s=300, max_total_s=400)
    write_config(config, settings)
    return ScheduledBCTrain(config, tmp_path / "scheduled")


def test_schedule_retains_all_samples_beyond_the_old_event_history_limit(tmp_path):
    request = request_for(tmp_path)
    resources = AlternatingResources(request.output_dir)
    result = run_experiment(request, learning_resources=resources)
    summary = result.summary["learning_schedule"]
    assert summary["state"] == "stopped" and summary["stop_reason"] == "requested_stop"
    assert summary["sample_count"] == 1103
    assert summary["pressure_counts"] == {"collection_backlog": 552}
    assert summary["steps_completed"] == 0 and summary["candidate"] is None
    assert resources.closed
    history = summary["event_history"]
    assert history["status"] == "complete" and history["records"] == 1103
    path = request.output_dir / history["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == history["sha256"]
    with path.open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream]
    assert len(events) == 1103
    assert events[0]["sample"]["collector"]["seen_rows"] == 1
    assert events[-1]["sample"]["collector"]["seen_rows"] == 1103
    assert events[0]["reasons"] == events[-1]["reasons"] == ["collection_backlog"]
    assert events[1]["reasons"] == []
    assert summary["events_omitted"] == 0
    assert summary["events_unverified"] == 0
    assert "events" not in summary
    assert json.loads(result.report_path.read_bytes()) == summary
    assert result.report_path.stat().st_size < path.stat().st_size / 10


@pytest.mark.parametrize(
    "extra", ["x" * 32768, float("nan"), object()], ids=["large_text", "nonfinite", "non_json"]
)
def test_optional_resource_details_do_not_control_admission_or_inflate_each_event(tmp_path, extra):
    request = request_for(tmp_path)

    class ExtraDetails(AlternatingResources):
        def sample(self):
            sample = super().sample()
            sample["debug"] = extra
            sample["collector"]["debug"] = extra
            return sample

    summary = run_experiment(
        request, learning_resources=ExtraDetails(request.output_dir, samples=3)
    ).summary["learning_schedule"]
    assert summary["stop_reason"] == "requested_stop"
    assert summary["sample_count"] == 3
    assert summary["pressure_counts"] == {"collection_backlog": 2}
    history = summary["event_history"]
    assert history["status"] == "complete" and history["records"] == 3
    with (request.output_dir / history["path"]).open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream]
    assert len(events) == 3
    assert "debug" not in events[0]["sample"]
    assert "debug" not in events[0]["sample"]["collector"]
    assert events[1]["reasons"] == []


@pytest.mark.parametrize("operation", ["open", "write", "flush"])
@pytest.mark.parametrize("error_type", [OSError, MemoryError])
def test_optional_schedule_journal_failure_preserves_resource_checks_and_stop(
    tmp_path, monkeypatch, operation, error_type
):
    request = request_for(tmp_path)
    opening = Path.open
    failures = []

    def fail():
        failures.append(operation)
        raise error_type("optional event storage unavailable")

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, value):
            if operation == "write":
                fail()
            return self.stream.write(value)

        def flush(self):
            if operation == "flush":
                fail()
            return self.stream.flush()

    def unavailable(path, mode="r", *args, **kwargs):
        if path == request.output_dir / "diagnostics/schedule-events.jsonl" and mode == "xb":
            if operation == "open":
                fail()
            return FailingStream(opening(path, mode, *args, **kwargs))
        return opening(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unavailable)
    resources = AlternatingResources(request.output_dir, samples=3)
    result = run_experiment(request, learning_resources=resources)
    summary = result.summary["learning_schedule"]
    assert failures == [operation]
    assert summary["stop_reason"] == "requested_stop"
    assert summary["sample_count"] == 3
    assert summary["pressure_counts"] == {"collection_backlog": 2}
    assert summary["event_history"]["status"] == "unavailable"
    assert "optional event storage unavailable" in summary["event_history"]["error"]
    assert summary["event_history"]["sha256"] is None
    written = 3 if operation == "flush" else 0
    assert summary["event_history"]["records"] == written
    assert summary["events_unverified"] == written
    assert summary["events_omitted"] == 3 - written
    assert summary["steps_completed"] == 0 and summary["candidate"] is None
    assert resources.closed
    assert json.loads(result.report_path.read_bytes()) == summary


@pytest.mark.parametrize(
    ("section", "field", "value", "reason"),
    [
        (None, "observed_ns", float("nan"), "resource_status_stale"),
        (None, "process_private_bytes", float("inf"), "resource_status_missing"),
        (None, "free_disk_bytes", [], "resource_status_missing"),
        ("collector", "heartbeat_ns", float("nan"), "collection_status_stale"),
        ("collector", "latest_image_source_ns", float("inf"), "collection_images_stale"),
        ("collector", "pending_bytes", float("nan"), "collection_backlog"),
        ("collector", "dropped_rows", 1.5, "collection_counters_invalid"),
    ],
)
def test_invalid_required_resource_fields_remain_fail_closed_and_recordable(
    tmp_path, section, field, value, reason
):
    request = request_for(tmp_path)

    class InvalidMetrics(AlternatingResources):
        def sample(self):
            sample = super().sample()
            target = sample if section is None else sample[section]
            target[field] = value
            return sample

    summary = run_experiment(
        request, learning_resources=InvalidMetrics(request.output_dir, samples=3)
    ).summary["learning_schedule"]
    assert summary["stop_reason"] == "requested_stop"
    assert summary["pressure_counts"][reason] == 3
    assert summary["steps_completed"] == 0
    assert summary["event_history"]["status"] == "complete"


def test_large_integer_resource_measurement_does_not_require_float_conversion(tmp_path):
    request = request_for(tmp_path)

    class LargeIntegerMetric(AlternatingResources):
        def sample(self):
            sample = super().sample()
            sample["free_disk_bytes"] = 10**400
            return sample

    summary = run_experiment(
        request, learning_resources=LargeIntegerMetric(request.output_dir, samples=3)
    ).summary["learning_schedule"]
    assert summary["stop_reason"] == "requested_stop"
    assert summary["sample_count"] == 3
    assert summary["pressure_counts"] == {"collection_backlog": 2}
    history = summary["event_history"]
    with (request.output_dir / history["path"]).open(encoding="utf-8") as stream:
        first = json.loads(next(stream))
    assert first["sample"]["free_disk_bytes"] == 10**400


def test_optional_counter_encoding_failure_does_not_stop_resource_checks(tmp_path):
    request = request_for(tmp_path)
    digit_limit = sys.get_int_max_str_digits()
    if not digit_limit:
        pytest.skip("Interpreter has no decimal integer encoding limit")

    class HugeOptionalCounter(AlternatingResources):
        def sample(self):
            sample = super().sample()
            # Valid Python integer exceeds this interpreter's JSON digit interface.
            # The optional record must degrade, not change resource admission.
            sample["collector"]["seen_rows"] = 10**digit_limit
            return sample

    resources = HugeOptionalCounter(request.output_dir, samples=3)
    summary = run_experiment(request, learning_resources=resources).summary["learning_schedule"]
    assert summary["stop_reason"] == "requested_stop"
    assert summary["sample_count"] == 3 and resources.closed
    assert summary["pressure_counts"] == {"collection_backlog": 2}
    assert summary["steps_completed"] == 0
    assert summary["event_history"]["status"] == "unavailable"
    assert "ValueError" in summary["event_history"]["error"]
    assert summary["events_omitted"] == 3 and summary["events_unverified"] == 0


def test_resource_limit_sample_is_retained_with_its_terminal_stop_reason(tmp_path):
    request = request_for(tmp_path)

    class MemoryExhausted(AlternatingResources):
        def sample(self):
            sample = super().sample()
            sample["process_private_bytes"] = 5 * 1024**3
            return sample

    summary = run_experiment(
        request, learning_resources=MemoryExhausted(request.output_dir)
    ).summary["learning_schedule"]
    assert summary["stop_reason"] == "resource_limit:process_private_bytes"
    assert summary["sample_count"] == 1
    history = summary["event_history"]
    assert history["status"] == "complete" and history["records"] == 1
    with (request.output_dir / history["path"]).open(encoding="utf-8") as stream:
        event = json.loads(next(stream))
        assert list(stream) == []
    assert event["reasons"] == ["resource_limit:process_private_bytes"]
    assert event["sample"]["process_private_bytes"] == 5 * 1024**3
    assert summary["events_omitted"] == 0
