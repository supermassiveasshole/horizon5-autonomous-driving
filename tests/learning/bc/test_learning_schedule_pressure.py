"""Recoverable resource pressure at the public CPU training and resume seams."""

import json
from pathlib import Path

import pytest

from fh5.experiment import run_experiment
from fh5.learning.bc.schedule import ScheduledBCResume, ScheduledBCTrain
from tests.learning.bc.test_bc_resume_resources import resume_inputs
from tests.learning.bc.test_learning_schedule import Resources


class HostPressure(Resources):
    def __init__(self, field, *, clear_after_s=None):
        super().__init__()
        self.field = field
        self.clear_after_s = clear_after_s

    def sample(self):
        value = super().sample()
        # Admission and the first two complete optimizer updates are healthy.
        if self.reads >= 5 and (self.clear_after_s is None or self.waited < self.clear_after_s):
            value[self.field] = 16 * 1024**3 if self.field == "process_private_bytes" else 0
        return value


@pytest.mark.parametrize("field", ["process_private_bytes", "free_disk_bytes"])
def test_transient_host_pressure_resumes_the_same_cpu_learner(tmp_path, field):
    torch = pytest.importorskip("torch")
    schedule, _ = resume_inputs(tmp_path)
    frozen_configuration = schedule.read_bytes()
    reference = run_experiment(
        ScheduledBCTrain(schedule, tmp_path / "reference"), learning_resources=Resources()
    ).summary["learning_schedule"]
    resources = HostPressure(field, clear_after_s=0.2)
    recovered = run_experiment(
        ScheduledBCTrain(schedule, tmp_path / "recovered"), learning_resources=resources
    ).summary["learning_schedule"]

    assert recovered["state"] == "completed"
    assert recovered["steps_completed"] == recovered["durable_steps_completed"] == 6
    assert recovered["pauses"] == 1 and resources.waited == pytest.approx(0.2)
    assert recovered["pressure_counts"]["resource_limit:" + field] == 2
    assert resources.closed and schedule.read_bytes() == frozen_configuration
    assert recovered["config"]["budget"] == reference["config"]["budget"]
    assert (
        recovered["learner_checkpoint"]["learner_state_sha256"]
        == reference["learner_checkpoint"]["learner_state_sha256"]
    )
    states = [
        torch.load(
            Path(result["learner_checkpoint"]["directory"]) / "learner.pt", weights_only=True
        )
        for result in (reference, recovered)
    ]
    for key in ("actor", "optimizer", "rng", "step"):
        torch.testing.assert_close(states[1][key], states[0][key], rtol=0, atol=0)
    events = [
        json.loads(line)
        for line in (tmp_path / "recovered" / recovered["event_history"]["path"])
        .read_text()
        .splitlines()
    ]
    assert [event["steps_completed"] for event in events if event["reasons"]] == [2, 2]
    reports = [
        json.loads((tmp_path / name / "candidate/report.json").read_bytes())
        for name in ("reference", "recovered")
    ]
    assert reports[0]["decisions"] == reports[1]["decisions"]


@pytest.mark.parametrize("field", ["process_private_bytes", "free_disk_bytes"])
@pytest.mark.parametrize("limit", ["wait", "zero_wait", "total"])
def test_persistent_host_pressure_seals_at_the_explicit_budget_and_resumes(tmp_path, field, limit):
    schedule, _ = resume_inputs(tmp_path)
    options = json.loads(schedule.read_bytes())
    if limit == "zero_wait":
        options["budget"]["max_wait_s"] = 0
    elif limit == "total":
        options["budget"]["max_total_s"] = 0.15
    schedule.write_text(json.dumps(options))
    reference = run_experiment(
        ScheduledBCTrain(schedule, tmp_path / "reference"), learning_resources=Resources()
    ).summary["learning_schedule"]
    resources = HostPressure(field)
    stopped = run_experiment(
        ScheduledBCTrain(schedule, tmp_path / "stopped"), learning_resources=resources
    ).summary["learning_schedule"]

    assert stopped["state"] == "stopped"
    assert stopped["stop_reason"] == (
        "total_time_limit" if limit == "total" else "resource_wait_timeout"
    )
    assert stopped["steps_completed"] == stopped["durable_steps_completed"] == 2
    assert stopped["candidate"] is None and resources.closed
    assert stopped["pressure_counts"]["resource_limit:" + field] > 0
    if limit == "zero_wait":
        assert resources.waited == 0
    else:
        assert (
            0
            < resources.waited
            <= options["budget"]["max_total_s" if limit == "total" else "max_wait_s"]
        )
    learner = Path(stopped["learner_checkpoint"]["directory"]) / "learner.pt"
    preserved = learner.read_bytes()
    resumed = run_experiment(
        ScheduledBCResume(tmp_path / "stopped", tmp_path / "resumed"),
        learning_resources=Resources(),
    ).summary["learning_schedule"]
    assert resumed["state"] == "completed"
    assert resumed["steps_completed"] == resumed["durable_steps_completed"] == 6
    assert resumed["steps_this_run"] == 4
    assert resumed["config"]["budget"] == stopped["config"]["budget"]
    assert (
        resumed["learner_checkpoint"]["learner_state_sha256"]
        == reference["learner_checkpoint"]["learner_state_sha256"]
    )
    assert learner.read_bytes() == preserved


def test_optional_diagnostic_failure_does_not_break_pressure_recovery(tmp_path, monkeypatch):
    schedule, _ = resume_inputs(tmp_path)
    reference = run_experiment(
        ScheduledBCTrain(schedule, tmp_path / "reference"), learning_resources=Resources()
    ).summary["learning_schedule"]
    open_file = Path.open

    def unavailable_diagnostic(path, mode="r", *args, **kwargs):
        if mode == "xb" and path.name in ("schedule-events.jsonl", "losses.jsonl"):
            raise OSError("optional diagnostic storage unavailable")
        return open_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unavailable_diagnostic)
    stopped = run_experiment(
        ScheduledBCTrain(schedule, tmp_path / "stopped"),
        learning_resources=HostPressure("free_disk_bytes"),
    ).summary["learning_schedule"]
    assert stopped["stop_reason"] == "resource_wait_timeout"
    assert stopped["durable_steps_completed"] == 2
    assert stopped["event_history"]["status"] == "unavailable"
    assert stopped["pressure_counts"]["resource_limit:free_disk_bytes"] > 0
    resumed = run_experiment(
        ScheduledBCResume(tmp_path / "stopped", tmp_path / "resumed"),
        learning_resources=HostPressure("free_disk_bytes", clear_after_s=0.2),
    ).summary["learning_schedule"]
    assert resumed["state"] == "completed" and resumed["pauses"] == 1
    assert resumed["steps_completed"] == resumed["durable_steps_completed"] == 6
    assert resumed["event_history"]["status"] == "unavailable"
    assert (
        resumed["learner_checkpoint"]["learner_state_sha256"]
        == reference["learner_checkpoint"]["learner_state_sha256"]
    )
    manifest = json.loads((tmp_path / "resumed/candidate/model.json").read_bytes())
    assert manifest["training"]["loss_history"]["status"] == "unavailable"


@pytest.mark.parametrize("field", ["process_private_bytes", "free_disk_bytes"])
def test_stop_request_during_host_pressure_wait_seals_the_completed_updates(tmp_path, field):
    schedule, _ = resume_inputs(tmp_path)

    class StopDuringWait(HostPressure):
        def wait(self, seconds):
            super().wait(seconds)
            (tmp_path / "stopped/stop.request").touch()

    resources = StopDuringWait(field)
    stopped = run_experiment(
        ScheduledBCTrain(schedule, tmp_path / "stopped"), learning_resources=resources
    ).summary["learning_schedule"]
    assert stopped["state"] == "stopped" and stopped["stop_reason"] == "requested_stop"
    assert stopped["steps_completed"] == stopped["durable_steps_completed"] == 2
    assert stopped["candidate"] is None and resources.closed
    assert 0 < resources.waited < stopped["config"]["budget"]["max_wait_s"]


def test_collector_failure_remains_terminal_during_host_pressure(tmp_path):
    schedule, _ = resume_inputs(tmp_path)

    class FailedCollector(HostPressure):
        def sample(self):
            sample = super().sample()
            if self.reads >= 5:
                sample["collector"]["archive_error"] = "writer_failed"
            return sample

    resources = FailedCollector("process_private_bytes")
    stopped = run_experiment(
        ScheduledBCTrain(schedule, tmp_path / "stopped"), learning_resources=resources
    ).summary["learning_schedule"]
    assert stopped["state"] == "stopped" and stopped["stop_reason"] == "collector_failed"
    assert stopped["steps_completed"] == stopped["durable_steps_completed"] == 2
    assert stopped["candidate"] is None and resources.closed and resources.waited == 0
