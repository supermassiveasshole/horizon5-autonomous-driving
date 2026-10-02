"""Explicit waiting budgets include the time spent obtaining resource observations."""

import json

import pytest
from test_learning_schedule import Resources, configuration

from fh5.experiment import run_experiment
from fh5.learning_schedule import ScheduledBCTrain


@pytest.mark.parametrize("pressure_at", [1, 3], ids=["admission", "before_update"])
def test_slow_healthy_observation_cannot_resume_after_wait_budget(tmp_path, pressure_at):
    class SlowRecoveryObservation(Resources):
        was_pressured = False
        delayed = False

        def sample(self):
            sample = super().sample()
            pressured = sample["collector"]["pending_bytes"] > 0
            if self.was_pressured and not pressured and not self.delayed:
                self.delayed = True
                self.time += 1.0
                # The query is slow, but every returned measurement is fresh.
                # Only elapsed waiting, not stale telemetry, must stop dispatch.
                now = self.now_ns()
                sample["observed_ns"] = now
                for field in ("heartbeat_ns", "last_poll_ns", "latest_image_source_ns"):
                    sample["collector"][field] = now
            self.was_pressured = pressured
            return sample

    config, _, _ = configuration(tmp_path, max_wait_s=1, poll_interval_s=0.1)
    resources = SlowRecoveryObservation(pressures=(pressure_at,))
    output = tmp_path / "scheduled"
    summary = run_experiment(
        ScheduledBCTrain(config, output), learning_resources=resources
    ).summary["learning_schedule"]
    assert resources.delayed and resources.closed
    assert summary["state"] == "stopped"
    assert summary["stop_reason"] == "resource_wait_timeout"
    assert summary["steps_completed"] == summary["durable_steps_completed"] == 0
    assert summary["candidate"] is None
    with (output / summary["event_history"]["path"]).open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream]
    assert events[-1]["sample"]["collector"]["pending_bytes"] == 0
    assert events[-1]["reasons"] == ["resource_wait_timeout"]
