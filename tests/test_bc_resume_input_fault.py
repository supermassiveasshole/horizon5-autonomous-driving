"""A pixel read failure must preserve the pre-sampling learner boundary."""

import json
from pathlib import Path

import pytest
from test_bc_resume_resources import resume_inputs
from test_learning_schedule import Resources

from fh5.experiment import run_experiment
from fh5.learning_schedule import ScheduledBCTrain


@pytest.mark.parametrize(
    "error_type",
    [OSError, MemoryError, RuntimeError, pytest.importorskip("torch").OutOfMemoryError],
)
@pytest.mark.parametrize("fault_location", ["pixels", "resource_probe"])
def test_preupdate_io_failure_resumes_the_same_unconsumed_random_batch(
    tmp_path, monkeypatch, error_type, fault_location
):
    schedule, dataset = resume_inputs(tmp_path)
    reference = run_experiment(
        ScheduledBCTrain(schedule, tmp_path / "reference"), learning_resources=Resources()
    ).summary["learning_schedule"]
    armed, failures = False, 0
    open_file = Path.open

    class InputUnavailableAfterTwoUpdates(Resources):
        def sample(self):
            nonlocal armed, failures
            if self.reads >= 4 and not failures:
                if fault_location == "resource_probe":
                    failures += 1
                    raise error_type("resource probe temporarily unavailable")
                armed = True
            return super().sample()

    def fail_selected_pixel(path, mode="r", *args, **kwargs):
        nonlocal armed, failures
        if armed and path.parent == dataset.parent and path.suffix == ".rgb" and mode == "rb":
            armed = False
            failures += 1
            raise error_type("selected pixel temporarily unavailable")
        return open_file(path, mode, *args, **kwargs)

    parent = tmp_path / "stopped"
    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", fail_selected_pixel)
        with pytest.raises(error_type, match="temporarily unavailable"):
            run_experiment(
                ScheduledBCTrain(schedule, parent),
                learning_resources=InputUnavailableAfterTwoUpdates(),
            )
    assert failures == 1
    stopped = json.loads((parent / "schedule.json").read_bytes())
    assert stopped["steps_completed"] == stopped["durable_steps_completed"] == 2
    assert stopped["candidate"] is None
    checkpoint = stopped["learner_checkpoint"]

    from fh5.learning_schedule import ScheduledBCResume

    resumed = run_experiment(
        ScheduledBCResume(parent, tmp_path / "resumed", checkpoint["manifest_sha256"]),
        learning_resources=Resources(),
    ).summary["learning_schedule"]
    assert resumed["steps_this_run"] == 4
    assert resumed["steps_completed"] == resumed["durable_steps_completed"] == 6
    assert (
        resumed["learner_checkpoint"]["learner_state_sha256"]
        == reference["learner_checkpoint"]["learner_state_sha256"]
    )
    reports = [
        json.loads((tmp_path / name / "candidate/report.json").read_bytes())
        for name in ("reference", "resumed")
    ]
    assert reports[0]["decisions"] == reports[1]["decisions"]
