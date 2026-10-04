"""Failed learner publication preserves the last durable parent generation."""

import hashlib
import json
from pathlib import Path

import pytest

from fh5.experiment import run_experiment
from fh5.learning.bc.schedule import ScheduledBCResume, ScheduledBCTrain
from tests.learning.bc.test_bc_resume_resources import PressureAfterTwoUpdates, resume_inputs
from tests.learning.bc.test_learning_schedule import Resources


@pytest.mark.parametrize("fault_kind", ["weights_write", "manifest_replace"])
def test_failed_bc_checkpoint_reports_durable_parent_and_can_retry(
    tmp_path, monkeypatch, fault_kind
):
    schedule, _ = resume_inputs(tmp_path)
    parent = tmp_path / "parent"
    stopped = run_experiment(
        ScheduledBCTrain(schedule, parent), learning_resources=PressureAfterTwoUpdates()
    ).summary["learning_schedule"]
    checkpoint = stopped["learner_checkpoint"]
    preserved = {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.rglob("*") if p.is_file()
    }
    child = tmp_path / "failed-child"
    open_file, replace = Path.open, Path.replace

    def fail_weights(path, mode="r", *args, **kwargs):
        if path == child / "learner/learner.pt" and mode == "xb":
            raise OSError("learner storage unavailable")
        return open_file(path, mode, *args, **kwargs)

    def fail_manifest(path, target):
        if Path(target) == child / "learner/learner.json":
            raise OSError("learner storage unavailable")
        return replace(path, target)

    with monkeypatch.context() as fault:
        if fault_kind == "weights_write":
            fault.setattr(Path, "open", fail_weights)
        else:
            fault.setattr(Path, "replace", fail_manifest)
        with pytest.raises(OSError, match="learner storage unavailable"):
            run_experiment(
                ScheduledBCResume(parent, child, checkpoint["manifest_sha256"]),
                learning_resources=Resources(),
            )
    failed = json.loads((child / "schedule.json").read_bytes())
    assert failed["steps_completed"] == 6
    assert failed["durable_steps_completed"] == 2
    assert failed["learner_checkpoint"] == checkpoint
    assert failed["candidate"] is None
    assert not (child / "learner/learner.json").exists()
    # Retry through the failed run's public descriptor. Its incomplete child
    # must never replace the published parent as the recovery source.
    manifest = Path(checkpoint["directory"]) / "learner.json"
    original = manifest.read_bytes()
    manifest.write_bytes(original + b"\n")
    try:
        with pytest.raises(ValueError):
            run_experiment(
                ScheduledBCResume(child, tmp_path / "rejected"), learning_resources=Resources()
            )
        assert not (tmp_path / "rejected").exists()
    finally:
        manifest.write_bytes(original)
    retried = run_experiment(
        ScheduledBCResume(child, tmp_path / "retried"),
        learning_resources=Resources(),
    ).summary["learning_schedule"]
    assert retried["steps_this_run"] == 4
    assert retried["steps_completed"] == retried["durable_steps_completed"] == 6
    assert retried["state"] == "completed"
    for path, digest in preserved.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
