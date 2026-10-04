"""Candidate publication failure must not discard a completed CPU learner."""

import hashlib
import json
from pathlib import Path

import pytest

from fh5.experiment import run_experiment
from fh5.learning.bc.schedule import ScheduledBCTrain
from tests.learning.bc.test_learning_schedule import Resources, configuration


def test_candidate_rename_failure_resumes_verification_without_retraining(tmp_path, monkeypatch):
    config, _, _ = configuration(tmp_path)
    parent = tmp_path / "parent"
    rename = Path.rename

    def fail_publication(path, target):
        if path == parent / ".candidate":
            raise OSError("candidate publication unavailable")
        return rename(path, target)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "rename", fail_publication)
        with pytest.raises(OSError, match="candidate publication unavailable"):
            run_experiment(ScheduledBCTrain(config, parent), learning_resources=Resources())
    stopped = json.loads((parent / "schedule.json").read_bytes())
    assert stopped["state"] == "stopped"
    assert stopped["candidate"] is None
    assert stopped["steps_completed"] == stopped["durable_steps_completed"] == 3
    checkpoint = stopped["learner_checkpoint"]
    preserved = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in parent.rglob("*")
        if path.is_file()
    }

    from fh5.learning.bc.schedule import ScheduledBCResume

    result = run_experiment(
        ScheduledBCResume(parent, tmp_path / "continued", checkpoint["manifest_sha256"]),
        learning_resources=Resources(),
    ).summary["learning_schedule"]
    assert result["state"] == "completed"
    assert result["steps_this_run"] == 0
    assert result["steps_completed"] == result["durable_steps_completed"] == 3
    assert (
        result["learner_checkpoint"]["learner_state_sha256"] == checkpoint["learner_state_sha256"]
    )
    assert (tmp_path / "continued/candidate/report.json").is_file()
    for path, digest in preserved.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("report_state", ["missing", "truncated"])
def test_published_bc_learner_resumes_without_a_readable_schedule_report(
    tmp_path, monkeypatch, report_state
):
    config, _, _ = configuration(tmp_path)
    parent = tmp_path / "parent"
    open_file = Path.open

    def fail_report(path, mode="r", *args, **kwargs):
        if path == parent / "schedule.json" and mode == "xb":
            raise OSError("schedule report unavailable")
        return open_file(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", fail_report)
        completed = run_experiment(ScheduledBCTrain(config, parent), learning_resources=Resources())
    assert completed.summary["learning_schedule"]["state"] == "completed"
    assert completed.summary["learning_schedule"]["schedule_report"]["status"] == "unavailable"
    assert completed.report_path == parent / "learner/learner.json"
    manifest_path = parent / "learner/learner.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["steps_completed"] == 3
    assert (parent / "candidate").is_dir()
    if report_state == "truncated":
        (parent / "schedule.json").write_bytes(b'{"unfinished":')
    preserved = {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.rglob("*") if p.is_file()
    }

    from fh5.learning.bc.schedule import ScheduledBCResume

    continued = run_experiment(
        ScheduledBCResume(parent, tmp_path / "continued"),
        learning_resources=Resources(),
    ).summary["learning_schedule"]
    assert continued["state"] == "completed"
    assert continued["steps_this_run"] == 0
    assert (
        continued["learner_checkpoint"]["learner_state_sha256"] == manifest["learner_state_sha256"]
    )
    for path, digest in preserved.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
