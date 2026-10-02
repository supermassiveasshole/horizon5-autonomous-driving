"""Growing optional update diagnostics must not accumulate in training summaries."""

import hashlib
import json
from pathlib import Path

import pytest
from test_sac_learning import warm_start

from fh5.experiment import run_experiment
from fh5.sac_learning import SACResume, SACTrain


def test_updates_are_incremental_records_with_compact_training_summary(tmp_path):
    replay = warm_start(tmp_path)
    output = tmp_path / "candidate"
    result = run_experiment(SACTrain(tmp_path / "warm", replay, output, steps=4))
    summary = result.summary["sac_learning"]
    descriptor = summary["updates"]
    assert isinstance(descriptor, dict), "Update history must not grow inside the summary"
    assert descriptor["format"] == "sac-update-jsonl-v1"
    assert descriptor["status"] == "complete"
    assert descriptor["records"] == 4
    raw = (output / descriptor["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
    records = [json.loads(line) for line in raw.splitlines()]
    assert [record["step"] for record in records] == [1, 2, 3, 4]
    assert [record["step"] for record in records if "actor_loss" in record] == [2, 4]
    report = json.loads((output / "training-report.json").read_bytes())
    assert report["updates"] == descriptor
    assert report["steps_completed"] == 4


def test_update_log_is_written_before_training_finishes(tmp_path):
    replay = warm_start(tmp_path)
    output = tmp_path / "candidate"
    observed = []

    def stop_after_observation(completed):
        if completed != 64:
            return False
        # These records exceed ordinary file buffering, not a production quota.
        raw = (output / "diagnostics/updates.jsonl").read_bytes()
        observed.extend(json.loads(line) for line in raw.splitlines()[:-1])
        assert not (output / "policy.json").exists()
        return True

    result = run_experiment(
        SACTrain(tmp_path / "warm", replay, output, steps=100),
        sac_stop_requested=stop_after_observation,
    ).summary["sac_learning"]
    assert observed and observed[0]["step"] == 1
    assert result["steps_completed"] == result["updates"]["records"] == 64


def test_explicit_large_update_budget_can_stop_and_resume_without_a_hidden_ceiling(tmp_path):
    replay = warm_start(tmp_path)
    result = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "candidate", steps=10_001),
        sac_stop_requested=lambda completed: completed == 3,
    ).summary["sac_learning"]
    assert result["steps_requested"] == 10_001
    assert result["steps_completed"] == 3
    assert result["stop_reason"] == "stop_requested"
    resumed = run_experiment(
        SACResume(tmp_path / "candidate", tmp_path / "resumed", steps=2)
    ).summary["sac_learning"]
    whole = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "whole", steps=5)
    ).summary["sac_learning"]
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]


@pytest.mark.parametrize("failure", [OSError, MemoryError])
def test_optional_journal_failure_keeps_real_updates_and_continuation(
    tmp_path, monkeypatch, failure
):
    replay = warm_start(tmp_path)
    output = tmp_path / "candidate"
    original_open = Path.open

    class FailingSink:
        def __init__(self, stream):
            self.stream = stream
            self.writes = 0

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, raw):
            self.writes += 1
            if self.writes == 2:
                raise failure("optional update sink exhausted")
            return self.stream.write(raw)

    def failed_diagnostic(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if path == output / "diagnostics/updates.jsonl" and mode == "xb":
            return FailingSink(stream)
        return stream

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", failed_diagnostic)
        trained = run_experiment(SACTrain(tmp_path / "warm", replay, output, steps=3)).summary[
            "sac_learning"
        ]
    assert trained["steps_completed"] == 3
    assert trained["updates"]["status"] == "unavailable"
    assert trained["updates"]["records"] == 1
    assert trained["updates"]["sha256"] is None
    assert "optional update sink exhausted" in trained["updates"]["error"]
    resumed = run_experiment(SACResume(output, tmp_path / "resumed", steps=2)).summary[
        "sac_learning"
    ]
    whole = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "whole", steps=5)
    ).summary["sac_learning"]
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]
