"""Critic diagnostic growth must not bound or discard completed warm-up."""

import hashlib
import json
from pathlib import Path

import pytest

from fh5.experiment import run_experiment
from fh5.learning.sac.critic import SACCriticResume, SACCriticWarmup
from tests.learning.sac.test_critic_resume import warm_inputs
from tests.support.checkpoint_files import prediction_records


def test_critic_predictions_and_target_actions_are_one_optional_stream(tmp_path):
    model, replay, digest = warm_inputs(tmp_path)
    output = tmp_path / "warm"
    summary = run_experiment(SACCriticWarmup(model, replay, digest, output, steps=2)).summary["sac"]
    descriptor = summary["predictions"]
    assert isinstance(descriptor, dict), "Predictions must not embed the entire replay corpus"
    assert descriptor["format"] == "critic-predictions-v1"
    assert descriptor["records"] == summary["transitions"] == 2
    rows = prediction_records(output, summary)
    assert len(rows[0]["q"]) == len(rows[0]["target_action"]) == 2
    assert rows[1]["target_action"] == [0, 0], "Terminal observations cannot bootstrap"
    assert "target_actions" not in summary
    (output / descriptor["diagnostic"]["path"]).unlink()
    restored = run_experiment(SACCriticResume(output, tmp_path / "restored")).summary["sac"]
    assert restored["learner_state_sha256"] == summary["learner_state_sha256"]
    assert prediction_records(tmp_path / "restored", restored) == rows


@pytest.mark.parametrize("failure", [OSError, MemoryError])
@pytest.mark.parametrize("stage", ["open", "write", "flush"])
def test_optional_prediction_storage_cannot_discard_critic_progress(
    tmp_path, monkeypatch, failure, stage
):
    model, replay, digest = warm_inputs(tmp_path)
    output = tmp_path / "first"
    original_open = Path.open

    class FailingSink:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, raw):
            if stage == "write":
                raise failure("optional prediction storage unavailable")
            return self.stream.write(raw)

        def flush(self):
            if stage == "flush":
                raise failure("optional prediction storage unavailable")
            return self.stream.flush()

    def unavailable(path, mode="r", *args, **kwargs):
        if path == output / "diagnostics/predictions.jsonl" and mode == "xb":
            if stage == "open":
                raise failure("optional prediction storage unavailable")
            return FailingSink(original_open(path, mode, *args, **kwargs))
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", unavailable)
        trained = run_experiment(
            SACCriticWarmup(model, replay, digest, output, steps=5),
            sac_stop_requested=lambda step: step == 3,
        ).summary["sac"]
    assert trained["steps_completed"] == 3
    assert trained["predictions"]["status"] == "complete"
    assert trained["predictions"]["diagnostic"]["status"] == "unavailable"
    assert "optional prediction storage" in trained["predictions"]["diagnostic"]["error"]
    resumed = run_experiment(SACCriticResume(output, tmp_path / "resumed")).summary["sac"]
    whole = run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "whole", steps=5)
    ).summary["sac"]
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]
    assert prediction_records(tmp_path / "resumed", resumed) == prediction_records(
        tmp_path / "whole", whole
    )


def test_critic_updates_are_streamed_without_duplicate_loss_and_target_arrays(tmp_path):
    pytest.importorskip("torch")
    model, replay, digest = warm_inputs(tmp_path)
    output = tmp_path / "warm"
    summary = run_experiment(
        SACCriticWarmup(model, replay, digest, output, steps=3, batch_size=1)
    ).summary["sac"]
    descriptor = summary["updates"]
    assert isinstance(descriptor, dict), "Critic update history must not grow in the summary"
    assert descriptor["format"] == "critic-update-jsonl-v1"
    assert descriptor["status"] == "complete"
    assert descriptor["records"] == summary["steps_completed"] == 3
    raw = (output / descriptor["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert all(len(row["targets"]) == len(row["transition_ids"]) == 1 for row in rows)
    assert all(row["loss"] >= 0 for row in rows)
    assert "losses" not in summary and "target_values" not in summary
    report = json.loads((output / "training-report.json").read_bytes())
    assert report["updates"] == descriptor


def test_explicit_warmup_budget_above_old_ceiling_can_stop_and_resume(tmp_path):
    pytest.importorskip("torch")
    model, replay, digest = warm_inputs(tmp_path)
    first = run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "first", steps=10_001),
        sac_stop_requested=lambda step: step == 3,
    ).summary["sac"]
    assert first["total_steps"] == 3 and first["warmup_remaining_steps"] == 9998
    second = run_experiment(
        SACCriticResume(tmp_path / "first", tmp_path / "second", steps=2)
    ).summary["sac"]
    whole = run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "whole", steps=10_001),
        sac_stop_requested=lambda step: step == 5,
    ).summary["sac"]
    assert second["total_steps"] == 5 and second["warmup_remaining_steps"] == 9996
    assert second["phase_status"] == "warming"
    assert second["learner_state_sha256"] == whole["learner_state_sha256"]
    assert second["predictions"] == whole["predictions"]


@pytest.mark.parametrize("failure", [OSError, MemoryError])
@pytest.mark.parametrize("stage", ["open", "write", "flush"])
def test_optional_critic_log_failure_preserves_updates_and_exact_resume(
    tmp_path, monkeypatch, failure, stage
):
    pytest.importorskip("torch")
    model, replay, digest = warm_inputs(tmp_path)
    output = tmp_path / "first"
    original_open = Path.open

    class FailingSink:
        def __init__(self, stream):
            self.stream = stream
            self.writes = 0

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, raw):
            self.writes += 1
            if stage == "write" and self.writes == 2:
                raise failure("optional critic diagnostic exhausted")
            return self.stream.write(raw)

        def flush(self):
            if stage == "flush":
                raise failure("optional critic diagnostic exhausted")
            return self.stream.flush()

    def unavailable_log(path, mode="r", *args, **kwargs):
        if path == output / "diagnostics/updates.jsonl" and mode == "xb":
            if stage == "open":
                raise failure("optional critic diagnostic exhausted")
            return FailingSink(original_open(path, mode, *args, **kwargs))
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", unavailable_log)
        trained = run_experiment(
            SACCriticWarmup(model, replay, digest, output, steps=5),
            sac_stop_requested=lambda step: step == 3,
        ).summary["sac"]
    assert trained["steps_completed"] == trained["total_steps"] == 3
    assert trained["updates"]["status"] == "unavailable"
    assert trained["updates"]["sha256"] is None
    assert trained["updates"]["records"] == {"open": 0, "write": 1, "flush": 3}[stage]
    assert "optional critic diagnostic exhausted" in trained["updates"]["error"]
    resumed = run_experiment(SACCriticResume(output, tmp_path / "resumed")).summary["sac"]
    whole = run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "whole", steps=5)
    ).summary["sac"]
    assert resumed["phase_status"] == "complete"
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]
    assert resumed["predictions"] == whole["predictions"]
    assert resumed["actor_change_max"] == resumed["reload_max_abs_error"] == 0


def test_critic_records_exist_before_completion_and_are_optional_for_resume(tmp_path):
    pytest.importorskip("torch")
    model, replay, digest = warm_inputs(tmp_path)
    output = tmp_path / "first"
    observed = []

    def stop_after_observation(step):
        if step != 128:
            return False
        # Enough output to exceed normal file buffering, not a learner limit.
        raw = (output / "diagnostics/updates.jsonl").read_bytes()
        observed.extend(json.loads(line) for line in raw.splitlines()[:-1])
        assert not (output / "critic.json").exists()
        return True

    result = run_experiment(
        SACCriticWarmup(model, replay, digest, output, steps=129),
        sac_stop_requested=stop_after_observation,
    ).summary["sac"]
    assert observed and observed[0]["step"] == 1
    assert result["steps_completed"] == result["updates"]["records"] == 128
    # Optional diagnostics are not a dependency of the sealed learner.
    (output / "diagnostics/updates.jsonl").unlink()
    resumed = run_experiment(SACCriticResume(output, tmp_path / "resumed")).summary["sac"]
    whole = run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "whole", steps=129)
    ).summary["sac"]
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]
