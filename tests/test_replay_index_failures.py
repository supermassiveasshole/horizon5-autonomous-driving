"""Optional reads from replay indexes cannot discard completed learner updates."""

import sqlite3
import tempfile
from pathlib import Path

import pytest
from test_critic_resume import warm_inputs

from fh5.experiment import run_experiment
from fh5.sac import SACCriticResume, SACCriticWarmup
from fh5.sac_learning import SACResume, SACTrain


@pytest.mark.parametrize(
    "stage", ["critic_ids", "sac_ids", "sac_predictions", "sac_prediction_iteration"]
)
def test_optional_index_read_failure_preserves_updates_and_exact_continuation(
    tmp_path, monkeypatch, stage
):
    model, replay, digest = warm_inputs(tmp_path)
    output = tmp_path / "interrupted_diagnostics"
    if stage == "critic_ids":
        request = SACCriticWarmup(model, replay, digest, output, steps=5)
        continued = SACCriticResume(output, tmp_path / "resumed")
        complete = SACCriticWarmup(model, replay, digest, tmp_path / "whole", steps=5)
        result_key = "sac"
    else:
        warm = tmp_path / "warm"
        run_experiment(SACCriticWarmup(model, replay, digest, warm, steps=1))
        request = SACTrain(warm, replay, output, steps=5)
        continued = SACResume(output, tmp_path / "resumed", steps=2)
        complete = SACTrain(warm, replay, tmp_path / "whole", steps=5)
        result_key = "sac_learning"
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    connected, opened = sqlite3.connect, Path.open
    armed, faults = False, []
    prediction_stage = stage in ("sac_predictions", "sac_prediction_iteration")

    class UnavailableCursor(sqlite3.Cursor):
        fetched = 0

        def __next__(self):
            if self.fetched == 1:
                faults.append(True)
                raise sqlite3.OperationalError("optional replay index read unavailable")
            self.fetched += 1
            return super().__next__()

    class UnavailableIndex(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if armed and sql.lstrip().upper().startswith("SELECT"):
                if stage == "sac_prediction_iteration":
                    return self.cursor(factory=UnavailableCursor).execute(sql, *args, **kwargs)
                faults.append(True)
                raise sqlite3.OperationalError("optional replay index read unavailable")
            return super().execute(sql, *args, **kwargs)

    def connect(database, *args, **kwargs):
        if Path(database).parent.name.startswith("fh5-replay-document-"):
            kwargs["factory"] = UnavailableIndex
        return connected(database, *args, **kwargs)

    def open_file(path, mode="r", *args, **kwargs):
        nonlocal armed
        if prediction_stage and path == output / "diagnostics/predictions.jsonl":
            armed = True
        return opened(path, mode, *args, **kwargs)

    def stop(step):
        nonlocal armed
        if not prediction_stage:
            armed = True
        return step == 3

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "tempdir", str(temporary))
        patch.setattr(sqlite3, "connect", connect)
        patch.setattr(Path, "open", open_file)
        trained = run_experiment(request, sac_stop_requested=stop).summary[result_key]
    assert faults
    assert not list(temporary.iterdir())
    assert trained["steps_completed"] == trained["total_steps"] == 3
    diagnostic = trained["predictions" if prediction_stage else "updates"]
    assert diagnostic["status"] == "unavailable"
    assert diagnostic["sha256"] is None
    assert "optional replay index read unavailable" in diagnostic["error"]
    if stage == "sac_prediction_iteration":
        assert diagnostic["records"] == 1
    resumed = run_experiment(continued).summary[result_key]
    whole = run_experiment(complete).summary[result_key]
    assert resumed["total_steps"] == 5
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]
