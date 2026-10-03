"""Growing critic inputs remain on disk behind the public experiment boundary."""

import gc
import hashlib
import json
import sqlite3
import tempfile
import tracemalloc
from copy import deepcopy
from pathlib import Path

import pytest

from fh5.cli import main
from fh5.experiment import run_experiment
from fh5.learning.sac.critic import SACCriticReplay, SACCriticResume, SACCriticWarmup
from tests.learning.sac.test_critic_resume import warm_inputs
from tests.support.checkpoint_files import update_records


def test_critic_observation_metadata_is_not_retained_as_a_resident_corpus(tmp_path):
    model, replay, _ = warm_inputs(tmp_path)
    template = json.loads(replay.read_bytes())
    retained, states = [], []
    count, padding_size = 128, 64 * 1024
    for name, padding in (("small", 0), ("large", padding_size)):
        document = deepcopy(template)
        terminal = document["transitions"][-1]
        document["transitions"] = []
        for index in range(count):
            row = deepcopy(terminal)
            row["id"] = f"terminal-{index}"
            row["current"]["decision_id"] = f"observation-{index}"
            row["current"]["diagnostic"] = row["id"] + ":" + "x" * padding
            document["transitions"].append(row)
        path = replay.with_name(name + ".json")
        path.write_text(json.dumps(document), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        del document, row, terminal
        gc.collect()

        def observe(step):
            if step == 0:
                retained.append(tracemalloc.get_traced_memory()[0])
            return False

        tracemalloc.start()
        try:
            result = run_experiment(
                SACCriticWarmup(model, path, digest, tmp_path / name, steps=1, batch_size=2),
                sac_stop_requested=observe,
            ).summary["sac"]
        finally:
            tracemalloc.stop()
        assert result["steps_completed"] == 1 and result["transitions"] == count
        states.append(result["learner_state_sha256"])
    assert states[0] == states[1], "Diagnostic metadata cannot change numerical learning"
    # The added corpus is 8 MiB. A per-record implementation need not retain
    # even half of it. This is a regression contrast, not a production quota.
    assert retained[1] - retained[0] < count * padding_size // 2, retained


@pytest.mark.parametrize("failure", [sqlite3.OperationalError, MemoryError])
def test_critic_input_failure_seals_completed_updates_and_retries_the_same_sample(
    tmp_path, monkeypatch, failure
):
    model, replay, digest = warm_inputs(tmp_path)
    connected = sqlite3.connect
    armed, faults = False, []
    temporary = tmp_path / "temporary"
    temporary.mkdir()

    class UnavailableIndex(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if armed and sql.lstrip().upper().startswith("SELECT"):
                faults.append(True)
                raise failure("critic input unavailable")
            return super().execute(sql, *args, **kwargs)

    def connect(database, *args, **kwargs):
        if Path(database).parent.name.startswith("fh5-learning-data-"):
            kwargs["factory"] = UnavailableIndex
        return connected(database, *args, **kwargs)

    def stop(step):
        nonlocal armed
        armed = step == 3
        return False

    output = tmp_path / "stopped"
    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "tempdir", str(temporary))
        patch.setattr(sqlite3, "connect", connect)
        trained = run_experiment(
            SACCriticWarmup(model, replay, digest, output, steps=5, batch_size=1),
            sac_stop_requested=stop,
        ).summary["sac"]
    assert faults and not list(temporary.iterdir())
    assert trained["steps_completed"] == trained["total_steps"] == 3
    assert trained["warmup_remaining_steps"] == 2
    assert trained["stop_reason"] == "training_data_unavailable"
    assert "critic input unavailable" in trained["training_error"]
    assert trained["predictions"]["status"] == "unavailable"
    assert trained["q_change_max"] is None
    resumed = run_experiment(SACCriticResume(output, tmp_path / "resumed")).summary["sac"]
    whole = run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "whole", steps=5, batch_size=1)
    ).summary["sac"]
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]
    assert update_records(output) + update_records(tmp_path / "resumed") == update_records(
        tmp_path / "whole"
    )


def test_critic_cli_reports_recoverable_input_stop_instead_of_budget_success(
    tmp_path, monkeypatch, capsys
):
    model, replay, digest = warm_inputs(tmp_path)
    connected, opened = sqlite3.connect, Path.open
    armed = False
    output = tmp_path / "stopped"

    class UnavailableIndex(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if armed and sql.lstrip().upper().startswith("SELECT"):
                raise sqlite3.OperationalError("critic input unavailable")
            return super().execute(sql, *args, **kwargs)

    def connect(database, *args, **kwargs):
        if Path(database).parent.name.startswith("fh5-learning-data-"):
            kwargs["factory"] = UnavailableIndex
        return connected(database, *args, **kwargs)

    def open_file(path, mode="r", *args, **kwargs):
        nonlocal armed
        if path == output / "diagnostics/updates.jsonl" and mode == "xb":
            armed = True
        return opened(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect)
        patch.setattr(Path, "open", open_file)
        code = main(
            [
                "sac-warmup",
                "--model",
                str(model),
                "--replay",
                str(replay),
                "--replay-sha256",
                digest,
                "--output",
                str(output),
                "--steps",
                "2",
            ]
        )
    assert code == 4
    summary = json.loads(capsys.readouterr().out)
    assert summary["stop_reason"] == "training_data_unavailable"
    assert summary["steps_completed"] == 0 and summary["warmup_remaining_steps"] == 2
    resumed = run_experiment(SACCriticResume(output, tmp_path / "resumed")).summary["sac"]
    assert resumed["total_steps"] == 2 and resumed["phase_status"] == "complete"


def test_frozen_critic_replay_cannot_claim_success_without_numerical_predictions(
    tmp_path, monkeypatch
):
    model, replay, digest = warm_inputs(tmp_path)
    checkpoint = tmp_path / "warm"
    run_experiment(SACCriticWarmup(model, replay, digest, checkpoint, steps=1))
    report = tmp_path / "replay.html"
    connected, opened = sqlite3.connect, Path.open
    armed = False

    class UnavailableIndex(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if armed and sql.lstrip().upper().startswith("SELECT"):
                raise sqlite3.OperationalError("critic verification input unavailable")
            return super().execute(sql, *args, **kwargs)

    def connect(database, *args, **kwargs):
        if Path(database).parent.name.startswith("fh5-learning-data-"):
            kwargs["factory"] = UnavailableIndex
        return connected(database, *args, **kwargs)

    def open_file(path, mode="r", *args, **kwargs):
        nonlocal armed
        if path == report.with_suffix(".predictions.jsonl") and mode == "xb":
            armed = True
        return opened(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect)
        patch.setattr(Path, "open", open_file)
        with pytest.raises(OSError, match="critic verification input unavailable"):
            run_experiment(SACCriticReplay(checkpoint, replay, report))
    assert not report.exists()
