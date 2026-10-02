"""Parent continuation retains credit after required learner input I/O fails."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from test_candidate_store import candidates as candidates
from test_critic_resume import warm_inputs
from test_evaluation import sha
from test_learning_loop import SharedBackend, loop_request
from test_learning_loop import seeded_loop as seeded_loop

from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue
from fh5.sac import SACCriticWarmup


@contextmanager
def unavailable_after_update(monkeypatch, output):
    connected, opened = sqlite3.connect, Path.open
    connections, faults = [], []

    class UnavailableIndex(sqlite3.Connection):
        unavailable = False

        def execute(self, sql, *args, **kwargs):
            if self.unavailable and sql.lstrip().upper().startswith("SELECT"):
                faults.append(True)
                raise sqlite3.OperationalError("temporary learning volume unavailable")
            return super().execute(sql, *args, **kwargs)

    def connect(database, *args, **kwargs):
        indexed = Path(database).parent.name.startswith("fh5-learning-data-")
        if indexed:
            kwargs["factory"] = UnavailableIndex
        connection = connected(database, *args, **kwargs)
        if indexed:
            connections.append(connection)
        return connection

    class Journal:
        def __init__(self, stream):
            self.stream = stream
            self.connection = connections[-1]

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, raw):
            written = self.stream.write(raw)
            self.connection.unavailable = True
            return written

    def open_file(path, mode="r", *args, **kwargs):
        stream = opened(path, mode, *args, **kwargs)
        if path == output / "diagnostics/updates.jsonl" and mode == "xb":
            return Journal(stream)
        return stream

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect)
        patch.setattr(Path, "open", open_file)
        yield
    assert faults


def test_parent_resumes_remaining_credit_after_input_failures_in_sampling_and_continuation(
    tmp_path, seeded_loop, monkeypatch
):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    root = request.output_dir
    backend = SharedBackend(seeded_loop[0])
    with unavailable_after_update(monkeypatch, root / "round-000/learning/candidate-000"):
        stopped = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert stopped["learner_updates"] == 1
    assert stopped["stop_reason"] == "sampling_training_data_unavailable", stopped.get("error")
    assert stopped["resources_released"]
    assert len(backend.leases) == 1
    assert stopped["rounds_completed"] == 0
    replay = root / "round-000/learning/attempt-000/prepared/replay.json"
    original = replay.read_bytes()
    backend = SharedBackend(seeded_loop[0])
    with unavailable_after_update(monkeypatch, root / "round-000/updates-000"):
        stopped_again = run_experiment(
            LearningContinue(root, sha(root / "state.json")), learning_environment=backend
        ).summary["learning_loop"]
    assert stopped_again["learner_updates"] == 2, stopped_again.get("error")
    assert stopped_again["stop_reason"] == "training_data_unavailable"
    assert stopped_again["resources_released"]
    assert not backend.leases
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed", resumed.get("error")
    assert resumed["rounds_completed"] == 1
    assert resumed["learner_updates"] == resumed["eligible_transitions"] == 3
    assert len(backend.leases) == 1  # Evaluation only; no duplicate sampling.
    assert replay.read_bytes() == original


def test_cli_reports_saved_but_incomplete_training(tmp_path, monkeypatch, capsys):
    from fh5.cli import main

    model, replay, digest = warm_inputs(tmp_path)
    warm = tmp_path / "warm"
    run_experiment(SACCriticWarmup(model, replay, digest, warm, steps=1))
    config = tmp_path / "sac.json"
    config.write_text(
        json.dumps({"version": 1, "warmup": str(warm), "replay": str(replay), "steps": 3})
    )
    output = tmp_path / "stopped"
    with unavailable_after_update(monkeypatch, output):
        code = main(["sac-train", "--config", str(config), "--output", str(output)])
    assert code == 4
    summary = json.loads(capsys.readouterr().out)
    assert summary["steps_completed"] == 1
    assert summary["stop_reason"] == "training_data_unavailable"
    assert (output / "policy.json").exists()
    assert (
        main(
            [
                "sac-resume",
                "--checkpoint",
                str(output),
                "--output",
                str(tmp_path / "next"),
                "--steps",
                "2",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["total_steps"] == 3
