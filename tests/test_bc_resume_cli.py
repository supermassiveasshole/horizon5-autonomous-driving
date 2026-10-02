"""Resume a sealed learner through the CLI without repeating completed updates."""

import hashlib
import json

from test_learning_schedule import Resources, configuration

from fh5.cli import main
from fh5.experiment import run_experiment
from fh5.learning_schedule import ScheduledBCTrain


def test_cli_completed_bc_checkpoint_only_repeats_verification(tmp_path, monkeypatch, capsys):
    config, _, _ = configuration(tmp_path)
    parent = tmp_path / "parent"
    completed = run_experiment(
        ScheduledBCTrain(config, parent), learning_resources=Resources()
    ).summary["learning_schedule"]
    checkpoint = completed["learner_checkpoint"]
    parent_files = {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.rglob("*") if p.is_file()
    }

    # Substitute only the external resource probe, not the learner or dispatcher.
    import fh5.learning_resources

    monkeypatch.setattr(
        fh5.learning_resources, "NativeLearningResources", lambda *a, **kw: Resources()
    )
    output = tmp_path / "continued"
    result = main(
        [
            "collection-bc-resume",
            "--run",
            str(parent),
            "--output",
            str(output),
            "--checkpoint-sha256",
            checkpoint["manifest_sha256"],
        ]
    )
    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["state"] == "completed"
    assert summary["steps_this_run"] == 0
    assert summary["steps_completed"] == summary["durable_steps_completed"] == 3
    assert (
        summary["learner_checkpoint"]["learner_state_sha256"] == checkpoint["learner_state_sha256"]
    )
    assert (output / "candidate/report.json").is_file()
    for path, digest in parent_files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
