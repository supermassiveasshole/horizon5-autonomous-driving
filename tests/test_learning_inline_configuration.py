"""A single collection-aware training file owns the experiment's settings."""

import hashlib
import json

from test_learning_schedule import Resources, configuration

from fh5.cli import main
from fh5.experiment import run_experiment
from fh5.learning_schedule import ScheduledBCTrain


def test_one_configuration_trains_without_a_separate_training_file(tmp_path, monkeypatch, capsys):
    schedule, training, dataset = configuration(tmp_path)
    options = json.loads(schedule.read_bytes())
    options["version"] = 2
    options["training"] = json.loads(training.read_bytes())
    options["training"]["dataset"] = dataset.relative_to(schedule.parent).as_posix()
    del options["training_config"]
    del options["training_config_sha256"]
    schedule.write_text(json.dumps(options))
    training.unlink()
    original = schedule.read_bytes()
    resources = Resources()
    monkeypatch.setattr(
        "fh5.learning_resources.NativeLearningResources", lambda *a, **kw: resources
    )
    output = tmp_path / "trained"

    assert main(["collection-bc-train", "--config", str(schedule), "--output", str(output)]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "completed"
    assert result["steps_completed"] == result["durable_steps_completed"] == 3
    assert result["candidate"] == "candidate"
    assert resources.closed and not result["commands_sent"]
    frozen = json.loads((output / "schedule-config.json").read_bytes())
    assert frozen["version"] == 2
    assert frozen["training"]["dataset"] == str(dataset.resolve())
    assert not {"training_config", "training_config_sha256"} & frozen.keys()
    assert not (output / "requested-training.json").exists()
    assert (output / "learner/learner.json").is_file()
    assert (output / "candidate/actor.pt").is_file()
    assert schedule.read_bytes() == original


def test_legacy_training_paths_are_relative_to_the_training_file(tmp_path):
    from test_training_config_resources import (
        InterruptedResources,
        schedule_config,
        training_config,
    )

    folder = tmp_path / "old-training"
    folder.mkdir()
    training = training_config(folder)
    values = json.loads(training.read_bytes())
    values["dataset"] = "../snapshot/dataset.json"
    training.write_text(json.dumps(values))
    schedule = schedule_config(tmp_path)
    options = json.loads(schedule.read_bytes())
    options["training_config"] = "old-training/train.json"
    options["training_config_sha256"] = hashlib.sha256(training.read_bytes()).hexdigest()
    schedule.write_text(json.dumps(options))
    output = tmp_path / "interrupted"

    result = run_experiment(
        ScheduledBCTrain(schedule, output), learning_resources=InterruptedResources()
    ).summary["learning_schedule"]

    assert result["stop_reason"] == "interrupted"
    assert result["config"]["version"] == 2
    assert result["config"]["training"]["dataset"] == str(
        (tmp_path / "snapshot/dataset.json").resolve()
    )
    assert result["steps_completed"] == 0
