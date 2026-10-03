"""Scheduled BC continuation through real public experiment runs and artifacts."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from fh5.experiment import run_experiment
from fh5.learning.bc.schedule import ScheduledBCTrain
from fh5.learning.bc.training import TemporalBCReplay
from tests.learning.bc.test_learning_schedule import Resources, configuration
from tests.learning.bc.test_temporal_bc import temporal_fixture


class PressureAfterTwoUpdates(Resources):
    def sample(self):
        # Admission, snapshot validation, and the first two update admissions
        # are healthy; the next public resource observation stays backlogged.
        if self.reads >= 4:
            self.permanent = "backlog"
        return super().sample()


def resume_inputs(tmp_path):
    training, dataset = temporal_fixture(tmp_path)
    document = json.loads(dataset.read_bytes())
    original = document["decisions"][0]
    rows = []
    for index in range(5):
        row = deepcopy(original)
        row["decision_id"] = f"attempt-0:{index}"
        row["decision_ns"] += index * 1_000_000_000
        row["supervision"]["action"] = [(index - 2) / 5, (index + 1) / 6]
        pixels = bytes((17 + 20 * index, 91 - 7 * index, 29 + index)) * (64 * 36)
        image = dataset.parent / f"training-{index}.rgb"
        image.write_bytes(pixels)
        for slot, frame in enumerate(row["frames"]):
            frame.update(
                frame_id=f"attempt-0:{index}:{slot}",
                path=image.name,
                sha256=hashlib.sha256(pixels).hexdigest(),
            )
            for key in ("source_time_ns", "capture_received_ns", "preprocess_ready_ns"):
                frame[key] += index * 1_000_000_000
        for actor in row["views"].values():
            actor["ego"]["speed_mps"] = 5 + 11 * index
            actor["ego"]["velocity_car_mps"][2] = 5 + 11 * index
        rows.append(row)
    document["decisions"] = rows + document["decisions"][2:]
    dataset.write_text(json.dumps(document), encoding="utf-8")
    settings = json.loads(training.read_bytes())
    settings.update(
        steps=6,
        batch_size=4,
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
    )
    training.write_text(json.dumps(settings), encoding="utf-8")
    # Reuse the public scheduler's resource fixture, keeping its independent
    # preparation data separate from the varied sampling corpus above.
    schedule_inputs = tmp_path / "schedule-inputs"
    schedule_inputs.mkdir()
    schedule, _, _ = configuration(schedule_inputs)
    options = json.loads(schedule.read_bytes())
    options.update(
        training_config=str(training),
        training_config_sha256=hashlib.sha256(training.read_bytes()).hexdigest(),
    )
    schedule.write_text(json.dumps(options), encoding="utf-8")
    return schedule, dataset


@pytest.mark.parametrize("sealed_schedule_version", [1, 2])
def test_scheduled_bc_pressure_stop_resumes_exact_remaining_updates_without_loss_journal(
    tmp_path, sealed_schedule_version
):
    torch = pytest.importorskip("torch")
    schedule, dataset = resume_inputs(tmp_path)
    options = json.loads(schedule.read_bytes())
    training_path = Path(options.pop("training_config"))
    options.pop("training_config_sha256")
    options["version"] = 2
    options["training"] = json.loads(training_path.read_bytes())
    options["training"]["dataset"] = str(dataset.resolve())
    if sealed_schedule_version == 2:
        options["training"].pop("dataset_sha256")
        options["training"]["dataset"] = dataset.relative_to(
            schedule.parent, walk_up=True
        ).as_posix()
    schedule.write_text(json.dumps(options))
    requested_schedule = schedule.read_bytes()
    original_dataset = dataset.read_bytes()
    dataset_sha256 = hashlib.sha256(original_dataset).hexdigest()
    reference = run_experiment(
        ScheduledBCTrain(schedule, tmp_path / "reference"), learning_resources=Resources()
    ).summary["learning_schedule"]

    stopped = run_experiment(
        ScheduledBCTrain(schedule, tmp_path / "stopped"),
        learning_resources=PressureAfterTwoUpdates(),
    ).summary["learning_schedule"]
    assert stopped["state"] == "stopped"
    assert stopped["stop_reason"] == "resource_wait_timeout"
    assert stopped["steps_completed"] == stopped["durable_steps_completed"] == 2
    assert stopped["candidate"] is None
    parent = stopped["learner_checkpoint"]
    parent_dir = Path(parent["directory"])
    frozen_schedule = tmp_path / "stopped/schedule-config.json"
    frozen_training = tmp_path / "stopped/training.json"
    for settings in (
        json.loads(frozen_schedule.read_bytes())["training"],
        json.loads(frozen_training.read_bytes()),
    ):
        assert settings["dataset"] == str(dataset.resolve())
        assert settings["dataset_sha256"] == dataset_sha256
    checkpoint = json.loads((parent_dir / "learner.json").read_bytes())
    assert checkpoint["dataset"] == {"path": str(dataset.resolve()), "sha256": dataset_sha256}
    assert checkpoint["model_metadata"]["dataset_sha256"] == dataset_sha256
    assert checkpoint["model_metadata"]["config"]["dataset_sha256"] == dataset_sha256
    assert schedule.read_bytes() == requested_schedule
    assert (tmp_path / "stopped/requested-schedule.json").read_bytes() == requested_schedule
    if sealed_schedule_version == 1:
        # An already sealed v1 run locates its training file relative to the
        # schedule, then resolves the dataset relative to that training file.
        legacy = json.loads(frozen_schedule.read_bytes())
        legacy.pop("training")
        legacy.update(
            version=1,
            training_config="training.json",
            training_config_sha256=hashlib.sha256(
                (tmp_path / "stopped/training.json").read_bytes()
            ).hexdigest(),
        )
        frozen_schedule.write_text(json.dumps(legacy))
        schedule.write_text("developer changed the original configuration")
    preserved = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            parent_dir / "learner.json",
            parent_dir / "learner.pt",
            tmp_path / "stopped/schedule.json",
            frozen_schedule,
            frozen_training,
            tmp_path / "stopped/requested-schedule.json",
        )
    }
    for journal in (tmp_path / "stopped").rglob("losses.jsonl"):
        journal.unlink()
    assert not list((tmp_path / "stopped").rglob("losses.jsonl"))

    from fh5.learning.bc.schedule import ScheduledBCResume

    if sealed_schedule_version == 2:
        # Even valid changes to the snapshot's JSON bytes must fail the saved
        # binding; omission applies only to the original experiment request.
        dataset.write_bytes(original_dataset + b"\n")
        try:
            with pytest.raises(ValueError, match="Frozen numerical dataset hash mismatch"):
                run_experiment(
                    ScheduledBCResume(
                        tmp_path / "stopped", tmp_path / "rejected", parent["manifest_sha256"]
                    ),
                    learning_resources=Resources(),
                )
            assert not (tmp_path / "rejected/candidate").exists()
            assert not (tmp_path / "rejected/learner/learner.json").exists()
        finally:
            dataset.write_bytes(original_dataset)

    continued = run_experiment(
        ScheduledBCResume(
            tmp_path / "stopped",
            tmp_path / "continued",
            parent["manifest_sha256"] if sealed_schedule_version == 1 else None,
        ),
        learning_resources=Resources(),
    ).summary["learning_schedule"]
    assert continued["state"] == "completed"
    assert continued["steps_completed"] == continued["durable_steps_completed"] == 6
    assert continued["steps_this_run"] == 4
    assert (
        json.loads((tmp_path / "continued/learner/learner.json").read_bytes())[
            "parent_checkpoint_sha256"
        ]
        == parent["manifest_sha256"]
    )
    assert json.loads((tmp_path / "continued/schedule-config.json").read_bytes())["version"] == 2
    assert continued["config"]["training"]["dataset_sha256"] == dataset_sha256
    assert (
        continued["learner_checkpoint"]["learner_state_sha256"]
        == reference["learner_checkpoint"]["learner_state_sha256"]
    )
    states = [
        torch.load(
            Path(summary["learner_checkpoint"]["directory"]) / "learner.pt", weights_only=True
        )
        for summary in (reference, continued)
    ]
    for key in ("actor", "optimizer", "rng", "step"):
        torch.testing.assert_close(states[1][key], states[0][key], rtol=0, atol=0)
    for path, digest in preserved.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    if sealed_schedule_version == 2:
        assert schedule.read_bytes() == requested_schedule
    assert dataset.read_bytes() == original_dataset
    reports = [
        json.loads((tmp_path / name / "candidate/report.json").read_bytes())
        for name in ("reference", "continued")
    ]
    assert reports[1]["decisions"] == reports[0]["decisions"]
    replay = run_experiment(
        TemporalBCReplay(tmp_path / "continued/candidate", dataset, tmp_path / "replayed.html")
    ).summary["temporal_bc"]
    assert replay["decisions"] == reports[0]["decisions"]


def test_bc_resume_rejects_missing_adam_history_despite_valid_artifact_bindings(tmp_path):
    torch = pytest.importorskip("torch")
    schedule, _ = resume_inputs(tmp_path)
    source = tmp_path / "stopped"
    summary = run_experiment(
        ScheduledBCTrain(schedule, source), learning_resources=PressureAfterTwoUpdates()
    ).summary["learning_schedule"]
    assert summary["durable_steps_completed"] == 2
    root = Path(summary["learner_checkpoint"]["directory"])
    saved = torch.load(root / "learner.pt", map_location="cpu", weights_only=True)
    assert saved["optimizer"]["state"]
    saved["optimizer"]["state"].clear()
    expected = rewrite_learner_artifact(torch, source, summary, saved)
    assert_resume_rejected_before_admission(source, tmp_path / "rejected", expected)


def rewrite_learner_artifact(torch, source, summary, saved):
    from fh5.learning.sac.checkpoint import state_digest

    root = Path(summary["learner_checkpoint"]["directory"])
    # Build a coherently bound malformed public artifact, so byte/hash checks
    # cannot substitute for validating that completed Adam updates still exist.
    metadata = saved.pop("metadata")
    metadata["learner_state_sha256"] = state_digest(torch, saved)
    torch.save({"metadata": metadata, **saved}, root / "learner.pt")
    manifest = {
        **metadata,
        "weights_sha256": hashlib.sha256((root / "learner.pt").read_bytes()).hexdigest(),
    }
    manifest_path = root / "learner.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    summary["learner_checkpoint"].update(
        manifest_sha256=expected, learner_state_sha256=metadata["learner_state_sha256"]
    )
    (source / "schedule.json").write_text(json.dumps(summary), encoding="utf-8")
    return expected


def assert_resume_rejected_before_admission(source, output, expected):
    from fh5.learning.bc.schedule import ScheduledBCResume

    summary = json.loads((source / "schedule.json").read_bytes())
    root = Path(summary["learner_checkpoint"]["directory"])
    manifest_path = root / "learner.json"
    before = {path: path.read_bytes() for path in (manifest_path, root / "learner.pt")}
    resources = Resources()
    with pytest.raises(ValueError, match="Adam|optimizer"):
        run_experiment(
            ScheduledBCResume(source, output, expected),
            learning_resources=resources,
        )
    assert resources.reads == 0
    assert not output.exists()
    for path, original in before.items():
        assert path.read_bytes() == original


@pytest.mark.parametrize(("option", "value"), [("lr", 0.002), ("betas", (0.8, 0.99))])
def test_bc_resume_rejects_adam_options_that_change_the_frozen_update_rule(tmp_path, option, value):
    torch = pytest.importorskip("torch")
    schedule, _ = resume_inputs(tmp_path)
    source = tmp_path / "stopped"
    summary = run_experiment(
        ScheduledBCTrain(schedule, source), learning_resources=PressureAfterTwoUpdates()
    ).summary["learning_schedule"]
    root = Path(summary["learner_checkpoint"]["directory"])
    saved = torch.load(root / "learner.pt", map_location="cpu", weights_only=True)
    saved["optimizer"]["param_groups"][0][option] = value
    expected = rewrite_learner_artifact(torch, source, summary, saved)
    assert_resume_rejected_before_admission(source, tmp_path / "rejected", expected)


@pytest.mark.parametrize(("moment", "fault"), [("exp_avg", "shape"), ("exp_avg_sq", "dtype")])
def test_bc_resume_rejects_adam_moments_incompatible_with_their_actor_parameter(
    tmp_path, moment, fault
):
    torch = pytest.importorskip("torch")
    schedule, _ = resume_inputs(tmp_path)
    source = tmp_path / "stopped"
    summary = run_experiment(
        ScheduledBCTrain(schedule, source), learning_resources=PressureAfterTwoUpdates()
    ).summary["learning_schedule"]
    root = Path(summary["learner_checkpoint"]["directory"])
    saved = torch.load(root / "learner.pt", map_location="cpu", weights_only=True)
    history = next(iter(saved["optimizer"]["state"].values()))
    history[moment] = (
        history[moment].unsqueeze(-1) if fault == "shape" else history[moment].to(torch.float64)
    )
    expected = rewrite_learner_artifact(torch, source, summary, saved)
    assert_resume_rejected_before_admission(source, tmp_path / "rejected", expected)


def test_bc_resume_rejects_a_parameter_update_count_behind_durable_progress(tmp_path):
    torch = pytest.importorskip("torch")
    schedule, _ = resume_inputs(tmp_path)
    source = tmp_path / "stopped"
    summary = run_experiment(
        ScheduledBCTrain(schedule, source), learning_resources=PressureAfterTwoUpdates()
    ).summary["learning_schedule"]
    assert summary["durable_steps_completed"] == 2
    root = Path(summary["learner_checkpoint"]["directory"])
    saved = torch.load(root / "learner.pt", map_location="cpu", weights_only=True)
    history = next(iter(saved["optimizer"]["state"].values()))
    assert history["step"].item() == 2
    history["step"].sub_(1)
    expected = rewrite_learner_artifact(torch, source, summary, saved)
    assert_resume_rejected_before_admission(source, tmp_path / "rejected", expected)


def test_bc_resume_rejects_adam_history_reassigned_between_same_shape_parameters(tmp_path):
    torch = pytest.importorskip("torch")
    schedule, _ = resume_inputs(tmp_path)
    source = tmp_path / "stopped"
    summary = run_experiment(
        ScheduledBCTrain(schedule, source), learning_resources=PressureAfterTwoUpdates()
    ).summary["learning_schedule"]
    root = Path(summary["learner_checkpoint"]["directory"])
    saved = torch.load(root / "learner.pt", map_location="cpu", weights_only=True)
    keys = list(saved["actor"])
    left, right = keys.index("encoder.11.bias"), keys.index("state.0.bias")
    assert saved["actor"][keys[left]].shape == saved["actor"][keys[right]].shape == (64,)
    keys[left], keys[right] = keys[right], keys[left]
    saved["actor"] = {key: saved["actor"][key] for key in keys}
    history = saved["optimizer"]["state"]
    history[left], history[right] = history[right], history[left]
    expected = rewrite_learner_artifact(torch, source, summary, saved)
    assert_resume_rejected_before_admission(source, tmp_path / "rejected", expected)
