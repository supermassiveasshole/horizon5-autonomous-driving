"""Continuous native adapters with CPU learning and simulated external devices."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from fh5.evaluation.candidate_store import CandidateHistory, CandidateRecord
from fh5.evaluation.native import NativeEvaluationEnvironment
from fh5.evaluation.prepare import EvaluationPrepare, EvaluationReview
from fh5.experiment import run_experiment
from fh5.learning.loop.runner import LearningContinue, LearningLoop
from tests.driving.test_numeric_drive_cli import drive_config
from tests.driving.test_numeric_drive_cli import eligible_model as eligible_model
from tests.evaluation.test_attempts import evidence
from tests.evaluation.test_evaluation import sha
from tests.evaluation.test_native_candidate_store import native_record_config
from tests.learning.sac.test_native_sac_cycle import SamplingDevices
from tests.learning.sac.test_native_sac_drive import native_candidate as native_candidate
from tests.learning.sac.test_native_sac_drive import qualified_sac_config
from tests.learning.sac.test_native_sac_evaluation import (
    Devices,
    ShortCourseWorld,
    bind_native_validity,
    sac_evaluation,
)
from tests.support.native_device_timing import native_device_timing as native_device_timing


@pytest.fixture(scope="module")
def seeded_native(tmp_path_factory, native_candidate):
    """One independently qualified default, without requiring a speed comparison."""
    root = tmp_path_factory.mktemp("native-loop-seed")
    registry = root / "usage.sqlite"
    original = sac_evaluation(root, native_candidate)
    # Freeze the same route used by the learner, adding only the automatic start
    # evidence required by evaluation. A second toy route is not interchangeable.
    evaluation_config = json.loads((root / "evaluation.json").read_bytes())
    evaluation_task = json.loads((root / "task.json").read_bytes())
    source_task = json.loads((native_candidate.parent / "task.json").read_bytes())
    source_task.update(
        version=2,
        start_mode="automatic_event_ready",
        automatic_start=evaluation_task["automatic_start"],
    )
    source_task["route_file"] = str((native_candidate.parent / source_task["route_file"]).resolve())
    source_task["automatic_start"]["event_file"] = str(original.event_config_file)
    task_file = root / "native-task.json"
    task_file.write_text(json.dumps(source_task))
    evaluation_config["task"] = {"file": str(task_file), "sha256": sha(task_file)}
    (root / "evaluation.json").write_text(json.dumps(evaluation_config))
    batch = root / "registered-batch"
    run_experiment(EvaluationPrepare(root / "evaluation.json", batch, registry))
    operation = replace(
        original,
        batch_dir=batch,
        batch_sha256=sha(batch / "batch.json"),
        registry_file=registry,
        seconds=10,
    )
    drive = root / "drive"
    drive.mkdir()
    config = qualified_sac_config(
        drive, native_candidate, route_file=batch / "route/route.json", end_margin_m=0
    )
    devices = Devices(world_factory=ShortCourseWorld)
    environment = NativeEvaluationEnvironment(
        config, menu_factory=devices.menu, driving_factory=devices.drive
    )
    try:
        result = run_experiment(operation, evaluation_environment=environment).summary[
            "evaluation_run"
        ]
        assert result["stop_reason"] == "plan_complete", result
        ledger = bind_native_validity(operation, devices)
        reviewed = run_experiment(
            EvaluationReview(batch, ledger, root / "reviewed", registry)
        ).summary["evaluation"]
        assert all(row["outcome"] == "valid_complete" for row in reviewed["attempts"]), reviewed
    finally:
        devices.cleanup()
    candidate_config = json.loads((root / "evaluation.json").read_bytes())
    for slot in candidate_config["plan"]:
        slot["id"] = "candidate-" + slot["id"]
    candidate_file = root / "candidate-evaluation.json"
    candidate_file.write_text(json.dumps(candidate_config))
    candidate_batch = root / "candidate-batch"
    run_experiment(EvaluationPrepare(candidate_file, candidate_batch, registry))
    empty = root / "empty-ledger.json"
    empty.write_text(
        json.dumps(
            {"version": 1, "batch_sha256": sha(candidate_batch / "batch.json"), "entries": []}
        )
    )
    bindings = {
        name: {
            "batch": str(batch),
            "batch_sha256": operation.batch_sha256,
            "ledger": str(path),
            "ledger_sha256": sha(path),
        }
        for name, path in (("incumbent", ledger), ("candidate", empty))
    }
    bindings["candidate"].update(
        batch=str(candidate_batch), batch_sha256=sha(candidate_batch / "batch.json")
    )
    return root, bindings, {name: str(native_candidate) for name in bindings}, registry


def test_native_loop_requires_explicit_live_opt_in_before_opening_devices(tmp_path):
    from dataclasses import asdict

    import pytest

    from fh5.driving.realtime.model import RealtimeConfig
    from fh5.observation.numeric import PixelContract

    runtime = RealtimeConfig(pixels=PixelContract(size=(64, 36)), reference_count=1)
    config = tmp_path / "loop.json"
    config.write_text(
        json.dumps(
            {
                "version": 4,
                "store": {"directory": "versions", "revision": "0" * 64},
                "registry": "usage.sqlite",
                "recording": "record.json",
                "task": "task.json",
                "reward": "reward.json",
                # These explicit budgets exceed the old arbitrary parser caps;
                # validation must reach live opt-in without opening any device.
                "rounds": 11,
                "sampling": {
                    "runtime": {**asdict(runtime), "pixels": runtime.pixels.metadata()},
                    "seconds": 601,
                    "max_updates": 1001,
                },
                "evaluation_seconds": 5,
                "seed": 191,
                "acquisition_retry": {"max_retries": 4, "delay_seconds": 6},
                "sampling_retry": {"max_retries": 4},
            }
        )
    )

    class Unopened:
        source_kind = "native"

        def sampling(self, identity):
            raise AssertionError("No acquisition before explicit opt-in")

        evaluation = sampling

        def review(self, recording):
            raise AssertionError("No recording before explicit opt-in")

        def close(self):
            return {"resources_released": True}

    with pytest.raises(ValueError, match="live opt-in"):
        run_experiment(LearningLoop(config, tmp_path / "loop"), learning_environment=Unopened())
    assert not (tmp_path / "loop").exists()


def native_loop(tmp_path, pair, protocols, *, rounds=2):
    store = tmp_path / "versions"
    saved = run_experiment(
        CandidateRecord(
            native_record_config(tmp_path, pair),
            store,
            None,
            pair[3],
        )
    ).summary["candidate_store"]
    batch = Path(pair[1]["incumbent"]["batch"])
    frozen = json.loads((batch / "batch.json").read_bytes())["config"]
    record = tmp_path / "policy-record.json"
    record.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "control_source": "policy",
                "snapshot": frozen["conditions"]["snapshot"],
            }
        )
    )
    config = tmp_path / "loop.json"
    config.write_text(
        json.dumps(
            {
                "version": 4,
                "store": {"directory": str(store), "revision": saved["revision"]},
                "registry": str(pair[3]),
                "recording": str(record),
                "task": str(protocols / "task.json"),
                "reward": str(protocols / "reward.json"),
                "rounds": rounds,
                "sampling": {"runtime": frozen["runtime"], "seconds": 1.5, "max_updates": 2},
                "evaluation_seconds": 5,
                "seed": 191,
            }
        )
    )
    drive = drive_config(tmp_path, Path(pair[2]["incumbent"]) / "bc")
    template = json.loads(drive.read_bytes())
    template["decision"] = {k: v for k, v in frozen["runtime"].items() if k != "pixels"}
    template["task"].update(
        route_file=str(batch / "route/route.json"),
        expected_route_sha256=sha(batch / "route/route.json"),
        end_margin_m=0,
    )
    drive.write_text(json.dumps(template))
    return config, drive, batch / "start/event.json", saved


class LoopDevices(SamplingDevices):
    def menu(self, event_file, plan):
        menu = super().menu(event_file, plan)
        menu.screen = "driving"
        return menu

    def drive(self, plan):
        environment = super().drive(plan)
        world = self.worlds[-1]
        stop = world.stop
        observer = plan.request.output_dir.parent / "fixture-observer.json"

        def close_observer():
            stop()
            if not observer.exists():
                observer.write_text(
                    json.dumps(
                        {
                            "scope": "Simulated unobstructed straight world; NOT FH5 validity recognition",
                            "commands": world.commands,
                            "observations": world.observations,
                        }
                    )
                )

        world.stop = close_observer
        return environment

    def review(self, recording):
        observer = recording.parent / "fixture-observer.json"
        observed = json.loads(observer.read_bytes())
        assert observed["observations"] and observed["commands"]
        self.reviews.append(recording)
        path = evidence(recording.parent, recording)
        review = json.loads(path.read_bytes())
        review["items"] = [{"id": "review", "path": observer.name, "sha256": sha(observer)}]
        path.write_text(json.dumps(review))
        return path


def learning_backend(drive, event, devices, *, reviewed=True):
    from fh5.learning.loop.native import NativeLearningEnvironment

    return NativeLearningEnvironment(
        drive,
        event,
        shadow_seconds=2,
        handoff_timeout_s=5,
        review=devices.review if reviewed else None,
        shadow_factory=devices.shadow,
        menu_factory=devices.menu,
        driving_factory=devices.drive,
    )


def test_native_loop_requalifies_updated_models_and_retains_progress(
    tmp_path, seeded_native, native_candidate
):
    config, drive, event, original = native_loop(tmp_path, seeded_native, native_candidate.parent)
    devices = LoopDevices()
    environment = learning_backend(drive, event, devices)
    try:
        operation = LearningLoop(config, tmp_path / "loop", live=True)
        result = run_experiment(operation, learning_environment=environment).summary[
            "learning_loop"
        ]
        assert result["stop_reason"] == "budget_completed", result
        assert result["scope"] == "native_development_only"
        assert result["rounds_completed"] == 2 and result["learner_updates"] == 4
        first, second = result["rounds"]
        assert first["candidate_sha256"] == second["sampling_checkpoint_sha256"]
        assert first["candidate_sha256"] != second["candidate_sha256"]
        assert result["resources_released"] and result["commands_sent_to_game"]
        assert not result["real_driving_validated"] and not result["driving_improvement_validated"]
        history = run_experiment(CandidateHistory(tmp_path / "versions")).summary["candidate_store"]
        assert history["scope"] == "native_development_only"
        assert history["explorer"]["model_sha256"] == second["candidate_sha256"]
        assert history["revision"] != original["revision"]
        assert len(devices.worlds) == 6  # Two sampling attempts and two frozen pairs.
        assert all(w.actuator_closed and w.capture_closed for w in devices.worlds)
        sampling_hashes = [
            plan.model_hash for plan in devices.plans if plan.exploration_seed is not None
        ]
        assert sampling_hashes == [first["sampling_checkpoint_sha256"], first["candidate_sha256"]]
        frozen_hashes = [plan.model_hash for plan in devices.plans if plan.exploration_seed is None]
        assert frozen_hashes == [first["candidate_sha256"]] * 2 + [second["candidate_sha256"]] * 2
    finally:
        devices.cleanup()


def test_native_loop_without_review_keeps_raw_recording_and_does_not_learn(
    tmp_path, seeded_native, native_candidate
):
    config, drive, event, saved = native_loop(
        tmp_path, seeded_native, native_candidate.parent, rounds=1
    )
    devices = LoopDevices()
    environment = learning_backend(drive, event, devices, reviewed=False)
    try:
        operation = LearningLoop(config, tmp_path / "loop", live=True)
        result = run_experiment(operation, learning_environment=environment).summary[
            "learning_loop"
        ]
        assert result["stop_reason"] == "sampling_no_eligible_experience", result
        assert (
            result["eligible_transitions"]
            == result["learner_updates"]
            == result["rounds_completed"]
            == 0
        )
        assert result["resources_released"] and result["commands_sent_to_game"]
        assert len(devices.worlds) == 1 and not devices.reviews
        assert (
            operation.output_dir / "round-000/learning/attempt-000/recording/packets.jsonl"
        ).is_file()
        history = run_experiment(CandidateHistory(tmp_path / "versions")).summary["candidate_store"]
        assert history["revision"] == saved["revision"]
    finally:
        devices.cleanup()


@pytest.mark.parametrize("resume_fault", [None, "focus_lost"])
def test_native_loop_resumes_stopped_evaluation_qualification_without_relearning(
    tmp_path, seeded_native, native_candidate, resume_fault
):
    config, drive, event, _ = native_loop(
        tmp_path, seeded_native, native_candidate.parent, rounds=1
    )
    root = tmp_path / "loop"
    devices = LoopDevices()
    environment = learning_backend(drive, event, devices)

    def shadow(plan):
        observed = devices.shadow(plan)
        if plan.exploration_seed is None:
            camera = devices.shadows[-1][1]
            capture = camera.capture

            def stop_and_capture():
                (root / "stop.request").write_text("stop during candidate qualification")
                return capture()

            camera.capture = stop_and_capture
        return observed

    environment.shadow_factory = shadow
    try:
        first = run_experiment(
            LearningLoop(config, root, live=True), learning_environment=environment
        ).summary["learning_loop"]
        assert first["learner_updates"] == 2 and first["rounds_completed"] == 0, first
        assert first["resources_released"] and len(devices.worlds) == 1
        assert len(devices.shadows) == 2 and all(camera.closed for _, camera in devices.shadows)
        assert first["rounds"][0]["evaluation_interrupted_by_stop"]
        assert not (root / "round-000/evaluation").exists()
        preparation = root / "round-000/evaluation-preparation"
        assert preparation.is_dir()
        retained = {
            path: sha(path)
            for folder in (root / "round-000/learning", preparation)
            for path in folder.rglob("*")
            if path.is_file()
        }
    finally:
        devices.cleanup()
    (root / "stop.request").unlink()

    class ResumedDevices(LoopDevices):
        def drive(self, plan):
            observed = super().drive(plan)
            if resume_fault:
                world = self.worlds[-1]

                def focused():
                    with world.lock:
                        return not any(row["command"]["throttle_u8"] for row in world.commands)

                observed.observations.desktop.focused = focused
            return observed

    fresh = ResumedDevices()
    try:
        resumed = run_experiment(
            LearningContinue(root, live=True),
            learning_environment=learning_backend(drive, event, fresh),
        ).summary["learning_loop"]
        assert resumed["stop_reason"] == (
            "evaluation_execution_stopped" if resume_fault else "budget_completed"
        ), resumed
        assert resumed["learner_updates"] == 2 and resumed["rounds_completed"] == 1
        assert resumed["latest_learner"] == first["latest_learner"]
        assert all(sha(path) == digest for path, digest in retained.items())
        assert not resumed["rounds"][0]["evaluation_interrupted_by_stop"]
        assert not resumed["rounds"][0]["evaluation_interrupted_by_resource"]
        assert len(fresh.worlds) == (1 if resume_fault else 2)
        assert all(p.exploration_seed is None for p in fresh.plans)
        assert first["stop_reason"] == "stop_requested", first
    finally:
        fresh.cleanup()


@pytest.mark.parametrize("boundary", ["learned", "reviewing_evaluation"])
def test_native_loop_recovers_sealed_child_without_repeating_sampling(
    tmp_path, seeded_native, native_candidate, boundary
):
    config, drive, event, _ = native_loop(
        tmp_path, seeded_native, native_candidate.parent, rounds=1
    )
    root = tmp_path / "loop"
    script = r"""
import json, os, sys
from contextlib import contextmanager
from pathlib import Path
from fh5.experiment import run_experiment
from fh5.learning.loop.runner import LearningLoop
from tests.support.native_device_timing import native_device_timing
from tests.learning.loop.test_native_learning_loop import LoopDevices, learning_backend
root = Path(sys.argv[2])
original_replace = os.replace
def replace(source, destination, *args, **kwargs):
    if Path(destination).resolve() == root / 'state.json':
        state = json.loads(Path(source).read_bytes())
        if state['phase'] == sys.argv[5]:
            os._exit(73)
    return original_replace(source, destination, *args, **kwargs)
os.replace = replace
devices = LoopDevices()
with contextmanager(native_device_timing.__wrapped__)():
    result = run_experiment(LearningLoop(Path(sys.argv[1]), root, live=True),
        learning_environment=learning_backend(Path(sys.argv[3]), Path(sys.argv[4]), devices))
print(json.dumps(result.summary['learning_loop']))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(config), str(root), str(drive), str(event), boundary],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        timeout=120,
    )
    assert completed.returncode == 73, completed.stdout + completed.stderr
    child = root / "round-000/learning"
    original = {path: sha(path) for path in child.rglob("*") if path.is_file()}
    candidate_hash = sha(child / "candidate-000/policy.json")
    devices = LoopDevices()
    environment = learning_backend(drive, event, devices)
    try:
        state_hash = sha(root / "state.json")
        with pytest.raises(ValueError, match="live opt-in"):
            run_experiment(LearningContinue(root), learning_environment=environment)
        assert sha(root / "state.json") == state_hash
        assert not devices.plans and not devices.menus

        # Even an otherwise valid deployment edit needs a new experiment. Reject
        # it before reconciling a sealed child or acquiring any new game lease.
        original_drive = drive.read_bytes()
        drive.write_bytes(original_drive + b"\n")
        try:
            changed = learning_backend(drive, event, devices)
            with pytest.raises(ValueError, match="Frozen native learning environment changed"):
                run_experiment(LearningContinue(root, live=True), learning_environment=changed)
            assert sha(root / "state.json") == state_hash
            assert not devices.plans and not devices.menus
            assert all(sha(path) == digest for path, digest in original.items())
        finally:
            drive.write_bytes(original_drive)

        result = run_experiment(
            LearningContinue(root, sha(root / "state.json"), live=True),
            learning_environment=environment,
        ).summary["learning_loop"]
        assert result["stop_reason"] == "budget_completed", result
        assert result["rounds_completed"] == 1 and result["learner_updates"] == 2
        assert result["latest_learner"]["sha256"] == candidate_hash
        assert result["resources_released"]
        assert all(sha(path) == digest for path, digest in original.items())
        assert all(plan.exploration_seed is None for plan in devices.plans)
        assert len(devices.worlds) == (2 if boundary == "learned" else 0)
    finally:
        devices.cleanup()
