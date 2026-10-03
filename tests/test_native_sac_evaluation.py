"""Frozen SAC repeated evaluation with real models and simulated external devices."""

import json
import socket
from dataclasses import asdict, replace

import pytest
from test_evaluation import sha
from test_native_evaluation import ExternalDevices, native_request
from test_native_sac_drive import native_candidate as native_candidate
from test_native_sac_drive import qualified_sac_config
from test_numeric_drive_cli import eligible_model as eligible_model
from test_numeric_sac_assembly import Desktop, ExternalWorld

from fh5.evaluation import EvaluationPrepare, EvaluationReview
from fh5.evaluation_native import NativeEvaluationEnvironment
from fh5.experiment import run_experiment
from fh5.live import NEUTRAL
from fh5.realtime_driving import NumericDrivingEnvironment
from fh5.realtime_shadow import ShadowEnvironment
from fh5.realtime_udp import UDPTelemetry


def sac_evaluation(tmp_path, candidate):
    operation = native_request(tmp_path, candidate / "bc")
    path = tmp_path / "evaluation.json"
    config = json.loads(path.read_bytes())
    config["model"] = {
        "kind": "sac",
        "directory": str(candidate),
        "manifest_sha256": sha(candidate / "policy.json"),
        "device": "cpu",
    }
    path.write_text(json.dumps(config))
    batch = tmp_path / "sac-batch"
    run_experiment(EvaluationPrepare(path, batch))
    return replace(
        operation, batch_dir=batch, batch_sha256=sha(batch / "batch.json"), live=True, seconds=1.5
    )


class Devices(ExternalDevices):
    def __init__(self, fail_restart=False):
        super().__init__(fail_restart)
        self.worlds = []

    def drive(self, plan):
        assert self.menus[-1].closed
        assert all(world.actuator_closed and world.capture_closed for world in self.worlds)
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        telemetry = UDPTelemetry(receiver.getsockname()[1], receiver=receiver)
        world = ExternalWorld(receiver.getsockname())
        world.steer_feedback_scale = 0.4
        self.worlds.append(world)
        self.telemetry.append(telemetry)

        def capture():
            camera = world.start()
            camera.source_kind = "dxgi"  # Simulated identity, never actual game qualification.
            return camera

        observations = ShadowEnvironment(
            plan.request,
            plan.capture,
            capture,
            telemetry,
            Desktop(),
            plan.task,
            input_conditions=plan.bindings,
        )
        return NumericDrivingEnvironment(observations, lambda: world, configuration=plan)

    def cleanup(self):
        for world in self.worlds:
            world.stop()
        for telemetry in self.telemetry:
            telemetry.close()


def test_native_sac_repeats_frozen_driving_with_automatic_restart(tmp_path, native_candidate):
    operation = sac_evaluation(tmp_path, native_candidate)
    folder = tmp_path / "drive"
    folder.mkdir()
    config = qualified_sac_config(
        folder, native_candidate, route_file=operation.batch_dir / "route/route.json"
    )
    original = {name: sha(native_candidate / name) for name in ("policy.json", "policy.pt")}
    devices = Devices()
    environment = NativeEvaluationEnvironment(
        config, menu_factory=devices.menu, driving_factory=devices.drive
    )
    try:
        outcome = run_experiment(operation, evaluation_environment=environment)
        result = outcome.summary["evaluation_run"]
        assert result["stop_reason"] == "plan_complete", result
        assert result["started_slots"] == ["run-0", "run-1"]
        assert result["commands_sent_to_game"] and result["resources_released"]
        assert devices.menus[1].pulses == ["START", "X", "A", "A"]
        assert len(devices.worlds) == 2 and all(menu.closed for menu in devices.menus)
        for index, world in enumerate(devices.worlds):
            assert world.actuator_closed and world.capture_closed
            sent = [row["command"] for row in world.commands]
            assert any(command != asdict(NEUTRAL) for command in sent)
            assert sent[-1] == asdict(NEUTRAL)
            report = json.loads(
                (operation.output_dir / f"attempt-{index:04d}/execution/report.json").read_bytes()
            )
            assert report["actor_kind"] == "frozen-numeric-sac-v1"
            assert report["model"]["exploration"] is False
            assert report["model"]["sac_manifest_sha256"] == original["policy.json"]
            assert report["model"]["command_context"] == "successful-send-return-proxy-v1"
        reviewed = outcome.summary["evaluation"]
        assert reviewed["execution_metrics"]["bound_runs"] == 2
        assert reviewed["verified_starts"] == 2
        assert not reviewed["automatic_promotion_allowed"]
        assert original == {name: sha(native_candidate / name) for name in original}
        # Independent re-review only reads the frozen numerical and menu records.
        before = [len(world.commands) for world in devices.worlds]
        replayed = run_experiment(
            EvaluationReview(
                operation.output_dir / "frozen",
                operation.output_dir / "ledger.json",
                tmp_path / "reviewed",
            )
        ).summary["evaluation"]
        assert replayed["execution_metrics"]["bound_runs"] == 2
        assert replayed["verified_starts"] == 2
        assert before == [len(world.commands) for world in devices.worlds]
    finally:
        devices.cleanup()


def test_native_sac_evaluation_refuses_exploration_before_menu(tmp_path, native_candidate):
    operation = sac_evaluation(tmp_path, native_candidate)
    folder = tmp_path / "drive"
    folder.mkdir()
    config = qualified_sac_config(
        folder, native_candidate, seed=37, route_file=operation.batch_dir / "route/route.json"
    )
    devices = Devices()
    environment = NativeEvaluationEnvironment(
        config, menu_factory=devices.menu, driving_factory=devices.drive
    )
    with pytest.raises(ValueError, match="deterministic"):
        run_experiment(operation, evaluation_environment=environment)
    assert not devices.menus and not devices.worlds and not operation.output_dir.exists()


def test_native_sac_restart_failure_keeps_first_attempt(tmp_path, native_candidate):
    operation = sac_evaluation(tmp_path, native_candidate)
    folder = tmp_path / "drive"
    folder.mkdir()
    config = qualified_sac_config(
        folder, native_candidate, route_file=operation.batch_dir / "route/route.json"
    )
    devices = Devices(fail_restart=True)
    environment = NativeEvaluationEnvironment(
        config, menu_factory=devices.menu, driving_factory=devices.drive
    )
    try:
        outcome = run_experiment(operation, evaluation_environment=environment)
        result = outcome.summary["evaluation_run"]
        assert result["stop_reason"] == "ready_unconfirmed", result
        assert result["started_slots"] == ["run-0"]
        assert result["unstarted_slots"] == ["run-1"]
        assert result["resources_released"] and all(menu.closed for menu in devices.menus)
        assert len(devices.worlds) == 1
        world = devices.worlds[0]
        assert world.actuator_closed and world.capture_closed
        assert world.commands[-1]["command"] == asdict(NEUTRAL)
        assert outcome.summary["evaluation"]["metrics"]["all_attempts"] == 1
        assert not outcome.summary["evaluation"]["automatic_promotion_allowed"]
    finally:
        devices.cleanup()


def test_native_sac_batch_refuses_unsupported_device(tmp_path, native_candidate):
    sac_evaluation(tmp_path, native_candidate)
    config = tmp_path / "evaluation.json"
    value = json.loads(config.read_bytes())
    value["model"]["device"] = "cuda"
    config.write_text(json.dumps(value))
    output = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="CPU|cpu"):
        run_experiment(EvaluationPrepare(config, output))
    assert not output.exists()
