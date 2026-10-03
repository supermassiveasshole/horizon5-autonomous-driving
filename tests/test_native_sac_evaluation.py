"""Frozen SAC repeated evaluation with real models and simulated external devices."""

import json
import socket
import struct
from dataclasses import asdict, replace

import pytest
from native_device_timing import native_device_timing as native_device_timing
from test_attempts import evidence
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
    def __init__(self, fail_restart=False, world_factory=ExternalWorld):
        super().__init__(fail_restart)
        self.worlds = []
        self.world_factory = world_factory

    def drive(self, plan):
        assert self.menus[-1].closed
        assert all(world.actuator_closed and world.capture_closed for world in self.worlds)
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        telemetry = UDPTelemetry(receiver.getsockname()[1], receiver=receiver)
        world = self.world_factory(receiver.getsockname())
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


def bind_native_validity(operation, devices):
    """Bind the external world's raw observations, never execution outcome labels."""
    ledger_file = operation.output_dir / "ledger.json"
    ledger = json.loads(ledger_file.read_bytes())
    assert len(ledger["entries"]) == len(devices.worlds)
    for index, (entry, world) in enumerate(zip(ledger["entries"], devices.worlds)):
        assert world.actuator_closed and world.capture_closed and world.failure is None
        root = operation.output_dir / f"observer-{index:04d}"
        root.mkdir()
        observer = root / "observations.json"
        observer.write_text(
            json.dumps(
                {
                    "scope": "Simulated straight unobstructed world; NOT FH5 recognition",
                    "commands": world.commands,
                    "observations": world.observations,
                }
            )
        )
        proof = evidence(root, operation.output_dir / entry["recording"])
        reviewed = json.loads(proof.read_bytes())
        reviewed["items"] = [{"id": "review", "path": observer.name, "sha256": sha(observer)}]
        proof.write_text(json.dumps(reviewed))
        entry["evidence"] = {"file": str(proof), "sha256": sha(proof)}
    ledger_file.write_text(json.dumps(ledger))
    return ledger_file


class ShortCourseWorld(ExternalWorld):
    def speed_for(self, command):
        # This endpoint test uses a prompt, bounded external response, not FH5
        # physics. Keep the full 3 m task and actual UDP/command feedback; avoid
        # making endpoint coverage depend on ten seconds of host scheduling.
        return min(3.5, command.throttle_u8 / 255 * 20)


@pytest.mark.parametrize("end_margin_m", [0, 0.5], ids=["endpoint", "early-stop"])
def test_native_completion_requires_recorded_endpoint_not_early_stop(
    tmp_path, native_candidate, end_margin_m
):
    operation = replace(sac_evaluation(tmp_path, native_candidate), seconds=20)
    folder = tmp_path / "drive"
    folder.mkdir()
    config = qualified_sac_config(
        folder,
        native_candidate,
        route_file=operation.batch_dir / "route/route.json",
        end_margin_m=end_margin_m,
    )
    original = sha(native_candidate / "policy.pt")
    devices = Devices(world_factory=ShortCourseWorld)
    environment = NativeEvaluationEnvironment(
        config, menu_factory=devices.menu, driving_factory=devices.drive
    )
    try:
        outcome = run_experiment(operation, evaluation_environment=environment)
        result = outcome.summary["evaluation_run"]
        assert result["stop_reason"] == "plan_complete", result
        assert result["resources_released"]
        assert all(row["stop_reason"] == "local_end" for row in result["attempts"])
        for world in devices.worlds:
            assert world.commands[-1]["command"] == asdict(NEUTRAL)
        ledger = bind_native_validity(operation, devices)
        for entry in json.loads(ledger.read_bytes())["entries"]:
            packets = operation.output_dir / entry["recording"] / "packets.jsonl"
            positions = [
                struct.unpack_from("<f", bytes.fromhex(json.loads(line)["payload_hex"]), 244)[0]
                for line in packets.read_text().splitlines()
            ]
            if end_margin_m == 0:
                assert max(positions) >= 3
            else:
                assert 2.5 <= max(positions) < 3
        reviewed = run_experiment(
            EvaluationReview(operation.output_dir / "frozen", ledger, tmp_path / "reviewed")
        ).summary["evaluation"]
        assert reviewed["verified_starts"] == 2
        assert reviewed["execution_metrics"]["bound_runs"] == 2
        assert not reviewed["automatic_promotion_allowed"]
        for attempt in reviewed["attempts"]:
            if end_margin_m == 0:
                assert attempt["outcome"] == "valid_complete", attempt
                assert attempt["confirmed_progress_m"] == 3
                assert attempt["valid_duration_s"] > 0
            else:
                assert attempt["outcome"] == "driving_failed", attempt
                assert "task_not_completed" in attempt["reasons"]
                assert attempt["confirmed_progress_m"] < 3
        assert sha(native_candidate / "policy.pt") == original
    finally:
        devices.cleanup()


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
