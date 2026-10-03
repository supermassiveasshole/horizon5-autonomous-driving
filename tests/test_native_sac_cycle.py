"""Native sampling orchestration with real learners and external device fixtures."""

import json
import struct
from dataclasses import asdict, replace

import pytest
from test_attempts import evidence
from test_evaluation import sha
from test_event_run import verified_config
from test_native_sac_drive import native_candidate as native_candidate
from test_native_sac_evaluation import Devices
from test_numeric_drive_cli import drive_config
from test_numeric_drive_cli import eligible_model as eligible_model
from test_realtime_shadow import Capture, Desktop, Telemetry

from fh5.experiment import run_experiment
from fh5.live import NEUTRAL
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig
from fh5.realtime_shadow import ShadowEnvironment
from fh5.sac_cycle import SACRealtimeCycle
from fh5.sac_native import NativeSACSamplingEnvironment


def settings(tmp_path, policy):
    config = drive_config(tmp_path, policy / "bc")
    root = json.loads(config.read_bytes())
    task = policy.parent / "task.json"
    protocol = json.loads(task.read_bytes())
    root["task"]["route_file"] = str(task.parent / protocol["route_file"])
    config.write_text(json.dumps(root))
    menu_dir = tmp_path / "menu"
    menu_dir.mkdir()
    event = verified_config(menu_dir)
    menu = json.loads(event.read_bytes())
    menu["event_run"].update(
        expected_car_ordinal=2941, expected_pi=999, start_position_m=[0, 2, 0.2]
    )
    event.write_text(json.dumps(menu))
    record = tmp_path / "record.json"
    record.write_text(
        json.dumps({"schema_version": 1, "control_source": "policy", "snapshot": menu["snapshot"]})
    )
    request = SACRealtimeCycle(
        policy,
        record,
        task,
        policy.parent / "reward.json",
        tmp_path / "cycle",
        RealtimeConfig(pixels=PixelContract(size=(64, 36)), reference_count=1),
        seconds_per_attempt=1.5,
        cycles=2,
        max_updates_per_attempt=2,
        expected_checkpoint_sha256=sha(policy / "policy.json"),
        live=True,
    )
    return request, config, event


class SamplingDevices(Devices):
    def __init__(self):
        super().__init__()
        self.shadows, self.plans, self.reviews = [], [], []

    def shadow(self, plan):
        assert all(world.actuator_closed and world.capture_closed for world in self.worlds)
        capture = Capture()
        capture.source_kind = "dxgi"  # External fixture identity, not actual FH5 evidence.
        self.shadows.append((plan, capture))
        return ShadowEnvironment(
            plan.request,
            plan.capture,
            lambda: capture,
            Telemetry(),
            Desktop(),
            plan.task,
            input_conditions=plan.bindings,
        )

    def drive(self, plan):
        assert self.shadows[-1][1].closed
        self.plans.append(plan)
        return super().drive(plan)

    def review(self, recording):
        world = self.worlds[-1]
        assert world.actuator_closed and world.capture_closed and world.failure is None
        self.reviews.append(recording)
        observer = recording.parent / "fixture-observer.json"
        observer.write_text(
            json.dumps(
                {
                    "scope": "simulated straight world without obstacles; NOT FH5 recognition",
                    "commands": world.commands,
                    "observations": world.observations,
                }
            )
        )
        path = evidence(recording.parent, recording)
        review = json.loads(path.read_bytes())
        review["items"] = [{"id": "review", "path": observer.name, "sha256": sha(observer)}]
        path.write_text(json.dumps(review))
        return path


class ParkedAwayDevices(SamplingDevices):
    """A persistent external location which only a successful menu restart resets."""

    def __init__(self):
        super().__init__()
        self.parked_x = 3.0

    def menu(self, event_file, plan):
        menu = super().menu(event_file, plan)
        pulse = menu.pulse

        def apply(button):
            pulse(button)
            if menu.screen == "driving":
                self.parked_x = 0.0

        menu.pulse = apply
        return menu

    def shadow(self, plan):
        environment = super().shadow(plan)
        read = environment.telemetry.read

        def located(period_s):
            batch = read(period_s)
            packet = batch.packets[0]
            payload = bytearray(packet.payload)
            struct.pack_into("<f", payload, 244, self.parked_x)
            return replace(batch, packets=(replace(packet, payload=bytes(payload)),))

        environment.telemetry.read = located
        return environment

    def review(self, recording):
        path = super().review(recording)
        # Independent external scenario: after this short segment the car is
        # stopped elsewhere on the route, not magically back at its start.
        self.parked_x = 3.0
        return path


@pytest.mark.parametrize("reviewed", [True, False])
def test_two_native_attempts_requalify_updated_snapshot_before_restart(
    tmp_path, native_candidate, reviewed
):
    request, config, event = settings(tmp_path, native_candidate)
    devices = SamplingDevices()
    environment = NativeSACSamplingEnvironment(
        config,
        event,
        shadow_seconds=2,
        handoff_timeout_s=5,
        initial_operation="start_ready",
        review=devices.review if reviewed else None,
        shadow_factory=devices.shadow,
        menu_factory=devices.menu,
        driving_factory=devices.drive,
    )
    original = sha(native_candidate / "policy.pt")
    try:
        result = run_experiment(request, sac_realtime_environment=environment).summary["sac_cycle"]
        if not reviewed:
            assert result["stop_reason"] == "no_eligible_experience", result
            assert result["resources_released"] and result["commands_sent_to_game"]
            assert len(devices.worlds) == 1 and not devices.reviews
            assert result["attempts"][0]["eligible_transitions"] == 0
            assert not (request.output_dir / "candidate-000").exists()
            assert (request.output_dir / "attempt-000/recording/packets.jsonl").is_file()
            assert sha(native_candidate / "policy.pt") == original
            return
        assert result["stop_reason"] == "budget_completed", result
        assert result["source_kind"] == "native" and result["commands_sent_to_game"]
        assert result["resources_released"] and not result["real_driving_validated"]
        assert not result["default_changed"]
        assert len(devices.worlds) == len(devices.shadows) == len(devices.reviews) == 2
        first, second = result["attempts"]
        assert first["candidate_sha256"] == second["sampling_checkpoint_sha256"]
        assert first["sampling_checkpoint_sha256"] != second["sampling_checkpoint_sha256"]
        assert devices.menus[1].pulses == ["START", "X", "A", "A"]
        for index, row in enumerate(result["attempts"]):
            assert row["learner_updates"] == 2 and row["inference_reload_max_error"] == 0
            assert row["resources_released"]
            shadow_plan, capture = devices.shadows[index]
            assert capture.closed and devices.menus[index].closed
            assert shadow_plan.model_hash == row["sampling_checkpoint_sha256"]
            assert shadow_plan.exploration_seed == row["sampler_seed"]
            assert devices.plans[index].model_hash == shadow_plan.model_hash
            replay = json.loads((request.output_dir / row["replay"]).read_bytes())
            assert replay["source_kind"] == "native" and replay["transitions"]
            assert devices.worlds[index].commands[-1]["command"] == asdict(NEUTRAL)
        assert sha(native_candidate / "policy.pt") == original
    finally:
        devices.cleanup()


def test_restart_resets_external_position_before_each_candidates_shadow(tmp_path, native_candidate):
    request, config, event = settings(tmp_path, native_candidate)
    devices = ParkedAwayDevices()
    environment = NativeSACSamplingEnvironment(
        config,
        event,
        shadow_seconds=2,
        handoff_timeout_s=5,
        initial_operation="start_ready",
        review=devices.review,
        shadow_factory=devices.shadow,
        menu_factory=devices.menu,
        driving_factory=devices.drive,
    )
    try:
        result = run_experiment(request, sac_realtime_environment=environment).summary["sac_cycle"]
        assert result["stop_reason"] == "budget_completed", result
        assert len(devices.worlds) == len(devices.shadows) == 2
        assert all(row["learner_updates"] == 2 for row in result["attempts"])
        assert result["resources_released"]
    finally:
        devices.cleanup()


def test_native_cycle_requires_live_before_any_device(tmp_path, native_candidate):
    request, config, event = settings(tmp_path, native_candidate)

    def unexpected(*args):
        pytest.fail("Missing live authorization must not open even a read-only capture")

    environment = NativeSACSamplingEnvironment(
        config,
        event,
        shadow_seconds=2,
        handoff_timeout_s=5,
        shadow_factory=unexpected,
        menu_factory=unexpected,
        driving_factory=unexpected,
    )
    result = run_experiment(
        replace(request, live=False), sac_realtime_environment=environment
    ).summary["sac_cycle"]
    assert result["stop_reason"] == "interface_error"
    assert not result["attempts"] and not result["commands_sent_to_game"]
    assert "live" in result["error"].lower()


def test_failed_native_restart_retains_learning_without_opening_another_driver(
    tmp_path, native_candidate
):
    request, config, event = settings(tmp_path, native_candidate)
    devices = SamplingDevices()
    devices.fail_restart = True
    environment = NativeSACSamplingEnvironment(
        config,
        event,
        shadow_seconds=2,
        handoff_timeout_s=5,
        initial_operation="start_ready",
        review=devices.review,
        shadow_factory=devices.shadow,
        menu_factory=devices.menu,
        driving_factory=devices.drive,
    )
    try:
        result = run_experiment(request, sac_realtime_environment=environment).summary["sac_cycle"]
        assert result["stop_reason"] == "sampling_fault", result
        assert result["resources_released"]
        assert len(devices.worlds) == 1 and len(devices.shadows) == 1
        assert all(menu.closed for menu in devices.menus)
        assert result["attempts"][0]["learner_updates"] == 2
        assert result["attempts"][1]["error"]
        assert (request.output_dir / result["latest_candidate"] / "policy.pt").is_file()
        assert not (request.output_dir / "candidate-001").exists()
    finally:
        devices.cleanup()


def test_native_sampling_condition_mismatch_opens_no_capture_or_menu(tmp_path, native_candidate):
    request, config, event = settings(tmp_path, native_candidate)
    menu = json.loads(event.read_bytes())
    menu["snapshot"]["vehicle"]["value"] = "another vehicle"
    event.write_text(json.dumps(menu))
    devices = SamplingDevices()
    environment = NativeSACSamplingEnvironment(
        config,
        event,
        shadow_seconds=2,
        handoff_timeout_s=5,
        shadow_factory=devices.shadow,
        menu_factory=devices.menu,
        driving_factory=devices.drive,
    )
    result = run_experiment(request, sac_realtime_environment=environment).summary["sac_cycle"]
    assert result["stop_reason"] == "sampling_fault", result
    assert result["resources_released"]
    assert not devices.shadows and not devices.menus and not devices.worlds


@pytest.mark.parametrize("stop_source", ["file", "desktop"])
def test_stop_during_shadow_prevents_driving_and_learning(tmp_path, native_candidate, stop_source):
    request, config, event = settings(tmp_path, native_candidate)
    devices = SamplingDevices()

    def shadow(plan):
        environment = devices.shadow(plan)
        camera = devices.shadows[-1][1]
        capture = camera.capture

        def stop_and_capture():
            # External operator requests stop when the first camera frame arrives.
            if stop_source == "file":
                (request.output_dir / "stop.request").write_text(
                    "stop during passive qualification"
                )
            else:
                environment.desktop.stop_requested = lambda: True
            return capture()

        camera.capture = stop_and_capture
        return environment

    environment = NativeSACSamplingEnvironment(
        config,
        event,
        shadow_seconds=2,
        handoff_timeout_s=5,
        initial_operation="start_ready",
        shadow_factory=shadow,
        menu_factory=devices.menu,
        driving_factory=devices.drive,
    )
    result = run_experiment(request, sac_realtime_environment=environment).summary["sac_cycle"]
    assert result["stop_reason"] == "stop_requested", result
    assert result["resources_released"] and result["commands_sent_to_game"]
    assert devices.shadows and all(capture.closed for _, capture in devices.shadows)
    assert len(devices.menus) == 1 and devices.menus[0].closed
    assert not devices.worlds
    assert not (request.output_dir / "candidate-000").exists()
