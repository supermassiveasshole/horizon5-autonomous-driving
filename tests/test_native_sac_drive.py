"""SAC uses the qualified native driving path with external devices simulated."""

import json
import socket
import time
from dataclasses import asdict

import pytest
from test_attempts import evidence, protocol
from test_bc_sac_experience import native_bc_recording
from test_evaluation import sha
from test_numeric_drive_cli import drive_config, synthetic_shadow
from test_numeric_drive_cli import eligible_model as eligible_model
from test_numeric_sac_assembly import Desktop, ExternalWorld
from test_realtime_shadow import Capture, Telemetry
from test_rewards import reward_config

from fh5.capture import CaptureEvent
from fh5.cli import main
from fh5.experiment import Packet, Record, run_experiment
from fh5.live import NEUTRAL
from fh5.numeric_drive_config import NumericDriveConfiguration
from fh5.realtime_shadow import ShadowEnvironment
from fh5.realtime_udp import UDPTelemetry
from fh5.sac import SACCriticWarmup
from fh5.sac_actions import ActionBounds
from fh5.sac_learning import SACResume, SACTrain
from fh5.sac_realtime_experience import SACRealtimePrepare


@pytest.fixture(scope="module")
def native_candidate(tmp_path_factory, eligible_model):
    root = tmp_path_factory.mktemp("native-sac-device-fixture")
    request, plan, _ = native_bc_recording(root, eligible_model)
    run_experiment(request, numeric_actor=plan.actor())
    replay = request.output_dir / "replay.json"
    warm = root / "warm"
    run_experiment(
        SACCriticWarmup(
            eligible_model,
            replay,
            sha(replay),
            warm,
            steps=2,
            bounds=ActionBounds(max_steer=0.4, max_throttle=0.25, max_brake=0.5),
        )
    )
    run_experiment(SACTrain(warm, replay, root / "candidate", steps=2))
    return root / "candidate"


def qualified_sac_config(root, candidate, seed=None, route_file=None):
    config = drive_config(root, candidate / "bc")
    data = json.loads(config.read_bytes())
    data["model"] = {"kind": "sac", "directory": str(candidate), "device": "cpu"}
    if route_file is not None:
        data["task"]["route_file"] = str(route_file)
    if seed is not None:
        data["model"]["exploration_seed"] = seed
    config.write_text(json.dumps(data))
    plan = NumericDriveConfiguration(config, root / "shadow", 2, False, mode="shadow")
    capture = Capture()
    capture.source_kind = "dxgi"  # Test fixture identity, never real shadow qualification.
    env = ShadowEnvironment(
        plan.request,
        plan.capture,
        lambda: capture,
        Telemetry(),
        Desktop(),
        plan.task,
        input_conditions=plan.bindings,
    )
    report = run_experiment(
        plan.request,
        realtime_environment=env,
        numeric_actor_factory=plan.actor,
    ).summary["realtime"]
    assert report["stop_reason"] == "time_limit", report["stop_reason"]
    assert not report["commands"] and report["proposals"]
    data["shadow"] = {"directory": str(plan.request.output_dir)}
    config.write_text(json.dumps(data))
    return config


def install_external_devices(monkeypatch, fault=None):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    telemetry = UDPTelemetry(receiver.getsockname()[1], receiver=receiver)
    world = ExternalWorld(receiver.getsockname())
    world.steer_feedback_scale = 0.4

    def driving_started():
        with world.lock:
            return any(row["command"] != asdict(NEUTRAL) for row in world.commands)

    class Signals(Desktop):
        def focused(self):
            return not (fault == "focus_lost" and driving_started())

        def stop_requested(self):
            return fault == "user_stop" and driving_started()

    def capture(target):
        camera = world.start()
        camera.source_kind = "dxgi"  # All Win32 capture is replaced at this boundary.
        read = camera.capture

        def capture_frame():
            if fault == "image_loss" and driving_started():
                return CaptureEvent(time.perf_counter_ns())
            return read()

        camera.capture = capture_frame
        return camera

    monkeypatch.setattr("fh5.dxgi_windows.WindowsDXGIFrames", capture)
    monkeypatch.setattr("fh5.numeric_drive_config.UDPTelemetry", lambda port: telemetry)
    monkeypatch.setattr("fh5.live.WindowsDesktop", Signals)
    monkeypatch.setattr("fh5.live.XboxController", lambda: world)
    monkeypatch.setattr("fh5.capture_resources.WindowsResources", lambda: None)
    return world, telemetry


@pytest.mark.parametrize("seed", [None, 37], ids=["deterministic", "exploration"])
def test_qualified_sac_cli_sends_real_commands_and_replays_exactly(
    tmp_path, native_candidate, monkeypatch, capsys, seed
):
    config = qualified_sac_config(tmp_path, native_candidate, seed)
    original = {name: sha(native_candidate / name) for name in ("policy.json", "policy.pt")}
    world, telemetry = install_external_devices(monkeypatch)
    output = tmp_path / "drive"
    try:
        code = main(
            [
                "realtime-drive",
                "--config",
                str(config),
                "--output",
                str(output),
                "--seconds",
                "1.5",
                "--live",
            ]
        )
        console = capsys.readouterr()
        assert code == 0, console.out + console.err
        report = json.loads((output / "report.json").read_bytes())
        assert report["stop_reason"] == "time_limit"
        assert report["environment"]["qualification"]["eligible"]
        assert report["model"]["command_context"] == "successful-send-return-proxy-v1"
        assert report["model"]["exploration"] is (seed is not None)
        assert not report.get("proposals")
        assert report["commands_sent_to_game"] and not report["real_game_validation"]
        assert report["resources_released"] and world.actuator_closed and world.capture_closed
        accepted = [r for r in report["decisions"] if r["status"] == "accepted"]
        assert len(accepted) >= 2
        assert report["commands"][0]["owner"] == "initial_neutral"
        sent = [row["command"] for row in world.commands]
        assert sent[-1] == asdict(NEUTRAL)
        assert any(c != asdict(NEUTRAL) for c in sent)
        for row in accepted:
            context = row["command_context"]
            assert context["sent"] == report["commands"][context["command_index"]]["sent"]
            assert context["returned_ns"] < row["decision_ns"]
        assert all(c["sent"] in sent for c in report["commands"])
        replay = tmp_path / "replay.html"
        assert (
            main(
                [
                    "realtime-replay",
                    str(output),
                    "--model",
                    str(native_candidate),
                    "--report",
                    str(replay),
                    "--tolerance",
                    "0",
                ]
            )
            == 0
        )
        proof = json.loads(replay.with_suffix(".json").read_bytes())
        assert proof["verified"] and proof["verified_predictions"] >= len(accepted)
        assert original == {name: sha(native_candidate / name) for name in original}
    finally:
        world.stop()
        telemetry.close()


@pytest.mark.parametrize("fault", ["user_stop", "focus_lost", "image_loss"])
def test_native_sac_keeps_existing_stop_and_image_lease_protection(
    tmp_path, native_candidate, monkeypatch, capsys, fault
):
    config = qualified_sac_config(tmp_path, native_candidate, seed=37)
    output = tmp_path / "drive"
    world, telemetry = install_external_devices(monkeypatch, fault)
    try:
        code = main(
            [
                "realtime-drive",
                "--config",
                str(config),
                "--output",
                str(output),
                "--seconds",
                "2",
                "--live",
            ]
        )
        console = capsys.readouterr()
        report = json.loads((output / "report.json").read_bytes())
        assert code == (0 if fault == "user_stop" else 1), console
        expected = "decision_watchdog" if fault == "image_loss" else fault
        assert report["stop_reason"] == expected
        assert report["resources_released"] and world.actuator_closed and world.capture_closed
        sent = [row["command"] for row in world.commands]
        assert any(command != asdict(NEUTRAL) for command in sent)
        assert sent[-1] == asdict(NEUTRAL)
        first_driving = next(i for i, command in enumerate(sent) if command != asdict(NEUTRAL))
        first_release = next(
            i for i in range(first_driving + 1, len(sent)) if sent[i] == asdict(NEUTRAL)
        )
        assert all(command == asdict(NEUTRAL) for command in sent[first_release:])
    finally:
        world.stop()
        telemetry.close()


@pytest.mark.parametrize("changed", ["seed", "parent_bc"])
def test_native_sac_requires_its_own_matching_shadow_before_devices(
    tmp_path, native_candidate, monkeypatch, capsys, changed
):
    if changed == "seed":
        config = qualified_sac_config(tmp_path, native_candidate, seed=37)
        data = json.loads(config.read_bytes())
        data["model"]["exploration_seed"] = 38
    else:
        config = drive_config(tmp_path, native_candidate / "bc")
        synthetic_shadow(tmp_path, native_candidate / "bc", config, native_file_fixture=True)
        data = json.loads(config.read_bytes())
        data["model"] = {"kind": "sac", "directory": str(native_candidate), "device": "cpu"}
    config.write_text(json.dumps(data))

    def unexpected_device(*args):
        pytest.fail("An ineligible candidate must not acquire devices")

    monkeypatch.setattr("fh5.dxgi_windows.WindowsDXGIFrames", unexpected_device)
    monkeypatch.setattr("fh5.realtime_udp.socket.socket", unexpected_device)
    monkeypatch.setattr("fh5.live.XboxController", unexpected_device)
    output = tmp_path / "rejected"
    assert main(["realtime-drive", "--config", str(config), "--output", str(output), "--live"]) == 2
    assert "shadow_candidate_mismatch" in capsys.readouterr().err
    assert not output.exists()


def recorded_experience(root, execution, plan):
    report = json.loads((execution / "report.json").read_bytes())
    events = [
        json.loads(line)
        for line in (execution / report["journal"]["path"]).read_bytes().splitlines()
    ]
    packets = [
        Packet(
            e["data"]["received_monotonic_ns"],
            e["data"]["received_utc"],
            bytes.fromhex(e["data"]["payload_hex"]),
        )
        for e in events
        if e["kind"] == "packet"
    ]
    config = root / "record.json"
    data = json.loads((root / "reference.json").read_bytes())
    data["control_source"] = "policy"
    config.write_text(json.dumps(data))
    recording = root / "recording"
    run_experiment(Record(config, recording, source_kind="udp"), packets=packets)
    task = protocol(root, plan.task.route_file)
    data = json.loads(task.read_bytes())
    data["control_owner"] = "policy"
    task.write_text(json.dumps(data))
    return SACRealtimePrepare(
        recording,
        execution,
        task,
        reward_config(root),
        root / "experience",
        evidence(root, recording),
    )


def test_native_sac_commands_feed_the_next_learning_update(
    tmp_path, native_candidate, monkeypatch, capsys
):
    route_file = native_candidate.parent / "route/route.json"
    config = qualified_sac_config(tmp_path, native_candidate, seed=37, route_file=route_file)
    output = tmp_path / "drive"
    plan = NumericDriveConfiguration(config, output, 1.5, True)
    before = {name: sha(native_candidate / name) for name in ("policy.json", "policy.pt")}
    world, telemetry = install_external_devices(monkeypatch)
    try:
        assert (
            main(
                [
                    "realtime-drive",
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                    "--seconds",
                    "1.5",
                    "--live",
                ]
            )
            == 0
        ), capsys.readouterr()
        request = recorded_experience(tmp_path, output, plan)
        prepared = run_experiment(request, numeric_actor=plan.actor()).summary["sac_replay"]
        assert prepared["eligible_transitions"] >= 2, prepared
        replay = request.output_dir / "replay.json"
        data = json.loads(replay.read_bytes())
        assert data["source_kind"] == "native" and not data["real_driving_validated"]
        assert data["game_application"] == "unverified"
        assert all(
            row["action_time_basis"] == "asynchronous_send_return_proxy_v1"
            for row in data["transitions"]
        )
        candidate = tmp_path / "updated"
        learned = run_experiment(
            SACResume(
                native_candidate,
                candidate,
                steps=2,
                additions=((replay, prepared["replay_sha256"]),),
            )
        ).summary["sac_learning"]
        assert learned["source_kind"] == "native" and learned["steps_completed"] == 2
        assert learned["critic_change_max"] > 0 and not learned["real_driving_validated"]
        assert before == {name: sha(native_candidate / name) for name in before}
        assert sha(candidate / "policy.pt") != before["policy.pt"]
        next_root = tmp_path / "next"
        next_root.mkdir()
        next_config = qualified_sac_config(next_root, candidate, seed=37, route_file=route_file)
        next_plan = NumericDriveConfiguration(next_config, next_root / "drive", 1.5, True)
        assert next_plan.actor().manifest["sac_manifest_sha256"] == sha(candidate / "policy.json")
    finally:
        world.stop()
        telemetry.close()
