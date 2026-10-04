"""Control behavior through the agreed experiment entry with an external game fixture."""

import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from fh5.driving.control import Command, Control, ControlInput
from fh5.experiment import Packet, Replay, run_experiment
from tests.telemetry.test_experiment import config_file, sample_packet


def control_config(tmp_path: Path) -> Path:
    path = config_file(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["control_source"] = "calibration"
    config["control"] = {
        "version": 1,
        "action_version": "steer-signed-longitudinal-v1",
        "expected_car_ordinal": 123,
        "expected_pi": 900,
        "rate_hz": 20,
        "max_speed_kmh": 30,
        "start_speed_kmh": 5,
        "max_steer": 0.25,
        "max_throttle": 0.15,
        "max_brake": 0.5,
        "telemetry_timeout_s": 0.25,
        "ready_timeout_s": 10,
        "steps": [
            {"seconds": 0.1, "steer": 0.8, "longitudinal": 0.4},
            {"seconds": 0.1, "steer": -0.8, "longitudinal": -0.6},
        ],
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class GameFixture:
    """An external clock, telemetry stream and controller transport; never a real driver."""

    source_kind = "synthetic"

    def __init__(self) -> None:
        self.time_ns = 1_000_000_000
        self.sent: list[Command] = []
        self.closed = False
        self.events = []

    def now_ns(self) -> int:
        return self.time_ns

    def read(self, period_s: float) -> ControlInput:
        self.time_ns += round(period_s * 1e9)
        data = bytearray(sample_packet())
        struct.pack_into("<I", data, 4, self.time_ns // 1_000_000)
        struct.pack_into("<f", data, 256, 0.0)
        return ControlInput(
            packets=(Packet(self.time_ns, "2026-09-28T12:00:00+00:00", bytes(data)),),
            focused=True,
        )

    def send(self, command: Command) -> None:
        self.sent.append(command)

    def close(self) -> None:
        self.closed = True


def test_control_records_limited_commands_separately_from_policy_and_game(tmp_path: Path) -> None:
    game = GameFixture()
    result = run_experiment(
        Control(control_config(tmp_path), tmp_path / "control"), environment=game
    )
    replay = run_experiment(Replay(tmp_path / "control", tmp_path / "replay.html"))
    commands = result.summary["control"]["commands"]
    positive = next(c for c in commands if c.get("requested", {}).get("longitudinal") == 0.4)
    assert positive["sent"] == {"steer_i16": 8192, "throttle_u8": 38, "brake_u8": 0}
    braking = next(c for c in commands if c.get("requested", {}).get("longitudinal") == -0.6)
    assert braking["sent"] == {"steer_i16": -8192, "throttle_u8": 0, "brake_u8": 128}
    assert all(c["owner"] in ("calibration", "stop_guard") for c in commands)
    assert commands[-1]["sent"] == {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0}
    assert game.sent[-1] == Command(0, 0, 0)
    assert game.closed
    assert result.summary["control"]["stop_reason"] == "completed"
    assert result.summary["control"] == replay.summary["control"]
    assert replay.samples[0]["telemetry_controls"]["accel"] == 128
    assert replay.samples[0]["command"] is None


def test_transport_failure_records_failed_send_and_reports_release_failure(tmp_path: Path) -> None:
    class BrokenController(GameFixture):
        def send(self, command: Command) -> None:
            if len(self.sent) >= 1:
                raise OSError("controller disconnected")
            super().send(command)

    result = run_experiment(
        Control(control_config(tmp_path), tmp_path / "broken"), environment=BrokenController()
    )
    control = result.summary["control"]
    assert control["stop_reason"] == "interface_error"
    assert control["release_sent"] is False
    assert control["commands"][-1]["status"] == "failed"
    assert control["commands"][-1]["sent"] is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("steps", []),
        ("rate_hz", 0),
        ("max_throttle", 1),
        ("ready_timeout_s", float("inf")),
        ("version", 2),
        ("expected_pi", True),
    ],
)
def test_invalid_configuration_never_sends_input(tmp_path: Path, key: str, value: object) -> None:
    path = control_config(tmp_path)
    config = json.loads(path.read_text())
    config["control"][key] = value
    path.write_text(json.dumps(config))
    game = GameFixture()
    with pytest.raises(ValueError):
        run_experiment(Control(path, tmp_path / "invalid"), environment=game)
    assert game.sent == []
    assert game.closed


@pytest.mark.parametrize(
    "fault,reason", [("stall", "control_stalled"), ("frozen", "game_time_stalled")]
)
def test_stalled_clocks_release_input(tmp_path: Path, fault: str, reason: str) -> None:
    class StalledGame(GameFixture):
        def read(self, period_s: float) -> ControlInput:
            if fault == "stall" and len(self.sent) >= 2:
                self.time_ns += 1_000_000_000
            frame = super().read(period_s)
            if fault == "frozen":
                data = bytearray(frame.packets[0].payload)
                struct.pack_into("<I", data, 4, 1000)
                frame = replace(frame, packets=(replace(frame.packets[0], payload=bytes(data)),))
            return frame

    path = control_config(tmp_path)
    config = json.loads(path.read_text())
    config["control"]["steps"][0]["seconds"] = 2
    path.write_text(json.dumps(config))
    game = StalledGame()
    result = run_experiment(Control(path, tmp_path / "stalled"), environment=game)
    assert result.summary["control"]["stop_reason"] == reason
    assert game.sent[-1] == Command(0, 0, 0)


def test_focus_wait_times_out_with_only_neutral_commands(tmp_path: Path) -> None:
    class UnfocusedGame(GameFixture):
        def read(self, period_s: float) -> ControlInput:
            return replace(super().read(period_s), focused=False)

    game = UnfocusedGame()
    result = run_experiment(
        Control(control_config(tmp_path), tmp_path / "waiting"), environment=game
    )
    assert result.summary["control"]["stop_reason"] == "ready_timeout"
    assert all(command == Command(0, 0, 0) for command in game.sent)


def test_pause_inside_one_receive_batch_stops_even_if_last_packet_is_active(tmp_path: Path) -> None:
    class BriefPause(GameFixture):
        def read(self, period_s: float) -> ControlInput:
            frame = super().read(period_s)
            if len(self.sent) < 2:
                return frame
            paused = bytearray(frame.packets[0].payload)
            struct.pack_into("<i", paused, 0, 0)
            return replace(
                frame, packets=(replace(frame.packets[0], payload=bytes(paused)), *frame.packets)
            )

    result = run_experiment(
        Control(control_config(tmp_path), tmp_path / "brief"), environment=BriefPause()
    )
    assert result.summary["control"]["stop_reason"] == "inactive"


def test_interrupted_control_journal_replays_as_incomplete(tmp_path: Path) -> None:
    run_dir = tmp_path / "partial"
    run_experiment(Control(control_config(tmp_path), run_dir), environment=GameFixture())
    # Model process death: initial metadata survived, command journal ended mid-write.
    (run_dir / "control.json").write_text(
        json.dumps({"version": 1, "stop_reason": "running", "release_sent": False})
    )
    journal = run_dir / "commands.jsonl"
    rows = journal.read_text().splitlines()
    journal.write_text("\n".join(rows[:-1]) + '\n{"issued_ns":')
    replay = run_experiment(Replay(run_dir, tmp_path / "partial.html"))
    control = replay.summary["control"]
    assert control["stop_reason"] == "incomplete"
    assert control["release_sent"] is False
    assert len(control["commands"]) == len(rows) - 1
    assert control["artifact_errors"]


@pytest.mark.parametrize("tail", ["missing_release", "empty"])
def test_whole_line_journal_loss_cannot_preserve_release_success(tmp_path: Path, tail: str) -> None:
    run_dir = tmp_path / "lost-journal"
    run_experiment(Control(control_config(tmp_path), run_dir), environment=GameFixture())
    journal = run_dir / "commands.jsonl"
    rows = journal.read_text().splitlines()
    journal.write_text("\n".join(rows[:-1]) + "\n" if tail == "missing_release" else "")
    replay = run_experiment(Replay(run_dir, tmp_path / "lost.html"))
    assert replay.summary["control"]["stop_reason"] == "incomplete"
    assert replay.summary["control"]["release_sent"] is False


@pytest.mark.parametrize("damage", ["torn", "missing"])
def test_damaged_control_summary_preserves_command_evidence(tmp_path: Path, damage: str) -> None:
    run_dir = tmp_path / "damaged-summary"
    original = run_experiment(Control(control_config(tmp_path), run_dir), environment=GameFixture())
    path = run_dir / "control.json"
    if damage == "torn":
        path.write_text('{"version": 1, "stop_reason":')
    else:
        path.unlink()
    replay = run_experiment(Replay(run_dir, tmp_path / "damaged.html"))
    assert replay.summary["control"]["stop_reason"] == "incomplete"
    assert replay.summary["control"]["release_sent"] is False
    assert replay.summary["control"]["commands"] == original.summary["control"]["commands"]


def test_observed_response_delay_uses_telemetry_not_send_return_time(tmp_path: Path) -> None:
    class ResponsiveGame(GameFixture):
        def read(self, period_s: float) -> ControlInput:
            frame = super().read(period_s)
            command = self.sent[-1]
            data = bytearray(frame.packets[0].payload)
            data[315], data[316] = command.throttle_u8, command.brake_u8
            struct.pack_into("<b", data, 320, round(command.steer_i16 / 32767 * 127))
            return replace(frame, packets=(replace(frame.packets[0], payload=bytes(data)),))

    result = run_experiment(
        Control(control_config(tmp_path), tmp_path / "responsive"), environment=ResponsiveGame()
    )
    observations = result.summary["control"]["response_observations"]
    assert any(
        o["field"] == "accel" and o["observed_value"] == 38 and o["delay_ms"] == 50
        for o in observations
    )
    assert any(
        o["field"] == "brake" and o["observed_value"] == 128 and o["delay_ms"] == 50
        for o in observations
    )
    assert result.summary["control"]["game_response_validation"] == "unverified"


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("stop", "user_stop"),
        ("focus", "focus_lost"),
        ("pause", "inactive"),
        ("stale", "telemetry_stale"),
        ("jump", "position_jump"),
        ("clock", "game_time_jump"),
        ("car", "vehicle_changed"),
        ("speed", "speed_limit"),
        ("invalid", "invalid_telemetry"),
    ],
)
def test_faults_end_control_and_release_inputs(tmp_path: Path, fault: str, reason: str) -> None:
    class FaultyGame(GameFixture):
        def read(self, period_s: float) -> ControlInput:
            frame = super().read(period_s)
            if len(self.sent) < 2:
                return frame
            if fault == "stop":
                return replace(frame, stop_requested=True)
            if fault == "focus":
                return replace(frame, focused=False)
            if fault == "stale":
                return replace(frame, packets=())
            data = bytearray(frame.packets[0].payload)
            if fault == "pause":
                struct.pack_into("<i", data, 0, 0)
            if fault == "jump":
                struct.pack_into("<f", data, 244, 1000.0)
            if fault == "clock":
                struct.pack_into("<I", data, 4, 100)
            if fault == "car":
                struct.pack_into("<i", data, 212, 124)
            if fault == "speed":
                struct.pack_into("<f", data, 256, 20.0)
            if fault == "invalid":
                struct.pack_into("<f", data, 256, float("nan"))
            return replace(frame, packets=(replace(frame.packets[0], payload=bytes(data)),))

    path = control_config(tmp_path)
    config = json.loads(path.read_text())
    config["control"]["steps"][0]["seconds"] = 2
    path.write_text(json.dumps(config))
    game = FaultyGame()
    result = run_experiment(Control(path, tmp_path / "fault"), environment=game)
    assert result.summary["control"]["stop_reason"] == reason
    assert result.summary["control"]["release_sent"] is True
    assert game.sent[-1] == Command(0, 0, 0)
    assert game.closed
