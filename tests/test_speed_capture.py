"""Bounded collection assistance through the existing experiment-run seam."""

import json
import struct
from dataclasses import replace

import pytest
from test_control import GameFixture, control_config

from fh5.control import Command, Control
from fh5.experiment import run_experiment


def speed_config(tmp_path):
    path = control_config(tmp_path)
    root = json.loads(path.read_text())
    root["control"].update(
        max_speed_kmh=15,
        max_throttle=0.24,
        max_brake=0.4,
        max_steer=0,
        start_speed_kmh=1,
        spatial_guard={
            "start_position_m": [10, 2, -30],
            "start_radius_m": 1,
            "heading_rad": 0,
            "max_heading_error_rad": 0.1,
            "max_lateral_m": 1,
            "max_distance_m": 55,
        },
        steps=[
            {"seconds": 0.2, "steer": 0, "target_speed_kmh": 8},
            {"seconds": 0.3, "steer": 0, "target_speed_kmh": 0},
        ],
    )
    path.write_text(json.dumps(root))
    return path


def test_feedback_uses_current_speed_and_releases_brake_at_a_stop(tmp_path):
    class SpeedReadings(GameFixture):
        def read(self, period_s):
            frame = super().read(period_s)
            data = bytearray(frame.packets[0].payload)
            # Independent input sequence: low, on target, too fast, low,
            # braking phase while moving, then stopped.
            values = [0, 8, 10, 6, 4, 0]
            index = min(len(self.sent) - 1, len(values) - 1)
            struct.pack_into("<f", data, 256, values[index] / 3.6)
            return replace(frame, packets=(replace(frame.packets[0], payload=bytes(data)),))

    game = SpeedReadings()
    result = run_experiment(Control(speed_config(tmp_path), tmp_path / "run"), environment=game)
    commands = [c for c in result.summary["control"]["commands"] if c["owner"] == "calibration"]
    assert [c["sent"] for c in commands[:6]] == [
        {"steer_i16": 0, "throttle_u8": 61, "brake_u8": 0},
        {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0},
        {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 102},
        {"steer_i16": 0, "throttle_u8": 61, "brake_u8": 0},
        {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 102},
        {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0},
    ]
    assert all(c["sent"] == {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0} for c in commands[6:])
    assert commands[2]["requested"]["target_speed_kmh"] == 8
    assert commands[2]["requested"]["observed_speed_kmh"] == pytest.approx(10)
    assert result.summary["control"]["feedback_version"] == "bounded-speed-v1"
    assert game.sent[-1] == Command(0, 0, 0)
    assert result.summary["control"]["stop_reason"] == "completed"


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("wrong_start", "start_position_mismatch"),
        ("heading", "heading_limit"),
        ("lateral", "lateral_limit"),
        ("distance", "distance_limit"),
        ("motion", "invalid_motion"),
    ],
)
def test_survey_spatial_guard_releases_and_does_not_restart(tmp_path, fault, reason):
    class SpatialFault(GameFixture):
        def read(self, period_s):
            frame = super().read(period_s)
            data = bytearray(frame.packets[0].payload)
            trigger = fault == "wrong_start" or len(self.sent) >= 2
            if trigger:
                if fault in ("wrong_start", "lateral"):
                    struct.pack_into("<f", data, 244, 11.5)
                elif fault == "heading":
                    struct.pack_into("<f", data, 56, 0.2)
                elif fault == "distance":
                    struct.pack_into("<f", data, 252, 26)
                else:
                    struct.pack_into("<f", data, 56, float("nan"))
            return replace(frame, packets=(replace(frame.packets[0], payload=bytes(data)),))

    game = SpatialFault()
    result = run_experiment(Control(speed_config(tmp_path), tmp_path / "run"), environment=game)
    assert result.summary["control"]["stop_reason"] == reason
    assert game.sent[-1] == Command(0, 0, 0)
    if fault == "wrong_start":
        assert all(c == Command(0, 0, 0) for c in game.sent)


@pytest.mark.parametrize("fault", ["guard", "both", "steer", "margin", "end", "mixed"])
def test_invalid_feedback_never_sends_input(tmp_path, fault):
    path = speed_config(tmp_path)
    root = json.loads(path.read_text())
    config = root["control"]
    if fault == "guard":
        del config["spatial_guard"]
    elif fault == "both":
        config["steps"][0]["longitudinal"] = 0.1
    elif fault == "steer":
        config["steps"][0]["steer"] = 0.1
    elif fault == "margin":
        config["steps"][0]["target_speed_kmh"] = 14
    elif fault == "end":
        config["steps"][-1]["target_speed_kmh"] = 8
    else:
        config["steps"][0] = {"seconds": 0.2, "steer": 0, "longitudinal": 0.2}
    path.write_text(json.dumps(root))
    game = GameFixture()
    with pytest.raises(ValueError):
        run_experiment(Control(path, tmp_path / "invalid"), environment=game)
    assert game.sent == []
    assert game.closed


@pytest.mark.parametrize("backwards", [False, True])
def test_feedback_does_not_claim_stopped_or_continue_backwards(tmp_path, backwards):
    class MovingGame(GameFixture):
        def read(self, period_s):
            frame = super().read(period_s)
            data = bytearray(frame.packets[0].payload)
            if len(self.sent) >= 2:
                struct.pack_into("<f", data, 256, 2 / 3.6)
                if backwards:
                    struct.pack_into("<f", data, 252, -31)
            return replace(frame, packets=(replace(frame.packets[0], payload=bytes(data)),))

    game = MovingGame()
    result = run_experiment(Control(speed_config(tmp_path), tmp_path / "run"), environment=game)
    assert result.summary["control"]["stop_reason"] == (
        "reverse_motion" if backwards else "not_stopped"
    )
    assert game.sent[-1] == Command(0, 0, 0)


@pytest.mark.parametrize(
    "feedback,throttle,allowed", [(True, 0.35, True), (True, 0.36, False), (False, 0.35, False)]
)
def test_collection_throttle_budget_does_not_widen_fixed_calibration(
    tmp_path, feedback, throttle, allowed
):
    path = speed_config(tmp_path) if feedback else control_config(tmp_path)
    root = json.loads(path.read_text())
    root["control"]["max_throttle"] = throttle
    path.write_text(json.dumps(root))
    game = GameFixture()
    if allowed:
        result = run_experiment(Control(path, tmp_path / "run"), environment=game)
        assert any(c.throttle_u8 == 89 for c in game.sent)
        assert result.summary["control"]["stop_reason"] == "completed"
    else:
        with pytest.raises(ValueError):
            run_experiment(Control(path, tmp_path / "run"), environment=game)
        assert not game.sent


def test_sub_tick_braking_phase_is_rejected_before_sending(tmp_path):
    path = speed_config(tmp_path)
    root = json.loads(path.read_text())
    root["control"]["steps"][0]["seconds"] = 0.16
    root["control"]["steps"][-1]["seconds"] = 0.02
    path.write_text(json.dumps(root))
    game = GameFixture()
    with pytest.raises(ValueError):
        run_experiment(Control(path, tmp_path / "invalid"), environment=game)
    assert not game.sent
