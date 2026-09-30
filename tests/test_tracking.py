"""Route following through the experiment seam and an external vehicle simulation."""

import hashlib
import json
import math
import struct
from dataclasses import replace

import pytest
from test_routes import recording

from fh5.control import Command, ControlInput
from fh5.experiment import Packet, Replay, run_experiment
from fh5.routes import BuildRoute
from fh5.tracking import TrackingDrive


def tracking_config(tmp_path, points=None):
    points = points or [(0, z) for z in range(31)]
    points = [struct.unpack("<ff", struct.pack("<ff", *point)) for point in points]
    source = recording(tmp_path, points)
    (tmp_path / "survey.txt").write_text("Synthetic rectangular test arena; no game evidence.")
    review = {"status": "verified", "evidence": ["survey.txt"]}
    reference = run_experiment(BuildRoute(source, tmp_path / "unreviewed", 0, len(points) - 1, 0.5))
    length = reference.summary["route"]["length_m"]
    notes = {
        "version": 1,
        "reference_review": review,
        "checkpoints_review": review,
        "corridors": [
            {
                "id": "arena",
                "s_start_m": 0,
                "s_end_m": length,
                "polygon_xz": [[-3, -2], [9, -2], [9, 35], [-3, 35]],
                "y_min_m": 1,
                "y_max_m": 3,
                **review,
            }
        ],
        "checkpoints": [],
    }
    notes_file = tmp_path / "notes.json"
    notes_file.write_text(json.dumps(notes))
    route_file = tmp_path / "route/route.json"
    run_experiment(BuildRoute(source, route_file.parent, 0, len(points) - 1, 0.5, notes_file))
    root = json.loads((tmp_path / "source.json").read_text())
    root["control_source"] = "policy"
    root["tracking"] = {
        "version": 1,
        "action_version": "steer-signed-longitudinal-v1",
        "route_file": "route/route.json",
        "route_sha256": hashlib.sha256(route_file.read_bytes()).hexdigest(),
        "expected_car_ordinal": 2941,
        "expected_pi": 999,
        "rate_hz": 20,
        "target_speed_kmh": 12,
        "max_speed_kmh": 15,
        "max_steer": 0.5,
        "max_throttle": 0.25,
        "max_brake": 0.5,
        "max_duration_s": 30,
        "ready_timeout_s": 1,
        "braking_s": 3,
        "telemetry_timeout_s": 0.25,
        "no_progress_timeout_s": 3,
        "recovery_timeout_s": 3,
        "end_margin_m": 2,
        "wheelbase_m": 2.6,
        "steering_scale_rad": 0.55,
        "lateral_accel_mps2": 0.6,
        "braking_decel_mps2": 1.5,
    }
    path = tmp_path / "tracking.json"
    path.write_text(json.dumps(root))
    return path


class Vehicle:
    """Synthetic lagged kinematic vehicle, deliberately not calibrated to FH5."""

    source_kind = "synthetic"

    def __init__(self):
        self.time_ns = 1_000_000_000
        self.x = self.z = self.yaw = self.speed = self.wheel = 0.0
        self.command = Command(0, 0, 0)
        self.sent = []
        self.events = []
        self.closed = False

    def now_ns(self):
        return self.time_ns

    def send(self, command):
        self.command = command
        self.sent.append(command)

    def close(self):
        self.closed = True

    def read(self, period_s):
        dt = period_s / 10
        for _ in range(10):
            acceleration = 7 * self.command.throttle_u8 / 255 - 10 * self.command.brake_u8 / 255
            self.speed = max(0, self.speed + dt * (acceleration - 0.1 * self.speed))
            requested_wheel = 0.58 * self.command.steer_i16 / 32767
            self.wheel += (requested_wheel - self.wheel) * min(1, dt / 0.12)
            self.yaw += dt * self.speed / 2.7 * math.tan(self.wheel)
            self.x += dt * self.speed * math.sin(self.yaw)
            self.z += dt * self.speed * math.cos(self.yaw)
        self.time_ns += round(period_s * 1e9)
        raw = bytearray(324)
        struct.pack_into("<iI", raw, 0, 1, self.time_ns // 1_000_000)
        struct.pack_into("<iii", raw, 212, 2941, 5, 999)
        struct.pack_into("<ffff", raw, 244, self.x, 2, self.z, self.speed)
        struct.pack_into("<f", raw, 40, self.speed)
        struct.pack_into("<f", raw, 56, self.yaw)
        raw[315], raw[316] = self.command.throttle_u8, self.command.brake_u8
        struct.pack_into("<b", raw, 320, round(self.command.steer_i16 / 32767 * 127))
        return ControlInput((Packet(self.time_ns, "2026-09-30T00:00:00+00:00", bytes(raw)),), True)


def test_baseline_drives_to_local_target_stops_and_replays_actual_commands(tmp_path):
    vehicle = Vehicle()
    result = run_experiment(
        TrackingDrive(tracking_config(tmp_path), tmp_path / "drive"), environment=vehicle
    )
    control = result.summary["control"]
    assert control["stop_reason"] == "local_end"
    assert 27 < vehicle.z < 30
    assert vehicle.speed * 3.6 < 0.5
    assert vehicle.sent[-1] == Command(0, 0, 0)
    assert vehicle.closed
    assert control["controller_kind"] == "route-feedback-v1"
    assert control["release_sent"]
    assert any(
        c["owner"] == "baseline" and c["sent"]["throttle_u8"] > 0 for c in control["commands"]
    )
    assert all(not (c.throttle_u8 and c.brake_u8) for c in vehicle.sent)
    assert result.summary["route"]["low_speed_ready"]
    assert result.samples[-1]["route"]["reference_s_m"] > 27
    assert control["timing"]["max_interval_ms"] <= 51
    for row in control["commands"]:
        if row["owner"] != "baseline":
            continue
        sample = result.samples[row["requested"]["telemetry_packet_index"]]
        assert sample["received_monotonic_ns"] == row["requested"]["telemetry_received_ns"]
        assert sample["received_monotonic_ns"] <= row["issued_ns"] <= row["returned_ns"]
    (tmp_path / "route/route.json").rename(tmp_path / "route/hidden.json")
    repeated = run_experiment(Replay(tmp_path / "drive", tmp_path / "repeat.html"))
    assert repeated.summary["control"] == control
    assert repeated.samples == result.samples
    assert repeated.summary["route"] == result.summary["route"]


def test_stalled_vehicle_stops_with_no_progress_without_infinite_throttle(tmp_path):
    class StalledVehicle(Vehicle):
        def read(self, period_s):
            self.command = Command(0, 0, 0)
            return super().read(period_s)

    vehicle = StalledVehicle()
    result = run_experiment(
        TrackingDrive(tracking_config(tmp_path), tmp_path / "stalled"), environment=vehicle
    )
    assert result.summary["control"]["stop_reason"] == "no_progress"
    assert vehicle.time_ns <= 4_100_000_000
    assert vehicle.sent[-1] == Command(0, 0, 0)
    assert vehicle.closed


def test_baseline_follows_opposite_bends_and_slows_for_curvature(tmp_path):
    points = [(2 * (1 - math.cos(math.pi * z / 15)), z) for z in range(31)]
    vehicle = Vehicle()
    result = run_experiment(
        TrackingDrive(tracking_config(tmp_path, points), tmp_path / "bends"), environment=vehicle
    )
    control = result.summary["control"]
    assert control["stop_reason"] == "local_end"
    assert vehicle.z > 27
    steering = [c.steer_i16 for c in vehicle.sent]
    assert max(steering) > 1000 and min(steering) < -1000
    assert (
        max(
            min(math.dist((s["position_m"][0], s["position_m"][2]), p) for p in points)
            for s in result.samples
        )
        < 1.0
    )
    targets = [
        c["requested"]["target_speed_kmh"]
        for c in control["commands"]
        if c["owner"] == "baseline" and 3 < c["requested"]["reference_s_m"] < 23
    ]
    assert min(targets) < 10 and max(targets) > 11
    assert max(s["speed_kmh"] for s in result.samples) < 15


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("focus", "focus_lost"),
        ("stop", "user_stop"),
        ("adapter", "watchdog_timeout"),
        ("car", "vehicle_changed"),
        ("pause", "inactive"),
        ("speed", "speed_limit"),
        ("lateral", "lateral_limit"),
        ("heading", "heading_limit"),
        ("inverse_time", "route_discontinuity"),
        ("motion", "invalid_motion"),
        ("drop", "telemetry_stale"),
        ("future", "invalid_receive_time"),
        ("malformed", "invalid_telemetry"),
    ],
)
def test_faults_including_transient_intermediate_packets_stop_commands(tmp_path, fault, reason):
    class FaultyVehicle(Vehicle):
        def read(self, period_s):
            frame = super().read(period_s)
            if self.z < 1:
                return frame
            if fault == "focus":
                return replace(frame, focused=False)
            if fault == "stop":
                return replace(frame, stop_requested=True)
            if fault == "adapter":
                return replace(frame, fault="watchdog_timeout")
            if fault == "drop":
                return replace(frame, packets=())
            packet = frame.packets[0]
            raw = bytearray(packet.payload)
            if fault == "car":
                struct.pack_into("<i", raw, 212, 99)
            if fault == "pause":
                struct.pack_into("<i", raw, 0, 0)
            if fault == "speed":
                struct.pack_into("<f", raw, 256, 20)
            if fault == "lateral":
                struct.pack_into("<f", raw, 244, 2)
            if fault == "heading":
                struct.pack_into("<f", raw, 56, 1.4)
            if fault == "inverse_time":
                struct.pack_into("<I", raw, 4, 10)
            if fault == "motion":
                struct.pack_into("<f", raw, 56, float("nan"))
            if fault == "malformed":
                raw = raw[:10]
            bad = replace(packet, payload=bytes(raw))
            if fault == "future":
                bad = replace(bad, received_monotonic_ns=self.time_ns + 1)
            # Even a valid latest packet must not erase an intermediate fault.
            return replace(frame, packets=(bad, packet))

    vehicle = FaultyVehicle()
    result = run_experiment(
        TrackingDrive(tracking_config(tmp_path), tmp_path / "fault"), environment=vehicle
    )
    control = result.summary["control"]
    assert control["stop_reason"] == reason
    assert vehicle.closed and vehicle.sent[-1] == Command(0, 0, 0)
    assert control["release_sent"]
    assert vehicle.z < 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_throttle", 0.9),
        ("max_steer", 1),
        ("max_speed_kmh", 100),
        ("target_speed_kmh", 15),
        ("rate_hz", 0),
        ("max_duration_s", float("inf")),
        ("ready_timeout_s", False),
        ("telemetry_timeout_s", 2),
        ("end_margin_m", 30),
        ("wheelbase_m", 0),
        ("version", True),
        ("action_version", "other"),
        ("no_progress_timeout_s", -1),
        ("route_sha256", "0" * 64),
        ("expected_pi", 1000),
    ],
)
def test_invalid_configuration_never_sends_input(tmp_path, key, value):
    path = tracking_config(tmp_path)
    config = json.loads(path.read_text())
    config["tracking"][key] = value
    path.write_text(json.dumps(config))
    vehicle = Vehicle()
    with pytest.raises(ValueError):
        run_experiment(TrackingDrive(path, tmp_path / "invalid"), environment=vehicle)
    assert vehicle.closed and vehicle.sent == []
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("arrives", [False, True])
def test_waits_neutral_for_foreground_start_with_bounded_deadline(tmp_path, arrives):
    class WaitingVehicle(Vehicle):
        def read(self, period_s):
            frame = super().read(period_s)
            if not arrives or self.time_ns < 1_300_000_000:
                assert self.command == Command(0, 0, 0)
                return ControlInput()
            return frame

    vehicle = WaitingVehicle()
    result = run_experiment(
        TrackingDrive(tracking_config(tmp_path), tmp_path / "waiting"), environment=vehicle
    )
    assert result.summary["control"]["stop_reason"] == ("local_end" if arrives else "ready_timeout")
    assert vehicle.z > 27 if arrives else vehicle.z == 0
    assert vehicle.closed and vehicle.sent[-1] == Command(0, 0, 0)


def test_refuses_moving_start_before_any_driving_command(tmp_path):
    vehicle = Vehicle()
    vehicle.speed = 1.0
    result = run_experiment(
        TrackingDrive(tracking_config(tmp_path), tmp_path / "moving"), environment=vehicle
    )
    assert result.summary["control"]["stop_reason"] == "start_not_stopped"
    assert all(command == Command(0, 0, 0) for command in vehicle.sent)


@pytest.mark.parametrize("stuck", [False, True])
def test_small_offset_recovers_with_slowdown_or_stops_within_recovery_budget(tmp_path, stuck):
    class DisplacedVehicle(Vehicle):
        displaced = False

        def read(self, period_s):
            if self.z > 3 and not self.displaced:
                self.x = 0.9
                self.displaced = True
            if stuck and self.displaced:
                self.x, self.yaw, self.wheel = 0.9, 0.0, 0.0
            return super().read(period_s)

    vehicle = DisplacedVehicle()
    result = run_experiment(
        TrackingDrive(tracking_config(tmp_path), tmp_path / "recovery"), environment=vehicle
    )
    control = result.summary["control"]
    assert control["stop_reason"] == ("recovery_timeout" if stuck else "local_end")
    recovery = [c for c in control["commands"] if c["requested"].get("mode") == "recovering"]
    assert recovery and max(c["requested"]["target_speed_kmh"] for c in recovery) <= 6
    if stuck:
        assert recovery[-1]["issued_ns"] - recovery[0]["issued_ns"] <= 3_000_000_000
        assert vehicle.z < 12
    else:
        assert vehicle.z > 27 and abs(vehicle.x) < 0.3
    assert vehicle.sent[-1] == Command(0, 0, 0)


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("frozen", "game_time_stalled"),
        ("read_stall", "control_stalled"),
        ("send_stall", "send_stalled"),
        ("send_error", "interface_error"),
        ("release_error", "interface_error"),
        ("close_error", "interface_error"),
    ],
)
def test_timing_and_interface_failures_leave_auditable_outcomes(tmp_path, fault, reason):
    class BrokenInterface(Vehicle):
        frozen_time = None

        def read(self, period_s):
            frame = super().read(period_s)
            if self.z < 1:
                return frame
            if fault == "frozen":
                raw = bytearray(frame.packets[0].payload)
                if self.frozen_time is None:
                    self.frozen_time = self.time_ns // 1_000_000
                struct.pack_into("<I", raw, 4, self.frozen_time)
                return replace(frame, packets=(replace(frame.packets[0], payload=bytes(raw)),))
            if fault == "read_stall":
                self.time_ns += 300_000_000
            return frame

        def send(self, command):
            super().send(command)
            if self.z >= 1 and command != Command(0, 0, 0):
                if fault == "send_stall":
                    self.time_ns += 300_000_000
                if fault == "send_error":
                    raise OSError("test transport send failed")
            if self.z >= 1 and command == Command(0, 0, 0) and fault == "release_error":
                raise OSError("test neutral send failed")

        def close(self):
            super().close()
            if fault == "close_error":
                raise OSError("test close failed")

    vehicle = BrokenInterface()
    result = run_experiment(
        TrackingDrive(tracking_config(tmp_path), tmp_path / "broken"), environment=vehicle
    )
    control = result.summary["control"]
    assert control["stop_reason"] == reason
    assert control["release_sent"] == (fault != "release_error")
    assert vehicle.closed
    if fault in ("send_error", "release_error"):
        assert any(c["status"] == "failed" and c["sent"] is None for c in control["commands"])
