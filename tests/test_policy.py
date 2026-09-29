"""Bounded visual driving through the agreed experiment-run boundary."""

import json
import math
import struct
from dataclasses import replace
from io import BytesIO

import pytest
from PIL import Image
from test_route_check import route

from fh5.control import Command
from fh5.experiment import Packet, Replay, run_experiment
from fh5.policy import PolicyDrive, PolicyInput
from fh5.vision import ColorFrame

SETTINGS = {
    "version": 2,
    "period_ms": 100,
    "history_offsets_ms": [200, 100, 0],
    "max_image_age_ms": 200,
    "max_telemetry_age_ms": 100,
    "waypoint_distances_m": [5, 10, 20, 40, 80],
    "reference_mode": "optional",
    "action_history_offsets_ms": [200, 100, 0],
    "max_action_age_ms": 200,
    "navigation": {"display": "unknown", "visibility": "unknown", "evidence": []},
}


class VisualActor:
    kind = "synthetic_test_actor"
    manifest = {
        "contract": {"observation": SETTINGS, "action": "xinput-lx-rt-lt-v1"},
        "weights_sha256": "fixture",
    }

    def __init__(self):
        self.inputs = []

    def predict(self, actor, images):
        self.inputs.append((actor, images))
        with Image.open(BytesIO(images[-1])) as pixels:
            assert pixels.getpixel((0, 0)) == (17, 34, 51)
        return [0.8, 0.6]


class DrivingGame:
    source_kind = "synthetic"

    def __init__(self):
        self.time_ns = 1_000_000_000
        self.x = 0.0
        self.command = Command(0, 0, 0)
        self.sent = []
        self.events = []
        self.closed = False
        buf = BytesIO()
        Image.new("RGB", (16, 9), (17, 34, 51)).save(buf, format="PNG")
        self.pixels = buf.getvalue()

    def now_ns(self):
        return self.time_ns

    def read(self, period_s):
        self.time_ns += int(period_s * 1e9)
        speed = 2 if self.command.throttle_u8 else 0
        self.x += speed * period_s
        raw = bytearray(324)
        struct.pack_into("<iI", raw, 0, 1, self.time_ns // 1_000_000)
        struct.pack_into("<iii", raw, 212, 2941, 6, 999)
        struct.pack_into("<ffff", raw, 244, self.x, 2, 0.2, speed)
        struct.pack_into("<f", raw, 56, math.pi / 2)
        return PolicyInput(
            (Packet(self.time_ns, "2026-09-30T00:00:00+00:00", bytes(raw)),),
            ColorFrame(
                self.time_ns, self.time_ns, self.time_ns, self.pixels, "png", (16, 9), (16, 9)
            ),
            True,
        )

    def send(self, command):
        self.command = command
        self.sent.append(command)

    def close(self):
        self.closed = True
        return True


def config(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle = route(tmp_path)
    root = json.loads((tmp_path / "reference.json").read_text())
    root["control_source"] = "policy"
    root["policy"] = {
        "version": 1,
        "model_dir": "unused-fixture",
        "device": "cpu",
        "evaluation_route_file": str(bundle),
        "reference_file": None,
        "reference_mode": "disabled",
        "camera_mode": "chase_far",
        "expected_car_ordinal": 2941,
        "expected_pi": 999,
        "max_speed_kmh": 15,
        "start_speed_kmh": 1,
        "max_steer": 0.2,
        "max_throttle": 0.25,
        "max_brake": 0.5,
        "max_duration_s": 5,
        "ready_timeout_s": 5,
        "inference_timeout_ms": 80,
        "max_command_age_ms": 150,
        "braking_s": 1,
        "navigation": "visible_no_junction",
    }
    path = tmp_path / "policy-config.json"
    path.write_text(json.dumps(root))
    return path


@pytest.mark.parametrize("steer_limit, sent_steer", [(0.2, 6553), (0.4, 13107)])
def test_visual_policy_sends_bounded_actions_with_causal_history_and_stops(
    tmp_path, steer_limit, sent_steer
):
    game, actor = DrivingGame(), VisualActor()
    path = config(tmp_path)
    root = json.loads(path.read_text())
    root["policy"]["max_steer"] = steer_limit
    path.write_text(json.dumps(root))
    result = run_experiment(
        PolicyDrive(path, tmp_path / "drive"),
        policy_environment=game,
        policy_actor=actor,
    )
    p = result.summary["policy"]
    assert p["stop_reason"] == "local_end"
    assert p["geometry_completed"] is True
    assert p["formal_validity"] == "pending_independent_review"
    decisions = [d for d in p["decisions"] if d["prediction"] is not None]
    assert decisions
    assert decisions[0]["prediction"] == [0.8, 0.6]
    assert decisions[0]["sent"] == {"steer_i16": sent_steer, "throttle_u8": 64, "brake_u8": 0}
    assert decisions[0]["observation"]["actor"]["action_mask"] == [False, False, False]
    assert all(d["observation"]["actor"]["reference"]["mask"] == [False] * 5 for d in decisions)
    assert any(d["observation"]["actor"]["action_mask"][-1] for d in decisions[1:])
    assert game.sent[-1] == Command(0, 0, 0)
    assert game.closed and p["release_sent"]
    assert (tmp_path / "drive/policy-decisions.jsonl").exists()
    replay = run_experiment(Replay(tmp_path / "drive", tmp_path / "replayed.html"))
    assert replay.summary["policy"]["decisions"] == p["decisions"]
    assert replay.summary["policy"]["artifact_errors"] == []
    assert replay.summary["vision"]["integrity_errors"] == []
    assert replay.summary["control"]["artifact_errors"] == []
    assert replay.summary["control"]["timing"]["max_interval_ms"] is not None


def test_policy_rejects_steering_above_the_calibrated_range_before_sending(tmp_path):
    path = config(tmp_path)
    root = json.loads(path.read_text())
    root["policy"]["max_steer"] = 0.51
    path.write_text(json.dumps(root))
    game = DrivingGame()
    with pytest.raises(ValueError, match="max_steer"):
        run_experiment(
            PolicyDrive(path, tmp_path / "drive"),
            policy_environment=game,
            policy_actor=VisualActor(),
        )
    assert game.sent == [] and game.closed


def test_missing_optional_reference_masks_prior_but_required_mode_refuses(tmp_path):
    path = config(tmp_path)
    root = json.loads(path.read_text())
    root["policy"].update(reference_mode="optional", reference_file="absent.json")
    path.write_text(json.dumps(root))
    result = run_experiment(
        PolicyDrive(path, tmp_path / "drive"),
        policy_environment=DrivingGame(),
        policy_actor=VisualActor(),
    )
    assert result.summary["policy"]["stop_reason"] == "local_end"
    assert result.summary["policy"]["reference_error"]
    assert not any(
        result.summary["policy"]["decisions"][0]["observation"]["actor"]["reference"]["mask"]
    )
    root["policy"]["reference_mode"] = "required"
    path.write_text(json.dumps(root))
    game = DrivingGame()
    with pytest.raises((OSError, ValueError)):
        run_experiment(
            PolicyDrive(path, tmp_path / "required"),
            policy_environment=game,
            policy_actor=VisualActor(),
        )
    assert game.sent == [] and game.closed


def test_capture_and_decision_phase_drift_does_not_expire_a_healthy_image(tmp_path):
    class DelayedCapture(DrivingGame):
        def __init__(self):
            super().__init__()
            self.last_capture = 0

        def read(self, period_s):
            batch = super().read(period_s)
            # A 100 ms camera cadence with 80 ms capture/encoding latency,
            # independently clocked from the 100 ms policy period.
            capture = ((self.time_ns - 80_000_000) // 100_000_000) * 100_000_000
            if capture == self.last_capture:
                return replace(batch, frame=None)
            self.last_capture = capture
            frame = replace(
                batch.frame,
                capture_start_ns=capture,
                capture_end_ns=capture + 40_000_000,
                available_ns=capture + 80_000_000,
            )
            return replace(batch, frame=frame)

    game = DelayedCapture()

    class TimedActor(VisualActor):
        def predict(self, actor, images):
            game.time_ns += 40_000_000
            return super().predict(actor, images)

    result = run_experiment(
        PolicyDrive(config(tmp_path), tmp_path / "drive"),
        policy_environment=game,
        policy_actor=TimedActor(),
    )
    p = result.summary["policy"]
    assert p["stop_reason"] == "local_end"
    assert len([d for d in p["decisions"] if d["sent"]]) > 5
    assert p["release_sent"] and game.closed


def test_declared_manual_start_inside_verified_region_is_not_a_fake_trajectory(tmp_path):
    path = config(tmp_path)
    root = json.loads(path.read_text())
    root["policy"].update(start_station_m=1.5, start_tolerance_m=0.25)
    path.write_text(json.dumps(root))
    game = DrivingGame()
    game.x = 1.5
    result = run_experiment(
        PolicyDrive(path, tmp_path / "drive"), policy_environment=game, policy_actor=VisualActor()
    )
    assert result.summary["policy"]["stop_reason"] == "local_end"
    assert result.summary["policy"]["start_station_m"] == 1.5
    assert result.samples[0]["position_m"][0] == 1.5


def test_deadline_crossed_in_final_receive_brakes_without_an_extra_prediction_send(tmp_path):
    path = config(tmp_path)
    root = json.loads(path.read_text())
    root["policy"]["max_duration_s"] = 0.108
    path.write_text(json.dumps(root))
    result = run_experiment(
        PolicyDrive(path, tmp_path / "drive"),
        policy_environment=DrivingGame(),
        policy_actor=VisualActor(),
    )
    p = result.summary["policy"]
    assert p["stop_reason"] == "time_limit"
    assert len([c for c in p["commands"] if c["owner"] == "policy"]) == 1
    assert p["decisions"][-1]["rejection"] == "time_limit"


def test_stalled_game_clock_never_arms_even_with_fresh_rgb_and_packets(tmp_path):
    class FrozenGame(DrivingGame):
        def read(self, period_s):
            batch = super().read(period_s)
            raw = bytearray(batch.packets[0].payload)
            struct.pack_into("<I", raw, 4, 1000)
            return replace(batch, packets=(replace(batch.packets[0], payload=bytes(raw)),))

    result = run_experiment(
        PolicyDrive(config(tmp_path), tmp_path / "drive"),
        policy_environment=FrozenGame(),
        policy_actor=VisualActor(),
    )
    assert not result.summary["policy"]["decisions"]


def test_blocked_image_disk_does_not_block_control_release(tmp_path, monkeypatch):
    import threading
    from pathlib import Path

    path = config(tmp_path)
    unblock, done = threading.Event(), threading.Event()
    original = Path.write_bytes

    def write_bytes(file, value):
        if file.parent.name == "frames":
            unblock.wait(5)
        return original(file, value)

    monkeypatch.setattr(Path, "write_bytes", write_bytes)
    game, results = DrivingGame(), []

    def run():
        try:
            results.append(
                run_experiment(
                    PolicyDrive(path, tmp_path / "drive"),
                    policy_environment=game,
                    policy_actor=VisualActor(),
                )
            )
        finally:
            done.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert done.wait(2), "Blocked image persistence stalled the experiment runner"
        assert results[0].summary["policy"]["stop_reason"] == "image_writer_backpressure"
        assert results[0].summary["policy"]["release_sent"] and game.closed
        assert not results[0].summary["policy"]["image_writer_released"]
    finally:
        unblock.set()
        worker.join(2)


@pytest.mark.parametrize("value", [[float("nan"), 0], [0, float("inf")], [0], [False, 0], [0, 2]])
def test_invalid_prediction_is_a_retained_rejection(tmp_path, value):
    class BadActor(VisualActor):
        def predict(self, actor, images):
            return value

    game = DrivingGame()
    result = run_experiment(
        PolicyDrive(config(tmp_path), tmp_path / "drive"),
        policy_environment=game,
        policy_actor=BadActor(),
    )
    assert result.summary["policy"]["stop_reason"] == "invalid_prediction"
    assert all(c == Command(0, 0, 0) for c in game.sent)


def test_late_inference_is_saved_but_never_sent(tmp_path):
    game = DrivingGame()

    class SlowActor(VisualActor):
        def predict(self, actor, images):
            game.time_ns += 200_000_000
            return super().predict(actor, images)

    result = run_experiment(
        PolicyDrive(config(tmp_path), tmp_path / "drive"),
        policy_environment=game,
        policy_actor=SlowActor(),
    )
    p = result.summary["policy"]
    assert p["stop_reason"] == "inference_timeout"
    assert not any(c.throttle_u8 or c.steer_i16 for c in game.sent)
    assert p["decisions"][0]["sent"] is None
    assert p["release_sent"] and game.closed


@pytest.mark.parametrize(
    "fault, reason",
    [
        ("focus", "focus_lost"),
        ("image", "incomplete_image_history"),
        ("telemetry", "stale_telemetry"),
        ("f8", "user_stop"),
        ("car", "unexpected_vehicle"),
        ("position", "task_location_untrusted"),
    ],
)
def test_fault_after_arming_releases_and_retains_failed_attempt(tmp_path, fault, reason):
    class FaultyGame(DrivingGame):
        def read(self, period_s):
            batch = super().read(period_s)
            if not any(c.throttle_u8 for c in self.sent):
                return batch
            if fault == "focus":
                return replace(batch, focused=False)
            if fault == "image":
                return replace(batch, frame=None)
            if fault == "telemetry":
                return replace(batch, packets=())
            if fault == "f8":
                return replace(batch, stop_requested=True)
            raw = bytearray(batch.packets[0].payload)
            if fault == "car":
                struct.pack_into("<i", raw, 212, 99)
            if fault == "position":
                struct.pack_into("<f", raw, 252, 20)
            return replace(batch, packets=(replace(batch.packets[0], payload=bytes(raw)),))

    game = FaultyGame()
    result = run_experiment(
        PolicyDrive(config(tmp_path), tmp_path / "drive"),
        policy_environment=game,
        policy_actor=VisualActor(),
    )
    p = result.summary["policy"]
    assert p["stop_reason"] == reason
    assert not p["geometry_completed"]
    assert p["release_sent"] and game.sent[-1] == Command(0, 0, 0)


def test_hung_inference_does_not_block_release(tmp_path):
    import threading

    unblock = threading.Event()

    class HungActor(VisualActor):
        def predict(self, actor, images):
            unblock.wait(5)
            return [0.9, 0.9]

    game = DrivingGame()
    try:
        result = run_experiment(
            PolicyDrive(config(tmp_path), tmp_path / "drive"),
            policy_environment=game,
            policy_actor=HungActor(),
        )
        p = result.summary["policy"]
        assert p["stop_reason"] == "inference_timeout"
        assert p["release_sent"] and not p["inference_worker_released"]
        assert all(c == Command(0, 0, 0) for c in game.sent)
    finally:
        unblock.set()


def test_environment_change_during_inference_prevents_dispatch(tmp_path):
    game = DrivingGame()

    class MovingActor(VisualActor):
        def predict(self, actor, images):
            game.x = 200
            return [0.9, 0.9]

    result = run_experiment(
        PolicyDrive(config(tmp_path), tmp_path / "drive"),
        policy_environment=game,
        policy_actor=MovingActor(),
    )
    assert result.summary["policy"]["stop_reason"] == "task_location_untrusted"
    assert all(c == Command(0, 0, 0) for c in game.sent)


def test_wrong_initial_heading_never_arms(tmp_path):
    class WrongHeading(DrivingGame):
        def read(self, period_s):
            batch = super().read(period_s)
            raw = bytearray(batch.packets[0].payload)
            struct.pack_into("<f", raw, 56, -math.pi / 2)
            return replace(batch, packets=(replace(batch.packets[0], payload=bytes(raw)),))

    game = WrongHeading()
    result = run_experiment(
        PolicyDrive(config(tmp_path), tmp_path / "drive"),
        policy_environment=game,
        policy_actor=VisualActor(),
    )
    assert result.summary["policy"]["stop_reason"] == "ready_timeout"
    assert not result.summary["policy"]["decisions"]
    assert all(c == Command(0, 0, 0) for c in game.sent)


def test_warmup_cannot_reuse_images_across_pause_in_one_received_batch(tmp_path):
    class PausedWarmup(DrivingGame):
        def __init__(self):
            super().__init__()
            self.reads = 0

        def read(self, period_s):
            self.reads += 1
            batch = super().read(period_s)
            raw = bytearray(batch.packets[0].payload)
            if self.reads < 20:
                struct.pack_into("<f", raw, 256, 1)  # Prevent arming while images accumulate.
                return replace(batch, packets=(replace(batch.packets[0], payload=bytes(raw)),))
            if self.reads == 20:
                struct.pack_into("<i", raw, 0, 0)
                paused = replace(
                    batch.packets[0], received_monotonic_ns=self.time_ns - 1, payload=bytes(raw)
                )
                return replace(batch, packets=(paused, batch.packets[0]))
            return replace(batch, stop_requested=self.reads >= 24)

    result = run_experiment(
        PolicyDrive(config(tmp_path), tmp_path / "drive"),
        policy_environment=PausedWarmup(),
        policy_actor=VisualActor(),
    )
    assert not result.summary["policy"]["decisions"]


def test_frozen_actor_loads_real_trained_weights_and_rejects_incompatible_version(tmp_path):
    pytest.importorskip("torch")
    from test_bc import setup_bc

    from fh5.policy_actor import FrozenActor

    training = setup_bc(tmp_path)
    run_experiment(training)
    path = config(tmp_path / "policy")
    root = json.loads(path.read_text())
    manifest = json.loads((training.output_dir / "model.json").read_text())
    root["snapshot"] = manifest["contract"]["conditions"]
    root["policy"].update(
        model_dir=str(training.output_dir),
        expected_model_sha256=manifest["weights_sha256"],
        expected_car_ordinal=123,
        expected_pi=900,
    )
    path.write_text(json.dumps(root))

    class TrainingCarGame(DrivingGame):
        def read(self, period_s):
            batch = super().read(period_s)
            raw = bytearray(batch.packets[0].payload)
            struct.pack_into("<iii", raw, 212, 123, 5, 900)
            return replace(batch, packets=(replace(batch.packets[0], payload=bytes(raw)),))

    game = TrainingCarGame()
    result = run_experiment(PolicyDrive(path, tmp_path / "drive"), policy_environment=game)
    assert result.summary["policy"]["actor_kind"] == "frozen_bc_v1"
    assert result.summary["policy"]["decisions"]
    assert any(d["sent"] is not None for d in result.summary["policy"]["decisions"])
    manifest["contract"]["action"] = "unvalidated-mapping"
    (training.output_dir / "model.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract|metadata|version"):
        FrozenActor(path)
