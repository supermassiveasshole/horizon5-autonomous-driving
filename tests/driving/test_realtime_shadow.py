"""Raw capture and telemetry integration through the agreed experiment-run seam."""

import hashlib
import json
import math
import socket
import struct
import threading
import time

import pytest

from fh5.capture.pipeline import CaptureConfig, CaptureEvent, QpcMapping, RawCapture
from fh5.driving.realtime.model import RealtimeConfig, RealtimeRun
from fh5.driving.realtime.shadow import LocalTask, ShadowEnvironment, TelemetryBatch
from fh5.driving.realtime.udp import UDPTelemetry
from fh5.experiment import Packet, run_experiment
from fh5.observation.numeric import PixelContract
from fh5.observation.routes import BuildRoute
from tests.observation.test_route_check import record, route


class Desktop:
    def focused(self):
        return True

    def stop_requested(self):
        return False


class Telemetry:
    def __init__(self):
        self.closed = False

    def read(self, period_s):
        time.sleep(period_s)
        now = time.perf_counter_ns()
        raw = bytearray(324)
        struct.pack_into("<iI", raw, 0, 1, (now // 1_000_000) % 2**32)
        struct.pack_into("<iii", raw, 212, 2941, 6, 999)
        struct.pack_into("<ffff", raw, 244, 0, 2, 0.2, 0)
        struct.pack_into("<fff", raw, 32, 1, 2, 3)
        struct.pack_into("<fff", raw, 44, 0.1, 0.2, 0.3)
        struct.pack_into("<f", raw, 56, math.pi / 2)
        return TelemetryBatch((Packet(now, "2026-09-30T00:00:00+00:00", bytes(raw)),))

    def close(self):
        self.closed = True


class Capture:
    source_kind = "synthetic-capture"

    def __init__(self):
        self.closed = False

    def capture(self):
        now = time.perf_counter_ns()
        return CaptureEvent(
            now,
            RawCapture(
                now,
                QpcMapping(now, now, 1_000_000_000, 0),
                (2, 1),
                bytes([51, 34, 17, 255] * 2),
                {"client": "synthetic"},
            ),
        )

    def close(self):
        self.closed = True


class Actor:
    kind = "synthetic-service"
    manifest = {}

    def __init__(self):
        self.inputs = []
        self.active = threading.Event()

    def predict(self, actor, frames):
        self.inputs.append((actor, frames))
        if len(self.inputs) > 1:
            self.active.set()
        return [0.1, 0.2]


def setup(tmp_path, telemetry=None, capture=None, actor=None):
    task_file = route(tmp_path)
    task = LocalTask(
        task_file, hashlib.sha256(task_file.read_bytes()).hexdigest(), end_margin_m=0.5
    )
    request = RealtimeRun(
        tmp_path / "shadow",
        RealtimeConfig(pixels=PixelContract(size=(2, 1))),
        seconds=0.8,
    )
    telemetry, capture, actor = telemetry or Telemetry(), capture or Capture(), actor or Actor()
    env = ShadowEnvironment(
        request,
        CaptureConfig(pixels=request.config.pixels),
        lambda: capture,
        telemetry,
        Desktop(),
        task,
    )
    return request, env, actor, telemetry, capture


def test_shadow_combines_raw_pixels_and_telemetry_without_sending_candidate_actions(tmp_path):
    request, env, actor, telemetry, capture = setup(tmp_path)
    result = run_experiment(request, realtime_environment=env, numeric_actor_factory=lambda: actor)
    r = result.summary["realtime"]
    assert r["stop_reason"] == "time_limit"
    decisions = [d for d in r["decisions"] if d["status"] == "accepted"]
    assert decisions
    assert r["commands_sent_to_game"] is False
    assert r["evidence"]["training_eligible"] is False
    for observed, frames in actor.inputs[1:]:
        assert bytes(frames[-1].pixels) == bytes([17, 34, 51] * 2)
        assert observed["ego"]["velocity_car_mps"] == [1, 2, 3]
        assert observed["action_mask"] == [False] * 3
        assert observed["reference"]["mask"] == [False] * 5
    assert decisions[0]["safety_at_decision"]["task_location"]["status"] == "matched"
    assert decisions[0]["safety_at_decision"]["telemetry_packet_index"] >= 2
    assert r["environment"]["task"]["route_sha256"] == env.task.expected_route_sha256
    assert r["environment"]["telemetry"]["packets"] >= 3
    assert r["environment"]["capture"]["preprocessed"] >= 3
    assert r["resources_released"] and telemetry.closed and capture.closed


class BurstTelemetry(Telemetry):
    def __init__(self, actor, fault):
        super().__init__()
        self.actor, self.fault = actor, fault
        self.injected = False

    def read(self, period_s):
        batch = super().read(period_s)
        if self.injected or not self.actor.active.is_set():
            return batch
        self.injected = True
        good = batch.packets[0]
        raw = bytearray(good.payload)
        if self.fault == "inactive":
            struct.pack_into("<i", raw, 0, 0)
        elif self.fault == "game_clock_discontinuity":
            struct.pack_into("<I", raw, 4, 0)
        elif self.fault == "unexpected_vehicle":
            struct.pack_into("<i", raw, 212, 9999)
        elif self.fault == "speed_limit":
            struct.pack_into("<f", raw, 256, 5)
        elif self.fault == "invalid_motion":
            struct.pack_into("<f", raw, 56, float("nan"))
        elif self.fault == "invalid_telemetry":
            raw = raw[:42]
        elif self.fault == "task_location_untrusted":
            struct.pack_into("<f", raw, 252, 20)
        bad = Packet(good.received_monotonic_ns - 1, good.received_utc, bytes(raw))
        return TelemetryBatch((bad, good))


@pytest.mark.parametrize(
    "fault",
    [
        "inactive",
        "game_clock_discontinuity",
        "unexpected_vehicle",
        "speed_limit",
        "invalid_motion",
        "invalid_telemetry",
        "task_location_untrusted",
    ],
)
def test_transient_fault_in_a_raw_batch_is_not_hidden_by_its_last_healthy_packet(tmp_path, fault):
    actor = Actor()
    telemetry = BurstTelemetry(actor, fault)
    request, env, actor, _, _ = setup(tmp_path, telemetry=telemetry, actor=actor)
    r = run_experiment(
        request, realtime_environment=env, numeric_actor_factory=lambda: actor
    ).summary["realtime"]
    assert r["stop_reason"] == fault
    assert r["environment"]["fault"] == fault
    assert r["resources_released"]
    assert telemetry.injected
    journal = (request.output_dir / "realtime-events.jsonl").read_text().splitlines()
    raw_rows = [json.loads(line) for line in journal if json.loads(line)["kind"] == "packet"]
    assert len(raw_rows) == r["environment"]["telemetry"]["packets"]


def test_capture_boundary_rejects_pending_result_before_new_history_is_available(tmp_path):
    changed = threading.Event()
    actor = Actor()

    class BoundaryCapture(Capture):
        def capture(self):
            if actor.active.is_set() and not changed.is_set():
                changed.set()
                return CaptureEvent(time.perf_counter_ns(), boundary="mode_changed")
            return super().capture()

    class DelayedActor(Actor):
        def predict(self, observed, frames):
            result = super().predict(observed, frames)
            if len(self.inputs) == 2:
                assert changed.wait(0.1)
                time.sleep(0.015)
            return result

    actor = DelayedActor()
    request, env, _, _, _ = setup(tmp_path, capture=BoundaryCapture(), actor=actor)
    r = run_experiment(
        request, realtime_environment=env, numeric_actor_factory=lambda: actor
    ).summary["realtime"]
    inferred = [d for d in r["decisions"] if d["prediction"] is not None]
    assert inferred[0]["status"] == "discard_capture_epoch"
    assert not any(c["decision_id"] == inferred[0]["decision_id"] for c in r["commands"])
    assert any(d["status"] == "accepted" for d in inferred[1:])


def test_menu_frames_cannot_be_used_after_active_driving_resumes(tmp_path):
    class MenuTelemetry(Telemetry):
        def __init__(self):
            super().__init__()
            self.started_ns = None
            self.first_active_ns = None

        def read(self, period_s):
            packet = super().read(period_s).packets[0]
            if self.started_ns is None:
                self.started_ns = packet.received_monotonic_ns
            if packet.received_monotonic_ns - self.started_ns < 180_000_000:
                raw = bytearray(packet.payload)
                struct.pack_into("<i", raw, 0, 0)
                return TelemetryBatch(
                    (Packet(packet.received_monotonic_ns, packet.received_utc, bytes(raw)),)
                )
            if self.first_active_ns is None:
                self.first_active_ns = packet.received_monotonic_ns
            return TelemetryBatch((packet,))

    telemetry = MenuTelemetry()
    request, env, actor, _, _ = setup(tmp_path, telemetry=telemetry)
    r = run_experiment(
        request, realtime_environment=env, numeric_actor_factory=lambda: actor
    ).summary["realtime"]
    assert any(d["status"] == "accepted" for d in r["decisions"])
    assert actor.inputs[1][1][0].source_time_ns >= telemetry.first_active_ns


def test_loopback_udp_is_recorded_and_released_by_the_shadow_run(tmp_path):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    address = receiver.getsockname()
    done = threading.Event()

    def produce():
        source = Telemetry()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            while not done.is_set():
                packet = source.read(0.005).packets[0]
                sender.sendto(packet.payload, address)

    producer = threading.Thread(target=produce, daemon=True)
    request, env, actor, _, _ = setup(tmp_path, telemetry=UDPTelemetry(receiver=receiver))
    producer.start()
    try:
        r = run_experiment(
            request, realtime_environment=env, numeric_actor_factory=lambda: actor
        ).summary["realtime"]
    finally:
        done.set()
        producer.join(timeout=1)
    assert any(d["status"] == "accepted" for d in r["decisions"])
    assert r["environment"]["telemetry"]["packets"] >= 3
    assert r["resources_released"] and receiver.fileno() == -1


def test_udp_backlog_never_starts_inference_or_claims_complete_evidence(tmp_path):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
    receiver.bind(("127.0.0.1", 0))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        payload = Telemetry().read(0).packets[0].payload
        for _ in range(65):
            sender.sendto(payload, receiver.getsockname())
    request, env, actor, _, _ = setup(tmp_path, telemetry=UDPTelemetry(receiver=receiver))
    r = run_experiment(
        request, realtime_environment=env, numeric_actor_factory=lambda: actor
    ).summary["realtime"]
    assert r["stop_reason"] == "telemetry_backlog"
    assert not any(d["prediction"] is not None for d in r["decisions"])
    assert r["commands"] == [] and not r["evidence"]["recording_complete"]
    assert r["resources_released"]


def test_restart_gap_before_preparation_clears_old_frames(tmp_path):
    class GappedTelemetry(Telemetry):
        def __init__(self):
            super().__init__()
            self.start = None
            self.resumed_ns = None

        def read(self, period_s):
            batch = super().read(period_s)
            now = batch.packets[0].received_monotonic_ns
            if self.start is None:
                self.start = now
            elapsed = now - self.start
            if 60_000_000 < elapsed < 240_000_000:
                return TelemetryBatch()
            if elapsed >= 240_000_000 and self.resumed_ns is None:
                self.resumed_ns = now
            return batch

    telemetry = GappedTelemetry()
    request, env, actor, _, _ = setup(tmp_path, telemetry=telemetry)
    r = run_experiment(
        request, realtime_environment=env, numeric_actor_factory=lambda: actor
    ).summary["realtime"]
    assert any(d["status"] == "accepted" for d in r["decisions"])
    assert actor.inputs[1][1][0].source_time_ns >= telemetry.resumed_ns


@pytest.mark.parametrize("yaw,accepted", [(0, True), (math.pi / 2, False)])
def test_nonzero_task_start_uses_local_road_heading(tmp_path, yaw, accepted):
    class CornerTelemetry(Telemetry):
        def read(self, period_s):
            packet = super().read(period_s).packets[0]
            raw = bytearray(packet.payload)
            struct.pack_into("<fff", raw, 244, 2, 2, 1)
            struct.pack_into("<f", raw, 56, yaw)
            return TelemetryBatch(
                (Packet(packet.received_monotonic_ns, packet.received_utc, bytes(raw)),)
            )

    # Build an independently reviewed synthetic L-shaped task at the experiment seam.
    request, _, actor, _, capture = setup(tmp_path)
    source = record(tmp_path, "curved", [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3)])
    annotation = tmp_path / "annotations.json"
    notes = json.loads(annotation.read_text())
    notes["corridors"][0].update(s_end_m=5, polygon_xz=[[-1, -1], [4, -1], [4, 4], [-1, 4]])
    annotation.write_text(json.dumps(notes))
    run_experiment(BuildRoute(source, tmp_path / "curve-task", 0, 5, annotations_file=annotation))
    path = tmp_path / "curve-task/route.json"
    task = LocalTask(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), start_station_m=3, end_margin_m=0.5
    )
    env = ShadowEnvironment(
        request,
        CaptureConfig(pixels=request.config.pixels),
        lambda: capture,
        CornerTelemetry(),
        Desktop(),
        task,
    )
    r = run_experiment(
        request, realtime_environment=env, numeric_actor_factory=lambda: actor
    ).summary["realtime"]
    assert any(d["status"] == "accepted" for d in r["decisions"]) is accepted
