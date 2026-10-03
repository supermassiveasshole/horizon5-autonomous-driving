"""SAC through the real numeric driving adapter; external world remains synthetic."""

import hashlib
import json
import math
import socket
import struct
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

from fh5.capture.pipeline import CaptureConfig, CaptureEvent, QpcMapping, RawCapture
from fh5.driving.realtime.driving import NumericDrivingEnvironment
from fh5.driving.realtime.model import RealtimeRun
from fh5.driving.realtime.shadow import LocalTask, ShadowEnvironment
from fh5.driving.realtime.udp import UDPTelemetry
from fh5.driving.windows import NEUTRAL
from fh5.experiment import run_experiment
from tests.evaluation.test_attempts import evidence
from tests.learning.sac.test_sac_evaluation import sac_policy as sac_policy
from tests.learning.sac.test_sac_realtime_cycle import request


class ExternalWorld:
    """Small command-responsive loopback world, not a model of FH5 physics."""

    def __init__(self, address):
        self.address = address
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.thread = None
        self.command = NEUTRAL
        self.commands = []
        self.observations = []
        self.position = 0.0
        self.capture_closed = self.actuator_closed = False
        self.failure = None
        self.steer_feedback_scale = 1.0

    def start(self):
        self.thread = threading.Thread(target=self.publish, name="synthetic-udp-world", daemon=True)
        self.thread.start()
        return Camera(self)

    def speed_for(self, command):
        return command.throttle_u8 / 255 * 4.0

    def publish(self):
        previous = time.perf_counter_ns()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                while not self.done.is_set():
                    now = time.perf_counter_ns()
                    with self.lock:
                        command = self.command
                        speed = self.speed_for(command)
                        self.position += speed * (now - previous) / 1e9
                        position = self.position
                        raw = bytearray(324)
                        struct.pack_into("<iI", raw, 0, 1, (now // 1_000_000) % 2**32)
                        struct.pack_into("<iii", raw, 212, 2941, 6, 999)
                        struct.pack_into("<ffff", raw, 244, position, 2, 0.2, speed)
                        struct.pack_into("<f", raw, 40, speed)
                        struct.pack_into("<f", raw, 56, math.pi / 2)
                        struct.pack_into("<BB", raw, 315, command.throttle_u8, command.brake_u8)
                        struct.pack_into(
                            "<b",
                            raw,
                            320,
                            round(command.steer_i16 / 32767 * 127 * self.steer_feedback_scale),
                        )
                        self.observations.append(
                            {
                                "published_ns": now,
                                "position_x_m": position,
                                "speed_mps": speed,
                                "command": asdict(command),
                                "payload_hex": bytes(raw).hex(),
                            }
                        )
                        sender.sendto(raw, self.address)
                    previous = now
                    self.done.wait(0.005)
        except Exception as error:
            self.failure = repr(error)

    def send(self, command):
        with self.lock:
            assert not self.actuator_closed
            self.command = command
            self.commands.append({"sent_ns": time.perf_counter_ns(), "command": asdict(command)})

    def close(self):
        self.actuator_closed = True

    def stop(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(2)
            assert not self.thread.is_alive(), "Synthetic UDP publisher failed to stop"


class Camera:
    source_kind = "synthetic-capture"

    def __init__(self, world):
        self.world = world

    def capture(self):
        now = time.perf_counter_ns()
        with self.world.lock:
            # BGRA is supplied at the external device seam; production does RGB conversion.
            color = bytes([51, 34, 17 + self.world.command.throttle_u8, 255])
        return CaptureEvent(
            now,
            RawCapture(
                now,
                QpcMapping(now, now, 1_000_000_000, 0),
                (2, 1),
                color * 2,
                {"client": "synthetic-loopback-world"},
            ),
        )

    def close(self):
        self.world.stop()
        self.world.capture_closed = True


class Desktop:
    def focused(self):
        return True

    def stop_requested(self):
        return False


class AdapterAttempt:
    source_kind = "synthetic"

    def __init__(self, settings):
        self.settings = settings
        self.world = self.telemetry = self.drive = None
        self.closed = False
        self.reviewed_after_release = False

    def start(self, attempt):
        assert self.world is None, "The tracer requests exactly one attempt"
        runtime = attempt.request.config
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        self.world = ExternalWorld(receiver.getsockname())
        self.telemetry = UDPTelemetry(receiver.getsockname()[1], receiver=receiver)
        task = json.loads(self.settings.task_file.read_bytes())
        route = Path(task["route_file"])
        if not route.is_absolute():
            route = self.settings.task_file.parent / route
        observations = ShadowEnvironment(
            RealtimeRun(
                self.settings.output_dir / "attempt-000/execution",
                runtime,
                seconds=self.settings.seconds_per_attempt,
            ),
            CaptureConfig(pixels=runtime.pixels),
            self.world.start,
            self.telemetry,
            Desktop(),
            LocalTask(route, task["route_sha256"], end_margin_m=0.5),
        )
        self.drive = NumericDrivingEnvironment(
            observations, lambda: self.world, source_kind="synthetic"
        )
        return self.drive

    def finish(self, recording):
        assert self.world.capture_closed and self.world.actuator_closed and self.telemetry.closed
        assert self.world.failure is None
        self.reviewed_after_release = True
        # Independent fixture-world observations accompany the existing validity fixture.
        # Neither these observations nor the validity labels use policy predictions.
        observer = recording.parent / "external-observer.json"
        observer.write_text(
            json.dumps(
                {
                    "source_kind": "synthetic",
                    "scope": "fixture truth only; not FH5 feedback or driving validation",
                    "commands": self.world.commands,
                    "observations": self.world.observations,
                    "devices_released_before_review": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        review = evidence(recording.parent, recording)
        document = json.loads(review.read_bytes())
        document["items"].append(
            {
                "id": "world",
                "path": observer.name,
                "sha256": hashlib.sha256(observer.read_bytes()).hexdigest(),
            }
        )
        document["coverage"][0]["evidence"].append("world")
        review.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return review

    def close(self):
        self.closed = True
        try:
            return self.drive.close() if self.drive else {"resources_released": True}
        finally:
            if self.world:
                self.world.stop()
            if self.telemetry:
                self.telemetry.close()


def test_sac_acts_through_numeric_adapter_and_learns_from_loopback_feedback(tmp_path, sac_policy):
    parent = {
        name: hashlib.sha256((sac_policy / name).read_bytes()).hexdigest()
        for name in ("policy.json", "policy.pt")
    }
    settings = replace(
        request(tmp_path, sac_policy),
        cycles=1,
        seconds_per_attempt=1.5,
        max_updates_per_attempt=1,
    )
    environment = AdapterAttempt(settings)
    result = run_experiment(settings, sac_realtime_environment=environment).summary["sac_cycle"]
    assert result["stop_reason"] == "budget_completed", result
    assert environment.closed and environment.reviewed_after_release
    assert result["resources_released"] and not result["commands_sent_to_game"]
    assert result["source_kind"] == "synthetic" and not result["real_driving_validated"]
    assert not result["default_changed"]
    attempt = result["attempts"][0]
    assert attempt["learner_updates"] == 1 and attempt["eligible_transitions"] >= 1
    assert attempt["inference_reload_max_error"] == 0
    assert attempt["candidate_sha256"] != attempt["sampling_checkpoint_sha256"]
    root = settings.output_dir / "attempt-000"
    execution = json.loads((root / "execution/report.json").read_bytes())
    assert execution["actor_kind"] == "frozen-numeric-sac-sampling-v1"
    assert execution["environment"]["mode"] == "numeric_driving"
    assert execution["environment"]["capture"]["source_kind"] == "synthetic-capture"
    assert execution["environment"]["telemetry"]["packets"] > 2
    assert execution["inference"]["inference_device"] == "cpu"
    policy_commands = [c for c in execution["commands"] if c["owner"] == "policy"]
    assert policy_commands and any(c["sent"] != asdict(NEUTRAL) for c in policy_commands)
    observed_commands = [c["command"] for c in environment.world.commands]
    assert all(command["sent"] in observed_commands for command in policy_commands)
    assert observed_commands[-1] == asdict(NEUTRAL)
    replay = json.loads((root / "prepared/replay.json").read_bytes())
    assert replay["source_kind"] == "synthetic" and replay["transitions"]
    assert (
        replay["source_hashes"]["execution"]
        == hashlib.sha256((root / "execution/realtime-manifest.json").read_bytes()).hexdigest()
    )
    frames = replay["transitions"][0]["current"]["frames"]
    rgb = (root / "prepared" / frames[-1]["path"]).read_bytes()
    assert len(rgb) == 64 * 36 * 3 and rgb[1:3] == bytes([34, 51])
    candidate = settings.output_dir / attempt["candidate"]
    learned = json.loads((candidate / "training-report.json").read_bytes())
    assert learned["steps_completed"] == 1 and learned["actor_updates"] == 1
    assert learned["critic_change_max"] > 0 and learned["actor_change_max"] > 0
    assert (candidate / "policy.pt").is_file()
    assert parent == {
        name: hashlib.sha256((sac_policy / name).read_bytes()).hexdigest() for name in parent
    }
