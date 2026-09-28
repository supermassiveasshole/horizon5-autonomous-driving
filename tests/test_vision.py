"""Color recording behavior at the agreed experiment-run seam."""

import base64
import json
import socket
import struct
import time
from pathlib import Path

import pytest
from test_experiment import changed_packet, config_file

from fh5.experiment import Packet, Replay, run_experiment
from fh5.vision import ColorFrame, VisionInput, VisionRecord

# One red RGB pixel, encoded independently of the recorder.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def packet(ms: int, game_ms: int = 1000) -> Packet:
    return Packet(ms * 1_000_000, "2026-09-29T00:00:00+00:00", changed_packet(game_ms, 10))


def frame(start: int = 1000, end: int = 1020, available: int = 1030) -> ColorFrame:
    return ColorFrame(
        start * 1_000_000, end * 1_000_000, available * 1_000_000, PNG, "png", (1, 1), (1920, 1080)
    )


class Observations:
    source_kind = "synthetic"

    def __init__(self, batches: list[VisionInput]):
        self.batches = iter(batches)
        self.clock = 900_000_000
        self.closed = False

    def now_ns(self) -> int:
        return self.clock

    def read(self, period_s: float) -> VisionInput:
        self.clock += 200_000_000
        return next(self.batches, VisionInput(stop_requested=True))

    def close(self) -> bool:
        self.closed = True
        return True


def test_color_replay_distinguishes_available_telemetry_from_future_packets(tmp_path: Path):
    env = Observations(
        [
            VisionInput(packets=(packet(990), packet(1025), packet(1040)), frame=frame()),
        ]
    )
    directory = tmp_path / "rgb"
    result = run_experiment(VisionRecord(config_file(tmp_path), directory), vision_environment=env)
    replay = run_experiment(Replay(directory, tmp_path / "elsewhere" / "replay.html"))
    visual = replay.summary["vision"]
    saved = visual["frames"][0]
    assert saved["online"]["sample"]["packet_index"] == 1
    assert saved["online"]["telemetry_age_ms"] == 5
    assert saved["posthoc_after_capture"]["packet_index"] == 1
    assert saved["capture_start_ns"] == 1_000_000_000
    assert saved["available_ns"] == 1_030_000_000
    assert saved["stored_ns"] >= saved["available_ns"]
    assert saved["image_url"].endswith("rgb/frames/000000.png")
    assert (directory / "frames/000000.png").read_bytes() == PNG
    assert visual["integrity_errors"] == []
    assert visual["session"]["camera_pose"] == "dynamic_unknown"
    assert result.summary["vision"]["frame_count"] == 1
    assert env.closed
    report = json.loads(replay.report_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["summary"]["vision"]["frames"][0]["online"] == saved["online"]


@pytest.mark.parametrize("disruption", ["bad_packet", "restart", "pause", "focus", "gap"])
def test_visual_pairing_rejects_discontinuities_inside_capture(tmp_path: Path, disruption: str):
    middle = packet(1010, 10 if disruption == "restart" else 1010)
    events = ()
    first = packet(400 if disruption == "gap" else 990, 990)
    if disruption == "bad_packet":
        middle = Packet(middle.received_monotonic_ns, middle.received_utc, b"invalid")
    if disruption == "pause":
        paused = bytearray(middle.payload)
        struct.pack_into("<i", paused, 0, 0)
        middle = Packet(middle.received_monotonic_ns, middle.received_utc, bytes(paused))
    if disruption == "focus":
        events = ({"kind": "focus_lost", "observed_ns": 1_010_000_000},)
    env = Observations([VisionInput(packets=(first, middle), frame=frame(), events=events)])
    result = run_experiment(
        VisionRecord(config_file(tmp_path), tmp_path / "run"), vision_environment=env
    )
    saved = result.summary["vision"]["frames"][0]
    assert saved["online"]["usable"] is False
    assert saved["online"]["reason"]
    assert saved["posthoc_after_capture"] is None


@pytest.mark.parametrize("end", ["fault", "budget", "interrupt"])
def test_stopping_preserves_written_color_and_telemetry(tmp_path: Path, end: str):
    class Stopping(Observations):
        def read(self, period_s: float) -> VisionInput:
            if end == "interrupt" and self.clock >= 1_100_000_000:
                raise KeyboardInterrupt
            return super().read(period_s)

    last = VisionInput(fault="camera_disconnected")
    if end == "budget":
        last = VisionInput(
            frame=ColorFrame(
                1200000000, 1210000000, 1220000000, PNG * 100, "png", (1, 1), (1920, 1080)
            )
        )
    env = Stopping([VisionInput(packets=(packet(990),), frame=frame()), last])
    directory = tmp_path / "partial"
    result = run_experiment(
        VisionRecord(config_file(tmp_path), directory, max_bytes=4096), vision_environment=env
    )
    vision = result.summary["vision"]
    assert vision["frame_count"] == 1
    assert (
        vision["session"]["stop_reason"]
        == {"fault": "camera_disconnected", "budget": "byte_limit", "interrupt": "interrupted"}[end]
    )
    assert vision["session"]["resources_released"] is True
    assert result.summary["valid_packets"] == 1
    assert env.closed
    replay = run_experiment(Replay(directory, tmp_path / "partial.html"))
    assert replay.summary["vision"]["frame_count"] == 1


@pytest.mark.parametrize("missing", ["image", "telemetry", "stale_image", "future_only"])
def test_partial_observations_do_not_manufacture_synchronized_data(tmp_path: Path, missing: str):
    packets = () if missing == "telemetry" else (packet(990), packet(1025))
    if missing == "future_only":
        packets = (packet(1040),)
    picture = None if missing == "image" else frame(800 if missing == "stale_image" else 1000)
    env = Observations([VisionInput(packets=packets, frame=picture)])
    result = run_experiment(
        VisionRecord(config_file(tmp_path), tmp_path / "run"), vision_environment=env
    )
    visual = result.summary["vision"]
    if missing == "image":
        assert visual["frame_count"] == 0
        assert result.summary["valid_packets"] == 2
    else:
        assert visual["frame_count"] == 1
        assert visual["frames"][0]["online"]["usable"] is False
        if missing in ("telemetry", "future_only"):
            assert visual["frames"][0]["online"]["sample"] is None


@pytest.mark.parametrize("corruption", ["truncated", "handoff_time"])
def test_replay_recovers_partial_journal_and_detects_tampered_images(
    tmp_path: Path, corruption: str
):
    directory = tmp_path / "run"
    run_experiment(
        VisionRecord(config_file(tmp_path), directory),
        vision_environment=Observations(
            [VisionInput(packets=(packet(990), packet(1025)), frame=frame())]
        ),
    )
    (directory / "frames/000000.png").write_bytes(b"damaged")
    with (directory / "vision.jsonl").open("ab") as file:
        if corruption == "truncated":
            file.write(b'{"kind":')
        else:
            broken = json.loads((directory / "vision.jsonl").read_bytes().splitlines()[0])
            broken["delivered_ns"] = "damaged"
            file.write(json.dumps(broken).encode("utf-8") + b"\n")
    result = run_experiment(Replay(directory, tmp_path / "recovered.html"))
    visual = result.summary["vision"]
    assert visual["frame_count"] == 1
    assert len(visual["integrity_errors"]) >= 2
    assert visual["frames"][0]["online"]["usable"] is False
    assert visual["frames"][0]["image_url"] is None


def test_live_adapter_records_loopback_udp_and_releases_read_only_capture(tmp_path: Path):
    from fh5.live_vision import LiveVisionEnvironment

    class Desktop:
        def focused(self):
            return True

        def stop_requested(self):
            return False

    class Camera:
        closed = False

        def capture(self):
            clock = time.perf_counter_ns()
            return ColorFrame(clock, clock, clock, PNG, "png", (1, 1), (1920, 1080))

        def close(self):
            self.closed = True

    camera = Camera()
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender,
    ):
        receiver.bind(("127.0.0.1", 0))
        env = LiveVisionEnvironment(receiver, Desktop(), 0.1, frame_factory=lambda: camera)
        sender.sendto(packet(990).payload, receiver.getsockname())
        result = run_experiment(
            VisionRecord(config_file(tmp_path), tmp_path / "live", seconds=0.3),
            vision_environment=env,
        )
    assert result.summary["valid_packets"] == 1
    assert result.summary["vision"]["frame_count"] >= 1
    assert result.summary["vision"]["session"]["resources_released"] is True
    assert camera.closed


@pytest.mark.parametrize("fault", ["capture", "close"])
def test_live_camera_failures_are_reported_without_losing_telemetry(tmp_path: Path, fault: str):
    from fh5.live_vision import LiveVisionEnvironment

    class Desktop:
        def focused(self):
            return True

        def stop_requested(self):
            return False

    class Camera:
        def capture(self):
            if fault == "capture":
                raise OSError("camera disconnected")
            stamp = time.perf_counter_ns()
            return ColorFrame(stamp, stamp, stamp, PNG, "png", (1, 1), (1920, 1080))

        def close(self):
            if fault == "close":
                raise OSError("cannot release camera")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        env = LiveVisionEnvironment(receiver, Desktop(), 0.1, frame_factory=Camera)
        result = run_experiment(
            VisionRecord(config_file(tmp_path), tmp_path / "run", seconds=0.2),
            vision_environment=env,
        )
    session = result.summary["vision"]["session"]
    if fault == "capture":
        assert session["stop_reason"] == "capture_error"
    else:
        assert session["resources_released"] is False


def test_delayed_image_handoff_keeps_original_time_and_is_not_usable(tmp_path: Path):
    class Delayed(Observations):
        def read(self, period_s: float) -> VisionInput:
            result = super().read(period_s)
            self.clock += 900_000_000
            return result

    env = Delayed([VisionInput(packets=(packet(990), packet(1025)), frame=frame())])
    result = run_experiment(
        VisionRecord(config_file(tmp_path), tmp_path / "delayed"), vision_environment=env
    )
    saved = result.summary["vision"]["frames"][0]
    assert saved["available_ns"] == 1_030_000_000
    assert saved["online"]["usable"] is False
    assert saved["online"]["reason"] == "stale_observation"


def test_future_diagnostic_packet_cannot_cross_a_focus_boundary(tmp_path: Path):
    env = Observations(
        [
            VisionInput(
                packets=(packet(990), packet(1040)),
                frame=frame(),
                events=({"kind": "focus_lost", "observed_ns": 1_035_000_000},),
            )
        ]
    )
    result = run_experiment(
        VisionRecord(config_file(tmp_path), tmp_path / "focus"), vision_environment=env
    )
    saved = result.summary["vision"]["frames"][0]
    assert saved["online"]["sample"]["packet_index"] == 0
    assert saved["online"]["usable"] is True
    assert saved["posthoc_after_capture"] is None
