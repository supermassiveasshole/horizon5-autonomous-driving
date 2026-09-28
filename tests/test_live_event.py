"""Real UDP and clock; external pixels and virtual-device writes are substituted."""

import json
import socket
import struct
import threading
import time

import pytest
from test_event_run import PATTERNS, verified_config
from test_experiment import sample_packet
from test_live_control import Desktop

from fh5.control import Command
from fh5.events import EventRun, ScreenFrame
from fh5.experiment import run_experiment
from fh5.live import MenuButton
from fh5.live_event import BoundedFrames, LiveEventEnvironment


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "user_stop",
        "focus_lost",
        "capture_failed",
        "screen_stale",
        "user_stop_idle",
        "focus_lost_idle",
        "capture_blocked",
        "close_blocked",
    ],
)
def test_live_event_releases_buttons_and_never_drives(tmp_path, fault):
    class Device:
        def __init__(self):
            self.screen = "ready"
            self.sent = []
            self.closed = False

        def send(self, command):
            self.sent.append(command)
            if command == MenuButton("A"):
                self.screen = "driving"

        def close(self):
            self.closed = True

    device = Device()
    unblock = threading.Event()
    frame_closed = threading.Event()

    class Frames:
        closed = False
        transient_signal = False

        def capture(self):
            started = time.perf_counter_ns()
            if fault == "capture_blocked":
                unblock.wait(5)
            if fault in ("user_stop_idle", "focus_lost_idle"):
                self.transient_signal = True
                time.sleep(0.12)
                self.transient_signal = False
            if device.screen == "driving":
                if fault == "capture_failed":
                    raise OSError("capture unavailable")
                if fault == "screen_stale":
                    time.sleep(0.55)
            return ScreenFrame(started, 4, 2, PATTERNS[device.screen])

        def close(self):
            if fault == "close_blocked":
                unblock.wait(5)
            self.closed = True
            frame_closed.set()

    class FaultDesktop(Desktop):
        def focused(self):
            return not (
                (fault == "focus_lost" and device.screen == "driving")
                or (fault == "focus_lost_idle" and frames.transient_signal)
            )

        def stop_requested(self):
            return (fault == "user_stop" and device.screen == "driving") or (
                fault == "user_stop_idle" and frames.transient_signal
            )

    path = verified_config(tmp_path)
    root = json.loads(path.read_text(encoding="utf-8"))
    root["event_run"]["max_attempts"] = 1
    path.write_text(json.dumps(root), encoding="utf-8")
    frames = Frames()
    done = threading.Event()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        address = receiver.getsockname()

        def feed():
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                while not done.wait(0.01):
                    data = bytearray(sample_packet())
                    struct.pack_into("<fff", data, 244, 0, 0, 0)
                    struct.pack_into("<f", data, 256, 0)
                    struct.pack_into("<I", data, 4, (time.perf_counter_ns() // 1_000_000) % 2**32)
                    sender.sendto(data, address)

        thread = threading.Thread(target=feed)
        thread.start()
        started = time.monotonic()
        try:
            source = BoundedFrames(lambda: frames) if fault != "screen_stale" else frames
            environment = LiveEventEnvironment(receiver, device, FaultDesktop(), source)
            result = run_experiment(EventRun(path, tmp_path / "run"), event_environment=environment)
            elapsed = time.monotonic() - started
        finally:
            done.set()
            thread.join()
            unblock.set()
            assert frame_closed.wait(1)
    expected = {
        "capture_failed": "interface_error",
        "capture_blocked": "interface_error",
        "close_blocked": "attempt_limit",
    }.get(fault, (fault or "attempt_limit").removesuffix("_idle"))
    assert result.summary["event_run"]["stop_reason"] == expected
    assert result.summary["event_run"]["release_sent"] is True
    if fault in (None, "close_blocked"):
        assert result.summary["event_run"]["attempts"][0]["outcome"] == "failed"
    else:
        assert result.summary["event_run"]["attempts"] == []
    if expected in ("user_stop", "focus_lost"):
        assert any(
            e["kind"] == "adapter_event" and e["detail"]["reason"] == expected
            for e in result.events
        )
    if fault in ("user_stop_idle", "focus_lost_idle", "capture_blocked"):
        assert all(command == Command(0, 0, 0) for command in device.sent)
    else:
        assert MenuButton("A") in device.sent
    assert all(command == MenuButton("A") or command == Command(0, 0, 0) for command in device.sent)
    assert device.sent[-1] == Command(0, 0, 0)
    assert device.closed and frames.closed
    if fault in ("capture_blocked", "close_blocked"):
        assert elapsed < 3  # The blocked external operation needs five seconds without supervision.
