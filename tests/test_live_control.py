"""Exercise real UDP, scheduling and watchdog with external desktop/driver substitutes."""

import socket
import threading
import time
from pathlib import Path

import pytest
from test_control import control_config
from test_experiment import sample_packet

from fh5.control import Command, Control
from fh5.experiment import run_experiment
from fh5.live import LiveEnvironment


class Desktop:
    def focused(self) -> bool:
        return True

    def stop_requested(self) -> bool:
        return False


class Pad:
    def __init__(self) -> None:
        self.sent: list[tuple[float, Command]] = []
        self.closed = False

    def send(self, command: Command) -> None:
        self.sent.append((time.monotonic(), command))

    def close(self) -> None:
        self.closed = True
        self.closed_at = time.monotonic()


@pytest.mark.parametrize("failures", [0, 1, 10])
def test_watchdog_releases_during_blocked_control_loop(tmp_path: Path, failures: int) -> None:
    class FaultyPad(Pad):
        remaining_failures = failures

        def send(self, command: Command) -> None:
            if (
                self.remaining_failures
                and command == Command(0, 0, 0)
                and any(c.throttle_u8 for _, c in self.sent)
            ):
                self.remaining_failures -= 1
                raise OSError("transient release failure")
            super().send(command)

    pad = FaultyPad()

    class SlowEnvironment(LiveEnvironment):
        stalled = False

        def read(self, period_s: float):
            if not self.stalled and any(command.throttle_u8 for _, command in pad.sent):
                self.stalled = True
                time.sleep(0.4)
            return super().read(period_s)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        finished = threading.Event()

        def feed() -> None:
            import struct

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                while not finished.wait(0.01):
                    data = bytearray(sample_packet())
                    struct.pack_into("<f", data, 256, 0)
                    struct.pack_into("<I", data, 4, (time.perf_counter_ns() // 1_000_000) % 2**32)
                    sender.sendto(data, receiver.getsockname())

        feeder = threading.Thread(target=feed)
        feeder.start()
        environment = SlowEnvironment(receiver, pad, Desktop())
        try:
            result = run_experiment(
                Control(control_config(tmp_path), tmp_path / "watchdog"), environment=environment
            )
        finally:
            finished.set()
            feeder.join()
    assert result.summary["control"]["stop_reason"] == (
        "interface_error" if failures == 10 else "watchdog_timeout"
    )
    events = result.summary["control"]["adapter_events"]
    assert events[0]["status"] == ("failed" if failures else "sent")
    first_active = next(i for i, (_, command) in enumerate(pad.sent) if command.throttle_u8)
    active_at = pad.sent[first_active][0]
    if failures == 10:
        assert len(events) == 3
        assert events[-1]["detach_status"] == "closed"
        assert pad.closed
        assert 0.2 < pad.closed_at - active_at < 0.4
        assert result.summary["control"]["release_sent"] is False
        return
    assert events[-1]["status"] == "sent"
    released_at, command = pad.sent[first_active + 1]
    assert command == Command(0, 0, 0)
    assert 0.2 < released_at - active_at < 0.4
    assert all(command == Command(0, 0, 0) for _, command in pad.sent[first_active + 1 :])
    assert pad.closed
