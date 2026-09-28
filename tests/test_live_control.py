"""Exercise real UDP, scheduling and watchdog with external desktop/driver substitutes."""

import socket
import threading
import time
from pathlib import Path

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


def test_watchdog_releases_during_blocked_control_loop(tmp_path: Path) -> None:
    pad = Pad()

    class SlowEnvironment(LiveEnvironment):
        def read(self, period_s: float):
            if any(command.throttle_u8 for _, command in pad.sent):
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
    assert result.summary["control"]["stop_reason"] == "watchdog_timeout"
    assert result.summary["control"]["adapter_events"][0]["status"] == "sent"
    first_active = next(i for i, (_, command) in enumerate(pad.sent) if command.throttle_u8)
    active_at = pad.sent[first_active][0]
    released_at, command = pad.sent[first_active + 1]
    assert command == Command(0, 0, 0)
    assert 0.2 < released_at - active_at < 0.4
    assert all(command == Command(0, 0, 0) for _, command in pad.sent[first_active + 1 :])
    assert pad.closed
