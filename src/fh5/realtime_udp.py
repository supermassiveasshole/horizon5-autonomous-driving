"""One bounded loopback receiver; timestamps mean host delivery, not game time."""

from __future__ import annotations

import select
import socket
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fh5.realtime_shadow import TelemetryBatch

if TYPE_CHECKING:
    from fh5.experiment import Packet


class UDPTelemetry:
    def __init__(self, port: int = 5300, *, receiver: socket.socket | None = None) -> None:
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError("Invalid loopback telemetry port")
        self.port, self.receiver = port, receiver
        self.closed = False

    def read(self, period_s: float) -> TelemetryBatch:
        from fh5.experiment import Packet

        if self.closed:
            raise OSError("Telemetry receiver is closed")
        if self.receiver is None:
            self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.receiver.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            self.receiver.bind(("127.0.0.1", self.port))
        receiver = self.receiver
        receiver.setblocking(False)
        if not select.select([receiver], [], [], min(max(period_s, 0), 0.01))[0]:
            return TelemetryBatch()
        packets: list[Packet] = []
        for _ in range(64):
            try:
                payload, _ = receiver.recvfrom(65535)
            except BlockingIOError:
                return TelemetryBatch(tuple(packets))
            packets.append(Packet(time.perf_counter_ns(), datetime.now(UTC).isoformat(), payload))
        # Do not drain an unbounded socket backlog or call old queued packets fresh.
        fault = "telemetry_backlog" if select.select([receiver], [], [], 0)[0] else None
        return TelemetryBatch(tuple(packets), fault)

    def close(self) -> None:
        self.closed = True
        if self.receiver is not None:
            self.receiver.close()
