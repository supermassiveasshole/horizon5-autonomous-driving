"""Bounded priority journal; all encoding, packet hex conversion and writes run here."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any


class RealtimeJournal:
    def __init__(
        self, directory: Path, capacity: int, sink: Callable[[bytes], None] | None = None
    ) -> None:
        self.path = directory / "realtime-events.jsonl"
        self.sink = sink
        self.high: Queue[tuple[int, str, Any]] = Queue(capacity)
        self.low: Queue[tuple[int, str, Any]] = Queue(capacity)
        self.lock = threading.Lock()
        self.done, self.abort = threading.Event(), threading.Event()
        self.offered = 0
        self.written: set[int] = set()
        self.error: str | None = None
        self.worker = threading.Thread(target=self._run, name="fh5-priority-journal", daemon=True)
        self.worker.start()

    def submit(self, kind: str, row: Any) -> None:
        with self.lock:
            sequence = self.offered
            self.offered += 1
        if self.done.is_set() or self.error:
            return
        queue = self.low if kind == "packet" else self.high
        try:
            queue.put_nowait((sequence, kind, row))
        except Full:
            pass  # Every missing sequence is reported, even when the queue is full.

    def _write(self, sink: Callable[[bytes], Any]) -> None:
        while not self.abort.is_set():
            try:
                item = self.high.get_nowait()
            except Empty:
                try:
                    item = self.low.get_nowait()
                except Empty:
                    if self.done.wait(0.005):
                        break
                    continue
            sequence, kind, row = item
            if kind == "packet":
                row = {
                    "received_monotonic_ns": row.received_monotonic_ns,
                    "received_utc": row.received_utc,
                    "payload_hex": row.payload.hex(),
                }
            payload = (
                json.dumps({"sequence": sequence, "kind": kind, "data": row}, allow_nan=False)
                + "\n"
            ).encode()
            sink(payload)
            self.written.add(sequence)

    def _run(self) -> None:
        try:
            if self.sink:
                self._write(self.sink)
            else:
                with self.path.open("xb") as stream:
                    self._write(stream.write)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def close(self) -> dict[str, Any]:
        self.done.set()
        self.worker.join(timeout=1.5)
        released = not self.worker.is_alive()
        if not released:
            self.abort.set()
        missing = sorted(set(range(self.offered)) - self.written)
        return {
            "offered": self.offered,
            "written": len(self.written),
            "dropped": len(missing),
            "missing_sequences": missing,
            "error": self.error,
            "resources_released": released,
            "path": self.path.name if self.sink is None else None,
            "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()
            if released and self.path.is_file()
            else None,
            "external_sink": self.sink is not None,
        }
