"""Bounded RGB persistence, isolated from the actuator scheduling thread."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any


class PolicyImageWriter:
    def __init__(self, directory: Path, now_ns: Callable[[], int]) -> None:
        self.directory, self.now_ns = directory, now_ns
        self.queue: Queue[tuple[dict[str, Any], bytes | None]] = Queue(maxsize=8)
        self.finishing, self.aborted = threading.Event(), threading.Event()
        self.error: str | None = None
        self.worker = threading.Thread(
            target=self._write, daemon=True, name="fh5-policy-image-writer"
        )
        self.worker.start()

    def append(self, row: dict[str, Any], pixels: bytes | None = None) -> bool:
        if self.error or self.finishing.is_set():
            return False
        try:
            # Bounded enqueue only; no filesystem operation on the control thread.
            self.queue.put((dict(row), pixels), timeout=0.001)
        except Full:
            return False
        return True

    def _write(self) -> None:
        try:
            with (self.directory / "vision.jsonl").open("x", encoding="utf-8") as journal:
                while not self.aborted.is_set():
                    try:
                        row, pixels = self.queue.get(timeout=0.01)
                    except Empty:
                        if self.finishing.is_set():
                            break
                        continue
                    if pixels is not None:
                        (self.directory / row["path"]).write_bytes(pixels)
                        row["stored_ns"] = self.now_ns()
                    if self.aborted.is_set():
                        break
                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                    journal.flush()
        except Exception as error:
            self.error = str(error)

    def close(self) -> bool:
        self.finishing.set()
        self.worker.join(timeout=1)
        if self.worker.is_alive():
            self.aborted.set()
            self.error = "Image writer did not release; pending work quarantined"
            return False
        return self.error is None
