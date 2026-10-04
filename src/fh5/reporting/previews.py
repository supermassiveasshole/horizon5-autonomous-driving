"""Disposable previews with a budget independent of exact numerical evidence."""

from __future__ import annotations

import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from fh5.reporting.numeric import preview_png


class NumericPreviews:
    def __init__(self, root: Path, capacity: int, byte_limit: int) -> None:
        self.root, self.capacity, self.byte_limit = root, capacity, byte_limit
        self.queue: Queue[tuple[str, bytes, tuple[int, int]]] = Queue(capacity)
        self.lock = threading.Lock()
        self.pending = self.pending_bytes = self.peak_bytes = 0
        self.dropped = self.failed = 0
        self.known: set[str] = set()
        self.available: set[str] = set()
        self.finished, self.aborted = threading.Event(), threading.Event()
        self.worker = threading.Thread(target=self._work, name="fh5-numeric-preview", daemon=True)
        self.worker.start()

    def submit(self, digest: str, payload: bytes, size: tuple[int, int]) -> str | None:
        path = f"previews/{digest}.png"
        with self.lock:
            if path in self.known:
                return path
            if (
                self.finished.is_set()
                or self.pending >= self.capacity
                or self.pending_bytes + len(payload) > self.byte_limit
            ):
                self.dropped += 1
                return None
            self.known.add(path)
            self.pending += 1
            self.pending_bytes += len(payload)
            self.peak_bytes = max(self.peak_bytes, self.pending_bytes)
            self.queue.put_nowait((path, payload, size))
        return path

    def _work(self) -> None:
        while not self.aborted.is_set():
            try:
                path, payload, size = self.queue.get(timeout=0.01)
            except Empty:
                if self.finished.is_set():
                    break
                continue
            try:
                (self.root / path).write_bytes(preview_png(payload, size))
                with self.lock:
                    self.available.add(path)
            except Exception:
                with self.lock:
                    self.failed += 1
            finally:
                with self.lock:
                    self.pending -= 1
                    self.pending_bytes -= len(payload)
            del payload

    def close(self, timeout_s: float) -> tuple[dict[str, Any], set[str]]:
        self.finished.set()
        self.worker.join(timeout=max(0, timeout_s))
        released = not self.worker.is_alive()
        if not released:
            self.aborted.set()
        while True:
            try:
                _, payload, _ = self.queue.get_nowait()
            except Empty:
                break
            with self.lock:
                self.pending -= 1
                self.pending_bytes -= len(payload)
                self.dropped += 1
        with self.lock:
            return {
                "resources_released": released,
                "capacity": self.capacity,
                "byte_limit": self.byte_limit,
                "peak_pending_bytes": self.peak_bytes,
                "pending_after_close": self.pending,
                "dropped": self.dropped,
                "failed": self.failed,
            }, set(self.available)
