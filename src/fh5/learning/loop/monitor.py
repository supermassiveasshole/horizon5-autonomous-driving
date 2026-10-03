"""Bounded background disk observation; control callbacks only inspect memory."""

from __future__ import annotations

import math
import shutil
import threading
import time
from pathlib import Path
from typing import Any


def validate_monitor(value: Any) -> dict[str, float]:
    if (
        not isinstance(value, dict)
        or set(value) != {"interval_seconds", "timeout_seconds"}
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value.values())
        or not 0.05 <= value["interval_seconds"] <= 5
        or not 0.1 <= value["timeout_seconds"] <= 10
        or value["interval_seconds"] * 2 > value["timeout_seconds"]
    ):
        raise ValueError("Storage monitor requires finite interval and observation timeout")
    return {k: float(v) for k, v in value.items()}


class StorageMonitor:
    def __init__(self, root: Path, budget: dict[str, Any], config: dict[str, float]) -> None:
        self.root, self.config = root, config
        self.threshold = budget["min_free_bytes"] + budget["stop_reserve_bytes"]
        self.done, self.ready = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.last_completed_ns = time.perf_counter_ns()
        self.stop_reason: str | None = None
        self.error: str | None = None
        self.samples = 0
        self.minimum: int | None = None
        self.last_free: int | None = None
        self.maximum_query_seconds = 0.0
        self.thread = threading.Thread(
            target=self._run, name="fh5-learning-storage-monitor", daemon=True
        )

    def start(self) -> None:
        try:
            self.thread.start()
        except Exception as error:
            with self.lock:
                self.stop_reason = "storage_monitor_error"
                self.error = f"{type(error).__name__}: {error}"
            self.done.set()
            self.ready.set()
        self.ready.wait(self.config["timeout_seconds"])

    def reason(self) -> str | None:
        with self.lock:
            if self.stop_reason is None and (
                time.perf_counter_ns() - self.last_completed_ns
                >= self.config["timeout_seconds"] * 1e9
            ):
                self.stop_reason = "storage_monitor_stalled"
                self.done.set()
            return self.stop_reason

    def _run(self) -> None:
        while not self.done.is_set():
            started = time.perf_counter_ns()
            try:
                free = shutil.disk_usage(self.root).free
            except Exception as error:
                with self.lock:
                    if self.stop_reason is None:
                        self.stop_reason = "storage_monitor_error"
                        self.error = f"{type(error).__name__}: {error}"
                self.ready.set()
                self.done.set()
                return
            ended = time.perf_counter_ns()
            with self.lock:
                self.samples += 1
                self.last_free = free
                self.minimum = free if self.minimum is None else min(self.minimum, free)
                self.maximum_query_seconds = max(
                    self.maximum_query_seconds, (ended - started) / 1e9
                )
                if self.stop_reason is None:
                    if ended - started >= self.config["timeout_seconds"] * 1e9:
                        self.stop_reason = "storage_monitor_stalled"
                    elif free < self.threshold:
                        self.stop_reason = "storage_budget_exhausted"
                self.last_completed_ns = ended
                if self.stop_reason:
                    self.done.set()
            self.ready.set()
            self.done.wait(self.config["interval_seconds"])

    def close(self) -> dict[str, Any]:
        self.done.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=0.5)
        with self.lock:
            return {
                "version": 1,
                "configuration": self.config,
                "threshold_free_bytes": self.threshold,
                "samples": self.samples,
                "minimum_free_bytes": self.minimum,
                "last_free_bytes": self.last_free,
                "maximum_query_seconds": self.maximum_query_seconds,
                "stop_reason": self.stop_reason,
                "error": self.error,
                "resources_released": not self.thread.is_alive(),
                "close_timeout_seconds": 0.5,
                "reservation_kind": "observed_headroom_not_filesystem_quota",
                "files_deleted": 0,
            }
