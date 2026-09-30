"""Two capture workers with one pending slot; archive never blocks either worker."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any, Protocol

from fh5.capture import (
    CaptureConfig,
    CaptureEvent,
    CaptureRun,
    CaptureState,
    next_preprocess_time,
    preprocess,
)
from fh5.numeric_images import NumericFrame
from fh5.numeric_recording import NumericArchive
from fh5.numeric_report import write_numeric_report

if TYPE_CHECKING:
    from fh5.experiment import RunResult


class CaptureSource(Protocol):
    source_kind: str

    def capture(self) -> CaptureEvent: ...
    def close(self) -> None: ...


class LiveCapture:
    def __init__(self, config: CaptureConfig, factory: Callable[[], CaptureSource]) -> None:
        self.config, self.factory = config, factory
        self.state = CaptureState(config)
        self.condition = threading.Condition()
        self.done = threading.Event()
        self.fault: str | None = None
        self.source_kind = "uninitialized"
        self.release_failed = False
        self.capture_started_ns: int | None = None
        self.metrics: dict[str, deque[float]] = {
            name: deque(maxlen=72_000)
            for name in (
                "capture_call_ms",
                "pending_wait_ms",
                "preprocess_ms",
                "source_to_ready_ms",
            )
        }
        self.workers = [
            threading.Thread(target=self._capture, name="fh5-dxgi-capture", daemon=True),
            threading.Thread(target=self._preprocess, name="fh5-rgb-preprocess", daemon=True),
        ]
        for worker in self.workers:
            worker.start()

    def _offer(self, event: CaptureEvent) -> None:
        with self.condition:
            self.state.offer(event)
            self.condition.notify_all()

    def _capture(self) -> None:
        source = None
        attempts = 0
        try:
            next_tick = time.perf_counter_ns()
            while not self.done.is_set():
                self.capture_started_ns = time.perf_counter_ns()
                try:
                    if source is None:
                        source = self.factory()
                        self.source_kind = source.source_kind
                    event = source.capture()
                except Exception as error:
                    attempts += 1
                    self._offer(
                        CaptureEvent(
                            time.perf_counter_ns(),
                            boundary="capture_error",
                            reason=type(error).__name__,
                        )
                    )
                    if source is not None:
                        source.close()
                        source = None
                    if attempts >= 3:
                        self.fault = f"capture_retry_limit: {error}"
                        self.done.set()
                        break
                    self.done.wait(0.1)
                    continue
                finally:
                    started = self.capture_started_ns
                    if started is not None:
                        with self.condition:
                            self.metrics["capture_call_ms"].append(
                                (time.perf_counter_ns() - started) / 1e6
                            )
                    self.capture_started_ns = None
                if event.reason == "user_stop":
                    self.fault = "user_stop"
                    self.done.set()
                    break
                if not self.done.is_set():
                    self._offer(event)
                period = 1_000_000_000 // self.config.capture_hz
                next_tick = max(next_tick + period, time.perf_counter_ns())
                self.done.wait(max(0, (next_tick - time.perf_counter_ns()) / 1e9))
        except Exception as error:
            self.fault = f"capture_error: {error}"
            self.done.set()
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as error:
                    self.release_failed = True
                    self.fault = f"capture_close_error: {error}"
            with self.condition:
                self.condition.notify_all()

    def _preprocess(self) -> None:
        try:
            while not self.done.is_set():
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.state.pending is not None or self.done.is_set(), timeout=0.05
                    )
                    if self.done.is_set():
                        break
                    work = self.state.take()
                if work is None:
                    continue
                start = time.perf_counter_ns()
                # The frame bytes are independent of the raw producer's next write.
                frame = preprocess(work, self.config, start)
                ready = time.perf_counter_ns()
                frame = replace(frame, preprocess_ready_ns=ready)
                with self.condition:
                    if not self.done.is_set():
                        self.state.complete(work, frame)
                    self.metrics["pending_wait_ms"].append((start - work.event.received_ns) / 1e6)
                    self.metrics["preprocess_ms"].append((ready - start) / 1e6)
                    self.metrics["source_to_ready_ms"].append((ready - frame.source_time_ns) / 1e6)
                next_tick = next_preprocess_time(start, ready, self.config)
                self.done.wait(max(0, (next_tick - time.perf_counter_ns()) / 1e9))
                del work, frame
        except Exception as error:
            self.fault = f"preprocess_error: {error}"
            self.done.set()

    def snapshot(self) -> tuple[dict[str, Any], tuple[NumericFrame, ...]]:
        with self.condition:
            return self.state.select(time.perf_counter_ns())

    def close(self) -> dict[str, Any]:
        self.done.set()
        with self.condition:
            self.condition.notify_all()
        for worker in self.workers:
            worker.join(timeout=0.5)
        with self.condition:
            counts = dict(self.state.counts)
            events = list(self.state.events)
            metrics = {name: percentiles(list(values)) for name, values in self.metrics.items()}
            self.state.pending = None
            self.state.history.clear()
        released = not self.release_failed and all(not w.is_alive() for w in self.workers)
        return {
            **counts,
            "events": events,
            "metrics": metrics,
            "source_kind": self.source_kind,
            "resources_released": released,
            "fault": self.fault,
            "pending_capacity": 1,
            "history_capacity": self.config.history_capacity,
        }


def percentiles(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)

    def percentile(p: float) -> float | None:
        if not ordered:
            return None
        index = (len(ordered) - 1) * p
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return {
        "count": len(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(values) if values else None,
    }


def run_capture(
    request: CaptureRun,
    factory: Callable[[], CaptureSource],
    activity: Callable[[], dict[str, Any] | None] | None = None,
) -> RunResult:
    from fh5.experiment import RunResult, _write_json

    directory = request.output_dir
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("pixels", "previews", "inputs"):
        (directory / name).mkdir()
    archive = NumericArchive(directory, 8, 32 * 1024**2)
    pipeline = LiveCapture(request.config, factory)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter_ns()
    deadline = started + int(request.seconds * 1e9)
    next_tick = started
    archive_bytes = 0
    stop_reason = "duration_limit"
    execution_error: str | None = None
    try:
        while time.perf_counter_ns() < deadline and not pipeline.done.is_set():
            if (
                pipeline.capture_started_ns
                and time.perf_counter_ns() - pipeline.capture_started_ns > 2e9
            ):
                pipeline.fault = "capture_timeout"
                break
            context = activity() if activity is not None else None
            row, frames = pipeline.snapshot()
            row["index"] = len(rows)
            row["activity"] = context
            if frames:
                size = sum(frame.pixels.nbytes for frame in frames)
                if archive_bytes + size > request.archive_limit_bytes:
                    row["archive_reason"] = "archive_total_budget"
                else:
                    row["archive_reason"] = archive.submit(dict(row), frames)
                    if row["archive_reason"] is None:
                        archive_bytes += size
            rows.append(row)
            next_tick = max(
                next_tick + 1_000_000_000 // request.config.observation_hz, time.perf_counter_ns()
            )
            pipeline.done.wait(max(0, (next_tick - time.perf_counter_ns()) / 1e9))
        if pipeline.fault:
            stop_reason = pipeline.fault
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception as error:
        stop_reason = "observation_error"
        execution_error = str(error)
    finally:
        observation_ended = time.perf_counter_ns()
        stats = pipeline.close()
        archive_stats = archive.close()
    for row in rows:
        row["archive"] = archive.records.get(row["decision_id"])
        row["exact_replay_available"] = row["archive"] is not None
    summary = {
        "version": 1,
        "contract": request.config.pixels.metadata(),
        "config": asdict(request.config),
        "model": None,
        "stop_reason": stop_reason,
        "execution_error": execution_error,
        "timing_kind": "measured",
        "commands_sent": False,
        "input_conditions": request.input_conditions,
        "decisions": rows,
        "pipeline": stats,
        "archive": archive_stats,
        "resources_released": stats["resources_released"] and archive_stats["resources_released"],
        "elapsed_s": (observation_ended - started) / 1e9,
        "dynamic_game_validation": False,
        "moving_observations": sum(
            1
            for row in rows
            if row["status"] == "ready"
            and row["activity"]
            and row["activity"].get("fresh")
            and row["activity"].get("is_race_on") == 1
            and row["activity"].get("speed_mps", 0) > 1
        ),
    }
    _write_json(directory / "capture.json", summary)
    report = directory / "report.html"
    write_numeric_report(report, summary, directory)
    return RunResult({}, [], stats["events"], {"capture": summary}, report)
