"""Opt-in, bounded original BGRA samples for resolution diagnostics only."""

from __future__ import annotations

import hashlib
import importlib
import threading
from collections import Counter
from queue import Empty, Queue
from typing import Any

from fh5.capture.pipeline import CaptureRun, Work
from fh5.observation.numeric import NumericFrame
from fh5.reporting.numeric import preview_png


class RawSamples:
    def __init__(self, request: CaptureRun) -> None:
        self.request = request
        self.directory = request.output_dir / "source-samples"
        self.queue: Queue[tuple[Work, NumericFrame]] = Queue(maxsize=1)
        self.lock = threading.Lock()
        self.done, self.aborted = threading.Event(), threading.Event()
        self.pending = False
        self.accepted = self.accepted_bytes = 0
        self.last_source: int | None = None
        self.counts: Counter[str] = Counter()
        self.records: list[dict[str, Any]] = []
        self.error: str | None = None
        self.worker: threading.Thread | None = None
        if request.raw_sample_limit:
            self.directory.mkdir()
            self.worker = threading.Thread(
                target=self._work, name="fh5-source-samples", daemon=True
            )
            self.worker.start()

    def submit(self, work: Work, frame: NumericFrame) -> None:
        raw = work.event.frame
        assert raw is not None
        with self.lock:
            if self.done.is_set() or self.error or self.accepted >= self.request.raw_sample_limit:
                return
            if (
                self.last_source is not None
                and frame.source_time_ns - self.last_source
                < self.request.raw_sample_interval_s * 1e9
            ):
                return
            if self.pending:
                self.counts["busy_skipped"] += 1
                return
            if self.accepted_bytes + len(raw.bgra) > self.request.raw_sample_bytes:
                self.counts["byte_budget_skipped"] += 1
                return
            self.pending = True
            self.accepted += 1
            self.accepted_bytes += len(raw.bgra)
            self.last_source = frame.source_time_ns
            self.queue.put_nowait((work, frame))

    def _work(self) -> None:
        try:
            while not self.aborted.is_set():
                try:
                    work, frame = self.queue.get(timeout=0.02)
                except Empty:
                    if self.done.is_set():
                        break
                    continue
                raw = work.event.frame
                assert raw is not None
                base = self.directory / f"{work.epoch}-{work.frame_id}"
                base.with_suffix(".bgra").write_bytes(raw.bgra)
                image = importlib.import_module("PIL.Image")
                original = image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                original.save(base.with_suffix(".png"), compress_level=1)
                model_preview = base.with_name(base.name + "-model.png")
                model_preview.write_bytes(preview_png(bytes(frame.pixels), frame.size))
                record = {
                    "frame_id": frame.frame_id,
                    "epoch": frame.epoch,
                    "source_time_ns": frame.source_time_ns,
                    "time_quality": frame.time_quality,
                    "source_layout": frame.source_layout,
                    "size": list(raw.size),
                    "format": "BGRA uint8 HWC",
                    "path": base.with_suffix(".bgra")
                    .relative_to(self.request.output_dir)
                    .as_posix(),
                    "sha256": hashlib.sha256(raw.bgra).hexdigest(),
                    "preview": base.with_suffix(".png")
                    .relative_to(self.request.output_dir)
                    .as_posix(),
                    "model_preview": model_preview.relative_to(self.request.output_dir).as_posix(),
                }
                with self.lock:
                    if not self.aborted.is_set():
                        self.records.append(record)
                    self.pending = False
                del work, frame, raw, original
        except Exception as error:
            with self.lock:
                self.error = f"{type(error).__name__}: {error}"

    def close(self) -> dict[str, Any]:
        self.done.set()
        if self.worker is not None:
            self.worker.join(timeout=2)
            if self.worker.is_alive():
                self.aborted.set()
        with self.lock:
            return {
                "enabled": self.request.raw_sample_limit > 0,
                "accepted": self.accepted,
                "accepted_bytes": self.accepted_bytes,
                "sample_limit": self.request.raw_sample_limit,
                "byte_limit": self.request.raw_sample_bytes,
                "records": list(self.records),
                "error": self.error,
                "resources_released": self.worker is None or not self.worker.is_alive(),
                **self.counts,
            }
