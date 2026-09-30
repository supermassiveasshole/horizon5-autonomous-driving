"""Bounded archive worker with atomic, individually verifiable sealed blocks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

from fh5.collection import CollectionConfig, metadata_budget
from fh5.numeric_images import NumericFrame

WriteFile = Callable[[Path, bytes], None]


def read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Collection asset exceeds bounded limit: " + path.name)
    return payload


def collection_complete(status: dict[str, Any]) -> bool:
    environment = status.get("environment")
    return (
        type(status.get("seen_rows")) is int
        and status["seen_rows"] > 0
        and status.get("unsealed_rows") == 0
        and status.get("archive_error") is None
        and status.get("archive_released") is True
        and isinstance(environment, dict)
        and environment.get("resources_released") is True
        and status.get("error") is None
        and status.get("stop_reason") not in ("source_error", "source_fault", "archive_failure")
    )


def encode(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def write_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class CollectionArchive:
    def __init__(
        self, root: Path, config: CollectionConfig, binding: str, write: WriteFile = write_file
    ) -> None:
        self.root, self.config, self.binding, self.write = root, config, binding, write
        self.queue: Queue[tuple[dict[str, Any], tuple[NumericFrame, ...], int]] = Queue(
            config.queue_items
        )
        self.lock = threading.Lock()
        self.done, self.abort = threading.Event(), threading.Event()
        self.pending_bytes = self.peak_pending_bytes = 0
        self.offered = self.written = self.dropped = 0
        self.disk_bytes = 0
        self.error: str | None = None
        self.blocks: list[dict[str, Any]] = []
        self.progress: dict[str, Any] = {}
        self.partial_rows: list[bytes] = []
        self.partial_bytes = 0
        self.first = self.last = -1
        self.pixel_hashes: set[str] = set()
        self.partial: Path | None = None
        (root / "blocks").mkdir()
        (root / ".partial").mkdir()
        self.worker = threading.Thread(target=self._run, name="fh5-collection-archive", daemon=True)
        self.worker.start()

    def submit(
        self, row: dict[str, Any], frames: tuple[NumericFrame, ...], progress: dict[str, Any]
    ) -> bool:
        # Encoding and hashes happen only in the writer. Bound every retained field
        # before copying, including the source frame metadata kept alongside pixels.
        metadata = {k: v for k, v in row.items() if k != "packets"}
        metadata["frames"] = [f.metadata() for f in frames]
        size = (
            4096
            + metadata_budget(metadata)
            + metadata_budget(metadata["frames"])
            + metadata_budget([(p.received_monotonic_ns, p.received_utc) for p in row["packets"]])
            + sum(len(p.payload) for p in row["packets"])
            + sum(f.pixels.nbytes for f in frames)
        )
        with self.lock:
            self.offered += 1
            self.progress = deepcopy(progress)
            if (
                self.error
                or self.done.is_set()
                or self.pending_bytes + size > self.config.queue_bytes
            ):
                self.dropped += 1
                return False
            self.pending_bytes += size
            self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)
        owned = deepcopy(metadata)
        owned["packets"] = row["packets"]
        try:
            self.queue.put_nowait((owned, frames, size))
            return True
        except Full:
            with self.lock:
                self.pending_bytes -= size
                self.dropped += 1
            return False

    def _open(self) -> None:
        if len(self.blocks) >= self.config.max_blocks:
            raise OSError("collection_block_limit")
        self.partial = self.root / ".partial" / f"{len(self.blocks):06d}"
        self.partial.mkdir()
        (self.partial / "pixels").mkdir()

    def _seal(self) -> None:
        if not self.partial_rows or self.partial is None:
            return
        if self.abort.is_set():
            return
        payload = b"".join(self.partial_rows)
        self.write(self.partial / "rows.jsonl", payload)
        manifest = encode(
            {
                "version": 1,
                "session_sha256": self.binding,
                "index": len(self.blocks),
                "row_count": len(self.partial_rows),
                "first_sequence": self.first,
                "last_sequence": self.last,
                "rows_sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": self.partial_bytes,
            }
        )
        self.write(self.partial / "manifest.json", manifest)
        if self.abort.is_set():
            return
        destination = self.root / "blocks" / self.partial.name
        self.partial.rename(destination)
        self.blocks.append(
            {
                "path": str(destination.relative_to(self.root)).replace("\\", "/"),
                "sha256": hashlib.sha256(manifest).hexdigest(),
                "rows": len(self.partial_rows),
            }
        )
        self.written += len(self.partial_rows)
        self.disk_bytes += self.partial_bytes + len(manifest)
        atomic_json(
            self.root / "index.json",
            {"version": 1, "session_sha256": self.binding, "blocks": self.blocks},
        )
        self.partial, self.partial_rows, self.partial_bytes = None, [], 0
        self.pixel_hashes.clear()

    def _append(self, row: dict[str, Any], frames: tuple[NumericFrame, ...]) -> None:
        row["packets"] = [
            {
                "received_monotonic_ns": p.received_monotonic_ns,
                "received_utc": p.received_utc,
                "payload_hex": p.payload.hex(),
            }
            for p in row["packets"]
        ]
        pixels: dict[str, bytes] = {}
        for frame, metadata in zip(frames, row["frames"]):
            data = bytes(frame.pixels)
            digest = hashlib.sha256(data).hexdigest()
            metadata.update(path="pixels/" + digest + ".rgb", sha256=digest)
            pixels[digest] = data
        payload = encode(row)
        # Count full pixels before choosing a block, since the next block has a new hash cache.
        maximum = len(payload) + sum(len(p) for p in pixels.values())
        if maximum > self.config.block_bytes:
            raise OSError("collection_record_exceeds_block_budget")
        if self.partial_rows and (
            len(self.partial_rows) >= self.config.block_rows
            or self.partial_bytes + maximum > self.config.block_bytes
        ):
            self._seal()
        free = shutil.disk_usage(self.root).free
        if free < self.config.min_free_bytes + maximum + 4096:
            raise OSError("collection_disk_reserve")
        if self.disk_bytes + self.partial_bytes + maximum + 4096 > self.config.max_disk_bytes:
            raise OSError("collection_disk_budget")
        if self.partial is None:
            self._open()
        assert self.partial is not None
        for digest, data in pixels.items():
            if digest not in self.pixel_hashes:
                self.write(self.partial / "pixels" / (digest + ".rgb"), data)
                self.pixel_hashes.add(digest)
                self.partial_bytes += len(data)
        self.partial_rows.append(payload)
        self.partial_bytes += len(payload)
        if len(self.partial_rows) == 1:
            self.first = row["sequence"]
        self.last = row["sequence"]
        if len(self.partial_rows) >= self.config.block_rows:
            self._seal()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                **deepcopy(self.progress),
                "seen_rows": self.offered,
                "written_rows": self.written,
                "dropped_rows": self.dropped,
                "sealed_blocks": len(self.blocks),
                "disk_bytes": self.disk_bytes,
                "peak_pending_bytes": self.peak_pending_bytes,
                "pending_bytes": self.pending_bytes,
                "archive_error": self.error,
                "commands_sent": False,
                "complete_fraction": self.written / self.offered if self.offered else 0,
            }

    def _heartbeat(self) -> None:
        if self.abort.is_set():
            return
        atomic_json(
            self.root / "status.json",
            {
                **self.status(),
                "heartbeat_ns": time.perf_counter_ns(),
                "state": "draining" if self.done.is_set() else "recording",
                "free_bytes": shutil.disk_usage(self.root).free,
            },
        )

    def _run(self) -> None:
        heartbeat = 0.0
        try:
            while not self.abort.is_set():
                try:
                    row, frames, size = self.queue.get(timeout=0.05)
                except Empty:
                    if self.done.is_set():
                        break
                else:
                    try:
                        self._append(row, frames)
                    finally:
                        with self.lock:
                            self.pending_bytes -= size
                if time.monotonic() >= heartbeat:
                    self._heartbeat()
                    heartbeat = time.monotonic() + 1
            self._seal()
            self._heartbeat()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def close(self) -> dict[str, Any]:
        self.done.set()
        self.worker.join(timeout=2)
        released = not self.worker.is_alive()
        if not released:
            self.abort.set()
        while True:
            try:
                _, _, size = self.queue.get_nowait()
            except Empty:
                break
            with self.lock:
                self.pending_bytes -= size
        return {
            **self.status(),
            "archive_released": released,
            "unsealed_rows": self.offered - self.written,
            "blocks": list(self.blocks),
        }
