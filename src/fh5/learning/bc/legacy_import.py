"""Read saved legacy numerical input packages without image decoding."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

from fh5.observation.numeric import NumericDecision, NumericFrame, PixelContract


class PreparedNumericSource:
    """Offline disk reads are isolated from the numerical prediction caller."""

    def __init__(self, directory: Path, max_cache_bytes: int = 16 * 1024**2) -> None:
        if type(max_cache_bytes) is not int or not 1 <= max_cache_bytes <= 1024**3:
            raise ValueError("Invalid numerical source cache budget")
        self.directory = directory
        self.manifest = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
        self.contract = PixelContract.from_metadata(self.manifest["contract"])
        if self.manifest.get("version") != 1 or not 1 <= len(self.manifest["decisions"]) <= 5000:
            raise ValueError("Invalid prepared numerical source")
        identities: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.manifest["decisions"]:
            if len(row["frames"]) != len(self.contract.history_offsets_ms):
                raise ValueError("Prepared frame count differs from contract")
            for frame in row["frames"]:
                identity = (frame["epoch"], frame["frame_id"])
                if identity in identities and identities[identity] != frame:
                    raise ValueError("Prepared frame identity reused with different metadata")
                identities[identity] = frame
        self.max_cache_bytes = max_cache_bytes
        self.queue: Queue[NumericDecision | Exception | None] = Queue(2)
        self.stopped = threading.Event()
        self.closed = False
        self.peak_cached_bytes = 0
        self.worker: threading.Thread | None = None

    def _put(self, item: NumericDecision | Exception | None) -> bool:
        while not self.stopped.is_set():
            try:
                self.queue.put(item, timeout=0.02)
                return True
            except Full:
                pass
        return False

    def _read(self) -> None:
        from fh5.observation.recording import read_numeric_frame

        cache: OrderedDict[tuple[str, str], NumericFrame] = OrderedDict()
        size = 0
        try:
            for row in self.manifest["decisions"]:
                if self.stopped.is_set():
                    break
                chosen = []
                for frame in row["frames"]:
                    key = (frame["epoch"], frame["frame_id"])
                    if key not in cache:
                        loaded = read_numeric_frame(self.directory, frame, self.max_cache_bytes)
                        while cache and size + loaded.pixels.nbytes > self.max_cache_bytes:
                            _, previous = cache.popitem(last=False)
                            size -= previous.pixels.nbytes
                        cache[key] = loaded
                        size += loaded.pixels.nbytes
                        self.peak_cached_bytes = max(self.peak_cached_bytes, size)
                    cache.move_to_end(key)
                    chosen.append(cache[key])
                if not self._put(
                    NumericDecision(
                        row["decision_id"],
                        row["epoch"],
                        row["decision_ns"],
                        tuple(chosen),
                        row["actor"],
                        row.get("supervision"),
                    )
                ):
                    break
        except Exception as error:
            self._put(error)
        finally:
            cache.clear()
            self._put(None)

    def __iter__(self) -> Iterator[NumericDecision]:
        if self.worker is not None or self.closed:
            raise ValueError("A prepared numerical source can only be consumed once")
        self.worker = threading.Thread(target=self._read, name="fh5-numeric-source", daemon=True)
        self.worker.start()
        while True:
            try:
                item = self.queue.get(timeout=0.1)
            except Empty:
                if not self.worker.is_alive():
                    raise RuntimeError("Numerical source stopped without completion") from None
                continue
            if isinstance(item, Exception):
                raise item
            if item is None:
                break
            yield item

    def close(self) -> bool:
        self.stopped.set()
        if self.worker is not None:
            self.worker.join(timeout=2)
        while not self.queue.empty():
            self.queue.get_nowait()
        self.closed = self.worker is None or not self.worker.is_alive()
        return self.closed
