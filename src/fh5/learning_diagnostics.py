"""Stream learner diagnostics independently of recoverable learner state."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO


def record_bytes(entry: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            entry, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


class RecordJournal:
    """One JSON record per line; unavailable diagnostics never discard the learner."""

    def __init__(self, root: Path, path: str, format: str) -> None:
        self.path = path
        self.format = format
        self.stream: BinaryIO | None = None
        self.digest = hashlib.sha256()
        self.records = 0
        self.error: str | None = None
        try:
            destination = root / self.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.stream = destination.open("xb")
        except (OSError, MemoryError) as error:
            self.unavailable(error)

    def unavailable(self, error: OSError | MemoryError) -> None:
        if self.error is None:
            self.error = f"{type(error).__name__}: {error}"
        self.close()

    def close(self) -> None:
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.close()
            except (OSError, MemoryError) as error:
                if self.error is None:
                    self.error = f"{type(error).__name__}: {error}"

    def append(self, entry: dict[str, Any]) -> None:
        if self.stream is None:
            return
        try:
            self.write(record_bytes(entry))
        except MemoryError as error:
            self.unavailable(error)

    def write(self, raw: bytes) -> None:
        if self.stream is None:
            return
        try:
            if self.stream.write(raw) != len(raw):
                raise OSError("Incomplete diagnostic write")
            self.digest.update(raw)
            self.records += 1
        except (OSError, MemoryError) as error:
            self.unavailable(error)

    def checkpoint(self, *, close: bool = False) -> dict[str, Any]:
        """Bind a durable prefix while allowing later records in this segment."""
        if self.stream is not None:
            try:
                self.stream.flush()
                os.fsync(self.stream.fileno())
            except (OSError, MemoryError) as error:
                self.unavailable(error)
        if close:
            self.close()
        return {
            "format": self.format,
            "path": self.path,
            "status": "complete" if self.error is None else "unavailable",
            "records": self.records,
            "sha256": self.digest.hexdigest() if self.error is None else None,
            "error": self.error,
            "role": "optional_local_diagnostic; not required for checkpoint recovery",
        }

    def finish(self) -> dict[str, Any]:
        return self.checkpoint(close=True)


class PredictionRecorder:
    """Independent numerical fingerprint, even when optional storage is unavailable."""

    def __init__(
        self, journal: RecordJournal | None = None, *, format: str = "sac-predictions-v1"
    ) -> None:
        self.journal = journal
        self.format = format
        self.digest = hashlib.sha256()
        self.records = 0
        self.error: str | None = None

    def add(self, prediction: dict[str, Any]) -> None:
        raw = record_bytes(prediction)
        self.digest.update(raw)
        self.records += 1
        if self.journal is not None:
            self.journal.write(raw)

    def finish(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "status": "complete" if self.error is None else "unavailable",
            "records": self.records,
            "sha256": self.digest.hexdigest() if self.error is None else None,
            "error": self.error,
            **({"diagnostic": self.journal.finish()} if self.journal is not None else {}),
        }

    def unavailable(self, error: OSError | MemoryError) -> None:
        self.error = f"{type(error).__name__}: {error}"
        if self.journal is not None:
            self.journal.unavailable(error)


def prediction_identity(summary: dict[str, Any]) -> tuple[int, str]:
    if summary.get("format") != "sac-predictions-v1":
        raise ValueError("Unsupported SAC prediction summary")
    if summary.get("status") != "complete":
        raise ValueError("SAC numerical prediction check unavailable: " + str(summary.get("error")))
    return summary["records"], summary["sha256"]
