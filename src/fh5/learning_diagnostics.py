"""Stream optional update diagnostics independently of recoverable learner state."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO


class UpdateJournal:
    """One JSON record per update; unavailable diagnostics never discard the learner."""

    def __init__(self, root: Path) -> None:
        self.path = "diagnostics/updates.jsonl"
        self.stream: BinaryIO | None = None
        self.digest = hashlib.sha256()
        self.records = 0
        self.error: str | None = None
        try:
            destination = root / self.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.stream = destination.open("xb")
        except (OSError, MemoryError) as error:
            self._unavailable(error)

    def _unavailable(self, error: OSError | MemoryError) -> None:
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
            raw = (json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            if self.stream.write(raw) != len(raw):
                raise OSError("Incomplete update diagnostic write")
            self.digest.update(raw)
            self.records += 1
        except (OSError, MemoryError) as error:
            self._unavailable(error)

    def finish(self) -> dict[str, Any]:
        if self.stream is not None:
            try:
                self.stream.flush()
                os.fsync(self.stream.fileno())
            except (OSError, MemoryError) as error:
                self._unavailable(error)
        self.close()
        return {
            "format": "sac-update-jsonl-v1",
            "path": self.path,
            "status": "complete" if self.error is None else "unavailable",
            "records": self.records,
            "sha256": self.digest.hexdigest() if self.error is None else None,
            "error": self.error,
            "role": "optional_local_diagnostic; not required for checkpoint recovery",
        }
