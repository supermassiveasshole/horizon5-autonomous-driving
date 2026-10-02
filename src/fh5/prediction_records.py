"""Result-owned prediction records whose storage is independent of model publication."""

from __future__ import annotations

import json
import struct
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from threading import RLock
from typing import Any, BinaryIO, cast, overload
from weakref import finalize


def _close_files(data: BinaryIO, offsets: BinaryIO) -> None:
    try:
        data.close()
    finally:
        offsets.close()


class PredictionRecords(Sequence[dict[str, Any]]):
    """An immutable returned sequence, backed by delete-on-close temporary files."""

    def __init__(self) -> None:
        self._data = cast(BinaryIO, tempfile.TemporaryFile(mode="w+b"))
        try:
            self._offsets = cast(BinaryIO, tempfile.TemporaryFile(mode="w+b"))
        except BaseException:
            self._data.close()
            raise
        self._lock = RLock()
        self._release = finalize(self, _close_files, self._data, self._offsets)
        self._count = 0
        self.frozen = False

    def append(self, record: dict[str, Any]) -> None:
        if self.frozen:
            raise ValueError("Prediction records are already frozen")
        encoded = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode()
        with self._lock:
            self._data.seek(0, 2)
            offset = self._data.tell()
            self._data.write(encoded)
            self._offsets.seek(0, 2)
            self._offsets.write(struct.pack("<Q", offset))
            self._count += 1

    def freeze(self) -> None:
        self._data.flush()
        self._offsets.flush()
        self.frozen = True

    def close(self) -> None:
        self._release()

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, key: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, key: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, key: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        if key < 0:
            key += len(self)
        if not 0 <= key < len(self):
            raise IndexError(key)
        with self._lock:
            self._offsets.seek(key * 8)
            (offset,) = struct.unpack("<Q", self._offsets.read(8))
            self._data.seek(offset)
            record: dict[str, Any] = json.loads(self._data.readline())
        return record

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for position in range(len(self)):
            yield self[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))


@contextmanager
def prediction_spool() -> Iterator[PredictionRecords]:
    """Keep successful result data alive; close failed/incomplete output immediately."""
    records = PredictionRecords()
    try:
        yield records
    except BaseException as error:
        try:
            records.close()
        except (OSError, MemoryError) as cleanup_error:
            error.add_note(f"Prediction spool cleanup failed: {cleanup_error}")
        raise
    if not records.frozen:
        records.close()
