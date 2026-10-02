"""Stream artifact hashes; read metadata without an invented file-size ceiling."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any, BinaryIO, cast


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _stream_digest(stream)


def _stream_digest(stream: BinaryIO) -> str:
    # Standard I/O transfer quantum, never a limit on the artifact's total size.
    digest = hashlib.sha256()
    block = bytearray(io.DEFAULT_BUFFER_SIZE)
    view = memoryview(block)
    while size := cast(io.BufferedIOBase, stream).readinto(block):
        digest.update(view[:size])
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedFile:
    """A hash-bound artifact copied without retaining its payload in memory."""

    path: Path
    sha256: str

    def verify(self) -> None:
        if sha256_file(self.path) != self.sha256:
            raise ValueError("SAC continuation history changed: " + str(self.path))

    def copy_to(self, target: Path) -> None:
        with self.path.open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=io.DEFAULT_BUFFER_SIZE)
            destination.flush()
            os.fsync(destination.fileno())
        # Verify the copied bytes, not a source that might change between reads.
        if sha256_file(target) != self.sha256:
            raise ValueError("Checkpoint history changed during copy: " + str(self.path))

    @contextmanager
    def snapshot(self) -> Iterator[BinaryIO]:
        """Only deserialize the private copy whose digest was actually checked."""
        with TemporaryFile(mode="w+b") as frozen:
            with self.path.open("rb") as source:
                shutil.copyfileobj(source, frozen, length=io.DEFAULT_BUFFER_SIZE)
            frozen.seek(0)
            if _stream_digest(cast(BinaryIO, frozen)) != self.sha256:
                raise ValueError("Checkpoint weights changed: " + str(self.path))
            frozen.seek(0)
            yield cast(BinaryIO, frozen)


def copy_evidence(root: Path, artifacts: dict[str, bytes | VerifiedFile]) -> None:
    from fh5.collection_store import write_file
    from fh5.numeric_images import asset

    for name, value in artifacts.items():
        target = asset(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, VerifiedFile):
            value.copy_to(target)
        else:
            write_file(target, value)


def read_json(path: Path, *, expected_sha256: str | None = None) -> Any:
    """Metadata reader; growing records belong in indexed/streamed assets."""
    with path.open("rb") as stream:
        if expected_sha256 is None:
            return json.load(stream)
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("JSON artifact hash mismatch: " + str(path))
    return json.loads(payload)
