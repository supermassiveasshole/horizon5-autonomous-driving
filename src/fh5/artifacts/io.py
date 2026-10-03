"""Stream artifact hashes; read metadata without an invented file-size ceiling."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import time
from collections.abc import Callable, Iterator
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
                raise ValueError("Artifact hash mismatch: " + str(self.path))
            frozen.seek(0)
            yield cast(BinaryIO, frozen)


def copy_evidence(root: Path, artifacts: dict[str, bytes | VerifiedFile]) -> None:
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


WriteFile = Callable[[Path, bytes], None]


def read_bounded(path: Path, limit: int) -> bytes:
    """Legacy budgeted reader; allocate for received bytes, not the maximum budget."""
    payload = bytearray()
    with path.open("rb") as stream:
        while chunk := stream.read(min(io.DEFAULT_BUFFER_SIZE, limit + 1 - len(payload))):
            payload.extend(chunk)
            if len(payload) > limit:
                raise ValueError("Collection asset exceeds bounded limit: " + path.name)
    return bytes(payload)


def encode(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def write_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(
    path: Path, value: Any, *, publish: Callable[[Path, Path], object] | None = None
) -> None:
    if publish is None:
        publish = Path.replace
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())
    deadline = time.monotonic() + 0.25
    while True:
        try:
            publish(temporary, path)
            break
        except PermissionError as error:
            # Windows readers can briefly deny delete/rename sharing. Retry only
            # that contention, with a fixed bound; other disk failures remain fatal.
            if getattr(error, "winerror", None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def asset(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError("Numerical asset path escapes its recording")
    return path


def source_hashes() -> dict[str, str]:
    """Identify all package source files, including nested workflow packages."""
    package = Path(__file__).resolve().parents[1]
    return {
        path.relative_to(package).as_posix(): sha256_file(path)
        for path in sorted(package.rglob("*.py"))
    }
