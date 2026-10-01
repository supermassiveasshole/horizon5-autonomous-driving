"""Stream artifact hashes; read metadata without an invented file-size ceiling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path, *, expected_sha256: str | None = None) -> Any:
    """Metadata reader; growing records belong in indexed/streamed assets."""
    with path.open("rb") as stream:
        if expected_sha256 is None:
            return json.load(stream)
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("JSON artifact hash mismatch: " + str(path))
    return json.loads(payload)
