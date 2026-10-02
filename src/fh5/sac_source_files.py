"""Read immutable replay originals one document at a time, with disk path deduplication."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fh5.artifact_io import VerifiedFile
from fh5.numeric_images import asset


@dataclass(frozen=True)
class SourceReplay:
    name: str
    file: VerifiedFile
    document: dict[str, Any]


def source_replays(root: Path, replay: dict[str, Any]) -> Generator[SourceReplay, None, None]:
    """Verify the exact private copy parsed; original bytes are never retained across sources."""
    entries = replay.get("source_inventory", [])
    if not entries:
        return
    with TemporaryDirectory(prefix="fh5-source-index-") as temporary:
        try:
            with closing(sqlite3.connect(Path(temporary) / "sources.sqlite3")) as index:
                index.execute("CREATE TABLE sources (name TEXT PRIMARY KEY) WITHOUT ROWID")
                for item in entries:
                    name = item["path"]
                    if index.execute("SELECT 1 FROM sources WHERE name = ?", (name,)).fetchone():
                        raise ValueError("Duplicate SAC experience source manifest")
                    index.execute("INSERT INTO sources VALUES (?)", (name,))
                    original = VerifiedFile(asset(root, name), item["replay_sha256"])
                    try:
                        with original.snapshot() as stream:
                            document = json.load(stream)
                    except ValueError as error:
                        raise ValueError("SAC experience source manifest changed") from error
                    if not isinstance(document, dict) or (
                        document.get("source_hashes") != item["source_hashes"]
                    ):
                        raise ValueError("SAC experience source manifest changed")
                    yield SourceReplay(name, original, document)
        except sqlite3.Error as error:
            raise OSError("Cannot index SAC source manifests: " + str(error)) from error


def recording_origins(root: Path, replay: dict[str, Any]) -> set[str]:
    """Extract origin identities without collecting all source documents."""
    if not replay.get("source_inventory"):
        return {replay["source_hashes"]["packets"]}
    with closing(source_replays(root, replay)) as sources:
        return {source.document["source_hashes"]["packets"] for source in sources}
