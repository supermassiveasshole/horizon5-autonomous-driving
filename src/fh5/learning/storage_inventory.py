"""Disk-backed dependency accounting, with streamed external file details."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any


class StorageInventory:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        connection.executescript(
            "CREATE TABLE files (path TEXT PRIMARY KEY, bytes INTEGER, mtime_ns INTEGER);"
            "CREATE TABLE roles (path TEXT, role TEXT, PRIMARY KEY (path, role));"
            "CREATE TABLE documents (path TEXT PRIMARY KEY, sha256 TEXT);"
            "CREATE TABLE visits (kind TEXT, identity TEXT, PRIMARY KEY (kind, identity));"
            "CREATE TABLE directories (path TEXT, role TEXT, scanned INTEGER DEFAULT 0,"
            " PRIMARY KEY (path, role));"
            "CREATE INDEX pending_directories ON directories (scanned);"
        )

    def file(self, path: Path, status: os.stat_result, role: str) -> None:
        name = str(path)
        previous = self.connection.execute(
            "SELECT bytes, mtime_ns FROM files WHERE path = ?", (name,)
        ).fetchone()
        actual = status.st_size, status.st_mtime_ns
        if previous is None:
            self.connection.execute("INSERT INTO files VALUES (?, ?, ?)", (name, *actual))
        elif previous != actual:
            raise ValueError("Storage dependency changed during accounting: " + name)
        self.connection.execute("INSERT OR IGNORE INTO roles VALUES (?, ?)", (name, role))

    def document(self, path: Path, digest: str) -> None:
        name = str(path)
        previous = self.connection.execute(
            "SELECT sha256 FROM documents WHERE path = ?", (name,)
        ).fetchone()
        if previous is None:
            self.connection.execute("INSERT INTO documents VALUES (?, ?)", (name, digest))
        elif previous[0] != digest:
            raise ValueError("Storage manifest changed during accounting: " + name)

    def visited(self, kind: str, *identity: str) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO visits VALUES (?, ?)", (kind, json.dumps(identity))
        )
        return cursor.rowcount == 0

    def queue_directory(self, path: Path, role: str) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO directories (path, role) VALUES (?, ?)", (str(path), role)
        )

    def next_directory(self) -> tuple[Path, str] | None:
        row = self.connection.execute(
            "SELECT path, role FROM directories WHERE scanned = 0 LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        self.connection.execute(
            "UPDATE directories SET scanned = 1 WHERE path = ? AND role = ?", row
        )
        return Path(row[0]), row[1]

    def files(self) -> Iterator[dict[str, Any]]:
        for name, size, modified in self.connection.execute(
            "SELECT path, bytes, mtime_ns FROM files ORDER BY path"
        ):
            roles = [
                row[0]
                for row in self.connection.execute(
                    "SELECT role FROM roles WHERE path = ? ORDER BY role", (name,)
                )
            ]
            yield {"path": name, "bytes": size, "mtime_ns": modified, "roles": roles}

    def documents(self) -> Iterator[tuple[Path, str]]:
        for name, digest in self.connection.execute("SELECT path, sha256 FROM documents"):
            yield Path(name), digest

    def totals(self) -> dict[str, int]:
        total = count = 0
        for (size,) in self.connection.execute("SELECT bytes FROM files"):
            total += size
            count += 1
        return {"protected_bytes": total, "protected_files": count}

    def publish(self, path: Path) -> dict[str, Any]:
        digest = hashlib.sha256()
        count = 0
        with path.open("xb") as stream:
            for entry in self.files():
                raw = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
                stream.write(raw)
                digest.update(raw)
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        return {
            "format": "storage-files-v1",
            "path": path.name,
            "sha256": digest.hexdigest(),
            "count": count,
        }


@contextmanager
def storage_inventory(namespace: Path) -> Iterator[StorageInventory]:
    """Own the temporary database outside the retained source tree."""
    namespace = namespace.resolve()
    temporary_root = Path(gettempdir()).resolve()
    if temporary_root.is_relative_to(namespace):
        temporary_root = namespace.parent
        if temporary_root == namespace:
            raise ValueError("Storage temporary location must be outside the dependency namespace")
    with TemporaryDirectory(prefix="fh5-storage-index-", dir=temporary_root) as temporary:
        connection = sqlite3.connect(Path(temporary) / "inventory.sqlite3")
        try:
            with connection:
                yield StorageInventory(connection)
        finally:
            connection.close()
