"""Disk-backed asset references for immutable experience expansion."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fh5.artifacts.io import VerifiedFile, asset, encode, write_file
from fh5.observation.recording import read_numeric_frame, read_numeric_pixels


class ExperienceAssets:
    """Validate every input reference, then copy distinct assets one at a time."""

    def __init__(self, database: sqlite3.Connection) -> None:
        self.database = database
        database.execute("CREATE TABLE frames (name TEXT PRIMARY KEY, root TEXT, entry TEXT)")
        database.execute("CREATE TABLE sources (name TEXT PRIMARY KEY, path TEXT, sha256 TEXT)")
        self.frame_files = self.frame_bytes = 0

    def add_frame(self, root: Path, entry: dict[str, Any]) -> str:
        frame = read_numeric_frame(root, entry)
        name: str = "frames/" + entry["sha256"] + ".rgb"
        inserted = self.database.execute(
            "INSERT OR IGNORE INTO frames VALUES (?, ?, ?)",
            (name, str(root), encode(entry).decode("utf-8")),
        ).rowcount
        if inserted:
            self.frame_files += 1
            self.frame_bytes += frame.pixels.nbytes
        return name

    def add_source(self, original: VerifiedFile) -> None:
        name = f"sources/{original.sha256}.json"
        self.database.execute(
            "INSERT OR IGNORE INTO sources VALUES (?, ?, ?)",
            (name, str(original.path), original.sha256),
        )

    def copy_into(self, output: Path) -> dict[str, Any]:
        peak = 0
        for name, root, raw in self.database.execute("SELECT * FROM frames ORDER BY rowid"):
            entry = json.loads(raw)
            width, height = entry["size"]
            payload = read_numeric_pixels(Path(root), entry, width * height * 3)
            peak = max(peak, len(payload))
            target = asset(output, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_file(target, payload)
            del payload
        for name, path, sha in self.database.execute("SELECT * FROM sources ORDER BY rowid"):
            target = asset(output, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            VerifiedFile(Path(path), sha).copy_to(target)
        return {
            "mode": "verified-frame-stream",
            "frame_files": self.frame_files,
            "frame_bytes": self.frame_bytes,
            "copy_peak_frame_bytes": peak,
            "files_deleted": 0,
        }


@contextmanager
def experience_assets() -> Iterator[ExperienceAssets]:
    with TemporaryDirectory(prefix="fh5-expansion-assets-") as temporary:
        try:
            with closing(sqlite3.connect(Path(temporary) / "assets.sqlite3")) as database:
                yield ExperienceAssets(database)
        except sqlite3.Error as error:
            raise OSError("Cannot index SAC expansion assets: " + str(error)) from error
