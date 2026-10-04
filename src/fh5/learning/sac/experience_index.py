"""Incremental experience-union records and identity checks backed by disk."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fh5.artifacts.document import ReplayArray
from fh5.learning.sac.provenance_index import replay_identity_key


class ExperienceUnion:
    def __init__(self, database: sqlite3.Connection) -> None:
        self.database = database
        self.counts = {"transitions": 0, "source_inventory": 0}
        database.execute(
            "CREATE TABLE records (section TEXT, position INTEGER, data TEXT NOT NULL, "
            "PRIMARY KEY (section, position)) WITHOUT ROWID"
        )
        database.execute(
            "CREATE TABLE origins (sha256 TEXT PRIMARY KEY, packets TEXT UNIQUE, addition INTEGER)"
        )
        database.execute("CREATE TABLE transitions (identity TEXT PRIMARY KEY)")

    def _append(self, section: str, value: dict[str, Any]) -> None:
        self.database.execute(
            "INSERT INTO records VALUES (?, ?, ?)",
            (section, self.counts[section], json.dumps(value, allow_nan=False)),
        )
        self.counts[section] += 1

    def add_source(self, addition: int, entry: dict[str, Any]) -> None:
        # Re-reviewing or reformatting metadata does not create another interaction.
        added = self.database.execute(
            "INSERT OR IGNORE INTO origins VALUES (?, ?, ?)",
            (
                entry["replay_sha256"],
                replay_identity_key(entry["source_hashes"]["packets"]),
                addition,
            ),
        ).rowcount
        if not added:
            raise ValueError("Duplicate SAC experience cannot earn new update credit")
        self._append(
            "source_inventory", {**entry, "path": f"sources/{entry['replay_sha256']}.json"}
        )

    def contains_source(self, addition: int, sha256: str) -> bool:
        return (
            self.database.execute(
                "SELECT 1 FROM origins WHERE sha256 = ? AND addition = ?", (sha256, addition)
            ).fetchone()
            is not None
        )

    def add_transition(self, row: dict[str, Any]) -> None:
        added = self.database.execute(
            "INSERT OR IGNORE INTO transitions VALUES (?)", (row["id"],)
        ).rowcount
        if not added:
            raise ValueError("Duplicate SAC transition")
        self._append("transitions", row)

    def array(self, section: str) -> ReplayArray:
        return ReplayArray(self.database, section, self.counts[section])


@contextmanager
def experience_union() -> Iterator[ExperienceUnion]:
    with TemporaryDirectory(prefix="fh5-experience-union-") as temporary:
        try:
            with closing(sqlite3.connect(Path(temporary) / "union.sqlite3")) as database:
                yield ExperienceUnion(database)
        except sqlite3.Error as error:
            raise OSError("Cannot index SAC experience union: " + str(error)) from error
