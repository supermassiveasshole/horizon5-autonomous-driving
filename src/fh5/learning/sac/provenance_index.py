"""Disposable provenance and sampling-role indexes for prepared SAC replay."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


@contextmanager
def _index_errors() -> Iterator[None]:
    try:
        yield
    except sqlite3.Error as error:
        raise OSError("Cannot access SAC provenance index: " + str(error)) from error


class ReplayRoles:
    """Retain scalar counts; look up individual roles and original rows on demand."""

    def __init__(self, database: sqlite3.Connection) -> None:
        self.database = database
        self.counts = {"demonstration": 0, "online": 0}
        database.execute("CREATE TABLE identities (identity TEXT PRIMARY KEY)")
        database.execute(
            "CREATE TABLE originals (source TEXT, identity TEXT, data TEXT, role TEXT, "
            "used INTEGER, PRIMARY KEY (source, identity))"
        )
        database.execute(
            "CREATE TABLE roles (position INTEGER PRIMARY KEY, role TEXT, ordinal INTEGER, "
            "UNIQUE (role, ordinal))"
        )

    def __len__(self) -> int:
        return sum(self.counts.values())

    def __getitem__(self, position: int) -> str:
        with _index_errors():
            row = self.database.execute(
                "SELECT role FROM roles WHERE position = ?", (position,)
            ).fetchone()
        if row is None:
            raise IndexError(position)
        return str(row[0])

    def position(self, role: str, ordinal: int) -> int:
        with _index_errors():
            row = self.database.execute(
                "SELECT position FROM roles WHERE role = ? AND ordinal = ?", (role, ordinal)
            ).fetchone()
        if row is None:
            raise IndexError(ordinal)
        return int(row[0])

    def add_role(self, role: str) -> None:
        with _index_errors():
            self.database.execute(
                "INSERT INTO roles VALUES (?, ?, ?)", (len(self), role, self.counts[role])
            )
        self.counts[role] += 1

    def begin_document(self) -> None:
        with _index_errors():
            self.database.execute("DELETE FROM identities")

    def check_identity(self, identity: Any) -> None:
        with _index_errors():
            added = self.database.execute(
                "INSERT OR IGNORE INTO identities VALUES (?)", (replay_identity_key(identity),)
            ).rowcount
        if not added:
            raise ValueError("Duplicate SAC transition")

    def add_original(self, source: str, row: dict[str, Any], role: str) -> None:
        with _index_errors():
            self.database.execute(
                "INSERT OR REPLACE INTO originals VALUES (?, ?, ?, ?, 0)",
                (source, replay_identity_key(row["id"]), json.dumps(row), role),
            )

    def take_original(self, source: Any, identity: Any) -> tuple[dict[str, Any], str]:
        key = source, replay_identity_key(identity)
        with _index_errors():
            row = self.database.execute(
                "SELECT data, role, used FROM originals WHERE source = ? AND identity = ?", key
            ).fetchone()
            if row is None or row[2]:
                raise ValueError("SAC transition lacks a unique original source")
            self.database.execute(
                "UPDATE originals SET used = 1 WHERE source = ? AND identity = ?", key
            )
        return json.loads(row[0]), str(row[1])


def replay_identity_key(value: Any) -> str:
    # Preserve legacy dictionary/set equality for JSON scalar identities,
    # including 1 == 1.0 == True, without retaining an in-memory set.
    if isinstance(value, (int, float)):
        if isinstance(value, int):
            value = int(value)
        elif value.is_integer():
            value = int(value)
    elif not isinstance(value, (str, type(None))):
        raise ValueError("SAC replay identity must be a JSON scalar")
    return json.dumps(value)


@contextmanager
def provenance_index() -> Iterator[ReplayRoles]:
    with TemporaryDirectory(prefix="fh5-provenance-") as temporary, _index_errors():
        with closing(sqlite3.connect(Path(temporary) / "provenance.sqlite3")) as database:
            yield ReplayRoles(database)
