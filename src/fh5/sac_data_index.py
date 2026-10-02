"""Disposable disk indexes for validated SAC observations and numerical state."""

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
        raise OSError("Cannot access SAC learning data index: " + str(error)) from error


class LearningDataIndex:
    """Retain one record per read; the owning experiment controls its lifetime."""

    def __init__(self, database: sqlite3.Connection) -> None:
        self.database = database
        self.source_count = 0
        database.execute(
            "CREATE TABLE items (section TEXT, identity TEXT, data TEXT, "
            "PRIMARY KEY (section, identity))"
        )

    def get(self, section: str, identity: str | int) -> Any:
        with _index_errors():
            row = self.database.execute(
                "SELECT data FROM items WHERE section = ? AND identity = ?",
                (section, str(identity)),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def add(self, section: str, identity: str | int, value: Any) -> bool:
        with _index_errors():
            inserted = self.database.execute(
                "INSERT OR IGNORE INTO items VALUES (?, ?, ?)",
                (section, str(identity), json.dumps(value, allow_nan=False)),
            ).rowcount
        if inserted and section == "sources":
            self.source_count += 1
        return bool(inserted)

    def items(self, section: str) -> Iterator[tuple[str, Any]]:
        with _index_errors():
            cursor = self.database.execute(
                "SELECT identity, data FROM items WHERE section = ?", (section,)
            )
        with closing(cursor):
            while True:
                with _index_errors():
                    row = cursor.fetchone()
                if row is None:
                    return
                yield row[0], json.loads(row[1])


@contextmanager
def learning_data_index() -> Iterator[LearningDataIndex]:
    with TemporaryDirectory(prefix="fh5-learning-data-") as temporary, _index_errors():
        with closing(sqlite3.connect(Path(temporary) / "data.sqlite3")) as database:
            yield LearningDataIndex(database)
