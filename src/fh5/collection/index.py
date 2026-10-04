"""Hash-bound prefixes of append-only sealed-block references."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fh5.artifacts.document import ReplayArray

REFERENCE_FILE = "block-references.jsonl"


def validate_reference(ref: Any) -> None:
    if (
        not isinstance(ref, dict)
        or not isinstance(ref.get("path"), str)
        or re.fullmatch(r"blocks/[0-9]{6,}", ref["path"]) is None
        or not isinstance(ref.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is None
        or type(ref.get("rows")) is not int
        or not 1 <= ref["rows"] <= 4096
    ):
        raise ValueError("Invalid sealed block reference")


class BlockReferences(ReplayArray):
    """Ordered references with disk-backed lookup for recovery directory scans."""

    def get(self, path: str) -> dict[str, Any] | None:
        try:
            row = self.database.execute(
                "SELECT data FROM records WHERE path = ?", (path,)
            ).fetchone()
        except sqlite3.Error as error:
            raise OSError("Cannot read collection references: " + str(error)) from error
        if row is None:
            return None
        value: dict[str, Any] = json.loads(row[0])
        return value


@contextmanager
def read_references(
    root: Path, binding: str, document: dict[str, Any]
) -> Iterator[BlockReferences]:
    """Index a verified prefix or legacy array; later appended bytes are not part of it.

    The returned sequence is valid only in this context. All references, count
    and prefix bytes are checked before any caller can consume the sequence.
    """
    if document.get("version") != 1 or document.get("session_sha256") != binding:
        raise ValueError("Collection reference document binding differs")
    if ("blocks" in document) == ("references" in document):
        raise ValueError("Collection requires one reference representation")
    with TemporaryDirectory(prefix="fh5-collection-references-") as temporary:
        try:
            with closing(sqlite3.connect(Path(temporary) / "references.sqlite3")) as database:
                database.execute(
                    "CREATE TABLE records (section TEXT, position INTEGER, data TEXT NOT NULL, "
                    "path TEXT UNIQUE, PRIMARY KEY (section, position))"
                )
                count = 0

                def add(ref: Any) -> None:
                    nonlocal count
                    validate_reference(ref)
                    try:
                        database.execute(
                            "INSERT INTO records VALUES ('blocks', ?, ?, ?)",
                            (count, json.dumps(ref), ref["path"]),
                        )
                    except sqlite3.IntegrityError as error:
                        raise ValueError("Repeated sealed block reference") from error
                    count += 1

                if "blocks" in document:
                    blocks = document["blocks"]
                    if not isinstance(blocks, (list, ReplayArray)):
                        raise ValueError("Invalid collection reference list")
                    for ref in blocks:
                        add(ref)
                else:
                    prefix = document["references"]
                    if (
                        not isinstance(prefix, dict)
                        or prefix.get("file") != REFERENCE_FILE
                        or type(prefix.get("bytes")) is not int
                        or prefix["bytes"] < 0
                        or type(prefix.get("count")) is not int
                        or prefix["count"] < 0
                        or not isinstance(prefix.get("sha256"), str)
                        or re.fullmatch(r"[0-9a-f]{64}", prefix["sha256"]) is None
                    ):
                        raise ValueError("Invalid collection reference prefix")
                    digest = hashlib.sha256()
                    remaining = prefix["bytes"]
                    with (root / REFERENCE_FILE).open("rb") as stream:
                        if remaining > os.fstat(stream.fileno()).st_size:
                            raise ValueError("Truncated collection reference prefix")
                        while remaining:
                            line = stream.readline(remaining)
                            if not line or not line.endswith(b"\n"):
                                raise ValueError("Truncated collection reference prefix")
                            remaining -= len(line)
                            digest.update(line)
                            add(json.loads(line))
                    if count != prefix["count"] or digest.hexdigest() != prefix["sha256"]:
                        raise ValueError("Collection reference prefix changed")
                database.commit()
                yield BlockReferences(database, "blocks", count)
        except sqlite3.Error as error:
            raise OSError("Cannot index collection references: " + str(error)) from error
