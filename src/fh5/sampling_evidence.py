"""Bind retained sampling originals separately from derived learning experience."""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fh5.artifact_io import read_json, sha256_file
from fh5.collection_store import encode, write_file
from fh5.numeric_images import asset


class _SourceIndex(Mapping[str, str]):
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def __getitem__(self, name: str) -> str:
        row = self.connection.execute(
            "SELECT sha256 FROM assets WHERE path = ?", (name,)
        ).fetchone()
        if row is None:
            raise KeyError(name)
        return str(row[0])

    def __iter__(self) -> Iterator[str]:
        for row in self.connection.execute("SELECT path FROM assets ORDER BY path"):
            yield str(row[0])

    def __len__(self) -> int:
        return int(self.connection.execute("SELECT count(*) FROM assets").fetchone()[0])


@contextmanager
def sampling_sources(binding: dict[str, Any]) -> Iterator[Mapping[str, str]]:
    """Legacy inline inventories and immutable indexed inventories share lookup semantics."""
    if binding.get("kind") != "sampling-source-index-v1":
        yield binding
        return
    path = Path(binding["path"])
    if not path.is_absolute():
        raise ValueError("Retained sampling source index changed")
    # Query only the verified private snapshot, even if the original is replaced
    # while recovering. Copy in blocks; no full inventory needs to occupy RAM.
    with TemporaryDirectory(prefix="fh5-source-index-") as temporary:
        snapshot = Path(temporary) / "sources.sqlite3"
        with path.open("rb") as source, snapshot.open("xb") as target:
            shutil.copyfileobj(source, target)
        if sha256_file(snapshot) != binding["sha256"]:
            raise ValueError("Retained sampling source index changed")
        connection = sqlite3.connect(snapshot.as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            yield _SourceIndex(connection)
        finally:
            connection.close()


def _inventory(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        name = str(path.resolve())
        if name in result:
            continue
        result[name] = sha256_file(path)
    return result


def _source_paths(root: Path, review: Path | None) -> Iterable[Path]:
    # Called after sampling closes, before preparation creates derived files.
    for path in root.rglob("*"):
        if path.is_file():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Sampling original escapes its attempt directory")
            yield path
    if review is not None:
        yield review
        for item in read_json(review)["items"]:
            yield asset(review.parent, item["path"])


def seal_sampling_sources(root: Path, review: Path | None) -> dict[str, str]:
    return _inventory(_source_paths(root, review))


def _seal_index(root: Path, review: Path | None) -> dict[str, Any]:
    target = root.with_name(root.name + "-sources.sqlite3")
    temporary = target.with_suffix(".tmp")
    if target.exists() or temporary.exists():
        raise FileExistsError("Sampling source index already exists")
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("CREATE TABLE assets (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL)")
        with connection:
            for path in _source_paths(root, review):
                connection.execute(
                    "INSERT OR REPLACE INTO assets VALUES (?, ?)",
                    (str(path.resolve()), sha256_file(path)),
                )
    finally:
        connection.close()
    temporary.replace(target)
    return {
        "kind": "sampling-source-index-v1",
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
    }


def seal_sampling_attempt(
    root: Path, review: Path | None, *, indexed: bool = False
) -> dict[str, Any]:
    """Keep the review's role even if a later summary omits original file bindings."""
    proof = None
    if review is not None:
        name = str(review.resolve())
        proof = {"path": name, "sha256": _inventory([review])[name]}
    write_file(root / "sampling-sources.json", encode({"version": 1, "review": proof}))
    return _seal_index(root, review) if indexed else seal_sampling_sources(root, review)


def verify_sampling_sources(expected: dict[str, Any]) -> None:
    try:
        with sampling_sources(expected) as inventory:
            if not inventory:
                raise ValueError("Sampling originals differ")
            for name in inventory:
                if sha256_file(Path(name)) != inventory[name]:
                    raise ValueError("Sampling originals differ")
    except (OSError, ValueError, sqlite3.Error) as error:
        raise ValueError("Retained sampling originals changed or are unavailable") from error
