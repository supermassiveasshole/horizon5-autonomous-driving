"""Bind retained sampling originals separately from derived learning experience."""

from __future__ import annotations

import os
import shutil
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp
from typing import Any

from fh5.artifacts.document import consume_document_strings
from fh5.artifacts.io import VerifiedFile, asset, encode, read_json, sha256_file, write_file


class _SourceIndex(Mapping[str, str]):
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def __getitem__(self, name: str) -> str:
        row = self.connection.execute(
            "SELECT sha256 FROM assets WHERE path = ?", (name,)
        ).fetchone()
        if row is None:
            raise KeyError(name)
        if not isinstance(row[0], str):
            raise ValueError("Sampling inventory values must be strings")
        return row[0]

    def __iter__(self) -> Iterator[str]:
        for row in self.connection.execute("SELECT path FROM assets ORDER BY path"):
            if not isinstance(row[0], str):
                raise ValueError("Sampling inventory paths must be strings")
            yield row[0]

    def __len__(self) -> int:
        return int(self.connection.execute("SELECT count(*) FROM assets").fetchone()[0])


@contextmanager
def _inventory_document(source: VerifiedFile) -> Iterator[Mapping[str, str]]:
    """Import old flat JSON incrementally; only final duplicate values are validated."""
    with TemporaryDirectory(prefix="fh5-source-document-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "inventory.sqlite3")
        try:
            connection.execute("CREATE TABLE assets (path TEXT PRIMARY KEY, sha256 TEXT)")

            def consume(name: str, value: str | None) -> None:
                connection.execute("INSERT OR REPLACE INTO assets VALUES (?, ?)", (name, value))

            with connection:
                consume_document_strings(source, consume)
            yield _SourceIndex(connection)
        finally:
            connection.close()


@contextmanager
def sampling_sources(
    binding: dict[str, Any] | VerifiedFile, *, expected_index: Path | None = None
) -> Iterator[Mapping[str, str]]:
    """Legacy inline inventories and immutable indexed inventories share lookup semantics."""
    if isinstance(binding, VerifiedFile):
        with _inventory_document(binding) as document:
            if document.get("kind") == "sampling-source-index-v1":
                if len(document) != 3 or any(k not in document for k in ("path", "sha256")):
                    raise ValueError("Retained sampling source index descriptor changed")
                with sampling_sources(dict(document), expected_index=expected_index) as inventory:
                    yield inventory
            else:
                yield document
        return
    if binding.get("kind") != "sampling-source-index-v1":
        yield binding
        return
    if set(binding) != {"kind", "path", "sha256"} or not isinstance(binding["path"], str):
        raise ValueError("Retained sampling source index descriptor changed")
    path = Path(binding["path"])
    if not path.is_absolute() or (expected_index is not None and path != expected_index.resolve()):
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
            fields = connection.execute("PRAGMA table_info(assets)").fetchall()
            if [(row[1], row[2], row[3], row[5]) for row in fields] != [
                ("path", "TEXT", 0, 1),
                ("sha256", "TEXT", 1, 0),
            ]:
                raise ValueError("Retained sampling source index schema changed")
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


def seal_sampling_sources(
    root: Path, review: Path | None, *, indexed: bool = False
) -> dict[str, Any]:
    if indexed:
        return _seal_index(root, review, reuse_existing=review is None)
    return _inventory(_source_paths(root, review))


def _index_binding(target: Path) -> dict[str, Any]:
    return {
        "kind": "sampling-source-index-v1",
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
    }


def _seal_index(root: Path, review: Path | None, *, reuse_existing: bool = False) -> dict[str, Any]:
    target = root.with_name(root.name + "-sources.sqlite3")
    if target.exists():
        if not reuse_existing:
            raise FileExistsError("Sampling source index already exists")
        binding = _index_binding(target)
        verify_sampling_sources(binding, root=root)
        return binding
    if reuse_existing:
        # An interrupted staging file is not sealed evidence. Never overwrite
        # it while rebuilding; publish a separately named, complete transaction.
        descriptor, name = mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        os.close(descriptor)
        temporary = Path(name)
    else:
        temporary = target.with_suffix(".tmp")
        if temporary.exists():
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
    return _index_binding(target)


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


def verify_sampling_sources(
    expected: dict[str, Any] | VerifiedFile, *, root: Path | None = None
) -> None:
    try:
        target = root.with_name(root.name + "-sources.sqlite3") if root is not None else None
        with sampling_sources(expected, expected_index=target) as inventory:
            if not inventory:
                raise ValueError("Sampling originals differ")
            for name in inventory:
                path, digest = Path(name), inventory[name]
                if (
                    not path.is_absolute()
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                    or root is not None
                    and not path.resolve().is_relative_to(root.resolve())
                    or sha256_file(path) != digest
                ):
                    raise ValueError("Sampling originals differ")
            if root is not None:
                # Checking only indexed rows would accept new, unlisted originals.
                for path in _source_paths(root, None):
                    if str(path.resolve()) not in inventory:
                        raise ValueError("Sampling original inventory omits a retained file")
    except (OSError, ValueError, sqlite3.Error) as error:
        raise ValueError("Retained sampling originals changed or are unavailable") from error
