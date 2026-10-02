"""Verified replay metadata with top-level arrays indexed on disk."""

from __future__ import annotations

import io
import json
import os
import re
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TextIO, overload

from fh5.artifact_io import VerifiedFile

_NON_WHITESPACE = re.compile(r"[^ \t\r\n]")
_NON_SCALAR = object()


@contextmanager
def _index_errors() -> Iterator[None]:
    # Translate at the operation, before an optional caller decides to degrade.
    try:
        yield
    except sqlite3.Error as error:
        raise OSError("Cannot access replay document index: " + str(error)) from error


class ReplayArray(Sequence[Any]):
    """Load individual records while the owning replay document is open."""

    def __init__(self, index: sqlite3.Connection, section: str, length: int) -> None:
        self.database, self.section, self.length = index, section, length

    def __len__(self) -> int:
        return self.length

    @overload
    def __getitem__(self, key: int) -> Any: ...

    @overload
    def __getitem__(self, key: slice) -> list[Any]: ...

    def __getitem__(self, key: int | slice) -> Any:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(self.length))]
        if key < 0:
            key += self.length
        if not 0 <= key < self.length:
            raise IndexError(key)
        with _index_errors():
            row = self.database.execute(
                "SELECT data FROM records WHERE section = ? AND position = ?", (self.section, key)
            ).fetchone()
        return json.loads(row[0])

    def __iter__(self) -> Iterator[Any]:
        with (
            _index_errors(),
            closing(
                self.database.execute(
                    "SELECT data FROM records WHERE section = ? ORDER BY position", (self.section,)
                )
            ) as rows,
        ):
            for (raw,) in rows:
                yield json.loads(raw)


class _JSONInput:
    """Incrementally decode one JSON value; no file-size admission ceiling."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def more(self, characters: int = io.DEFAULT_BUFFER_SIZE) -> None:
        block = self.stream.read(characters)
        self.buffer = self.buffer[self.position :] + block
        self.position = 0
        self.eof = not block

    def peek(self) -> str:
        while True:
            found = _NON_WHITESPACE.search(self.buffer, self.position)
            if found is not None:
                self.position = found.start()
                return found.group()
            self.position = len(self.buffer)
            if self.eof:
                return ""
            self.more()

    def take(self, expected: str) -> None:
        if self.peek() != expected:
            raise ValueError("Malformed replay JSON: expected " + expected)
        self.position += 1

    def value(self) -> Any:
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof:
                    raise
            else:
                # A number may continue in the next block (12 -> 123 or 1e-2).
                # Never accept a partial numeric token or an invalid suffix.
                if end < len(self.buffer) and self.buffer[end] in " \t\r\n,:]}":
                    self.position = end
                    return value
                if self.eof:
                    if end != len(self.buffer):
                        raise ValueError("Malformed replay JSON value suffix")
                    self.position = end
                    return value
            # Retry a growing nested record geometrically, so failed decodes
            # revisit only linear text. This is not an admission size limit.
            self.more(max(io.DEFAULT_BUFFER_SIZE, len(self.buffer) - self.position))

    def document(self, index: sqlite3.Connection) -> dict[str, Any]:
        self.take("{")
        result: dict[str, Any] = {}
        if self.peek() != "}":
            while True:
                key = self.value()
                if not isinstance(key, str):
                    raise ValueError("Replay JSON object keys must be strings")
                self.take(":")
                # Match json.loads' last-key-wins semantics, including arrays.
                index.execute("DELETE FROM records WHERE section = ?", (key,))
                if self.peek() == "[":
                    result[key] = self.array(index, key)
                else:
                    result[key] = self.value()
                if self.peek() != ",":
                    break
                self.take(",")
        self.take("}")
        if self.peek():
            raise ValueError("Trailing data after replay JSON")
        return result

    def array(self, index: sqlite3.Connection, section: str) -> ReplayArray:
        self.take("[")
        count = 0
        if self.peek() != "]":
            while True:
                item = self.value()
                index.execute(
                    "INSERT INTO records VALUES (?, ?, ?)",
                    (section, count, json.dumps(item, ensure_ascii=True)),
                )
                count += 1
                if self.peek() != ",":
                    break
                self.take(",")
        self.take("]")
        return ReplayArray(index, section, count)

    def discard(self) -> None:
        """Validate an unused value without retaining its growing containers."""
        token = self.peek()
        if token not in ("[", "{"):
            self.value()
            return
        self.take(token)
        end = "]" if token == "[" else "}"
        if self.peek() != end:
            while True:
                if token == "{":
                    if not isinstance(self.value(), str):
                        raise ValueError("JSON object keys must be strings")
                    self.take(":")
                self.discard()
                if self.peek() != ",":
                    break
                self.take(",")
        self.take(end)

    def fields(self, wanted: set[str], *, reject_unknown: bool = False) -> dict[str, Any]:
        self.take("{")
        result: dict[str, Any] = {}
        unknown = False
        if self.peek() != "}":
            while True:
                key = self.value()
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                self.take(":")
                if key in wanted:
                    result[key] = self.value()
                else:
                    unknown = True
                    self.discard()
                if self.peek() != ",":
                    break
                self.take(",")
        self.take("}")
        if self.peek():
            raise ValueError("Trailing data after JSON document")
        if reject_unknown and unknown:
            raise ValueError("Unsupported configuration fields")
        return result

    def scalar_array(self) -> Iterator[Any]:
        """Validate containers without retaining them; reducers can reject the marker."""
        self.take("[")
        if self.peek() != "]":
            while True:
                if self.peek() in ("[", "{"):
                    self.discard()
                    yield _NON_SCALAR
                else:
                    yield self.value()
                if self.peek() != ",":
                    break
                self.take(",")
        self.take("]")

    def projected(
        self,
        reducers: dict[tuple[str, ...], Callable[[Iterator[Any]], Any]],
        path: tuple[str, ...] = (),
    ) -> Any:
        if path in reducers:
            if self.peek() == "[":
                values = self.scalar_array()
                result = reducers[path](values)
                # Every tail is still syntax-checked if a reducer returns early.
                for _ in values:
                    pass
                return result
            if self.peek() == "{":
                self.discard()
                return _NON_SCALAR
            return self.value()
        if self.peek() != "{" or not any(key[: len(path)] == path for key in reducers):
            return self.value()
        self.take("{")
        result = {}
        if self.peek() != "}":
            while True:
                key = self.value()
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                self.take(":")
                # Later keys replace the complete earlier projection, just as
                # json.loads replaces an earlier object or array value.
                result[key] = self.projected(reducers, (*path, key))
                if self.peek() != ",":
                    break
                self.take(",")
        self.take("}")
        return result


def read_document_fields(
    source: VerifiedFile, wanted: set[str], *, reject_unknown: bool = False
) -> dict[str, Any]:
    """Project verified metadata while validating/discarding unused histories.

    Selected fields and individual scalar values are decoded as usual. Growing
    unselected arrays/objects are visited incrementally, not materialized.
    Fixed-schema configuration callers can reject unknown fields after checking
    complete syntax, without retaining the unknown values or a list of their keys.
    """
    with source.snapshot() as frozen:
        with io.TextIOWrapper(frozen, encoding="utf-8-sig") as text:
            return _JSONInput(text).fields(wanted, reject_unknown=reject_unknown)


def read_document_projection(
    source: VerifiedFile, reducers: dict[tuple[str, ...], Callable[[Iterator[Any]], Any]]
) -> dict[str, Any]:
    """Reduce selected nested scalar arrays, retaining all other metadata normally.

    A selected container-shaped item is validated and supplied as a non-scalar
    marker. Reducers can record invalid values and let later duplicate keys
    replace that result before the caller validates the final projection.
    """
    with source.snapshot() as frozen:
        with io.TextIOWrapper(frozen, encoding="utf-8-sig") as text:
            parser = _JSONInput(text)
            value = parser.projected(reducers)
            if parser.peek():
                raise ValueError("Trailing data after JSON document")
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


@contextmanager
def replay_document(source: VerifiedFile) -> Iterator[dict[str, Any]]:
    """Parse only hash-verified private bytes; scope the on-demand array index."""
    with TemporaryDirectory(prefix="fh5-replay-document-") as temporary:
        try:
            with closing(sqlite3.connect(Path(temporary) / "records.sqlite3")) as index:
                index.execute(
                    "CREATE TABLE records (section TEXT, position INTEGER, data TEXT NOT NULL, "
                    "PRIMARY KEY (section, position)) WITHOUT ROWID"
                )
                with source.snapshot() as frozen:
                    with io.TextIOWrapper(frozen, encoding="utf-8-sig") as text:
                        document = _JSONInput(text).document(index)
                index.commit()
                yield document
        except sqlite3.Error as error:
            raise OSError("Cannot index replay document: " + str(error)) from error


def write_replay_document(path: Path, document: dict[str, Any]) -> None:
    """Write canonical legacy JSON without expanding indexed arrays or the whole text."""
    encoder = json.JSONEncoder(sort_keys=True, allow_nan=False, separators=(",", ":"))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("{")
        for number, key in enumerate(sorted(document)):
            if number:
                stream.write(",")
            stream.write(encoder.encode(key) + ":")
            value = document[key]
            if isinstance(value, ReplayArray):
                stream.write("[")
                for position, item in enumerate(value):
                    if position:
                        stream.write(",")
                    stream.writelines(encoder.iterencode(item))
                stream.write("]")
            else:
                stream.writelines(encoder.iterencode(value))
        stream.write("}\n")
        stream.flush()
        os.fsync(stream.fileno())
