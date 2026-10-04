"""Verified replay metadata with top-level and selected nested arrays indexed on disk."""

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

from fh5.artifacts.io import VerifiedFile

_NON_WHITESPACE = re.compile(r"[^ \t\r\n]")
_STRING_SPECIAL = re.compile(r'["\\\x00-\x1f]')
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

    def __init__(
        self,
        index: sqlite3.Connection,
        section: str,
        length: int,
        *,
        nested_references: bool = False,
    ) -> None:
        self.database, self.section, self.length = index, section, length
        self.nested_references = nested_references

    def __len__(self) -> int:
        return self.length

    def _decode(self, raw: str, references: str) -> Any:
        value = json.loads(raw)
        # References live beside the JSON record, never in source-owned keys.
        for path, section, length in json.loads(references):
            child = ReplayArray(self.database, section, length, nested_references=True)
            if not path:
                value = child
            else:
                parent = value
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = child
        return value

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
        columns = "data, refs" if self.nested_references else "data, '[]'"
        with _index_errors():
            row = self.database.execute(
                f"SELECT {columns} FROM records WHERE section = ? AND position = ?",
                (self.section, key),
            ).fetchone()
        return self._decode(row[0], row[1])

    def __iter__(self) -> Iterator[Any]:
        columns = "data, refs" if self.nested_references else "data, '[]'"
        with (
            _index_errors(),
            closing(
                self.database.execute(
                    f"SELECT {columns} FROM records WHERE section = ? ORDER BY position",
                    (self.section,),
                )
            ) as rows,
        ):
            for raw, references in rows:
                yield self._decode(raw, references)


def _indexed_record(value: Any) -> tuple[str, str]:
    references = []

    def visit(item: Any, path: tuple[str | int, ...]) -> Any:
        if isinstance(item, ReplayArray):
            references.append((path, item.section, len(item)))
            return None
        if isinstance(item, dict):
            return {key: visit(child, (*path, key)) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child, (*path, i)) for i, child in enumerate(item)]
        return item

    raw = json.dumps(visit(value, ()), ensure_ascii=True)
    return raw, json.dumps(references)


class _JSONInput:
    """Incrementally decode one JSON value; no file-size admission ceiling."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()
        self.sections = 0

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

    def object_keys(self) -> Iterator[str]:
        """Consume object syntax; each caller consumes the value after its key."""
        self.take("{")
        if self.peek() != "}":
            while True:
                key = self.value()
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                self.take(":")
                yield key
                if self.peek() != ",":
                    break
                self.take(",")
        self.take("}")

    def document(
        self, index: sqlite3.Connection, nested_arrays: set[tuple[str, ...]]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in self.object_keys():
            # Match json.loads' last-key-wins semantics, including arrays.
            if not nested_arrays:
                index.execute("DELETE FROM records WHERE section = ?", (key,))
            children = {path[1:] for path in nested_arrays if path and path[0] == key}
            if self.peek() == "[":
                section = self.section() if nested_arrays else key
                result[key] = self.array(index, section, children)
            else:
                result[key] = self.nested(index, children)
        if self.peek():
            raise ValueError("Trailing data after replay JSON")
        return result

    def section(self) -> str:
        self.sections += 1
        return str(self.sections)

    def nested(self, index: sqlite3.Connection, paths: set[tuple[str, ...]]) -> Any:
        if not paths:
            return self.value()
        if self.peek() == "{":
            return {
                key: self.nested(index, {path[1:] for path in paths if path and path[0] == key})
                for key in self.object_keys()
            }
        if self.peek() == "[":
            if () in paths:
                return self.array(index, self.section(), paths)
            return list(self.array_values(index, paths))
        return self.value()

    def array_values(self, index: sqlite3.Connection, paths: set[tuple[str, ...]]) -> Iterator[Any]:
        self.take("[")
        children = {path[1:] for path in paths if path and path[0] == "*"}
        if self.peek() != "]":
            while True:
                yield self.nested(index, children)
                if self.peek() != ",":
                    break
                self.take(",")
        self.take("]")

    def array(
        self, index: sqlite3.Connection, section: str, paths: set[tuple[str, ...]]
    ) -> ReplayArray:
        count = 0
        for item in self.array_values(index, paths):
            raw, references = (
                _indexed_record(item) if paths else (json.dumps(item, ensure_ascii=True), "[]")
            )
            index.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?)", (section, count, raw, references)
            )
            count += 1
        return ReplayArray(index, section, count, nested_references=True)

    def _discard_string(self) -> None:
        """Validate a discarded string without accumulating or decoding its text."""
        self.take('"')
        while True:
            special = _STRING_SPECIAL.search(self.buffer, self.position)
            if special is None:
                self.position = len(self.buffer)
                if self.eof:
                    raise ValueError("Unterminated discarded JSON string")
                self.more()
                continue
            self.position = special.end()
            token = special.group()
            if token == '"':
                return
            if token != "\\":
                raise ValueError("Control character in discarded JSON string")
            if self.position == len(self.buffer):
                self.more()
            if self.position == len(self.buffer):
                raise ValueError("Incomplete discarded JSON string escape")
            escape = self.buffer[self.position]
            self.position += 1
            if escape in '"\\/bfnrt':
                continue
            if escape != "u":
                raise ValueError("Invalid discarded JSON string escape")
            remaining = 4
            while remaining:
                if self.position == len(self.buffer):
                    self.more()
                digits = self.buffer[self.position : self.position + remaining]
                if not digits or any(c not in "0123456789abcdefABCDEF" for c in digits):
                    raise ValueError("Invalid discarded JSON Unicode escape")
                self.position += len(digits)
                remaining -= len(digits)

    def discard(self) -> None:
        """Validate unused strings and containers without retaining their payloads."""
        token = self.peek()
        if token == '"':
            self._discard_string()
            return
        if token not in ("[", "{"):
            self.value()
            return
        self.take(token)
        end = "]" if token == "[" else "}"
        if self.peek() != end:
            while True:
                if token == "{":
                    if self.peek() != '"':
                        raise ValueError("JSON object keys must be strings")
                    self._discard_string()
                    self.take(":")
                self.discard()
                if self.peek() != ",":
                    break
                self.take(",")
        self.take(end)

    def fields(self, wanted: set[str], *, reject_unknown: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {}
        unknown = False
        for key in self.object_keys():
            if key in wanted:
                result[key] = self.value()
            else:
                unknown = True
                self.discard()
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

    def selected(self, wanted: set[tuple[str, ...]]) -> Any:
        """Project nested scalar leaves; discard other containers incrementally."""
        token = self.peek()
        if () in wanted or token != "{":
            if token in ("[", "{"):
                self.discard()
                return _NON_SCALAR
            return self.value()
        result = {}
        for key in self.object_keys():
            children = {path[1:] for path in wanted if path[0] == key}
            if children:
                # Later duplicate parents replace their entire projection.
                result[key] = self.selected(children)
            else:
                self.discard()
        return result

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
        result = {}
        for key in self.object_keys():
            # Later keys replace the complete earlier projection, just as
            # json.loads replaces an earlier object or array value.
            result[key] = self.projected(reducers, (*path, key))
        return result


def consume_document_strings(
    source: VerifiedFile, consume: Callable[[str, str | None], None]
) -> None:
    """Visit every verified object field; None marks a fully parsed non-string value.

    Consumers apply last-key-wins before validating their final schema. No
    growing container is retained, and the complete document must be valid JSON.
    """
    with source.snapshot() as frozen:
        with io.TextIOWrapper(frozen, encoding="utf-8-sig") as text:
            parser = _JSONInput(text)
            for key in parser.object_keys():
                if parser.peek() == '"':
                    consume(key, parser.value())
                else:
                    parser.discard()
                    consume(key, None)
            if parser.peek():
                raise ValueError("Trailing data after JSON document")


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


def read_stream_paths(stream: TextIO, wanted: set[tuple[str, ...]]) -> dict[str, Any]:
    """Select scalar leaves from one open JSON document, validating all syntax."""
    parser = _JSONInput(stream)
    value = parser.selected(wanted)
    if parser.peek():
        raise ValueError("Trailing data after JSON document")
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")

    def validate_shape(item: Any) -> None:
        if item is _NON_SCALAR:
            raise ValueError("Expected a scalar JSON document field")
        if isinstance(item, dict):
            for child in item.values():
                validate_shape(child)

    validate_shape(value)
    return value


def read_document_paths(source: VerifiedFile, wanted: set[tuple[str, ...]]) -> dict[str, Any]:
    """Select nested scalar metadata from the private hash-verified file bytes."""
    with source.snapshot() as frozen:
        with io.TextIOWrapper(frozen, encoding="utf-8-sig") as stream:
            return read_stream_paths(stream, wanted)


@contextmanager
def replay_document(
    source: VerifiedFile, *, nested_arrays: set[tuple[str, ...]] | None = None
) -> Iterator[dict[str, Any]]:
    """Index top-level and opted-in nested arrays in hash-verified private bytes.

    Paths name object keys; '*' traverses array elements. Other nested arrays
    retain their list representation. Proxies are valid only inside this context.
    """
    with TemporaryDirectory(prefix="fh5-replay-document-") as temporary:
        try:
            with closing(sqlite3.connect(Path(temporary) / "records.sqlite3")) as index:
                index.execute(
                    "CREATE TABLE records (section TEXT, position INTEGER, data TEXT NOT NULL, "
                    "refs TEXT NOT NULL, "
                    "PRIMARY KEY (section, position)) WITHOUT ROWID"
                )
                with source.snapshot() as frozen:
                    with io.TextIOWrapper(frozen, encoding="utf-8-sig") as text:
                        document = _JSONInput(text).document(index, nested_arrays or set())
                index.commit()
                yield document
        except sqlite3.Error as error:
            raise OSError("Cannot index replay document: " + str(error)) from error


def write_replay_document(path: Path, document: dict[str, Any]) -> None:
    """Write canonical legacy JSON without expanding indexed arrays or the whole text."""
    encoder = json.JSONEncoder(sort_keys=True, allow_nan=False, separators=(",", ":"))
    with path.open("x", encoding="utf-8", newline="\n") as stream:

        def write(value: Any) -> None:
            if isinstance(value, dict):
                stream.write("{")
                for number, key in enumerate(sorted(value)):
                    if number:
                        stream.write(",")
                    stream.write(encoder.encode(key) + ":")
                    write(value[key])
                stream.write("}")
            elif isinstance(value, (list, tuple, ReplayArray)):
                stream.write("[")
                for position, item in enumerate(value):
                    if position:
                        stream.write(",")
                    write(item)
                stream.write("]")
            else:
                stream.writelines(encoder.iterencode(value))

        write(document)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
