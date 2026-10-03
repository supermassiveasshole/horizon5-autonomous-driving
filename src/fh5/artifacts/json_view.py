"""Write ordered report arrays without materializing their encoded document."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TextIO


@dataclass(frozen=True)
class JsonArray:
    """A single-pass projection used only while rendering a report."""

    values: Iterable[Any]


def _chunks(value: Any, level: int = 0) -> Iterator[str]:
    if isinstance(value, dict):
        yield "{"
        for position, (key, item) in enumerate(value.items()):
            # Delegate key coercion/rejection to the standard JSON encoder.
            encoded_key = (
                json.dumps(key, ensure_ascii=False)
                if isinstance(key, str)
                else json.dumps({key: None}, ensure_ascii=False, allow_nan=False)[1:].removesuffix(
                    ": null}"
                )
            )
            yield ("," if position else "") + "\n" + "  " * (level + 1)
            yield encoded_key + ": "
            yield from _chunks(item, level + 1)
        if value:
            yield "\n" + "  " * level
        yield "}"
    elif isinstance(value, JsonArray) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        yield "["
        populated = False
        for item in value.values if isinstance(value, JsonArray) else value:
            yield ("," if populated else "") + "\n" + "  " * (level + 1)
            yield from _chunks(item, level + 1)
            populated = True
        if populated:
            yield "\n" + "  " * level
        yield "]"
    else:
        yield json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_json(stream: TextIO, value: Any, *, script_safe: bool = False) -> None:
    """Stream records in source order; only one scalar/record is encoded at a time."""
    for chunk in _chunks(value):
        stream.write(chunk.replace("<", "\\u003c") if script_safe else chunk)
