"""Immutable indexed update bindings, streamed in their original order."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fh5.artifact_io import sha256_file
from fh5.collection_store import atomic_json, encode


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _entry(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"directory", "sha256"}
        or not isinstance(value["directory"], str)
        or not _digest(value["sha256"])
    ):
        raise ValueError("Invalid learning update binding")
    return value


class UpdateHistory:
    """A fixed-size head authenticates sequential files; old arrays stay readable."""

    def __init__(self, root: Path, saved: Any) -> None:
        self.root, self.saved = root, saved
        self.legacy = isinstance(saved, list)
        if self.legacy:
            self.count = len(saved)
        elif (
            isinstance(saved, dict)
            and set(saved) == {"format", "count", "head_sha256"}
            and saved["format"] == "learning-update-history-v1"
            and type(saved["count"]) is int
            and saved["count"] >= 0
            and (
                saved["head_sha256"] is None
                if saved["count"] == 0
                else _digest(saved["head_sha256"])
            )
        ):
            self.count = saved["count"]
        else:
            raise ValueError("Invalid learning update history index")

    def path(self, number: int) -> Path:
        return self.root / "update-history" / f"{number:06d}.json"

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.legacy:
            for value in self.saved:
                yield _entry(value)
            return
        previous = None
        for number in range(self.count):
            raw = self.path(number).read_bytes()
            node = json.loads(raw)
            if (
                not isinstance(node, dict)
                or set(node) != {"format", "previous_sha256", "entry"}
                or node["format"] != "learning-update-node-v1"
                or node["previous_sha256"] != previous
            ):
                raise ValueError("Learning update history chain changed")
            previous = hashlib.sha256(raw).hexdigest()
            yield _entry(node["entry"])
        if previous != self.saved["head_sha256"]:
            raise ValueError("Learning update history head changed")

    def _publish(self, number: int, previous: str | None, entry: dict[str, Any]) -> str:
        node = {
            "format": "learning-update-node-v1",
            "previous_sha256": previous,
            "entry": _entry(entry),
        }
        digest = hashlib.sha256(encode(node)).hexdigest()
        path = self.path(number)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if sha256_file(path) != digest:
                raise ValueError("Retained learning update history node changed")
        else:
            # Publish before the parent acknowledges it. A crash can leave a
            # complete unacknowledged node, which is verified and reused above.
            atomic_json(path, node)
        return digest

    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        previous = None
        for number, existing in enumerate(self):
            if self.legacy:
                previous = self._publish(number, previous, existing)
        if not self.legacy:
            previous = self.saved["head_sha256"]
        return {
            "format": "learning-update-history-v1",
            "count": self.count + 1,
            "head_sha256": self._publish(self.count, previous, entry),
        }
