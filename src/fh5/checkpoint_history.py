"""Portable hash-linked ancestry without growing arrays in each model manifest."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fh5.artifact_io import VerifiedFile, read_json
from fh5.collection_store import encode, write_file
from fh5.numeric_images import asset

NodeReader = Callable[[Path, str], dict[str, Any]]


def empty_history() -> dict[str, Any]:
    return {"kind": "linked-checkpoint-history-v1", "count": 0, "head": None}


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _entry(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"checkpoint", "checkpoint_sha256", "report", "report_sha256"}
        or any(
            not isinstance(value[kind], str) or not _digest(value[kind + "_sha256"])
            for kind in ("checkpoint", "report")
        )
    ):
        raise ValueError("Invalid checkpoint history entry")
    return value


def _count(value: Any) -> int:
    if (
        not isinstance(value, dict)
        or set(value) != {"kind", "count", "head"}
        or value["kind"] != "linked-checkpoint-history-v1"
        or type(value["count"]) is not int
        or value["count"] < 0
    ):
        raise ValueError("Invalid checkpoint history index")
    count: int = value["count"]
    head = value["head"]
    if count == 0:
        if head is not None:
            raise ValueError("Empty checkpoint history has a head")
    elif (
        not isinstance(head, dict)
        or set(head) != {"path", "sha256"}
        or not isinstance(head["path"], str)
        or not _digest(head["sha256"])
    ):
        raise ValueError("Invalid checkpoint history head")
    return count


def _read_node(path: Path, expected: str) -> dict[str, Any]:
    value = read_json(path, expected_sha256=expected)
    if not isinstance(value, dict):
        raise ValueError("Invalid checkpoint history node")
    return value


def history_assets(
    root: Path, history: Any, *, read_node: NodeReader = _read_node
) -> Iterator[tuple[str, str]]:
    """Enumerate linked records and original evidence, verifying one node at a time."""
    if isinstance(history, list):
        for value in history:
            entry = _entry(value)
            for kind in ("checkpoint", "report"):
                yield entry[kind], entry[kind + "_sha256"]
        return
    current = history
    count = _count(current)
    while count:
        head = current["head"]
        node = read_node(asset(root, head["path"]), head["sha256"])
        if (
            set(node) != {"kind", "entry", "previous"}
            or node["kind"] != "checkpoint-history-node-v1"
        ):
            raise ValueError("Invalid checkpoint history node")
        previous = node["previous"]
        if _count(previous) != count - 1:
            raise ValueError("Checkpoint history count is not continuous")
        entry = _entry(node["entry"])
        yield head["path"], head["sha256"]
        for kind in ("checkpoint", "report"):
            yield entry[kind], entry[kind + "_sha256"]
        current, count = previous, count - 1


def verify_history(root: Path, history: Any) -> None:
    for name, digest in history_assets(root, history):
        VerifiedFile(asset(root, name), digest).verify()


def _node(root: Path, previous: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    payload = encode({"kind": "checkpoint-history-node-v1", "entry": entry, "previous": previous})
    digest = hashlib.sha256(payload).hexdigest()
    name = f"history/{digest}-node.json"
    target = asset(root, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        VerifiedFile(target, digest).verify()
    else:
        write_file(target, payload)
    return {
        "kind": "linked-checkpoint-history-v1",
        "count": previous["count"] + 1,
        "head": {"path": name, "sha256": digest},
    }


@dataclass(frozen=True)
class HistorySource:
    root: Path
    history: Any
    parent: bytes
    report_sha256: str

    def retain(self, output: Path) -> dict[str, Any]:
        for name, digest in history_assets(self.root, self.history):
            target = asset(output, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                VerifiedFile(target, digest).verify()
            else:
                VerifiedFile(asset(self.root, name), digest).copy_to(target)
        if isinstance(self.history, list):
            binding = empty_history()
            for value in self.history:
                binding = _node(output, binding, _entry(value))
        else:
            binding = self.history
        parent_sha = hashlib.sha256(self.parent).hexdigest()
        entry = {
            "checkpoint": f"history/{parent_sha}-checkpoint.json",
            "checkpoint_sha256": parent_sha,
            "report": f"history/{parent_sha}-report.json",
            "report_sha256": self.report_sha256,
        }
        checkpoint = asset(output, entry["checkpoint"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        write_file(checkpoint, self.parent)
        VerifiedFile(self.root / "training-report.json", self.report_sha256).copy_to(
            asset(output, entry["report"])
        )
        return _node(output, binding, entry)
