"""Bind retained sampling originals separately from derived learning experience."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from fh5.collection_store import encode, read_bounded, write_file
from fh5.numeric_images import asset


def _inventory(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    total = 0
    for path in paths:
        name = str(path.resolve())
        if name in result:
            continue
        if len(result) >= 50000:
            raise ValueError("Sampling originals exceed 50000 files")
        raw = read_bounded(path, min(256 * 1024**2, 1024**3 - total))
        total += len(raw)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def seal_sampling_sources(root: Path, review: Path | None) -> dict[str, str]:
    def paths() -> Iterable[Path]:
        # Called after sampling closes, before preparation creates derived files.
        for path in root.rglob("*"):
            if path.is_file():
                if not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Sampling original escapes its attempt directory")
                yield path
        if review is not None:
            yield review
            proof = json.loads(read_bounded(review, 4 * 1024**2))
            for item in proof["items"]:
                yield asset(review.parent, item["path"])

    return _inventory(paths())


def seal_sampling_attempt(root: Path, review: Path | None) -> dict[str, str]:
    """Keep the review's role even if a later summary omits original file bindings."""
    proof = None
    if review is not None:
        name = str(review.resolve())
        proof = {"path": name, "sha256": _inventory([review])[name]}
    write_file(root / "sampling-sources.json", encode({"version": 1, "review": proof}))
    return seal_sampling_sources(root, review)


def verify_sampling_sources(expected: dict[str, str]) -> None:
    try:
        if not expected or _inventory(Path(name) for name in expected) != expected:
            raise ValueError("Sampling originals differ")
    except (OSError, ValueError) as error:
        raise ValueError("Retained sampling originals changed or are unavailable") from error
