"""Read sealed blocks only; an interrupted active block is never a dataset."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.artifact_io import VerifiedFile
from fh5.collection import CollectionReview
from fh5.collection_store import atomic_json, collection_complete, read_bounded, write_file
from fh5.numeric_images import PixelContract, validate_frame_history
from fh5.numeric_recording import read_numeric_frame

if TYPE_CHECKING:
    from fh5.experiment import RunResult


def collection_result(
    path: Path, result: dict[str, Any], *, title: str = "持续采集状态"
) -> RunResult:
    from fh5.experiment import RunResult

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    write_file(
        path,
        (
            '<!doctype html><html lang="zh"><meta charset="utf-8"><title>'
            + html.escape(title)
            + "</title>"
            "<style>body{font:16px system-ui;margin:32px;max-width:1000px}pre{white-space:pre-wrap}</style>"
            "<h1>"
            + html.escape(title)
            + "</h1><p>完整封存不等于优质示范；碰撞、离路、导航和尝试边界仍需核验。</p><pre>"
            + html.escape(data)
            + "</pre></html>"
        ).encode("utf-8"),
    )
    return RunResult({"source_kind": "passive_collection"}, [], [], {"collection": result}, path)


def _references(document: dict[str, Any], binding: str) -> dict[str, dict[str, Any]]:
    if document["version"] != 1 or document["session_sha256"] != binding:
        raise ValueError("Collection reference document binding differs")
    blocks = document["blocks"]
    if not isinstance(blocks, list) or len(blocks) > 8192:
        raise ValueError("Invalid bounded collection reference list")
    result = {}
    for ref in blocks:
        path, digest, rows = ref["path"], ref["sha256"], ref["rows"]
        if (
            not isinstance(path, str)
            or re.fullmatch(r"blocks/[0-9]{6}", path) is None
            or path in result
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(rows) is not int
            or not 1 <= rows <= 4096
        ):
            raise ValueError("Invalid or repeated sealed block reference")
        result[path] = ref
    return result


def sealed_block_rows(
    block: Path, manifest: dict[str, Any], contract: PixelContract
) -> Iterator[dict[str, Any]]:
    """Read verified bytes one row at a time; exhaust before accepting the block."""
    expected = manifest["row_count"]
    if type(expected) is not int or expected < 1:
        raise ValueError("Sealed row count mismatch")
    rows = VerifiedFile(block / "rows.jsonl", manifest["rows_sha256"])
    with rows.snapshot() as stream:
        count = 0
        for line in stream:
            row = json.loads(line)
            count += 1
            if count > expected:
                raise ValueError("Sealed row count mismatch")
            if (count == 1 and row["sequence"] != manifest["first_sequence"]) or (
                count == expected and row["sequence"] != manifest["last_sequence"]
            ):
                raise ValueError("Sealed sequence bounds differ")
            if row["frames"]:
                frames = tuple(
                    read_numeric_frame(block, f, byte_limit=contract.size[0] * contract.size[1] * 3)
                    for f in row["frames"]
                )
                reason = validate_frame_history(
                    row["capture_epoch"], row["at_ns"], frames, contract
                )
                if reason or any(f.source_time_ns < row["segment_start_ns"] for f in frames):
                    raise ValueError("Archived history crosses a segment or violates contract")
            yield row
        if count != expected:
            raise ValueError("Sealed row count mismatch")


def review_collection(request: CollectionReview) -> RunResult:
    root = request.recording_dir
    if request.report_path.exists() or request.report_path.with_suffix(".json").exists():
        raise FileExistsError(request.report_path)
    session_raw = read_bounded(root / "session.json", 1024**2)
    session = json.loads(session_raw)
    if (
        session["version"] != 1
        or session["kind"] != "continuous-numeric-collection-v1"
        or session["commands_sent"] is not False
    ):
        raise ValueError("Unsupported collection session")
    binding = hashlib.sha256(session_raw).hexdigest()
    contract = PixelContract.from_metadata(session["configuration"]["pixels"])
    result: dict[str, Any] = {
        "version": 1,
        "session_sha256": binding,
        "verified_blocks": 0,
        "rows": 0,
        "missing_rows": 0,
        "errors": [],
        "blocks": [],
        "commands_sent": False,
        "training_eligible": False,
        "complete": False,
        "active_blocks_ignored": len(list((root / ".partial").iterdir())),
    }
    # Snapshot documents before enumerating blocks: a concurrent seal may appear
    # after this index, and is recoverable even before its reference is published.
    references: dict[str, dict[str, dict[str, Any]]] = {}
    final = None
    result["final_status_present"] = (root / "final.json").is_file()
    for name in ("final.json", "index.json"):
        if not (root / name).is_file():
            continue
        if name == "final.json":
            result["final_status_present"] = True
        try:
            document = json.loads(read_bounded(root / name, 4 * 1024**2))
            references[name] = _references(document, binding)
            if name == "final.json":
                final = document
        except PermissionError as error:
            if name == "index.json" and not result["final_status_present"]:
                # Progress is replaceable while collection is open. Validate
                # immutable blocks independently; the unsealed tail stays unknown.
                result.setdefault("unavailable_snapshots", []).append(
                    {"file": name, "error": str(error)}
                )
            else:
                result["errors"].append({"file": name, "error": str(error)})
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["errors"].append({"file": name, "error": str(error)})
    blocks = sorted((root / "blocks").iterdir())
    if len(blocks) > 8192:
        raise ValueError("Too many collection blocks")
    seen_paths = {"blocks/" + block.name for block in blocks}
    for name, refs in references.items():
        for missing_path in refs.keys() - seen_paths:
            result["errors"].append(
                {"file": name, "error": "Missing sealed block: " + missing_path}
            )
    previous = -1
    for block in blocks:
        try:
            if (
                not block.is_dir()
                or len(block.name) != 6
                or not block.name.isdecimal()
                or block.is_symlink()
            ):
                raise ValueError("Unexpected sealed block entry")
            raw = read_bounded(block / "manifest.json", 4096)
            manifest = json.loads(raw)
            digest = hashlib.sha256(raw).hexdigest()
            for refs in references.values():
                ref = refs.get("blocks/" + block.name)
                if ref is not None and (
                    digest != ref["sha256"] or manifest["row_count"] != ref["rows"]
                ):
                    raise ValueError("Sealed manifest reference mismatch")
            if (
                manifest["version"] != 1
                or manifest["session_sha256"] != binding
                or manifest["index"] != int(block.name)
            ):
                raise ValueError("Sealed block belongs to another session or index")
            block_previous, missing, count = previous, 0, 0
            for row in sealed_block_rows(block, manifest, contract):
                sequence = row["sequence"]
                if type(sequence) is not int or sequence <= block_previous:
                    raise ValueError("Collection sequence repeated or moved backwards")
                missing += sequence - block_previous - 1
                block_previous = sequence
                count += 1
            previous = block_previous
            result["missing_rows"] += missing
            result["rows"] += count
            result["verified_blocks"] += 1
            result["blocks"].append(
                {
                    "path": "blocks/" + block.name,
                    "manifest_sha256": digest,
                    "rows": count,
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["errors"].append({"block": block.name, "error": str(error)})
    if final is not None:
        result["source_status"] = {
            key: final.get(key)
            for key in ("stop_reason", "error", "archive_error", "archive_released", "environment")
        }
        if final.get("complete") is True and (
            not collection_complete(final)
            or references["final.json"].keys() != seen_paths
            or final.get("written_rows") != result["rows"]
            or final.get("seen_rows") != result["rows"]
            or final.get("sealed_blocks") != result["verified_blocks"]
            or final.get("unsealed_rows") != 0
            or final.get("dropped_rows") != 0
        ):
            result["errors"].append({"error": "Complete final status differs from sealed evidence"})
        if type(final.get("seen_rows")) is not int or final["seen_rows"] < previous + 1:
            result["errors"].append({"error": "Final status binding or row count differs"})
        else:
            result["missing_rows"] += final["seen_rows"] - previous - 1
            result["complete"] = bool(
                final.get("complete") is True
                and not result["errors"]
                and not result["missing_rows"]
                and result["verified_blocks"]
            )
    else:
        result["recovery"] = "sealed_blocks_only; final tail size unknown"
    atomic_json(request.report_path.with_suffix(".json"), result)
    return collection_result(request.report_path, result)
