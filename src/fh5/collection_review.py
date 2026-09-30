"""Read sealed blocks only; an interrupted active block is never a dataset."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.collection import CollectionReview
from fh5.collection_store import atomic_json
from fh5.numeric_images import PixelContract, validate_frame_history
from fh5.numeric_recording import read_numeric_frame

if TYPE_CHECKING:
    from fh5.experiment import RunResult


def collection_result(path: Path, result: dict[str, Any]) -> RunResult:
    from fh5.experiment import RunResult

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>持续采集状态</title>'
        "<style>body{font:16px system-ui;margin:32px;max-width:1000px}pre{white-space:pre-wrap}</style>"
        "<h1>持续采集状态</h1><p>完整封存不等于优质示范；碰撞、离路、导航和尝试边界仍需核验。</p><pre>"
        + html.escape(data)
        + "</pre></html>",
        encoding="utf-8",
    )
    return RunResult({"source_kind": "passive_collection"}, [], [], {"collection": result}, path)


def _read(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Collection asset exceeds bounded limit")
    return payload


def review_collection(request: CollectionReview) -> RunResult:
    root = request.recording_dir
    if request.report_path.exists() or request.report_path.with_suffix(".json").exists():
        raise FileExistsError(request.report_path)
    session_raw = _read(root / "session.json", 1024**2)
    session = json.loads(session_raw)
    if (
        session["version"] != 1
        or session["kind"] != "continuous-numeric-collection-v1"
        or session["commands_sent"] is not False
    ):
        raise ValueError("Unsupported collection session")
    binding = hashlib.sha256(session_raw).hexdigest()
    contract = PixelContract.from_metadata(session["configuration"]["pixels"])
    blocks = sorted((root / "blocks").iterdir())
    if len(blocks) > 8192:
        raise ValueError("Too many collection blocks")
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
            raw = _read(block / "manifest.json", 4096)
            manifest = json.loads(raw)
            if (
                manifest["version"] != 1
                or manifest["session_sha256"] != binding
                or manifest["index"] != int(block.name)
            ):
                raise ValueError("Sealed block belongs to another session or index")
            rows_raw = _read(block / "rows.jsonl", 256 * 1024**2)
            if hashlib.sha256(rows_raw).hexdigest() != manifest["rows_sha256"]:
                raise ValueError("Sealed rows hash mismatch")
            rows = rows_raw.splitlines()
            if len(rows) != manifest["row_count"] or not 1 <= len(rows) <= 4096:
                raise ValueError("Sealed row count mismatch")
            first = last = -1
            for line in rows:
                row = json.loads(line)
                sequence = row["sequence"]
                if type(sequence) is not int or sequence <= previous:
                    raise ValueError("Collection sequence repeated or moved backwards")
                result["missing_rows"] += sequence - previous - 1
                previous = last = sequence
                if first == -1:
                    first = sequence
                if row["frames"]:
                    if len(row["frames"]) != len(contract.history_offsets_ms):
                        raise ValueError("Incomplete archived frame history")
                    frames = tuple(
                        read_numeric_frame(
                            block, f, byte_limit=contract.size[0] * contract.size[1] * 3
                        )
                        for f in row["frames"]
                    )
                    reason = validate_frame_history(
                        row["capture_epoch"], row["at_ns"], frames, contract
                    )
                    if reason or any(f.source_time_ns < row["segment_start_ns"] for f in frames):
                        raise ValueError("Archived history crosses a segment or violates contract")
                result["rows"] += 1
            if first != manifest["first_sequence"] or last != manifest["last_sequence"]:
                raise ValueError("Sealed sequence bounds differ")
            result["verified_blocks"] += 1
            result["blocks"].append(
                {
                    "path": str(block.relative_to(root)),
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    "rows": len(rows),
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["errors"].append({"block": block.name, "error": str(error)})
    final_path = root / "final.json"
    result["final_status_present"] = final_path.is_file()
    if final_path.is_file():
        final = json.loads(_read(final_path, 4 * 1024**2))
        if final["session_sha256"] != binding or final["seen_rows"] < previous + 1:
            result["errors"].append({"error": "Final status binding or row count differs"})
        else:
            result["missing_rows"] += final["seen_rows"] - previous - 1
            result["complete"] = bool(
                final["complete"]
                and not result["errors"]
                and not result["missing_rows"]
                and result["verified_blocks"]
            )
    else:
        result["recovery"] = "sealed_blocks_only; final tail size unknown"
    atomic_json(request.report_path.with_suffix(".json"), result)
    return collection_result(request.report_path, result)
