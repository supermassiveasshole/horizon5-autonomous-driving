"""Read sealed blocks only; an interrupted active block is never a dataset."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from fh5.artifacts.document import replay_document
from fh5.artifacts.io import VerifiedFile, atomic_json, read_bounded, sha256_file
from fh5.collection.index import BlockReferences, read_references
from fh5.collection.model import CollectionReview
from fh5.collection.store import collection_complete
from fh5.observation.numeric import PixelContract, validate_frame_history
from fh5.observation.recording import read_numeric_frame
from fh5.reporting.presentation import optional_report

if TYPE_CHECKING:
    from fh5.result import RunResult


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
    with ExitStack() as resources:
        return _review_collection(request, resources)


def _review_collection(request: CollectionReview, resources: ExitStack) -> RunResult:
    from fh5.result import RunResult

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
    references: dict[str, BlockReferences] = {}
    snapshots = Path(resources.enter_context(TemporaryDirectory(prefix="fh5-collection-review-")))
    final = None
    result["final_status_present"] = (root / "final.json").is_file()
    for name in ("final.json", "index.json"):
        if not (root / name).is_file():
            continue
        if name == "final.json":
            result["final_status_present"] = True
        try:
            frozen = snapshots / name
            with (root / name).open("rb") as origin, frozen.open("xb") as target:
                shutil.copyfileobj(origin, target)
            source = VerifiedFile(frozen, sha256_file(frozen))
            document = resources.enter_context(replay_document(source))
            references[name] = resources.enter_context(read_references(root, binding, document))
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
    blocks = sorted(
        (root / "blocks").iterdir(),
        key=lambda block: (0, int(block.name)) if block.name.isdecimal() else (1, block.name),
    )
    seen_paths = {"blocks/" + block.name for block in blocks}
    for name, refs in references.items():
        for ref in refs:
            if ref["path"] not in seen_paths:
                result["errors"].append(
                    {"file": name, "error": "Missing sealed block: " + ref["path"]}
                )
    previous = -1
    for block in blocks:
        try:
            if (
                not block.is_dir()
                or re.fullmatch(r"[0-9]{6,}", block.name) is None
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
            or len(references["final.json"]) != len(seen_paths)
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
    report = optional_report(
        request.report_path,
        "持续采集审核（完整封存不等于优质示范）",
        result,
        fallback=request.report_path.with_suffix(".json"),
        exclusive=True,
    )
    return RunResult({"source_kind": "passive_collection"}, [], [], {"collection": result}, report)
