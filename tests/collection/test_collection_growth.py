"""Growing sealed evidence remains usable through public preparation and review."""

import hashlib
import json
import shutil
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from fh5.artifacts.document import ReplayArray, replay_document
from fh5.artifacts.io import VerifiedFile
from fh5.collection.bc import CollectionBCPrepare
from fh5.collection.dataset import CollectionDatasetReview
from fh5.experiment import run_experiment
from tests.collection.test_collection import Stream, input_at, request


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def _shift_clocks(value, delta):
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key.endswith("_ns")
                and key not in ("segment_start_ns", "uncertainty_ns")
                and type(item) is int
            ):
                value[key] = item + delta
            else:
                _shift_clocks(item, delta)
    elif isinstance(value, list):
        for item in value:
            _shift_clocks(item, delta)


def _growing_source(
    root, *, attempt_rows=5006, attempt_count=11, clock_offset_ns=0, identity_prefix=""
):
    """Extend real recorder output with independently timestamped synthetic evidence."""
    recorded = root / "recorded"
    recorded.mkdir()
    req = replace(
        request(recorded, block_rows=4096),
        input_conditions={"camera": "chase_far", "blueprint": "105657219"},
    )
    result = run_experiment(
        req, collection_environment=Stream(input_at(ms) for ms in range(250, 951, 50))
    )
    assert result.summary["collection"]["complete"]
    original = req.output_dir / "blocks" / "000000"
    with (original / "rows.jsonl").open(encoding="utf-8") as stream:
        template = next(
            row for row in reversed([json.loads(line) for line in stream]) if row["frames"]
        )
    assert template["input_usable"] and not template["reasons"]

    source = root / "growing-session"
    (source / "blocks").mkdir(parents=True)
    (source / ".partial").mkdir()
    shutil.copyfile(req.output_dir / "session.json", source / "session.json")
    binding = _digest(source / "session.json")
    # Four initial histories and the following-label boundary are excluded in each
    # attempt; the remaining 5,001 observations exercise the configured reservoir.
    total = attempt_rows * attempt_count
    refs = []
    for index, first in enumerate(range(0, total, 4096)):
        block = source / "blocks" / f"{index:06d}"
        block.mkdir()
        shutil.copytree(original / "pixels", block / "pixels")
        last = min(first + 4096, total)
        with (block / "rows.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for sequence in range(first, last):
                row = deepcopy(template)
                _shift_clocks(row, clock_offset_ns + sequence * 50_000_000)
                row["sequence"] = sequence
                for frame in row["frames"]:
                    frame["frame_id"] = str(frame["source_time_ns"])
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        manifest = {
            "version": 1,
            "session_sha256": binding,
            "index": index,
            "row_count": last - first,
            "first_sequence": first,
            "last_sequence": last - 1,
            "rows_sha256": _digest(block / "rows.jsonl"),
            "bytes": sum(path.stat().st_size for path in block.rglob("*") if path.is_file()),
        }
        _json(block / "manifest.json", manifest)
        refs.append(
            {
                "path": f"blocks/{index:06d}",
                "sha256": _digest(block / "manifest.json"),
                "rows": last - first,
            }
        )
    _json(source / "index.json", {"version": 1, "session_sha256": binding, "blocks": refs})
    final = json.loads((req.output_dir / "final.json").read_bytes())
    final.pop("references", None)  # This fixture deliberately exercises legacy v1 arrays.
    final.update(
        blocks=refs,
        seen_rows=total,
        written_rows=total,
        sealed_blocks=len(refs),
        disk_bytes=sum(
            path.stat().st_size for path in (source / "blocks").rglob("*") if path.is_file()
        ),
    )
    _json(source / "final.json", final)
    attempts = []
    for index in range(attempt_count):
        start, end = index * attempt_rows, (index + 1) * attempt_rows
        attempts.append(
            {
                "id": f"{identity_prefix}attempt-{index:02d}",
                "group": f"{identity_prefix}group-{index:02d}",
                "split": "train" if index < 9 else "development" if index == 9 else "evaluation",
                "start_sequence": start,
                "end_sequence": end,
                "related_attempts": [],
                "intervals": [
                    {
                        "start_sequence": start,
                        "end_sequence": end,
                        "quality": "trusted",
                        "reasons": [],
                        "evidence": ["independent synthetic timeline"],
                        "road_kind": "unknown",
                    }
                ],
            }
        )
    review = root / "review.json"
    _json(
        review,
        {
            "version": 1,
            "session_sha256": binding,
            "conditions_verified": True,
            "evidence": ["synthetic fixture only"],
            "independence_evidence": [
                "eleven disjoint synthetic attempts, with independent history floors"
            ],
            "attempts": attempts,
        },
    )
    config = root / "config.json"
    _json(
        config,
        {
            "version": 2,
            "seed": 7,
            "sources": [{"recording": str(source), "review": str(review)}],
            "rules": {
                "max_samples_per_attempt": 5000,
                "speed_range_mps": [0, 100],
                "steering_limit": 1.0,
                "longitudinal_limit": 1.0,
                "max_label_delay_ms": 50,
            },
            "action_history_offsets_ms": [100, 50, 0],
            "max_action_age_ms": 100,
            "waypoint_distances_m": [5, 10, 20],
        },
    )
    return config


def test_more_than_fifty_thousand_selected_observations_prepare_and_review():
    # Use the system temporary disk, not the repository disk. Pixels are shared by
    # content and every source/output array is written or inspected incrementally.
    with TemporaryDirectory(prefix="fh5-collection-growth-") as temporary:
        root = Path(temporary)
        config = _growing_source(root)
        output = root / "prepared"
        result = run_experiment(CollectionBCPrepare(config, output))
        expected = {"train": 45000, "development": 5000, "evaluation": 5000}
        summary = result.summary["collection_bc"]
        assert summary["selection"]["bc_samples_by_split"] == expected
        assert summary["diagnostic_only"] and not summary["closed_loop_validated"]
        selection = output / "selection.json"
        assert _digest(selection) == summary["selection_sha256"]
        reformatted = root / "reformatted-selection.json"
        with replay_document(VerifiedFile(selection, _digest(selection))) as data:
            assert data["version"] == 1 and data["kind"] == "collection-dataset-snapshot-v1"
            assert len(data["samples"]) == 55000
            counts = Counter()
            previous = -1
            for sample in data["samples"]:
                assert sample["sequence"] > previous
                previous = sample["sequence"]
                assert sample["bc_eligible"] and not sample["q_eligible"]
                counts[sample["attempt"]] += 1
            assert counts == {f"attempt-{index:02d}": 5000 for index in range(11)}
            # Preserve the legacy object/array representation but change whitespace
            # and key order: review verifies content, not canonical byte layout.
            with reformatted.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write("{\n")
                for number, key in enumerate(reversed(list(data))):
                    if number:
                        stream.write(",\n")
                    stream.write(json.dumps(key) + ": ")
                    value = data[key]
                    if isinstance(value, ReplayArray):
                        stream.write("[\n")
                        for position, item in enumerate(value):
                            if position:
                                stream.write(",\n")
                            json.dump(item, stream)
                        stream.write("\n]")
                    else:
                        json.dump(value, stream)
                stream.write("\n}\n")
        assert _digest(reformatted) != _digest(selection)
        reviewed = run_experiment(CollectionDatasetReview(reformatted, root / "checked.html"))
        checked = reviewed.summary["collection_dataset"]
        assert checked["verified"] and checked["bc_samples_by_split"] == expected
        assert checked["dataset_sha256"] == _digest(reformatted)
        assert checked["evaluation"] == {"groups": 1, "coverage": "withheld"}

        for name, count, groups in (
            ("dataset.json", 50000, {f"group-{index:02d}" for index in range(10)}),
            ("evaluation.json", 5000, {"group-10"}),
        ):
            path = output / name
            if name == "dataset.json":
                assert path.stat().st_size > 128 * 1024**2
            digest = _digest(path)
            assert digest == summary["snapshot_sha256"][name]
            with replay_document(VerifiedFile(path, digest)) as data:
                assert len(data["decisions"]) == count
                assert {group["id"] for group in data["groups"]} == groups
                assert data["provenance"]["selection_sha256"] == _digest(selection)
                if name == "dataset.json":
                    assert (
                        data["provenance"]["final_dataset_sha256"]
                        == summary["snapshot_sha256"]["evaluation.json"]
                    )
                previous = -1
                for decision in data["decisions"]:
                    assert decision["source_sequence"] > previous
                    previous = decision["source_sequence"]
                    assert decision["group"] in groups and decision["bc_eligible"]
