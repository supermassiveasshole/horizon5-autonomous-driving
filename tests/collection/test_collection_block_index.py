"""Append-only block references through public collection and review operations."""

import hashlib
import json
import shutil
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from fh5.artifacts.io import write_file
from fh5.collection.model import CollectionConfig, CollectionInput, CollectionReview
from fh5.experiment import run_experiment
from tests.collection.test_collection import Stream, input_at, request


def test_collection_crosses_old_block_count_without_expanding_published_index(tmp_path):
    req = request(tmp_path, block_rows=1)
    sealed = threading.Semaphore(0)
    total = 8193

    def persist(path, payload):
        write_file(path, payload)
        if path.name == "manifest.json":
            sealed.release()

    def points():
        for number in range(total):
            if number:
                assert sealed.acquire(timeout=5), "Previous real block did not seal"
            yield CollectionInput((number + 1) * 50_000_000)

    result = run_experiment(
        req, collection_environment=Stream(points()), collection_write=persist
    ).summary["collection"]
    assert result["written_rows"] == total and result["sealed_blocks"] == total
    assert result["complete"] and result["archive_error"] is None
    index = json.loads((req.output_dir / "index.json").read_bytes())
    final = json.loads((req.output_dir / "final.json").read_bytes())
    assert "blocks" not in index and "blocks" not in final
    assert index["references"] == final["references"]
    assert index["references"]["count"] == total
    reviewed = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html"))
    evidence = reviewed.summary["collection"]
    assert evidence["verified_blocks"] == evidence["rows"] == total
    assert evidence["complete"] and evidence["errors"] == []


def test_frozen_synthetic_prepare_preserves_explicit_long_duration(tmp_path):
    from tests.collection.test_collection_process import prepare

    _, _, result = prepare(tmp_path, seconds=12 * 3600 + 1)
    assert result.summary["collection"]["collection"]["seconds"] == 12 * 3600 + 1
    assert result.summary["collection"]["commands_sent"] is False


def record_short(tmp_path, **config):
    req = request(tmp_path, block_rows=1, **config)
    result = run_experiment(
        req, collection_environment=Stream(input_at(ms) for ms in (250, 300, 350))
    ).summary["collection"]
    return req.output_dir, result


def test_old_published_prefix_survives_valid_and_unfinished_append_tail(tmp_path):
    root, _ = record_short(tmp_path)
    (root / "final.json").unlink()
    journal = root / "block-references.jsonl"
    first = journal.read_bytes().splitlines(keepends=True)[0]
    index = json.loads((root / "index.json").read_bytes())
    index["references"].update(bytes=len(first), sha256=hashlib.sha256(first).hexdigest(), count=1)
    (root / "index.json").write_text(json.dumps(index))
    with journal.open("ab") as stream:
        stream.write(b'{"interrupted append":')
    result = run_experiment(CollectionReview(root, tmp_path / "review.html")).summary["collection"]
    assert result["verified_blocks"] == result["rows"] == 3 and result["errors"] == []
    assert not result["complete"] and "tail size unknown" in result["recovery"]


@pytest.mark.parametrize(
    "fault", ["hash", "bytes", "huge_bytes", "count", "duplicate", "malformed"]
)
def test_changed_bound_reference_prefix_is_reported_before_claiming_completeness(tmp_path, fault):
    root, _ = record_short(tmp_path)
    journal = root / "block-references.jsonl"
    final = json.loads((root / "final.json").read_bytes())
    prefix = final["references"]
    if fault == "hash":
        prefix["sha256"] = "0" * 64
    elif fault == "bytes":
        prefix["bytes"] += 1
    elif fault == "huge_bytes":
        prefix["bytes"] = 2**80
    elif fault == "count":
        prefix["count"] += 1
    else:
        lines = journal.read_bytes().splitlines(keepends=True)
        lines[1] = lines[0] if fault == "duplicate" else b"not JSON\n"
        changed = b"".join(lines)
        journal.write_bytes(changed)
        prefix.update(bytes=len(changed), sha256=hashlib.sha256(changed).hexdigest())
    (root / "final.json").write_text(json.dumps(final))
    result = run_experiment(CollectionReview(root, tmp_path / "review.html")).summary["collection"]
    assert not result["complete"] and result["errors"]
    assert any(error.get("file") == "final.json" for error in result["errors"])


def test_failed_reference_append_keeps_renamed_tail_recoverable(tmp_path, monkeypatch):
    req = request(tmp_path, block_rows=1)
    opened = Path.open
    appends = 0

    def fail_append(path, mode="r", *args, **kwargs):
        nonlocal appends
        if path == req.output_dir / "block-references.jsonl" and mode == "ab":
            appends += 1
            if appends == 2:
                raise OSError("reference disk write failed")
        return opened(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_append)
    result = run_experiment(
        req, collection_environment=Stream([input_at(250), input_at(300)])
    ).summary["collection"]
    assert not result["complete"] and "reference disk write failed" in result["archive_error"]
    assert result["written_rows"] == result["sealed_blocks"] == 2
    assert result["references"]["count"] == 1
    reviewed = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html"))
    assert reviewed.summary["collection"]["verified_blocks"] == 2
    assert reviewed.summary["collection"]["errors"] == []
    assert not reviewed.summary["collection"]["complete"]


def test_legacy_reference_arrays_and_expanded_block_names_remain_readable(tmp_path):
    root, _ = record_short(tmp_path)
    refs = [
        json.loads(line) for line in (root / "block-references.jsonl").read_bytes().splitlines()
    ]
    for ref, name in zip(refs, ("999998", "999999", "1000000")):
        old = root / ref["path"]
        block = root / "blocks" / name
        old.rename(block)
        manifest = json.loads((block / "manifest.json").read_bytes())
        manifest["index"] = int(name)
        payload = json.dumps(manifest).encode()
        (block / "manifest.json").write_bytes(payload)
        ref.update(path="blocks/" + name, sha256=hashlib.sha256(payload).hexdigest())
    for name in ("index.json", "final.json"):
        document = json.loads((root / name).read_bytes())
        document.pop("references")
        document["blocks"] = refs
        (root / name).write_text(json.dumps(document))
    result = run_experiment(CollectionReview(root, tmp_path / "review.html")).summary["collection"]
    assert result["complete"] and result["errors"] == []
    assert [row["path"] for row in result["blocks"]] == [ref["path"] for ref in refs]


def test_explicit_block_budget_still_seals_progress_before_stopping(tmp_path):
    root, result = record_short(tmp_path, max_blocks=2)
    assert result["written_rows"] == result["sealed_blocks"] == 2
    assert not result["complete"] and "collection_block_budget" in result["archive_error"]
    reviewed = run_experiment(CollectionReview(root, tmp_path / "review.html"))
    assert reviewed.summary["collection"]["verified_blocks"] == 2
    assert reviewed.summary["collection"]["errors"] == []


def test_large_explicit_disk_and_duration_budgets_reach_actual_collection(tmp_path):
    _, result = record_short(
        tmp_path, max_blocks=8193, max_disk_bytes=2 * 1024**4, seconds=12 * 3600 + 1
    )
    assert result["complete"] and result["written_rows"] == 3


def test_large_disk_reserve_is_valid_but_available_space_still_stops_collection(tmp_path):
    _, result = record_short(tmp_path, min_free_bytes=2 * 1024**4)
    assert result["written_rows"] == 0 and not result["complete"]
    assert "collection_disk_reserve" in result["archive_error"]


@pytest.mark.parametrize("budget", ["default", "explicit_total", "explicit_reserve", "exhausted"])
def test_default_collection_uses_available_disk_and_honors_only_explicit_disk_budgets(
    tmp_path, monkeypatch, budget
):
    req = request(tmp_path, block_rows=1)
    defaults = CollectionConfig()
    config = replace(
        req.config,
        max_disk_bytes=1 if budget == "explicit_total" else defaults.max_disk_bytes,
        min_free_bytes=1024**3 if budget == "explicit_reserve" else defaults.min_free_bytes,
    )
    req = replace(req, config=config)
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda path: usage._replace(free=0 if budget == "exhausted" else 512 * 1024**2),
    )
    result = run_experiment(
        req, collection_environment=Stream([input_at(250), input_at(300), input_at(350)])
    ).summary["collection"]
    if budget == "default":
        assert result["complete"] and result["written_rows"] == 3
        frozen = json.loads((req.output_dir / "session.json").read_bytes())["configuration"]
        assert frozen["max_disk_bytes"] is None and frozen["min_free_bytes"] == 0
    else:
        assert not result["complete"] and result["written_rows"] == 0
        reason = (
            "collection_disk_budget" if budget == "explicit_total" else "collection_disk_reserve"
        )
        assert reason in result["archive_error"]
