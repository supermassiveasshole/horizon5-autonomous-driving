"""Growing dependency inventories through public storage planning."""

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from storage_files import storage_files
from test_candidate_store import candidates as candidates
from test_evaluation import sha
from test_learning_capacity import budget_request
from test_learning_loop import SharedBackend
from test_learning_loop import seeded_loop as seeded_loop
from test_learning_storage import recorded_storage as recorded_storage
from test_learning_storage import session_with_task, storage_request

from fh5.experiment import run_experiment


def test_storage_plan_streams_file_details_outside_its_summary(tmp_path, recorded_storage):
    request = storage_request(tmp_path, recorded_storage)
    result = run_experiment(request).summary["storage"]
    assert result["version"] == 2
    binding = result["files"]
    assert binding["format"] == "storage-files-v1"
    path = request.output_dir / binding["path"]
    assert sha(path) == binding["sha256"]
    total = count = 0
    previous = ""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            assert row["path"] > previous
            previous = row["path"]
            total += row["bytes"]
            count += 1
    assert count == binding["count"] == result["protected_files"] > 0
    assert total == result["protected_bytes"]
    assert "metadata_budget_bytes" not in result


def test_storage_plan_accepts_more_than_the_old_file_ceiling(tmp_path, recorded_storage):
    run, _ = recorded_storage
    config = json.loads((run / "config.json").read_bytes())
    session = session_with_task(tmp_path, recorded_storage, Path(config["task"]))
    request = storage_request(tmp_path, session)
    baseline = run_experiment(request).summary["storage"]
    evidence = session[0] / "many-originals"
    evidence.mkdir()
    for number in range(100_001):
        (evidence / f"{number:06d}.bin").touch(exist_ok=False)
    output = tmp_path / "grown-plan"
    plan = run_experiment(replace(request, output_dir=output)).summary["storage"]
    assert plan["status"] == "within_budget"
    assert plan["protected_bytes"] == baseline["protected_bytes"]
    assert plan["protected_files"] == baseline["protected_files"] + 100_001
    assert plan["files"]["count"] == plan["protected_files"]
    assert (evidence / "000000.bin").exists() and (evidence / "100000.bin").exists()


@pytest.mark.parametrize("entry", ["plan", "loop"])
def test_declared_budget_is_not_rejected_by_an_unrelated_numeric_ceiling(
    tmp_path, seeded_loop, recorded_storage, entry
):
    budget = 2**50 + 1
    if entry == "plan":
        request = storage_request(tmp_path, recorded_storage, budget_bytes=budget)
        result = run_experiment(request).summary["storage"]
        assert result["status"] == "within_budget"
        assert result["budget_bytes"] == budget
    else:
        request = budget_request(
            tmp_path, seeded_loop, recorded_storage[1], budget_bytes=budget, min_free_bytes=budget
        )
        backend = SharedBackend(seeded_loop[0])
        result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
        # A declared reserve greater than actual free space is a capacity stop,
        # not a configuration-format error at an invented upper number.
        assert result["stop_reason"] == "storage_budget_exhausted"
        assert result["storage_checks"][0]["reasons"] == ["disk_reserve"]
        assert not backend.leases and result["resources_released"]
        assert result["learner_updates"] == 0


def test_inventory_cannot_count_itself_when_temp_is_inside_the_source(tmp_path, recorded_storage):
    run, _ = recorded_storage
    config = json.loads((run / "config.json").read_bytes())
    session = session_with_task(tmp_path, recorded_storage, Path(config["task"]))
    source_temp = session[0] / "temporary"
    source_temp.mkdir()
    originals = {path: sha(path) for path in session[0].rglob("*") if path.is_file()}
    request = storage_request(tmp_path, session)
    with pytest.MonkeyPatch.context() as filesystem:
        filesystem.setattr(tempfile, "tempdir", str(source_temp))
        plan = run_experiment(request).summary["storage"]
    assert plan["status"] == "within_budget"
    rows = list(storage_files(request.output_dir, plan))
    assert all(Path(row["path"]).is_file() for row in rows)
    assert not any(Path(row["path"]).is_relative_to(source_temp) for row in rows)
    assert list(source_temp.iterdir()) == []
    assert all(sha(path) == digest for path, digest in originals.items())
