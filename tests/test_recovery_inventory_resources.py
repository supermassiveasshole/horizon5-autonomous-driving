"""Failed sampling archives remain recoverable as their original inventories grow."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from recovery_inventory_cases import FailedOriginals, digest, originals
from test_candidate_store import candidates as candidates
from test_learning_loop import SharedBackend, loop_request
from test_learning_loop import seeded_loop as seeded_loop

from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue

LEGACY_LIMIT = 64 * 1024**2  # Historical rejection boundary, never a production budget.


def failed_archive(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})
    backend = FailedOriginals(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "sampling_retries_exhausted", result.get("error")
    assert result["learner_updates"] == result["eligible_transitions"] == 0
    assert len(backend.leases) == 2 and backend.closed and result["resources_released"]
    assert all(lease.closed for lease in backend.leases)
    assert len(result["rounds"][0]["sampling_history"]) == 1
    return request, result


def archived_binding(root):
    state = json.loads((root / "state.json").read_bytes())
    return state, state["rounds"][0]["sampling_history"][0]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def rebind_legacy(root, *, padding=False, duplicate=False, bom=False, tail=b""):
    """Serialize a genuine older-format fixture, keeping actual original file hashes."""
    state, binding = archived_binding(root)
    path = Path(binding["originals_file"])
    inventory = originals(Path(binding["directory"]))
    with path.open("wb") as target:
        if bom:
            target.write(b"\xef\xbb\xbf")
        target.write(b"{")
        if duplicate:
            name = next(iter(inventory))
            target.write(json.dumps(name, ensure_ascii=False).encode("utf-8"))
            target.write(b':"' + b"0" * 64 + b'",')
        for number, (name, expected) in enumerate(inventory.items()):
            if number:
                target.write(b",")
            target.write(json.dumps(name, ensure_ascii=False).encode("utf-8"))
            target.write(b":" + json.dumps(expected).encode("ascii"))
        target.write(b"}\n")
        if padding:
            chunk = b" " * 1024**2
            while target.tell() <= LEGACY_LIMIT:
                target.write(chunk)
        target.write(tail)
    binding["originals_sha256"] = digest(path)
    write_json(root / "state.json", state)
    return path, inventory


def assert_exhausted(result, learner, *, history_size=1):
    assert result["stop_reason"] == "sampling_retries_exhausted", result.get("error")
    assert result["learner_updates"] == result["eligible_transitions"] == 0
    assert result["latest_learner"] == learner
    assert result["resources_released"]
    row = result["rounds"][0]
    assert row["sampling_attempt"] == history_size
    assert len(row["sampling_history"]) == history_size
    assert sum(item["kind"] == "sampling_retry" for item in result["recoveries"]) == history_size


def assert_native_index(binding):
    root = Path(binding["directory"])
    descriptor_path = root.with_name(root.name + "-originals.json")
    assert binding["originals_file"] == str(descriptor_path)
    assert binding["originals_sha256"] == digest(descriptor_path)
    descriptor = json.loads(descriptor_path.read_bytes())
    assert set(descriptor) == {"kind", "path", "sha256"}
    assert descriptor["kind"] == "sampling-source-index-v1"
    index = root.with_name(root.name + "-sources.sqlite3")
    assert descriptor["path"] == str(index.resolve())
    assert descriptor["sha256"] == digest(index)
    expected = originals(root)
    connection = sqlite3.connect(index.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        observed = dict(connection.execute("SELECT path, sha256 FROM assets ORDER BY path"))
    finally:
        connection.close()
    assert observed == expected
    assert len(expected) > 256
    assert all(Path(name).is_relative_to(root.resolve()) for name in observed)
    assert str(index.resolve()) not in observed and str(descriptor_path.resolve()) not in observed
    return descriptor_path, index, expected


def test_bound_legacy_inventory_above_64_mib_resumes_without_rebinding(
    tmp_path, seeded_loop, record_property
):
    request, initial = failed_archive(tmp_path, seeded_loop)
    path, expected = rebind_legacy(request.output_dir, padding=True)
    original_hash = digest(path)
    original_size = path.stat().st_size
    assert original_size > LEGACY_LIMIT
    record_property("legacy_inventory_bytes", original_size)
    record_property("real_inventory_entries", len(expected))
    state = request.output_dir / "state.json"
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir, digest(state)), learning_environment=backend
    ).summary["learning_loop"]
    assert_exhausted(resumed, initial["latest_learner"])
    assert not backend.leases and backend.closed
    assert digest(path) == original_hash and path.stat().st_size == original_size
    assert originals(Path(resumed["rounds"][0]["sampling_history"][0]["directory"])) == expected
    assert resumed["rounds"][0]["sampling_history"][0]["originals_sha256"] == original_hash


@pytest.mark.parametrize("bom", [False, True], ids=["utf8", "utf8-bom"])
def test_bound_legacy_inventory_preserves_last_duplicate_key(tmp_path, seeded_loop, bom):
    request, initial = failed_archive(tmp_path, seeded_loop)
    path, expected = rebind_legacy(request.output_dir, duplicate=True, bom=bom)
    before = digest(path)
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir, digest(request.output_dir / "state.json")),
        learning_environment=backend,
    ).summary["learning_loop"]
    assert_exhausted(resumed, initial["latest_learner"])
    assert not backend.leases and backend.closed
    assert digest(path) == before
    assert originals(Path(resumed["rounds"][0]["sampling_history"][0]["directory"])) == expected


def test_bound_legacy_inventory_rejects_invalid_trailing_json(tmp_path, seeded_loop):
    request, _ = failed_archive(tmp_path, seeded_loop)
    path, _ = rebind_legacy(request.output_dir, tail=b'{"unverified":true}')
    state = request.output_dir / "state.json"
    before, inventory_before = state.read_bytes(), digest(path)
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises(ValueError):
        run_experiment(
            LearningContinue(request.output_dir, digest(state)), learning_environment=backend
        )
    assert state.read_bytes() == before and digest(path) == inventory_before
    assert not backend.leases and backend.closed


def test_failed_sampling_publishes_native_index_and_reuses_it_on_continue(
    tmp_path, seeded_loop, monkeypatch, record_property
):
    request, initial = failed_archive(tmp_path, seeded_loop)
    _, binding = archived_binding(request.output_dir)
    descriptor, index, expected = assert_native_index(binding)
    retained = {path: digest(path) for path in (descriptor, index)}
    record_property("real_inventory_entries", len(expected))
    record_property("native_index_bytes", index.stat().st_size)
    replacements = []
    replace = os.replace

    def observe(source, target, *args, **kwargs):
        if Path(target).resolve() in retained:
            replacements.append(Path(target).resolve())
        return replace(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", observe)
    for _ in range(2):
        backend = SharedBackend(seeded_loop[0])
        result = run_experiment(
            LearningContinue(request.output_dir, digest(request.output_dir / "state.json")),
            learning_environment=backend,
        ).summary["learning_loop"]
        assert_exhausted(result, initial["latest_learner"])
        assert not backend.leases and backend.closed
    assert replacements == []
    assert all(digest(path) == expected_hash for path, expected_hash in retained.items())
    assert originals(Path(binding["directory"])) == expected


def test_new_index_survives_explicit_stop_and_does_not_reset_retry_budget(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})
    stop = request.output_dir / "stop.request"

    class StopRetry(FailedOriginals):
        def sampling(self, identity):
            if self.leases:
                self.stop_sampling_file = stop
                return SharedBackend.sampling(self, identity)
            return super().sampling(identity)

    backend = StopRetry(seeded_loop[0])
    stopped = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert stopped["stop_reason"] == "stop_requested", stopped.get("error")
    assert len(backend.leases) == 2 and backend.closed and stopped["resources_released"]
    assert all(lease.closed for lease in backend.leases)
    assert stopped["learner_updates"] == stopped["eligible_transitions"] == 0
    binding = stopped["rounds"][0]["sampling_history"][0]
    descriptor, index, expected = assert_native_index(binding)
    retained = {path: digest(path) for path in (descriptor, index)}
    stop.unlink()
    followup = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir, digest(request.output_dir / "state.json")),
        learning_environment=followup,
    ).summary["learning_loop"]
    assert_exhausted(resumed, stopped["latest_learner"])
    assert not followup.leases and followup.closed
    assert originals(Path(binding["directory"])) == expected
    assert all(digest(path) == expected_hash for path, expected_hash in retained.items())


@pytest.mark.parametrize(
    "changed",
    ["descriptor", "index", "original", "original_added", "original_removed", "binding_path"],
)
def test_bound_native_archive_rejects_changed_binding_before_sampling(
    tmp_path, seeded_loop, changed
):
    request, initial = failed_archive(tmp_path, seeded_loop)
    state, binding = archived_binding(request.output_dir)
    descriptor, index, _ = assert_native_index(binding)
    root = Path(binding["directory"])
    if changed == "descriptor":
        descriptor.write_bytes(descriptor.read_bytes() + b"\n")
    elif changed == "index":
        with index.open("ab") as target:
            target.write(b"changed source index")
    elif changed == "original":
        (root / "failure-note.bin").write_bytes(b"changed original")
    elif changed == "original_added":
        (root / "unlisted-after-sealing.bin").write_bytes(b"new original")
    elif changed == "original_removed":
        (root / "failure-note.bin").unlink()
    else:
        alias = descriptor.with_name("different-originals.json")
        alias.write_bytes(descriptor.read_bytes())
        binding["originals_file"] = str(alias)
        write_json(request.output_dir / "state.json", state)
    state_path = request.output_dir / "state.json"
    before = state_path.read_bytes()
    artifacts = {path: digest(path) for path in (descriptor, index)}
    learner = Path(initial["latest_learner"]["directory"]) / "policy.json"
    learner_hash = digest(learner)
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises(ValueError, match="originals|inventory|index|source"):
        run_experiment(
            LearningContinue(request.output_dir, digest(state_path)), learning_environment=backend
        )
    assert state_path.read_bytes() == before and digest(learner) == learner_hash
    assert all(digest(path) == expected for path, expected in artifacts.items())
    assert not backend.leases and backend.closed


def interrupt_archive(request, source, boundary):
    driver = Path(__file__).with_name("interrupted_learning_inventory.py")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(driver.parent), *sys.path])
    process = subprocess.run(
        [
            sys.executable,
            str(driver),
            str(request.config_file),
            str(request.output_dir),
            str(source),
            boundary,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert process.returncode == 73, process.stdout + process.stderr
    evidence = json.loads((request.output_dir / "inventory-interruption.json").read_bytes())
    assert evidence["boundary"] == boundary
    return evidence


@pytest.mark.parametrize(
    "boundary", ["before_index", "after_index", "after_descriptor", "before_parent"]
)
def test_unacknowledged_archive_reenters_without_rewriting_or_spending_retry_twice(
    tmp_path, seeded_loop, monkeypatch, boundary
):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})
    interruption = interrupt_archive(request, seeded_loop[0], boundary)
    root = request.output_dir / "round-000/learning"
    expected = originals(root)
    state = json.loads((request.output_dir / "state.json").read_bytes())
    assert state["rounds"][0].get("sampling_attempt", 0) == 0
    assert not state["rounds"][0].get("sampling_history")
    index = root.with_name("learning-sources.sqlite3")
    descriptor = root.with_name("learning-originals.json")
    retained = {path: digest(path) for path in (index, descriptor) if path.exists()}
    if boundary == "before_index":
        pending = Path(interruption["source"])
        retained[pending] = digest(pending)
        assert not index.exists() and not descriptor.exists()
    else:
        assert index.exists()
        assert descriptor.exists() is (boundary != "after_index")
    replacements = []
    replace = os.replace

    def observe(source, target, *args, **kwargs):
        if Path(target).resolve() in retained:
            replacements.append(Path(target).resolve())
        return replace(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", observe)
    backend = FailedOriginals(seeded_loop[0], extra_files=0)
    result = run_experiment(
        LearningContinue(request.output_dir, digest(request.output_dir / "state.json")),
        learning_environment=backend,
    ).summary["learning_loop"]
    assert_exhausted(result, state["latest_learner"])
    assert len(backend.leases) == 1 and backend.closed
    assert all(lease.closed for lease in backend.leases)
    assert originals(root) == expected
    assert replacements == []
    assert all(digest(path) == expected_hash for path, expected_hash in retained.items())
    assert_native_index(result["rounds"][0]["sampling_history"][0])
    followup = SharedBackend(seeded_loop[0])
    again = run_experiment(
        LearningContinue(request.output_dir, digest(request.output_dir / "state.json")),
        learning_environment=followup,
    ).summary["learning_loop"]
    assert_exhausted(again, state["latest_learner"])
    assert not followup.leases and followup.closed


@pytest.mark.parametrize("boundary", ["after_index", "after_descriptor"])
@pytest.mark.parametrize("changed", ["added", "modified", "removed"])
def test_unacknowledged_archive_cannot_reseal_changed_originals(
    tmp_path, seeded_loop, boundary, changed
):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})
    interrupt_archive(request, seeded_loop[0], boundary)
    root = request.output_dir / "round-000/learning"
    note = root / "failure-note.bin"
    if changed == "added":
        (root / "unlisted-after-interruption.bin").write_bytes(b"late original")
    elif changed == "modified":
        note.write_bytes(b"different closed receiver")
    else:
        note.unlink()
    expected = originals(root)
    state_path = request.output_dir / "state.json"
    state = json.loads(state_path.read_bytes())
    index, descriptor = (
        root.with_name(name) for name in ("learning-sources.sqlite3", "learning-originals.json")
    )
    retained = {path: digest(path) for path in (index, descriptor) if path.exists()}
    backend = SharedBackend(seeded_loop[0])
    result = run_experiment(
        LearningContinue(request.output_dir, digest(state_path)), learning_environment=backend
    ).summary["learning_loop"]
    # Existing restore first acknowledges continuation, then the retry operation
    # reports an interface failure. Progress and bindings must still stay put.
    assert result["stop_reason"] == "interface_error", result.get("error")
    assert any(word in result["error"].lower() for word in ("originals", "inventory", "source"))
    assert result["latest_learner"] == state["latest_learner"]
    assert result["learner_updates"] == result["eligible_transitions"] == 0
    assert result["rounds"][0].get("sampling_attempt", 0) == 0
    assert not result["rounds"][0].get("sampling_history")
    assert not (request.output_dir / "round-000/learning-001").exists()
    assert not backend.leases and backend.closed and result["resources_released"]
    assert all(digest(path) == expected_hash for path, expected_hash in retained.items())
    assert originals(root) == expected
    if boundary == "after_index":
        assert not descriptor.exists()
