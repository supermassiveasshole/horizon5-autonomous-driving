"""Real original inventories stream through archival and legacy continuation."""

import hashlib
import json
import sqlite3
import time
import tracemalloc
from contextlib import contextmanager

from fh5.experiment import run_experiment
from fh5.learning.loop.runner import LearningContinue
from tests.evaluation.test_candidate_store import candidates as candidates
from tests.learning.loop.test_learning_loop import SharedBackend, loop_request
from tests.learning.loop.test_learning_loop import seeded_loop as seeded_loop
from tests.support.recovery_inventory_cases import digest

HISTORICAL_LIMIT = 64 * 1024**2
DIAGNOSTIC_FILES = 6000
PATH_LEVELS = 16
CHINESE_PER_COMPONENT = 120


class ManyRootOriginals(SharedBackend):
    """Create actual external files before sealing, outside the per-attempt inventory."""

    def sampling(self, identity):
        lease = super().sampling(identity)
        lease.fail_at = 0
        finish = lease.finish

        def review(recording):
            if len(self.leases) == 1:
                folder = recording.parent.parent / "diagnostics"
                for level in range(PATH_LEVELS):
                    folder /= "采" * CHINESE_PER_COMPONENT + f"_{level:03d}"
                folder.mkdir(parents=True)
                for number in range(DIAGNOSTIC_FILES):
                    (folder / f"original-{number:06d}.bin").write_bytes(
                        number.to_bytes(16, "little")
                    )
            return finish(recording)

        lease.finish = review
        return lease


@contextmanager
def source_index(path):
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        yield connection
    finally:
        connection.close()


def json_entry(name, expected):
    # Match the old published representation: ensure_ascii=True, compact colon,
    # no added whitespace; object punctuation is counted by the caller.
    return (
        json.dumps(name, ensure_ascii=True).encode("ascii")
        + b":"
        + json.dumps(expected).encode("ascii")
    )


def inspect_originals(root, index):
    """Compare actual files one by one; never construct an expected inventory dict."""
    file_count = 0
    diagnostic_count = 0
    min_path_units = None
    max_path_units = 0
    legacy_bytes = 3  # Opening/closing braces plus the original terminal newline.
    resolved_root = root.resolve()
    diagnostic_root = (root / "diagnostics").resolve()
    with source_index(index) as connection:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            assert resolved.is_relative_to(resolved_root)
            name = str(resolved)
            row = connection.execute("SELECT sha256 FROM assets WHERE path = ?", (name,)).fetchone()
            assert row is not None, f"Original absent from archive: {name}"
            assert digest(path) == row[0], f"Original hash changed: {name}"
            legacy_bytes += len(json_entry(name, row[0])) + (1 if file_count else 0)
            units = len(name.encode("utf-16-le")) // 2
            min_path_units = units if min_path_units is None else min(min_path_units, units)
            max_path_units = max(max_path_units, units)
            file_count += 1
            if resolved.is_relative_to(diagnostic_root):
                diagnostic_count += 1
        assert connection.execute("SELECT count(*) FROM assets").fetchone()[0] == file_count
        # A small rolling digest makes the independently checked complete set
        # comparable across continuations without retaining thousands of paths.
        tree_digest = hashlib.sha256()
        for name, expected in connection.execute("SELECT path, sha256 FROM assets ORDER BY path"):
            tree_digest.update(name.encode("utf-8") + b"\0" + expected.encode("ascii") + b"\n")
    return {
        "entries": file_count,
        "diagnostic_entries": diagnostic_count,
        "legacy_json_bytes": legacy_bytes,
        "minimum_path_utf16_units": min_path_units,
        "maximum_path_utf16_units": max_path_units,
        "originals_sha256": tree_digest.hexdigest(),
    }


def measured_run(request, backend, label, record_property):
    # Seeded candidate construction has finished. Only this public invocation is
    # measured: independent index inspection/export below happens after stop().
    assert not tracemalloc.is_tracing()
    started = time.perf_counter()
    tracemalloc.start()
    try:
        result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        record_property(label + "_python_peak_bytes", peak)
        record_property(label + "_seconds", time.perf_counter() - started)
    assert backend.closed and result["resources_released"]
    assert all(lease.closed for lease in backend.leases)
    return result, peak


def assert_same_completed_failure(result, learner, binding):
    assert result["stop_reason"] == "sampling_retries_exhausted", result.get("error")
    assert result["latest_learner"] == learner
    assert result["learner_updates"] == result["eligible_transitions"] == 0
    assert result["rounds_completed"] == 0
    row = result["rounds"][0]
    assert row["sampling_attempt"] == 1
    assert row["sampling_history"] == [binding]
    assert sum(item["kind"] == "sampling_retry" for item in result["recoveries"]) == 1


def export_legacy_inventory(index, destination):
    """Write genuine path/hash entries incrementally, with no padding or materialized map."""
    with source_index(index) as connection, destination.open("wb") as target:
        target.write(b"{")
        for number, (name, expected) in enumerate(
            connection.execute("SELECT path, sha256 FROM assets ORDER BY path")
        ):
            if number:
                target.write(b",")
            target.write(json_entry(name, expected))
        target.write(b"}\n")


def test_real_original_entries_cross_64_mib_and_resume_as_index_then_legacy(
    tmp_path, seeded_loop, record_property
):
    request = loop_request(tmp_path, seeded_loop, rounds=1, sampling_retry={"max_retries": 1})
    root = request.output_dir / "round-000/learning"
    state_path = request.output_dir / "state.json"
    archive_path = root.with_name("learning-originals.json")
    index_path = root.with_name("learning-sources.sqlite3")
    backend = ManyRootOriginals(seeded_loop[0])
    initial, archive_peak = measured_run(request, backend, "native_archive", record_property)
    assert initial["stop_reason"] == "sampling_retries_exhausted", initial.get("error")
    assert len(backend.leases) == 2
    binding = initial["rounds"][0]["sampling_history"][0]
    assert_same_completed_failure(initial, initial["latest_learner"], binding)
    native_descriptor = archive_path.read_bytes()
    descriptor = json.loads(native_descriptor)
    index_hash = digest(index_path)
    assert descriptor == {
        "kind": "sampling-source-index-v1",
        "path": str(index_path.resolve()),
        "sha256": index_hash,
    }
    assert binding["originals_file"] == str(archive_path)
    assert binding["originals_sha256"] == digest(archive_path)
    assert binding["summary_sha256"] == digest(root / "summary.json")
    expected = inspect_originals(root, index_path)
    assert expected["diagnostic_entries"] == DIAGNOSTIC_FILES
    assert expected["entries"] > DIAGNOSTIC_FILES
    assert expected["legacy_json_bytes"] > HISTORICAL_LIMIT
    # Relational evidence of streaming: Python peak is smaller than the real
    # prior JSON payload; no new production admission threshold is introduced.
    assert archive_peak < expected["legacy_json_bytes"]
    for key, value in expected.items():
        record_property(key, value)
    record_property("native_index_bytes", index_path.stat().st_size)
    record_property("native_descriptor_bytes", len(native_descriptor))
    record_property("native_descriptor_sha256_before", digest(archive_path))
    record_property("native_index_sha256_before", index_hash)
    record_property("state_sha256_before_native_resume", digest(state_path))

    followup = SharedBackend(seeded_loop[0])
    continued, continue_peak = measured_run(
        LearningContinue(request.output_dir, digest(state_path)),
        followup,
        "native_resume",
        record_property,
    )
    assert_same_completed_failure(continued, initial["latest_learner"], binding)
    assert not followup.leases
    assert continue_peak < expected["legacy_json_bytes"]
    assert archive_path.read_bytes() == native_descriptor
    assert digest(index_path) == index_hash
    native_after = inspect_originals(root, index_path)
    assert native_after == expected
    record_property("native_descriptor_sha256_after", digest(archive_path))
    record_property("native_index_sha256_after_native_resume", digest(index_path))
    record_property("originals_sha256_after_native_resume", native_after["originals_sha256"])
    record_property("state_sha256_after_native_resume", digest(state_path))
    record_property("native_resume_stop_reason", continued["stop_reason"])
    record_property("native_resume_sampling_attempt", continued["rounds"][0]["sampling_attempt"])

    # Construct a correctly bound older-format fixture from the same verified
    # index. This intentional fixture conversion changes no original file.
    export_legacy_inventory(index_path, archive_path)
    assert archive_path.stat().st_size == expected["legacy_json_bytes"]
    legacy_hash = digest(archive_path)
    state = json.loads(state_path.read_bytes())
    legacy_binding = state["rounds"][0]["sampling_history"][0]
    legacy_binding["originals_sha256"] = legacy_hash
    state_path.write_text(json.dumps(state), encoding="utf-8")
    record_property("legacy_inventory_sha256_before", legacy_hash)
    record_property("state_sha256_before_legacy_resume", digest(state_path))
    legacy_backend = SharedBackend(seeded_loop[0])
    legacy, legacy_peak = measured_run(
        LearningContinue(request.output_dir, digest(state_path)),
        legacy_backend,
        "legacy_resume",
        record_property,
    )
    assert_same_completed_failure(legacy, initial["latest_learner"], legacy_binding)
    assert not legacy_backend.leases
    assert legacy_peak < expected["legacy_json_bytes"]
    assert digest(archive_path) == legacy_hash
    assert archive_path.stat().st_size == expected["legacy_json_bytes"]
    assert digest(index_path) == index_hash
    legacy_after = inspect_originals(root, index_path)
    assert legacy_after == expected
    record_property("legacy_inventory_sha256_after", digest(archive_path))
    record_property("native_index_sha256_after_legacy_resume", digest(index_path))
    record_property("originals_sha256_after_legacy_resume", legacy_after["originals_sha256"])
    record_property("state_sha256_after_legacy_resume", digest(state_path))
    record_property("legacy_resume_stop_reason", legacy["stop_reason"])
    record_property("legacy_resume_sampling_attempt", legacy["rounds"][0]["sampling_attempt"])
