"""Candidate retention grows through SQLite rows, not application size ceilings."""

import hashlib
import json
import shutil
import sqlite3
import threading
import tracemalloc
from pathlib import Path

import pytest
from test_candidate_store import candidates as candidates
from test_candidate_store import retained_store as retained_store

from fh5.candidate_store import CandidateHistory, CandidateRollback
from fh5.experiment import run_experiment


@pytest.mark.parametrize("growth", ["events", "database"])
def test_growing_history_can_be_read_and_receive_a_verified_rollback(
    tmp_path, retained_store, candidates, growth
):
    store = tmp_path / "versions"
    shutil.copytree(retained_store, store)
    # Construct a self-consistent external history referencing the actual retained
    # learners/evaluation; no learner or internal validator is replaced.
    with sqlite3.connect(store / "state.sqlite") as db:
        original = json.loads(db.execute("SELECT payload FROM events").fetchone()[0])
        if growth == "events":
            revision = db.execute("SELECT revision FROM events").fetchone()[0]
            count = 1001
            for number in range(2, count + 1):
                event = dict(original, parent=revision)
                raw = json.dumps(event).encode()
                revision = hashlib.sha256(raw).hexdigest()
                db.execute("INSERT INTO events VALUES (?, ?, ?)", (number, revision, raw))
        else:
            count = 1
            raw = b" " * (33 * 1024**2) + json.dumps(original).encode()
            revision = hashlib.sha256(raw).hexdigest()
            db.execute("UPDATE events SET revision=?, payload=?", (revision, raw))
    before = run_experiment(CandidateHistory(store)).summary["candidate_store"]
    assert before["revision"] == revision
    assert len(before["history"]) == count
    assert before["default"] == original["default"]
    rolled = run_experiment(
        CandidateRollback(store, revision, revision, "Keep retained default", candidates[3])
    ).summary["candidate_store"]
    assert rolled["parent"] == revision
    assert rolled["default"] == original["default"]
    after = run_experiment(CandidateHistory(store)).summary["candidate_store"]
    assert after["revision"] == rolled["revision"]
    assert len(after["history"]) == count + 1


def test_history_paging_and_latest_only_keep_memory_independent_of_total_payload(
    tmp_path, retained_store
):
    store = tmp_path / "versions"
    shutil.copytree(retained_store, store)
    with sqlite3.connect(store / "state.sqlite") as db:
        first_revision, raw = db.execute("SELECT revision, payload FROM events").fetchone()
        original = json.loads(raw)
        revision = first_revision
        for number in range(2, 258):
            # A valid large reason makes accidentally materializing the entire
            # history observable without needing thousands of model updates.
            raw = json.dumps(dict(original, parent=revision, reasons=["evidence " * 8192])).encode()
            revision = hashlib.sha256(raw).hexdigest()
            db.execute("INSERT INTO events VALUES (?, ?, ?)", (number, revision, raw))
        payload_bytes = db.execute("SELECT sum(length(payload)) FROM events").fetchone()[0]
    del raw
    tracemalloc.start()
    try:
        latest = run_experiment(CandidateHistory(store, limit=0)).summary["candidate_store"]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert latest["revision"] == revision
    assert latest["history"] == []
    assert latest["history_count"] == 257
    assert peak < payload_bytes
    page = run_experiment(CandidateHistory(store, limit=1)).summary["candidate_store"]
    assert page["revision"] == revision
    assert [row["revision"] for row in page["history"]] == [first_revision]
    assert page["next_sequence"] == 1 and not page["history_complete"]
    tail = run_experiment(CandidateHistory(store, after_sequence=256, limit=1)).summary[
        "candidate_store"
    ]
    assert [row["revision"] for row in tail["history"]] == [revision]
    assert tail["history_complete"] and tail["next_sequence"] == 257
    # A page cannot hide corruption outside the requested range.
    with sqlite3.connect(store / "state.sqlite") as db:
        db.execute("UPDATE events SET payload=? WHERE sequence=257", (b"{}",))
    with pytest.raises(ValueError, match="history changed"):
        run_experiment(CandidateHistory(store, limit=1))


def test_history_cli_can_request_current_roles_without_expanding_history(retained_store, capsys):
    from fh5.cli import main

    assert main(["candidate-history", "--store", str(retained_store), "--limit", "0"]) == 0
    latest = json.loads(capsys.readouterr().out)
    assert latest["history"] == [] and latest["history_count"] == 1
    assert (
        main(
            [
                "candidate-history",
                "--store",
                str(retained_store),
                "--after-sequence",
                "1",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    beyond = json.loads(capsys.readouterr().out)
    assert beyond["revision"] == latest["revision"]
    assert beyond["history"] == [] and beyond["history_complete"]


@pytest.mark.parametrize("arguments", [{"limit": -1}, {"limit": True}, {"after_sequence": -1}])
def test_history_page_rejects_invalid_query_before_accessing_store(tmp_path, arguments):
    with pytest.raises(ValueError, match="nonnegative integers"):
        run_experiment(CandidateHistory(tmp_path / "not-created", **arguments))


def test_slow_history_diagnostics_do_not_block_candidate_commit(
    tmp_path, retained_store, candidates, monkeypatch
):
    store = tmp_path / "versions"
    shutil.copytree(retained_store, store)
    with sqlite3.connect(store / "state.sqlite") as db:
        revision, raw = db.execute("SELECT revision, payload FROM events").fetchone()
        original = json.loads(raw)
        for number in (2, 3):
            raw = json.dumps(dict(original, parent=revision)).encode()
            revision = hashlib.sha256(raw).hexdigest()
            db.execute("INSERT INTO events VALUES (?, ?, ?)", (number, revision, raw))
    waiting, release = threading.Event(), threading.Event()
    results, errors = [], []
    original_open = Path.open

    def slow_attachment(path, *args, **kwargs):
        if (
            threading.current_thread().name == "diagnostic-reader"
            and path == store / original["comparison"]
            and not release.is_set()
        ):
            waiting.set()
            release.wait()
        return original_open(path, *args, **kwargs)

    def read_history():
        try:
            results.append(run_experiment(CandidateHistory(store, limit=0)))
        except Exception as error:
            errors.append(error)
        finally:
            waiting.set()

    monkeypatch.setattr(Path, "open", slow_attachment)
    reader = threading.Thread(target=read_history, name="diagnostic-reader")
    reader.start()
    try:
        waiting.wait()
        assert not errors
        rolled = run_experiment(
            CandidateRollback(store, revision, revision, "Concurrent retention", candidates[3])
        ).summary["candidate_store"]
    finally:
        release.set()
        reader.join()
    assert not errors
    assert rolled["parent"] == revision
    assert results[0].summary["candidate_store"]["revision"] == revision
    latest = run_experiment(CandidateHistory(store, limit=0)).summary["candidate_store"]
    assert latest["revision"] == rolled["revision"]


def test_unavailable_snapshot_reports_an_error_without_changing_the_store(
    tmp_path, monkeypatch, capsys
):
    from fh5.cli import main

    store = tmp_path / "versions"
    store.mkdir()
    database = store / "state.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE preserved(value TEXT)")
        connection.execute("INSERT INTO preserved VALUES ('unchanged')")
    original = database.read_bytes()
    connect = sqlite3.connect

    def unavailable(path, *args, **kwargs):
        if Path(path).name == "history.sqlite":
            raise sqlite3.OperationalError("Snapshot storage unavailable")
        return connect(path, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", unavailable)
    assert main(["candidate-history", "--store", str(store), "--limit", "0"]) == 2
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["status"] == "error"
    assert "Snapshot storage unavailable" in output.err
    assert database.read_bytes() == original
