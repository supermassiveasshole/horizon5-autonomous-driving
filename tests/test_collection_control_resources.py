"""Public control-state behavior with OS identity and filesystem boundary fixtures.

No worker, game, or model is started; publication uses actual local files.
"""

import io
import json
import os
import shutil
import tracemalloc

import pytest

from fh5.artifact_io import sha256_file
from fh5.collection import CollectionControl
from fh5.collection_store import atomic_control_json, encode
from fh5.experiment import run_experiment

OLD_LIMITS = [
    ("recording/session.json", 1024**2),
    ("recording/status.json", 4 * 1024**2),
    ("recording/final.json", 4 * 1024**2),
    ("process.json", 16 * 1024),
    ("worker-state.json", 64 * 1024),
    ("launch-failed.json", 64 * 1024),
]


def write_document(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(value))


def add_padding(path, minimum):
    with path.open("ab") as stream:
        while stream.tell() <= minimum:
            stream.write(b" " * 8192)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "collector"
    write_document(
        root / "frozen.json",
        {
            "version": 1,
            "kind": "frozen-passive-collection-v1",
            "source": "synthetic",
            "files": {},
            "commands_sent": False,
        },
    )
    digest = sha256_file(root / "frozen.json")
    write_document(
        root / "recording/session.json",
        {
            "version": 1,
            "kind": "continuous-numeric-collection-v1",
            "session_id": "synthetic-control-fixture",
            "source_kind": "synthetic",
            "commands_sent": False,
            "software_snapshot": {"manifest_sha256": digest, "verified": True},
        },
    )
    write_document(
        root / "recording/status.json",
        {
            "version": 1,
            "state": "recording",
            "session_sha256": sha256_file(root / "recording/session.json"),
            "heartbeat_ns": 10_000_000_000,
            "last_poll_ns": 9_999_000_000,
            "latest_image_source_ns": 9_990_000_000,
            "seen_rows": 12,
            "written_rows": 10,
            "dropped_rows": 0,
            "sealed_blocks": 2,
            "pending_bytes": 48,
            "archive_error": None,
            "commands_sent": False,
            "complete": False,
        },
    )
    write_document(
        root / "process.json",
        {
            "pid": 4101,
            "birth": "launcher-birth",
            "token": "launch-token",
            "manifest_sha256": digest,
            "state": "launched",
            "commands_sent": False,
        },
    )
    write_document(
        root / "worker-state.json",
        {
            "pid": 4102,
            "birth": "worker-birth",
            "token": "launch-token",
            "manifest_sha256": digest,
            "state": "recording",
            "commands_sent": False,
            "software_snapshot_verified": True,
        },
    )
    # These are deterministic host identities, never actual PIDs to launch/stop.
    identities = {
        4101: {"state": "running", "birth": "launcher-birth"},
        4102: {"state": "running", "birth": "worker-birth"},
    }
    monkeypatch.setattr(
        "fh5.collection_process.process_identity", lambda pid: dict(identities[pid])
    )
    return root


def add_optional_file(bundle, relative):
    target = bundle / relative
    if relative == "recording/final.json":
        value = json.loads((bundle / "recording/status.json").read_bytes())
        value.update(state="stopped", complete=True, stop_reason="source_end")
        write_document(target, value)
    elif relative == "launch-failed.json":
        write_document(target, {"error": "synthetic diagnostic, retained for inspection"})
    return target


def control(root, *, stop=False):
    return run_experiment(CollectionControl(root, stop=stop)).summary["collection"]


@pytest.mark.parametrize("relative,old_limit", OLD_LIMITS)
def test_valid_control_file_crosses_previous_size_limit_without_changing_evidence(
    bundle, relative, old_limit
):
    target = add_optional_file(bundle, relative)
    baseline = control(bundle)
    add_padding(target, old_limit)
    assert target.stat().st_size > old_limit
    if relative == "recording/session.json":
        # Preserve the whole-byte session binding, including legal whitespace.
        status_path = bundle / "recording/status.json"
        status = json.loads(status_path.read_bytes())
        status["session_sha256"] = sha256_file(target)
        write_document(status_path, status)
        baseline["session_sha256"] = status["session_sha256"]
    before = {path: sha256_file(path) for path in bundle.rglob("*.json")}
    observed = control(bundle, stop=True)
    baseline["stop_requested"] = True
    assert observed == baseline
    assert (bundle / "stop.request").is_file()
    assert all(sha256_file(path) == digest for path, digest in before.items())


def test_large_recording_status_does_not_prevent_direct_stop_request(bundle):
    target = bundle / "recording/status.json"
    add_padding(target, 4 * 1024**2)
    status = control(bundle / "recording", stop=True)
    assert status["stop_requested"] is True
    assert status["seen_rows"] == 12 and status["pending_bytes"] == 48
    assert "not_checked" in status["process_liveness"]
    assert (bundle / "recording/stop.request").is_file()


@pytest.mark.parametrize("relative", ["recording/status.json", "worker-state.json"])
def test_unselected_diagnostic_array_is_validated_without_retaining_it(bundle, relative):
    target = bundle / relative
    original = target.read_bytes().rstrip()
    with target.open("wb") as stream:
        stream.write(original[:-1] + b',"optional_diagnostics":[')
        row = b'{"event":42,"detail":"' + b"x" * 800 + b'"}'
        for index in range(6000):
            if index:
                stream.write(b",")
            stream.write(row)
        stream.write(b"]}\n")
    assert target.stat().st_size > 4 * 1024**2
    tracemalloc.start()
    try:
        observed = control(bundle)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    source_status = observed if relative.startswith("recording/") else observed["worker-state"]
    assert "optional_diagnostics" not in source_status
    assert observed["seen_rows"] == 12 and observed["pid"] == 4102
    assert observed["worker_identity_matches"] is True
    assert observed["software_snapshot_verified"] is True
    assert peak < target.stat().st_size


@pytest.mark.parametrize("relative,old_limit", OLD_LIMITS)
def test_malformed_tail_in_each_control_document_is_rejected(bundle, relative, old_limit):
    target = add_optional_file(bundle, relative)
    with target.open("ab") as stream:
        stream.write(b"\n{}")
    assert target.stat().st_size < old_limit
    with pytest.raises(ValueError, match="JSON|document|Malformed|Extra data|Expecting"):
        control(bundle)


def test_large_discarded_array_still_rejects_malformed_tail(bundle):
    target = bundle / "recording/status.json"
    original = target.read_bytes().rstrip()
    with target.open("wb") as stream:
        stream.write(original[:-1] + b',"optional_diagnostics":[')
        for _ in range(4 * 128 + 1):
            stream.write(b" " * 8192)
        stream.write(b"0,]}\n")
    assert target.stat().st_size > 4 * 1024**2
    # An old byte-limit exception is not evidence that the tail was validated.
    with pytest.raises(ValueError, match="JSON|document|Malformed|Extra data|Expecting"):
        control(bundle)


@pytest.mark.parametrize("field", ["token", "manifest_sha256"])
def test_worker_identity_mismatch_remains_unknown_and_unverified(bundle, field):
    path = bundle / "worker-state.json"
    worker = json.loads(path.read_bytes())
    worker[field] = "other-token" if field == "token" else "0" * 64
    write_document(path, worker)
    status = control(bundle)
    assert status["worker_identity_matches"] is False
    assert status["software_snapshot_verified"] is False
    assert status["process_liveness"] == "unknown"
    assert status["launcher_liveness"] == "running"
    assert status["pid"] == status["launcher_pid"] == 4101
    assert status["abnormal_exit"] is False


def test_worker_birth_mismatch_is_exited_not_a_reused_live_process(bundle):
    path = bundle / "worker-state.json"
    worker = json.loads(path.read_bytes())
    worker["birth"] = "prior-process-birth"
    write_document(path, worker)
    status = control(bundle)
    assert status["worker_identity_matches"] is True
    assert status["pid"] == 4102 and status["process_role"] == "worker"
    assert status["process_liveness"] == "exited"
    assert status["abnormal_exit"] is True
    assert status["state"] == "interrupted_or_start_failed"
    assert status["complete"] is False


@pytest.mark.parametrize(
    "relative,long_path",
    [
        ("recording/status.json", False),
        ("process.json", False),
        ("worker-state.json", False),
        ("recording/status.json", True),
    ],
)
def test_dynamic_document_replacement_uses_one_consistent_opened_snapshot(
    bundle, monkeypatch, relative, long_path
):
    if long_path:
        nested = bundle.parent
        for index in range(3):
            nested = nested / (f"采集资料_{index}_" + "x" * 48)
        bundle = shutil.copytree(bundle, nested / "collector")
        assert len(str(bundle)) > 260
    target = bundle / relative
    baseline = control(bundle)
    newer = json.loads(target.read_bytes())
    if relative == "recording/status.json":
        newer.update(seen_rows=999, pending_bytes=4096)
    elif relative == "process.json":
        newer.update(token="next-launch", birth="next-launcher-birth")
    else:
        newer.update(token="next-worker", birth="next-worker-birth")
    opening = io.open
    old_identity = target.stat()
    opened = []
    replaced = []

    class ReplaceDuringRead:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def replace(self):
            if not replaced:
                assert not self.stream.closed
                # Exercise the project's real filesystem publisher while the
                # reader still owns the prior OS handle. The producer and reader
                # must support one another's Windows sharing semantics.
                atomic_control_json(target, newer)
                replaced.append(True)

        def read(self, *args):
            value = self.stream.read(*args)
            self.replace()
            return value

        def readinto(self, *args):
            value = self.stream.readinto(*args)
            self.replace()
            return value

        def read1(self, *args):
            value = self.stream.read1(*args)
            self.replace()
            return value

    def replace_during_read(file, mode="r", *args, **kwargs):
        stream = opening(file, mode, *args, **kwargs)
        identity = os.fstat(stream.fileno())
        is_source = (identity.st_dev, identity.st_ino) == (
            old_identity.st_dev,
            old_identity.st_ino,
        )
        if is_source and mode in ("r", "rb"):
            opened.append(mode)
            return ReplaceDuringRead(stream)
        return stream

    # io.open also covers os.fdopen of a Windows shared-delete OS handle.
    monkeypatch.setattr(io, "open", replace_during_read)
    observed = control(bundle)
    assert len(opened) == 1
    assert replaced == [True]
    assert observed == baseline
    with opening(target, "rb") as stream:
        assert json.load(stream) == newer


@pytest.mark.parametrize(
    "relative,field,value",
    [("recording/status.json", "seen_rows", []), ("worker-state.json", "pid", {})],
)
def test_selected_scalar_container_is_rejected_before_control_output(
    bundle, relative, field, value
):
    path = bundle / relative
    document = json.loads(path.read_bytes())
    document[field] = value
    write_document(path, document)
    with pytest.raises(ValueError):
        control(bundle)


def test_last_duplicate_scalar_replaces_prior_invalid_container(bundle):
    path = bundle / "recording/status.json"
    valid = path.read_bytes().strip()
    path.write_bytes(b'{"seen_rows":[],' + valid[1:] + b"\n")
    status = control(bundle)
    assert status["seen_rows"] == 12 and status["software_snapshot_verified"] is True
