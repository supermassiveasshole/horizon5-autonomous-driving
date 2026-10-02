"""Scalar control/status contracts; detailed diagnostics stay in source files."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fh5.artifact_io import VerifiedFile
from fh5.collection_file_io import control_file_reader
from fh5.replay_document import read_document_paths, read_stream_paths

SESSION_FIELDS: set[tuple[str, ...]] = {
    ("kind",),
    ("software_snapshot", "verified"),
    ("software_snapshot", "manifest_sha256"),
}
STATUS_FIELDS: set[tuple[str, ...]] = {
    (key,)
    for key in (
        "version",
        "state",
        "commands_sent",
        "complete",
        "session_sha256",
        "heartbeat_ns",
        "last_poll_ns",
        "latest_image_source_ns",
        "seen_rows",
        "written_rows",
        "dropped_rows",
        "sealed_blocks",
        "disk_bytes",
        "peak_pending_bytes",
        "pending_bytes",
        "free_bytes",
        "complete_fraction",
        "archive_error",
        "archive_released",
        "unsealed_rows",
        "stop_reason",
        "error",
        "active_driving_seconds",
        "segments",
        "confirmed_attempts",
        "attempts_status",
        "training_eligible",
    )
} | {("environment", "resources_released"), ("environment", "resource_lease_released")}
PROCESS_FIELDS: set[tuple[str, ...]] = {
    (key,)
    for key in (
        "pid",
        "birth",
        "token",
        "manifest_sha256",
        "state",
        "commands_sent",
        "software_snapshot_verified",
        "complete",
        "exit_code",
        "error",
    )
}


def read_control_status(path: Path, fields: set[tuple[str, ...]]) -> dict[str, Any]:
    """Atomic publication may change the pathname while this old handle is read."""
    with control_file_reader(path) as frozen:
        with io.TextIOWrapper(frozen, encoding="utf-8-sig") as stream:
            return read_stream_paths(stream, fields)


def read_collection_session(source: Path | VerifiedFile) -> dict[str, Any]:
    result = (
        read_document_paths(source, SESSION_FIELDS)
        if isinstance(source, VerifiedFile)
        else read_control_status(source, SESSION_FIELDS)
    )
    if result.get("kind") != "continuous-numeric-collection-v1":
        raise ValueError("Not a continuous collection session")
    return result


def control_source(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "role": "original diagnostics; mutable status paths may contain a later publication",
    }
