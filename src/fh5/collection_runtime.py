"""Passive stream runner; encoding and disk work stay in the archive worker."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from fh5.collection import CollectionControl, CollectionEnvironment, CollectionRun
from fh5.collection_state import CollectionState
from fh5.collection_store import CollectionArchive, WriteFile, atomic_json, encode, write_file
from fh5.demonstrations import _profile

if TYPE_CHECKING:
    from fh5.experiment import RunResult


def control_collection(request: CollectionControl) -> RunResult:
    import json

    from fh5.experiment import RunResult

    root = request.recording_dir
    with (root / "session.json").open("rb") as stream:
        payload = stream.read(1024**2 + 1)
    if len(payload) > 1024**2:
        raise ValueError("Collection session metadata exceeds bounded limit")
    session = json.loads(payload)
    if session.get("kind") != "continuous-numeric-collection-v1":
        raise ValueError("Not a continuous collection session")
    final = root / "final.json"
    status = final if final.exists() else root / "status.json"
    if status.is_file():
        with status.open("rb") as stream:
            data = stream.read(4 * 1024**2 + 1)
        if len(data) > 4 * 1024**2:
            raise ValueError("Collection status exceeds bounded limit")
        value = json.loads(data)
    else:
        value = {"state": "starting", "commands_sent": False}
    if request.stop and not final.exists():
        (root / "stop.request").touch(exist_ok=True)
    value.update(
        stop_requested=(root / "stop.request").exists(),
        process_liveness="not_checked; status file alone does not prove a running process",
        final_status_present=final.exists(),
    )
    return RunResult({"source_kind": "collection_control"}, [], [], {"collection": value}, status)


def collect(
    request: CollectionRun, environment: CollectionEnvironment, write: WriteFile | None = None
) -> RunResult:
    from fh5.collection_review import collection_result

    if environment.source_kind not in ("synthetic", "live_passive"):
        raise ValueError("Unsupported passive collection environment")
    profile = _profile(request.input_profile.read_bytes())
    if profile["calibration"]["status"] != "verified":
        raise ValueError("Continuous collection requires verified input calibration")
    cfg, root = request.config, request.output_dir
    session = {
        "version": 1,
        "kind": "continuous-numeric-collection-v1",
        "session_id": uuid.uuid4().hex,
        "configuration": {**asdict(cfg), "pixels": cfg.pixels.metadata()},
        "profile": profile,
        "software_snapshot": request.software_snapshot,
        "input_conditions": request.input_conditions,
        "source_kind": environment.source_kind,
        "commands_sent": False,
        "training_eligible": False,
    }
    payload = encode(session)
    if len(payload) > 1024**2:
        raise ValueError("Collection session metadata exceeds bounded limit")
    root.mkdir(parents=True, exist_ok=False)
    write_file(root / "session.json", payload)
    binding = hashlib.sha256(payload).hexdigest()
    archive = CollectionArchive(root, cfg, binding, write or write_file)
    state = CollectionState(cfg, profile)
    started, sequence, reason = time.monotonic(), 0, "source_end"
    error: str | None = None
    try:
        while time.monotonic() - started < cfg.seconds:
            if (root / "stop.request").exists():
                reason = "requested_stop"
                break
            if archive.error:
                reason = "archive_failure"
                break
            value = environment.read(1 / cfg.poll_hz)
            if value is None:
                break
            row, frames = state.sample(value, sequence)
            archive.submit(row, frames, state.progress())
            sequence += 1
            if value.stop_requested or value.fault:
                reason = "user_stop" if value.stop_requested else "source_fault"
                error = value.fault
                break
            if sequence >= int(cfg.seconds * cfg.poll_hz * 2):
                reason = "source_rate_limit"
                break
        else:
            reason = "time_limit"
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception as failure:
        reason, error = "source_error", f"{type(failure).__name__}: {failure}"
    finally:
        try:
            released = environment.close()
        except Exception as failure:
            released = {"resources_released": False, "error": str(failure)}
        result: dict[str, Any] = archive.close()
    if result["archive_error"]:
        reason = "archive_failure"
    result.update(
        version=1,
        session_sha256=binding,
        stop_reason=reason,
        error=error,
        state="stopped",
        environment=released,
        training_eligible=False,
        complete=result["unsealed_rows"] == 0
        and not result["archive_error"]
        and result["archive_released"]
        and released.get("resources_released", False)
        and sequence > 0
        and error is None,
    )
    atomic_json(root / "final.json", result)
    atomic_json(root / "status.json", result)
    return collection_result(root / "report.html", result)
