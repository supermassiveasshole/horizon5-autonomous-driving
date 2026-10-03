"""Passive stream runner; encoding and disk work stay in the archive worker."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from fh5.collection import CollectionControl, CollectionEnvironment, CollectionRun
from fh5.collection_state import CollectionState
from fh5.collection_status import (
    STATUS_FIELDS,
    control_source,
    read_collection_session,
    read_control_status,
)
from fh5.collection_store import (
    CollectionArchive,
    WriteFile,
    atomic_control_json,
    collection_complete,
    encode,
    write_file,
)
from fh5.demonstrations import _profile
from fh5.presentation import optional_report

if TYPE_CHECKING:
    from fh5.experiment import RunResult


def control_collection(request: CollectionControl) -> RunResult:
    from fh5.experiment import RunResult

    root = request.recording_dir
    if (root / "frozen.json").is_file():
        from fh5.collection_process import control_bundle

        return control_bundle(request)
    read_collection_session(root / "session.json")
    final = root / "final.json"
    status = final if final.exists() else root / "status.json"
    if status.is_file():
        value = read_control_status(status, STATUS_FIELDS)
    else:
        value = {"state": "starting", "commands_sent": False}
    if request.stop and not final.exists():
        (root / "stop.request").touch(exist_ok=True)
    value.update(
        stop_requested=(root / "stop.request").exists(),
        process_liveness="not_checked; status file alone does not prove a running process",
        final_status_present=final.exists(),
        source_documents={
            "session": control_source(root / "session.json"),
            "status": control_source(status),
        },
    )
    return RunResult({"source_kind": "collection_control"}, [], [], {"collection": value}, status)


def collect(
    request: CollectionRun, environment: CollectionEnvironment, write: WriteFile | None = None
) -> RunResult:
    from fh5.experiment import RunResult

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
            if (root / "stop.request").exists() or (
                request.stop_path is not None and request.stop_path.exists()
            ):
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
    )
    result["complete"] = collection_complete(result)
    atomic_control_json(root / "final.json", result)
    atomic_control_json(root / "status.json", result)
    report = optional_report(
        root / "report.html",
        "持续采集状态（完整封存不等于优质示范）",
        result,
        fallback=root / "final.json",
        exclusive=True,
    )
    return RunResult({"source_kind": "passive_collection"}, [], [], {"collection": result}, report)
