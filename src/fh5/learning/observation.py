"""Project admission evidence without retaining unconstrained resource diagnostics."""

from __future__ import annotations

import math
from typing import Any


def _number(value: Any) -> int | float | None:
    if type(value) is int or (type(value) is float and math.isfinite(value)):
        return value
    return None


def _choice(value: Any, choices: tuple[str, ...]) -> str | None:
    return value if isinstance(value, str) and value in choices else None


def resource_observation(sample: dict[str, Any], *, include_gpu: bool) -> dict[str, Any]:
    """Retain typed admission fields; missing/invalid metrics remain unavailable.

    Unknown text and nested diagnostics do not take part in resource admission
    or get duplicated in every scheduler event. Error details remain with their
    source; their presence still blocks admission through the existing checks.
    """
    raw_collector = sample.get("collector")
    collector = raw_collector if isinstance(raw_collector, dict) else {}
    result: dict[str, Any] = {
        key: _number(sample.get(key))
        for key in ("observed_ns", "process_private_bytes", "free_disk_bytes")
    }
    for key in (
        "process_working_set_bytes",
        "process_lifetime_peak_working_set_bytes",
        "process_cpu_cores",
        "memory_error",
    ):
        if key in sample:
            result[key] = _number(sample[key])
    evidence: dict[str, Any] = {
        key: _number(collector.get(key))
        for key in (
            "heartbeat_ns",
            "last_poll_ns",
            "latest_image_source_ns",
            "pending_bytes",
            "dropped_rows",
        )
    }
    evidence.update(
        process_liveness=_choice(
            collector.get("process_liveness"), ("running", "exited", "not_started", "unknown")
        ),
        state=_choice(
            collector.get("state"),
            ("recording", "prepared", "starting", "stopping", "stopped", "complete", "failed"),
        ),
        archive_error=bool(collector.get("archive_error")),
        abnormal_exit=bool(collector.get("abnormal_exit")),
    )
    for key in ("software_snapshot_verified", "final_status_present", "complete"):
        value = collector.get(key)
        evidence[key] = value if type(value) is bool else None
    for key in ("seen_rows", "pid"):
        if key in collector:
            evidence[key] = _number(collector[key])
    for key in ("session_sha256", "session_manifest_sha256", "worker_manifest_sha256"):
        value = collector.get(key)
        if (
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)
        ):
            evidence[key] = value
    result["collector"] = evidence
    if include_gpu:
        gpus = sample.get("gpus")
        result["gpus"] = (
            [
                {
                    key: _number(gpu.get(key)) if isinstance(gpu, dict) else None
                    for key in ("memory_used_mib", "utilization_percent")
                }
                for gpu in gpus
            ]
            if isinstance(gpus, list)
            else None
        )
    return result
