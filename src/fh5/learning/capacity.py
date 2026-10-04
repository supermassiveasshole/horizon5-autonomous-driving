"""Capacity admission at released learning boundaries; no deletion or device I/O."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fh5.learning.storage import measure_learning_storage


def validate_storage_budget(value: Any, base: Path) -> dict[str, Any]:
    fields = {"root", "budget_bytes", "min_free_bytes", "phase_reserve_bytes", "stop_reserve_bytes"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Learning storage requires an explicit capacity and reserves")
    if not isinstance(value["root"], str) or not value["root"].strip():
        raise ValueError("Learning storage requires an explicit dependency namespace")
    for field in fields - {"root"}:
        low = 0 if field == "min_free_bytes" else 1
        if type(value[field]) is not int or value[field] < low:
            raise ValueError("Invalid learning storage bound: " + field)
    return {**value, "root": str((base / value["root"]).resolve())}


def capacity_decision(
    run_dir: Path, state_sha256: str, budget: dict[str, Any], phase: str
) -> dict[str, Any]:
    snapshot = measure_learning_storage(Path(budget["root"]), run_dir, state_sha256)
    reserve = budget["phase_reserve_bytes"] + budget["stop_reserve_bytes"]
    free = shutil.disk_usage(run_dir).free
    reasons = []
    if snapshot["protected_bytes"] + reserve > budget["budget_bytes"]:
        reasons.append("logical_budget")
    if free < reserve + budget["min_free_bytes"]:
        reasons.append("disk_reserve")
    return {
        **snapshot,
        "phase": phase,
        "state_sha256": state_sha256,
        "budget_bytes": budget["budget_bytes"],
        "phase_reserve_bytes": budget["phase_reserve_bytes"],
        "stop_reserve_bytes": budget["stop_reserve_bytes"],
        "min_free_bytes": budget["min_free_bytes"],
        "free_bytes": free,
        "admitted": not reasons,
        "reasons": reasons,
        "reservation_kind": "admission_headroom_not_filesystem_quota",
        "files_deleted": 0,
    }
