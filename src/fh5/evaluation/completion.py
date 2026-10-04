"""Final evaluation publication and independent recovery of completed child runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from fh5.artifacts.io import atomic_json, read_bounded
from fh5.driving.realtime.numeric_replay import read_realtime_journal, read_realtime_recording
from fh5.evaluation.prepare import EvaluationReview, _read, read_evaluation_batch
from fh5.evaluation.start import event_payloads

if TYPE_CHECKING:
    from fh5.evaluation.run import EvaluationRun


_DOCUMENTS = (
    "run.json",
    "ledger.json",
    "run-protocol.json",
    "record.json",
    "review/batch-report.json",
    "review/ledger.json",
)


def _digests(root: Path) -> dict[str, str]:
    return {name: _read(root / name)[1] for name in _DOCUMENTS}


def seal_evaluation(root: Path) -> None:
    """Published only after all child work and its independent review return."""
    atomic_json(root / "completion.json", {"version": 1, "files": _digests(root)})


def completed_evaluation(
    request: EvaluationRun, *, expected_native: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Recover against parent-authenticated native bindings, never the child's declaration."""
    from fh5.evaluation.prepare import review_evaluation

    if request.live != (expected_native is not None):
        raise ValueError("Evaluation recovery requires matching parent native qualification")
    root = request.output_dir
    seal = _read(root / "completion.json", 4096)[0]
    if seal != {"version": 1, "files": _digests(root)}:
        raise ValueError("Completed evaluation publication changed")
    batch, _, _ = read_evaluation_batch(root / "frozen", request.batch_sha256)
    read_evaluation_batch(request.batch_dir, request.batch_sha256)
    if expected_native is not None and (
        batch["version"] != 3
        or expected_native.get("qualification", {}).get("eligible") is not True
        or expected_native.get("qualification", {}).get("reasons") != []
        or expected_native.get("bindings", {}).get("model_manifest_sha256")
        != batch["config"]["model"]["manifest_sha256"]
        or expected_native.get("bindings", {}).get("inference_device")
        != batch["config"]["model"]["device"]
        or expected_native.get("bindings", {}).get("conditions")
        != batch["config"]["conditions"]["numeric_input_conditions"]
    ):
        raise ValueError("Parent native qualification differs from the frozen evaluation")
    events = event_payloads(request.event_config_file)
    expected_protocol = {
        "version": 2 if request.live else 1,
        "batch_sha256": request.batch_sha256,
        "seconds_per_attempt": request.seconds,
        "event_files": {
            name.removeprefix("start/"): hashlib.sha256(raw).hexdigest()
            for name, raw in events.items()
        },
        "source_kind": "native" if request.live else "synthetic",
        **({"native": expected_native} if request.live else {}),
        "exploration": False,
        "rewind": False,
        "initial_operation": request.initial_operation,
    }
    if _read(root / "run-protocol.json")[0] != expected_protocol or any(
        read_bounded(root / name.removeprefix("start/"), 16 * 1024**2) != raw
        for name, raw in events.items()
    ):
        raise ValueError("Completed evaluation differs from its requested protocol")
    if _read(root / "record.json")[0] != {
        "schema_version": 1,
        "control_source": "policy",
        "snapshot": batch["config"]["conditions"]["snapshot"],
    }:
        raise ValueError("Completed evaluation recording conditions changed")
    result = _read(root / "run.json")[0]
    slots = [slot["id"] for slot in batch["config"]["plan"]]
    started = result.get("started_slots", [])
    count = len(started)
    if (
        result.get("version") != 1
        or result.get("stop_reason") not in ("plan_complete", "execution_stopped")
        or result.get("resources_released") is not True
        or type(result.get("commands_sent_to_game")) is not bool
        or not request.live
        and result["commands_sent_to_game"] is not False
        or result.get("environment", {}).get("resources_released") is not True
        or not 0 < count <= len(slots)
        or started != slots[:count]
        or result.get("unstarted_slots") != slots[count:]
        or len(result.get("attempts", [])) != count
        or len(result.get("preparations", [])) != count
        or any(result.get(key) for key in ("error", "review_error"))
    ):
        raise ValueError("Evaluation child did not seal a released plan prefix")
    ledger = _read(root / "ledger.json")[0]
    if (
        ledger["batch_sha256"] != request.batch_sha256
        or [entry["slot_id"] for entry in ledger["entries"]] != started
        or _read(root / "review/ledger.json")[0] != ledger
    ):
        raise ValueError("Completed evaluation slots differ from the frozen plan")
    final_stopped = False
    for index, slot in enumerate(started):
        directory = root / f"attempt-{index:04d}"
        execution = read_realtime_recording(directory / "execution")
        if execution["evidence_kind"] != ("native" if request.live else "synthetic") or (
            expected_native is not None
            and (
                execution["environment"].get("input_conditions") != expected_native["bindings"]
                or execution["environment"].get("qualification") != expected_native["qualification"]
            )
        ):
            raise ValueError("Completed evaluation execution differs from its requested source")
        read_realtime_journal(directory / "execution", execution, time_limit_s=request.seconds)
        preparation = _read(directory / "ready/event-run.json")[0]["summary"]
        if preparation.get("operation") != (
            request.initial_operation if index == 0 else "restart_ready"
        ):
            raise ValueError("Completed evaluation ready operation differs from its request")
        stopped = (
            execution["stop_reason"] not in ("time_limit", "local_end")
            or not execution["evidence"]["recording_complete"]
        )
        if (
            result["attempts"][index]
            != {
                "slot_id": slot,
                "stop_reason": execution["stop_reason"],
                "resources_released": execution["resources_released"],
            }
            or execution["resources_released"] is not True
            or (stopped and index != count - 1)
            or result["preparations"][index] != {"slot_id": slot, **preparation}
            or preparation["release_sent"] is not True
        ):
            raise ValueError("Completed evaluation summary differs from original execution")
        final_stopped = stopped
    if result["stop_reason"] != ("execution_stopped" if final_stopped else "plan_complete") or (
        not final_stopped and count != len(slots)
    ):
        raise ValueError("Completed evaluation stop differs from original execution")
    # Re-run inference and menu/recording checks, never just trust completion hashes.
    with TemporaryDirectory(prefix="evaluation-recovery-", dir=root.parent) as temporary:
        reviewed = review_evaluation(
            EvaluationReview(root / "frozen", root / "ledger.json", Path(temporary) / "review")
        ).summary["evaluation"]
    original = _read(root / "review/batch-report.json")[0]
    if (
        {key: value for key, value in reviewed.items() if key != "independence"}
        != {key: value for key, value in original.items() if key != "independence"}
        or reviewed["unresolved_recordings"]
        or reviewed["quarantined_executions"]
        or reviewed["execution_metrics"]["bound_runs"] != count
        or reviewed["verified_starts"] != count
    ):
        raise ValueError("Completed evaluation failed independent evidence replay")
    return result
