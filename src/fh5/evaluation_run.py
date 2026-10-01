"""Bounded repeated frozen-policy execution over explicitly synthetic external I/O."""

from __future__ import annotations

import hashlib
import html
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from fh5.collection_store import atomic_json, encode, read_bounded, write_file
from fh5.evaluation import EvaluationReview, _read, read_evaluation_batch
from fh5.evaluation_handoff import ReadyHandoff
from fh5.evaluation_model import asset_limit, evaluation_actor
from fh5.evaluation_start import event_payloads, seal_start
from fh5.events import EventEnvironment, EventRun, validate_event_file
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig, RealtimeEnvironment, RealtimeRun
from fh5.realtime_numeric_replay import read_realtime_journal

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class EvaluationRun:
    batch_dir: Path
    batch_sha256: str
    event_config_file: Path
    output_dir: Path
    seconds: float = 15
    registry_file: Path | None = None
    initial_operation: Literal["start_ready", "restart_ready"] = "start_ready"

    def __post_init__(self) -> None:
        if self.initial_operation not in ("start_ready", "restart_ready"):
            raise ValueError("Evaluation requires a declared initial ready operation")
        if (
            type(self.seconds) not in (int, float)
            or not math.isfinite(self.seconds)
            or not 0.1 <= self.seconds <= 600
        ):
            raise ValueError("Evaluation attempts require a finite 0.1..600 second bound")


class EvaluationEnvironment(Protocol):
    source_kind: Literal["synthetic"]

    def event(self, slot_id: str) -> EventEnvironment: ...
    def driving(self, slot_id: str, ready_state: dict[str, Any]) -> RealtimeEnvironment: ...
    def close(self) -> dict[str, Any]: ...


def _freeze(request: EvaluationRun) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    batch, _, _ = read_evaluation_batch(request.batch_dir, request.batch_sha256)
    plan = batch["config"]["plan"]
    if not 1 <= len(plan) <= 10 or any(p["reference_mode"] != "no_reference" for p in plan):
        raise ValueError("Current repeated runner supports 1..10 no-reference runs")
    event_snapshot = event_payloads(request.event_config_file)
    menu = json.loads(event_snapshot["start/event.json"])
    runtime = batch["config"]["runtime"]
    if (
        not menu["event_run"]["conditions_verified"]
        or menu["event_run"]["purpose"] != "event"
        or menu["snapshot"] != batch["config"]["conditions"]["snapshot"]
        or any(
            menu["event_run"][key] != runtime[key]
            for key in ("expected_car_ordinal", "expected_pi")
        )
        or _read(request.batch_dir / "task.json")[0]["control_owner"] != "policy"
    ):
        raise ValueError("Event conditions and frozen policy task disagree")
    root = request.output_dir
    if root.resolve().is_relative_to(request.batch_dir.resolve()):
        raise ValueError("Evaluation output must be outside frozen batch")
    root.mkdir(parents=True, exist_ok=False)
    for name in ["batch.json", *batch["files"]]:
        dest = root / "frozen" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_file(dest, read_bounded(request.batch_dir / name, asset_limit(name)))
    read_evaluation_batch(root / "frozen", request.batch_sha256)
    event_files = {}
    for name, payload in event_snapshot.items():
        relative = name.removeprefix("start/")
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_file(destination, payload)
        event_files[relative] = hashlib.sha256(payload).hexdigest()
    event_file = root / "event.json"
    validate_event_file(event_file)
    task = _read(root / "frozen/task.json")[0]
    if task["version"] == 2:
        frozen = {
            name: read_bounded(root / "frozen" / name, 16 * 1024**2)
            for name in batch["files"]
            if name.startswith("start/")
        }
        if event_snapshot != frozen:
            raise ValueError("Requested event differs from the frozen automatic start protocol")
    write_file(
        root / "run-protocol.json",
        encode(
            {
                "version": 1,
                "batch_sha256": request.batch_sha256,
                "seconds_per_attempt": request.seconds,
                "event_files": event_files,
                "source_kind": "synthetic",
                "exploration": False,
                "rewind": False,
                "initial_operation": request.initial_operation,
            }
        ),
    )
    return batch, event_file, menu["event_run"]


def _verify_event_protocol(root: Path, expected: str) -> None:
    protocol, digest, _ = _read(root / "run-protocol.json", 1024**2)
    if digest != expected:
        raise ValueError("Frozen run protocol changed")
    for name, wanted in protocol["event_files"].items():
        target = (root / name).resolve()
        if (
            not target.is_relative_to(root.resolve())
            or hashlib.sha256(read_bounded(target, 16 * 1024**2)).hexdigest() != wanted
        ):
            raise ValueError("Frozen event protocol changed: " + name)


def run_evaluation(request: EvaluationRun, environment: EvaluationEnvironment) -> RunResult:
    from fh5.experiment import Packet, Record, RunResult, run_experiment

    summary: dict[str, Any] = {
        "version": 1,
        "started_slots": [],
        "preparations": [],
        "attempts": [],
        "stop_reason": "interface_error",
        "resources_released": False,
        "commands_sent_to_game": False,
        "diagnostic_only": True,
    }
    entries: list[dict[str, Any]] = []
    batch: dict[str, Any] | None = None
    root = request.output_dir
    try:
        if environment.source_kind != "synthetic":
            raise ValueError("Repeated evaluation currently requires synthetic external I/O")
        batch, event_file, event_parameters = _freeze(request)
        protocol_sha = _read(root / "run-protocol.json")[1]
        settings = dict(batch["config"]["runtime"])
        settings["pixels"] = PixelContract.from_metadata(settings["pixels"])
        settings["action_offsets_ms"] = tuple(settings["action_offsets_ms"])
        config = RealtimeConfig(**settings)
        task = _read(root / "frozen/task.json")[0]
        record_config = root / "record.json"
        write_file(
            record_config,
            encode(
                {
                    "schema_version": 1,
                    "control_source": "policy",
                    "snapshot": batch["config"]["conditions"]["snapshot"],
                }
            ),
        )
        for i, slot in enumerate(batch["config"]["plan"]):
            read_evaluation_batch(root / "frozen", request.batch_sha256)
            _verify_event_protocol(root, protocol_sha)
            directory = root / f"attempt-{i:04d}"
            directory.mkdir()
            menu_environment = environment.event(slot["id"])
            try:
                _verify_event_protocol(root, protocol_sha)
            except (Exception, KeyboardInterrupt):
                try:
                    menu_environment.release()
                finally:
                    menu_environment.close()
                raise
            preparation = run_experiment(
                EventRun(
                    event_file,
                    directory / "ready",
                    operation=request.initial_operation if i == 0 else "restart_ready",
                ),
                event_environment=menu_environment,
            ).summary["event_run"]
            summary["preparations"].append({"slot_id": slot["id"], **preparation})
            if (
                not preparation["ready_verified"]
                or not preparation["release_sent"]
                or preparation["evidence_status"] != "complete"
            ):
                summary["stop_reason"] = "ready_unconfirmed"
                break
            read_evaluation_batch(root / "frozen", request.batch_sha256)
            _verify_event_protocol(root, protocol_sha)
            summary["started_slots"].append(slot["id"])
            attempt = {
                "slot_id": slot["id"],
                "stop_reason": "interface_error",
                "resources_released": False,
            }
            summary["attempts"].append(attempt)
            entry: dict[str, Any] = {
                "slot_id": slot["id"],
                "recording": f"attempt-{i:04d}/recording",
                "files": {},
                "evidence": None,
            }
            entries.append(entry)
            atomic_json(
                root / "ledger.json",
                {"version": 1, "batch_sha256": request.batch_sha256, "entries": entries},
            )
            drive = environment.driving(slot["id"], deepcopy(preparation["ready_state"]))
            try:
                if drive.source_kind != "synthetic":
                    raise ValueError("Driving environment must be synthetic")
                _verify_event_protocol(root, protocol_sha)
            except (Exception, KeyboardInterrupt):
                drive.close()
                raise
            executed = run_experiment(
                RealtimeRun(directory / "execution", config, seconds=request.seconds),
                realtime_environment=ReadyHandoff(
                    drive,
                    preparation["ready_state"],
                    event_parameters,
                    config,
                    deadline_ns=round(
                        preparation["ready_observed_ns"]
                        + task["automatic_start"]["handoff_timeout_s"] * 1e9
                    )
                    if task["version"] == 2
                    else None,
                ),
                numeric_actor_factory=lambda: evaluation_actor(
                    root / "frozen/model", batch["config"]["model"], batch["config"]["runtime"]
                ),
            ).summary["realtime"]
            attempt.update(
                stop_reason=executed["stop_reason"],
                resources_released=executed["resources_released"],
            )
            packets = read_realtime_journal(directory / "execution", executed)
            run_experiment(
                Record(record_config, directory / "recording"),
                packets=(
                    Packet(
                        p["received_monotonic_ns"],
                        p["received_utc"],
                        bytes.fromhex(p["payload_hex"]),
                    )
                    for p in packets
                ),
            )
            entry.update(
                files={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (directory / "recording").iterdir()
                    if p.suffix in (".json", ".jsonl")
                },
                execution={
                    "directory": f"attempt-{i:04d}/execution",
                    "manifest_sha256": hashlib.sha256(
                        (directory / "execution/realtime-manifest.json").read_bytes()
                    ).hexdigest(),
                },
            )
            if _read(root / "frozen/task.json")[0]["version"] == 2:
                entry["preparation"] = {
                    "directory": f"attempt-{i:04d}/ready",
                    "manifest_sha256": seal_start(
                        directory / "ready",
                        request.batch_sha256,
                        slot["id"],
                        entry["files"]["packets.jsonl"],
                        entry["execution"]["manifest_sha256"],
                    ),
                }
            atomic_json(
                root / "ledger.json",
                {"version": 1, "batch_sha256": request.batch_sha256, "entries": entries},
            )
            if (
                not executed["resources_released"]
                or not executed["evidence"]["recording_complete"]
                or executed["stop_reason"] not in ("time_limit", "local_end")
            ):
                summary["stop_reason"] = "execution_stopped"
                break
        else:
            summary["stop_reason"] = "plan_complete"
    except (Exception, KeyboardInterrupt) as error:
        if batch is None:
            raise
        summary.update(
            stop_reason="user_stop" if isinstance(error, KeyboardInterrupt) else "interface_error",
            error=str(error),
        )
    finally:
        try:
            summary["environment"] = environment.close()
        except Exception as error:
            summary["environment"] = {"resources_released": False, "error": str(error)}
            summary["stop_reason"] = "close_failed"
        summary["resources_released"] = summary["environment"].get(
            "resources_released", False
        ) and all(a["resources_released"] for a in summary["attempts"])
    assert batch is not None
    summary["unstarted_slots"] = [
        p["id"] for p in batch["config"]["plan"] if p["id"] not in summary["started_slots"]
    ]
    atomic_json(root / "run.json", summary)
    atomic_json(
        root / "ledger.json",
        {"version": 1, "batch_sha256": request.batch_sha256, "entries": entries},
    )
    reviewed_summary: dict[str, Any] = {}
    try:
        reviewed_summary = run_experiment(
            EvaluationReview(
                root / "frozen", root / "ledger.json", root / "review", request.registry_file
            )
        ).summary
    except (OSError, ValueError, KeyError, TypeError) as error:
        summary["review_error"] = str(error)
        summary["stop_reason"] = "review_failed"
    atomic_json(root / "run.json", summary)
    path = root / "report.html"
    path.write_text(
        '<!doctype html><meta charset="utf-8"><h1>合成重复评估</h1><p>非实机驾驶验收。</p>'
        + (
            '<a href="review/report.html">全部尝试与执行核验</a>'
            if reviewed_summary
            else "<p>评估失败，原始记录及尝试清单保留。</p>"
        )
        + "<pre>"
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>",
        encoding="utf-8",
    )
    return RunResult({}, [], [], {"evaluation_run": summary, **reviewed_summary}, path)
