"""Bounded repeated frozen-policy execution with declared external I/O sources."""

from __future__ import annotations

import hashlib
import html
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from fh5.artifacts.io import atomic_json, encode, read_bounded, write_file
from fh5.driving.events import EventEnvironment, EventRun, validate_event_file
from fh5.driving.realtime.model import RealtimeConfig, RealtimeEnvironment, RealtimeRun
from fh5.driving.realtime.numeric_replay import read_realtime_journal
from fh5.evaluation.completion import seal_evaluation
from fh5.evaluation.handoff import ReadyHandoff
from fh5.evaluation.model import asset_limit, evaluation_actor
from fh5.evaluation.prepare import EvaluationReview, _read, read_evaluation_batch
from fh5.evaluation.start import event_payloads, seal_start
from fh5.observation.numeric import PixelContract

if TYPE_CHECKING:
    from fh5.result import RunResult


@dataclass(frozen=True)
class EvaluationRun:
    batch_dir: Path
    batch_sha256: str
    event_config_file: Path
    output_dir: Path
    seconds: float = 15
    registry_file: Path | None = None
    initial_operation: Literal["start_ready", "restart_ready"] = "start_ready"
    live: bool = False

    def __post_init__(self) -> None:
        if type(self.live) is not bool:
            raise ValueError("Evaluation live opt-in must be a boolean")
        if self.initial_operation not in ("start_ready", "restart_ready"):
            raise ValueError("Evaluation requires a declared initial ready operation")
        if (
            type(self.seconds) not in (int, float)
            or not math.isfinite(self.seconds)
            or not 0.1 <= self.seconds <= 600
        ):
            raise ValueError("Evaluation attempts require a finite 0.1..600 second bound")


class EvaluationEnvironment(Protocol):
    @property
    def source_kind(self) -> Literal["synthetic", "native"]: ...

    def event(self, slot_id: str) -> EventEnvironment: ...
    def driving(self, slot_id: str, ready_state: dict[str, Any]) -> RealtimeEnvironment: ...
    def close(self) -> dict[str, Any]: ...


@runtime_checkable
class QualifiedEvaluationEnvironment(EvaluationEnvironment, Protocol):
    """Native environments qualify frozen inputs before acquiring any devices."""

    def prepare(self, request: EvaluationRun, batch: dict[str, Any]) -> dict[str, Any]: ...


def evaluation_inputs(
    request: EvaluationRun,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Validate all run/menu bindings without creating output or opening devices."""
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
    task = _read(request.batch_dir / "task.json")[0]
    if task["version"] == 2 and event_snapshot != {
        name: read_bounded(request.batch_dir / name, 16 * 1024**2)
        for name in batch["files"]
        if name.startswith("start/")
    }:
        raise ValueError("Requested event differs from the frozen automatic start protocol")
    return batch, event_snapshot, menu["event_run"]


def _freeze(
    request: EvaluationRun, native_binding: dict[str, Any] | None = None
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    batch, event_snapshot, event_parameters = evaluation_inputs(request)
    root = request.output_dir
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
    write_file(
        root / "run-protocol.json",
        encode(
            {
                "version": 2 if native_binding is not None else 1,
                "batch_sha256": request.batch_sha256,
                "seconds_per_attempt": request.seconds,
                "event_files": event_files,
                "source_kind": "native" if native_binding is not None else "synthetic",
                **({"native": native_binding} if native_binding is not None else {}),
                "exploration": False,
                "rewind": False,
                "initial_operation": request.initial_operation,
            }
        ),
    )
    return batch, event_file, event_parameters


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
    from fh5.driving.events import run_event
    from fh5.driving.realtime.runtime import run_realtime
    from fh5.evaluation.prepare import review_evaluation
    from fh5.reporting.recording import run_recording_report
    from fh5.result import RunResult
    from fh5.telemetry.packet import Packet, Record

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
        native_binding = None
        device = "cpu"
        source_batch, _, _ = evaluation_inputs(request)
        if environment.source_kind == "native":
            if (
                not request.live
                or request.seconds > 30
                or source_batch["version"] != 3
                or not isinstance(environment, QualifiedEvaluationEnvironment)
            ):
                raise ValueError(
                    "Native evaluation requires live opt-in, bounded v3 and qualification"
                )
            native_binding = environment.prepare(request, source_batch)
            device = source_batch["config"]["model"]["device"]
        elif environment.source_kind != "synthetic" or request.live:
            raise ValueError("Evaluation source and live opt-in disagree")
        elif source_batch["version"] == 3:
            raise ValueError("Version 3 execution requires a qualified native environment")
        batch, event_file, event_parameters = _freeze(request, native_binding)
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
            preparation: dict[str, Any] = {"slot_id": slot["id"], "release_sent": False}
            summary["preparations"].append(preparation)
            try:
                _verify_event_protocol(root, protocol_sha)
            except (Exception, KeyboardInterrupt):
                try:
                    menu_environment.release()
                finally:
                    menu_environment.close()
                raise
            preparation.update(
                run_event(
                    EventRun(
                        event_file,
                        directory / "ready",
                        operation=request.initial_operation if i == 0 else "restart_ready",
                        live=request.live,
                    ),
                    environment=menu_environment,
                ).summary["event_run"]
            )
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
                if drive.source_kind != environment.source_kind:
                    raise ValueError("Driving environment differs from evaluation source")
                _verify_event_protocol(root, protocol_sha)
            except (Exception, KeyboardInterrupt):
                drive.close()
                raise
            executed = run_realtime(
                RealtimeRun(
                    directory / "execution", config, seconds=request.seconds, live=request.live
                ),
                environment=ReadyHandoff(
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
                factory=lambda: evaluation_actor(
                    root / "frozen/model",
                    batch["config"]["model"],
                    batch["config"]["runtime"],
                    device=device,
                ),
            ).summary["realtime"]
            attempt.update(
                stop_reason=executed["stop_reason"],
                resources_released=executed["resources_released"],
            )
            summary["commands_sent_to_game"] |= executed["commands_sent_to_game"]
            packets = read_realtime_journal(directory / "execution", executed)
            run_recording_report(
                Record(
                    record_config, directory / "recording", "udp" if request.live else "synthetic"
                ),
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
        summary["resources_released"] = (
            summary["environment"].get("resources_released", False)
            and all(a["resources_released"] for a in summary["attempts"])
            and all(p["release_sent"] for p in summary["preparations"])
        )
        if environment.source_kind == "native":
            summary["commands_sent_to_game"] |= summary["environment"].get("menu_sends", 0) > 0
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
        reviewed_summary = review_evaluation(
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
        '<!doctype html><meta charset="utf-8"><h1>冻结策略重复评估</h1><p>执行记录不自动证明驾驶验收通过。</p>'
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
    if reviewed_summary and summary["resources_released"]:
        seal_evaluation(root)
    return RunResult({}, [], [], {"evaluation_run": summary, **reviewed_summary}, path)
