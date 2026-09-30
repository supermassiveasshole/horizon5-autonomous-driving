"""Bounded repeated frozen-policy execution over explicitly synthetic external I/O."""

from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from fh5.collection_store import atomic_json, encode, read_bounded, write_file
from fh5.evaluation import EvaluationReview, _read
from fh5.evaluation_handoff import ReadyHandoff
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

    def __post_init__(self) -> None:
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


def _batch(directory: Path, expected: str) -> dict[str, Any]:
    batch, digest, _ = _read(directory / "batch.json", 4 * 1024**2)
    if (
        digest != expected
        or batch.get("kind") != "frozen-local-evaluation-v1"
        or batch.get("version") != 1
    ):
        raise ValueError("Frozen evaluation batch changed or unsupported")
    for name, wanted in batch["files"].items():
        path = (directory / name).resolve()
        if (
            not path.is_relative_to(directory.resolve())
            or hashlib.sha256(read_bounded(path, 128 * 1024**2)).hexdigest() != wanted
        ):
            raise ValueError("Frozen evaluation dependency changed: " + name)
    return batch


def _freeze(request: EvaluationRun) -> tuple[dict[str, Any], Path]:
    batch = _batch(request.batch_dir, request.batch_sha256)
    plan = batch["config"]["plan"]
    if not 1 <= len(plan) <= 10 or any(p["reference_mode"] != "no_reference" for p in plan):
        raise ValueError("Current repeated runner supports 1..10 no-reference runs")
    menu = validate_event_file(request.event_config_file)
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
        write_file(dest, read_bounded(request.batch_dir / name, 128 * 1024**2))
    _batch(root / "frozen", request.batch_sha256)
    assets = root / "event-assets"
    assets.mkdir()
    references = [
        patch for patches in menu["event_run"]["signatures"].values() for patch in patches
    ]
    for i, patch in enumerate(references):
        name = f"event-assets/{i}.pgm"
        write_file(
            root / name,
            read_bounded(request.event_config_file.parent / patch["template"], 16 * 1024**2),
        )
        patch["template"] = name
    for i, source in enumerate(menu["event_run"]["verification_evidence"]):
        name = f"event-assets/evidence-{i}"
        write_file(
            root / name, read_bounded(request.event_config_file.parent / source, 16 * 1024**2)
        )
        menu["event_run"]["verification_evidence"][i] = name
    event_file = root / "event.json"
    write_file(event_file, encode(menu))
    validate_event_file(event_file)
    write_file(
        root / "run-protocol.json",
        encode(
            {
                "version": 1,
                "batch_sha256": request.batch_sha256,
                "seconds_per_attempt": request.seconds,
                "event_files": {
                    p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [event_file, *assets.iterdir()]
                },
                "source_kind": "synthetic",
                "exploration": False,
                "rewind": False,
            }
        ),
    )
    return batch, event_file


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
    from fh5.numeric_actor import FrozenNumericActor

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
        batch, event_file = _freeze(request)
        protocol_sha = _read(root / "run-protocol.json")[1]
        settings = dict(batch["config"]["runtime"])
        settings["pixels"] = PixelContract.from_metadata(settings["pixels"])
        settings["action_offsets_ms"] = tuple(settings["action_offsets_ms"])
        config = RealtimeConfig(**settings)
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
            _batch(root / "frozen", request.batch_sha256)
            _verify_event_protocol(root, protocol_sha)
            directory = root / f"attempt-{i:04d}"
            directory.mkdir()
            preparation = run_experiment(
                EventRun(
                    event_file,
                    directory / "ready",
                    operation="start_ready" if i == 0 else "restart_ready",
                ),
                event_environment=environment.event(slot["id"]),
            ).summary["event_run"]
            summary["preparations"].append({"slot_id": slot["id"], **preparation})
            if (
                not preparation["ready_verified"]
                or not preparation["release_sent"]
                or preparation["evidence_status"] != "complete"
            ):
                summary["stop_reason"] = "ready_unconfirmed"
                break
            _batch(root / "frozen", request.batch_sha256)
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
            drive = environment.driving(slot["id"], preparation["ready_state"])
            if drive.source_kind != "synthetic":
                drive.close()
                raise ValueError("Driving environment must be synthetic")
            executed = run_experiment(
                RealtimeRun(directory / "execution", config, seconds=request.seconds),
                realtime_environment=ReadyHandoff(
                    drive, preparation["ready_state"], _read(event_file)[0]["event_run"], config
                ),
                numeric_actor_factory=lambda: FrozenNumericActor(
                    root / "frozen/model", config.pixels
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
            EvaluationReview(root / "frozen", root / "ledger.json", root / "review")
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
