"""Collection-first admission and bounded cooperative scheduling for numeric BC."""

from __future__ import annotations

import hashlib
import importlib
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from fh5.artifacts.document import read_document_fields
from fh5.artifacts.io import VerifiedFile, encode, sha256_file, write_file
from fh5.learning.bc.checkpoint import BCRecovery, read_bc_checkpoint
from fh5.learning.bc.training import (
    TemporalBCTrain,
    _checked_configuration,
    _configuration,
    run_temporal_bc,
)
from fh5.learning.diagnostics import RecordJournal
from fh5.learning.observation import resource_observation

if TYPE_CHECKING:
    from fh5.result import RunResult


@dataclass(frozen=True)
class ScheduledBCTrain:
    config_file: Path
    output_dir: Path


@dataclass(frozen=True)
class ScheduledBCResume:
    run_dir: Path
    output_dir: Path
    expected_checkpoint_sha256: str | None = None


class LearningResources(Protocol):
    source_kind: str

    def now_ns(self) -> int: ...
    def sample(self) -> dict[str, Any]: ...
    def wait(self, seconds: float) -> None: ...
    def close(self) -> None: ...


class ScheduleStopped(Exception):
    pass


def _configuration_schedule(source: VerifiedFile) -> dict[str, Any]:
    shared_fields = {
        "version",
        "collector_bundle",
        "collector_manifest_sha256",
        "budget",
    }
    config = read_document_fields(
        source,
        shared_fields | {"training", "training_config", "training_config_sha256"},
        reject_unknown=True,
    )
    version = config.get("version")
    if not (
        version == 1
        and set(config) == shared_fields | {"training_config", "training_config_sha256"}
        or version == 2
        and set(config) | {"collector_manifest_sha256"} == shared_fields | {"training"}
    ):
        raise ValueError("Unsupported learning schedule")
    budget_fields = {
        "cpu_threads",
        "poll_interval_s",
        "max_wait_s",
        "max_total_s",
        "max_unit_s",
        "max_private_bytes",
        "min_free_disk_bytes",
        "max_status_age_ms",
        "max_image_age_ms",
        "max_pending_bytes",
        "max_gpu_memory_mib",
        "max_gpu_utilization_percent",
    }
    budget = config["budget"]
    if not isinstance(budget, dict) or set(budget) != budget_fields:
        raise ValueError("Incomplete learning resource budget")
    for name in budget_fields:
        value = budget[name]
        if (
            type(value) not in (int, float)
            or (type(value) is float and not math.isfinite(value))
            or value < 0
        ):
            raise ValueError("Invalid learning resource budget: " + name)
    if type(budget["cpu_threads"]) is not int or budget["cpu_threads"] == 0:
        raise ValueError("CPU thread budget must be a positive integer")
    if budget["poll_interval_s"] == 0:
        raise ValueError("Resource polling interval must be positive")
    # Utilization is a percentage; other maxima come from this experiment,
    # not a second hard-coded capacity or duration ceiling.
    if budget["max_gpu_utilization_percent"] > 100:
        raise ValueError("GPU utilization budget must be a percentage from 0 to 100")
    if "collector_manifest_sha256" not in config:
        config["collector_manifest_sha256"] = sha256_file(
            source.path.parent / config["collector_bundle"] / "frozen.json"
        )
    for key in (
        ("training_config_sha256", "collector_manifest_sha256")
        if version == 1
        else ("collector_manifest_sha256",)
    ):
        if (
            not isinstance(config[key], str)
            or len(config[key]) != 64
            or any(c not in "0123456789abcdef" for c in config[key])
        ):
            raise ValueError("Expected SHA-256 binding: " + key)
    if version == 1:
        training_path = source.path.parent / config["training_config"]
        training = _configuration(training_path, expected_sha256=config["training_config_sha256"])
        training_base = training_path.parent
    else:
        training = _checked_configuration(config["training"], source.path.parent)
        training_base = source.path.parent
    return {
        **{key: config[key] for key in shared_fields},
        "version": 2,
        "training": dict(training, dataset=str((training_base / training["dataset"]).resolve())),
    }


class LearningSchedule:
    def __init__(
        self, budget: dict[str, Any], device: str, source: LearningResources, stop_path: Path
    ) -> None:
        self.budget, self.device, self.source = budget, device, source
        self.stop_path = stop_path
        self.started = source.now_ns()
        self.unit_started: int | None = None
        self.completed = self.pauses = self.samples = 0
        self.last_dropped = 0
        self.events = RecordJournal(
            stop_path.parent, "diagnostics/schedule-events.jsonl", "learning-schedule-events-v1"
        )
        self.reasons: Counter[str] = Counter()
        self.max_unit_s = self.wait_s = 0.0

    def _reasons(self, sample: dict[str, Any], now: int) -> list[str]:
        cfg = self.budget

        def number(value: Any) -> bool:
            return type(value) is int or (type(value) is float and math.isfinite(value))

        if not number(sample.get("observed_ns")) or not (
            0 <= now - sample["observed_ns"] <= cfg["max_status_age_ms"] * 1_000_000
        ):
            return ["resource_status_stale"]
        reasons = []
        for field, limit, below in (
            ("process_private_bytes", cfg["max_private_bytes"], False),
            ("free_disk_bytes", cfg["min_free_disk_bytes"], True),
        ):
            if not number(sample.get(field)) or sample[field] < 0:
                return ["resource_status_missing"]
            if (sample[field] < limit) if below else (sample[field] > limit):
                reasons.append("resource_limit:" + field)
        collector = sample.get("collector", {})
        if (
            collector.get("archive_error")
            or collector.get("abnormal_exit")
            or (
                collector.get("process_liveness") == "exited"
                and collector.get("final_status_present") is True
                and collector.get("complete") is not True
            )
        ):
            raise ScheduleStopped("collector_failed")
        stopped = (
            collector.get("process_liveness") == "exited"
            and collector.get("software_snapshot_verified") is True
            and collector.get("final_status_present") is True
            and collector.get("complete") is True
        )
        if not stopped:
            if (
                collector.get("process_liveness") != "running"
                or collector.get("software_snapshot_verified") is not True
                or collector.get("state") != "recording"
            ):
                reasons.append("collector_state_unverified")
            for field in ("heartbeat_ns", "last_poll_ns"):
                if not number(collector.get(field)) or not (
                    0 <= now - collector[field] <= cfg["max_status_age_ms"] * 1_000_000
                ):
                    reasons.append("collection_status_stale")
            if (
                not number(collector.get("last_poll_ns"))
                or not number(collector.get("latest_image_source_ns"))
                or not (
                    0
                    <= collector["last_poll_ns"] - collector["latest_image_source_ns"]
                    <= cfg["max_image_age_ms"] * 1_000_000
                )
            ):
                reasons.append("collection_images_stale")
            pending = collector.get("pending_bytes")
            if not number(pending) or not 0 <= pending <= cfg["max_pending_bytes"]:
                reasons.append("collection_backlog")
            dropped = collector.get("dropped_rows")
            if type(dropped) is not int or dropped < self.last_dropped:
                reasons.append("collection_counters_invalid")
            else:
                if dropped > self.last_dropped:
                    reasons.append("collection_dropped_rows")
                self.last_dropped = dropped
            if self.device == "cuda":
                reasons.append("cuda_waits_for_collection_exit")
        if self.device == "cuda":
            gpus = sample.get("gpus")
            if not isinstance(gpus, list) or not gpus:
                reasons.append("gpu_status_missing")
            else:
                for gpu in gpus:
                    for field, threshold in (
                        ("memory_used_mib", cfg["max_gpu_memory_mib"]),
                        ("utilization_percent", cfg["max_gpu_utilization_percent"]),
                    ):
                        if not number(gpu.get(field)) or not 0 <= gpu[field] <= threshold:
                            reasons.append("gpu_pressure")
        return sorted(set(reasons))

    def checkpoint(
        self,
        phase: str,
        completed: int,
        suspend: Callable[[], None] | None = None,
        resume: Callable[[], None] | None = None,
    ) -> None:
        try:
            self._checkpoint(phase, completed, suspend, resume)
        except Exception:
            if suspend:
                suspend()
            raise

    def _check_limits(self, now: int, work_started: int | None = None) -> None:
        elapsed = 0.0
        if work_started is not None:
            elapsed = (now - work_started) / 1e9
            if elapsed < 0:
                raise ScheduleStopped("resource_clock_regressed")
            self.max_unit_s = max(self.max_unit_s, elapsed)
        if self.stop_path.exists():
            raise ScheduleStopped("requested_stop")
        if elapsed > self.budget["max_unit_s"]:
            raise ScheduleStopped("work_unit_overrun")
        if now < self.started or now - self.started >= self.budget["max_total_s"] * 1_000_000_000:
            raise ScheduleStopped("total_time_limit")

    def _check_wait_limit(self, now: int, waiting_since: int | None) -> None:
        if (
            waiting_since is not None
            and now - waiting_since >= self.budget["max_wait_s"] * 1_000_000_000
        ):
            raise ScheduleStopped("resource_wait_timeout")

    def _checkpoint(
        self,
        phase: str,
        completed: int,
        suspend: Callable[[], None] | None,
        resume: Callable[[], None] | None,
    ) -> None:
        self.completed = completed
        self._check_limits(self.source.now_ns(), self.unit_started)
        waiting_since: int | None = None
        while True:
            now = self.source.now_ns()
            self._check_limits(now)
            self._check_wait_limit(now, waiting_since)
            sample = resource_observation(self.source.sample(), include_gpu=self.device == "cuda")
            self.samples += 1
            now = self.source.now_ns()
            stopped: ScheduleStopped | None = None
            try:
                self._check_limits(now)
                self._check_wait_limit(now, waiting_since)
                reasons = self._reasons(sample, now)
            except ScheduleStopped as error:
                stopped, reasons = error, [str(error)]
            self.reasons.update(reasons)
            self.events.append(
                {
                    "at_ns": now,
                    "phase": phase,
                    "steps_completed": completed,
                    "reasons": reasons,
                    "sample": sample,
                }
            )
            if stopped is not None:
                raise stopped
            if not reasons:
                self.unit_started = self.source.now_ns()
                if waiting_since is not None and resume:
                    resume()
                self._check_limits(self.source.now_ns(), self.unit_started)
                return
            if waiting_since is None:
                waiting_since = now
                self.pauses += 1
                if suspend:
                    transfer_started = self.source.now_ns()
                    suspend()
                    self._check_limits(self.source.now_ns(), transfer_started)
            now = self.source.now_ns()
            self._check_limits(now)
            wait_ns = min(
                self.budget["poll_interval_s"] * 1_000_000_000,
                self.budget["max_wait_s"] * 1_000_000_000 - (now - waiting_since),
                self.budget["max_total_s"] * 1_000_000_000 - (now - self.started),
            )
            if wait_ns <= 0:
                raise ScheduleStopped("resource_wait_timeout")
            wait_s = wait_ns / 1_000_000_000
            self.source.wait(wait_s)
            self.wait_s += wait_s


def run_scheduled_bc(
    request: ScheduledBCTrain | ScheduledBCResume, resources: LearningResources | None
) -> RunResult:
    from fh5.result import RunResult

    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    config_file = (
        request.run_dir / "schedule-config.json"
        if isinstance(request, ScheduledBCResume)
        else request.config_file
    )
    config_source = VerifiedFile(config_file, sha256_file(config_file))
    config = _configuration_schedule(config_source)
    base = config_file.parent
    training = config["training"]
    dataset = Path(training["dataset"])
    parent = None
    if isinstance(request, ScheduledBCResume):
        if request.output_dir.resolve().is_relative_to(request.run_dir.resolve()):
            raise ValueError("BC continuation output must be outside its parent run")
        checkpoint_root = request.run_dir / "learner"
        expected = request.expected_checkpoint_sha256
        if not (checkpoint_root / "learner.json").is_file():
            # Runs stopped before a new seal can still refer to their durable
            # ancestor. A newly sealed learner never depends on this report,
            # which may be absent/partial after a later diagnostic I/O failure.
            report = request.run_dir / "schedule.json"
            previous = read_document_fields(
                VerifiedFile(report, sha256_file(report)), {"learner_checkpoint"}
            ).get("learner_checkpoint")
            if (
                not isinstance(previous, dict)
                or not isinstance(previous.get("manifest_sha256"), str)
                or (expected is not None and previous["manifest_sha256"] != expected)
            ):
                raise ValueError("Scheduled BC run has no matching durable learner checkpoint")
            checkpoint_root = Path(previous["directory"])
            expected = previous["manifest_sha256"]
        elif expected is None:
            expected = sha256_file(checkpoint_root / "learner.json")
        if request.output_dir.resolve().is_relative_to(checkpoint_root.resolve()):
            raise ValueError("BC continuation output must be outside its parent checkpoint")
        parent = read_bc_checkpoint(
            importlib.import_module("torch"),
            checkpoint_root,
            expected_sha256=expected,
            cpu_threads=config["budget"]["cpu_threads"],
        )
        effective = dict(training, dataset=str(dataset))
        if parent.manifest["model_metadata"]["config"] != effective:
            raise ValueError("Scheduled BC training configuration changed from its checkpoint")
    recovery = (
        BCRecovery(request.output_dir / "learner", parent) if training["device"] == "cpu" else None
    )
    starting_step = recovery.completed if recovery is not None else 0
    if request.output_dir.resolve().is_relative_to(dataset.parent):
        raise ValueError("Scheduled output must be separate from its frozen dataset")
    if request.output_dir.resolve().is_relative_to((base / config["collector_bundle"]).resolve()):
        raise ValueError("Scheduled output must be outside the frozen collector")
    if resources is None:
        from fh5.learning.resources import NativeLearningResources

        resources = NativeLearningResources(
            base / config["collector_bundle"],
            config["collector_manifest_sha256"],
            request.output_dir.parent,
            config["budget"]["poll_interval_s"],
            include_gpu=training["device"] == "cuda",
        )
    if resources.source_kind not in ("synthetic", "native_resources"):
        raise ValueError("Unknown learning resource source")
    request.output_dir.mkdir(parents=True)
    schedule = LearningSchedule(
        config["budget"], training["device"], resources, request.output_dir / "stop.request"
    )
    training["dataset"] = str(dataset)
    frozen_config = request.output_dir / "training.json"
    state, reason, failure = "stopped", None, None
    candidate_hash = None
    partial = request.output_dir / ".candidate"
    try:
        write_file(frozen_config, encode(training))
        config_source.copy_to(request.output_dir / "requested-schedule.json")
        config = dict(
            config,
            collector_bundle=str((base / config["collector_bundle"]).resolve()),
        )
        write_file(request.output_dir / "schedule-config.json", encode(config))
        schedule.checkpoint("admission", starting_step)
        result = run_temporal_bc(
            TemporalBCTrain(frozen_config, partial),
            schedule,
            config["budget"]["cpu_threads"],
            recovery,
        )
        candidate_hash = sha256_file(partial / "model.json")
        schedule.checkpoint("publish", training["steps"])
        state = "completed"
        diagnostic = (
            result.summary["temporal_bc"]["model"]["diagnostic_only"]
            or resources.source_kind == "synthetic"
        )
    except ScheduleStopped as error:
        reason = str(error)
        diagnostic = True
    except KeyboardInterrupt:
        reason, diagnostic = "interrupted", True
    except Exception as error:
        failure = error
        reason, diagnostic = f"{type(error).__name__}: {error}", True
    finally:
        try:
            resources.close()
        except Exception as error:
            state, reason, diagnostic = "stopped", "resource_close_failed:" + str(error), True
            failure = failure or error
    if state == "completed":
        try:
            partial.rename(request.output_dir / "candidate")
        except OSError as error:
            state, reason, diagnostic = "stopped", "publication_failed:" + str(error), True
            failure = failure or error
    completed = max(schedule.completed, recovery.completed if recovery is not None else 0)
    latest = recovery.latest if recovery is not None else None
    event_history = schedule.events.finish()
    event_history["sample_contract"] = (
        "typed admission evidence; unstructured source details omitted"
    )
    summary = {
        "version": 1,
        "state": state,
        "stop_reason": reason,
        "source_kind": resources.source_kind,
        "diagnostic_only": diagnostic,
        "commands_sent": False,
        "closed_loop_validated": False,
        "steps_completed": completed,
        "steps_this_run": completed - starting_step,
        "durable_steps_completed": latest["steps_completed"] if latest is not None else 0,
        "learner_checkpoint": latest,
        "recovery_scope": "CPU cooperative complete-update boundaries; not abrupt-kill or partial-Adam recovery",
        "pauses": schedule.pauses,
        "wait_s": schedule.wait_s,
        "max_work_unit_s": schedule.max_unit_s,
        "sample_count": schedule.samples,
        "event_history": event_history,
        "events_omitted": max(0, schedule.samples - event_history["records"]),
        "events_unverified": (
            event_history["records"] if event_history["status"] != "complete" else 0
        ),
        "pressure_counts": dict(schedule.reasons),
        "config": config,
        "candidate": "candidate" if state == "completed" else None,
        "partial_candidate": ".candidate" if partial.exists() else None,
        "candidate_manifest_sha256": candidate_hash if state == "completed" else None,
        "effective_training_config_sha256": hashlib.sha256(encode(training)).hexdigest(),
        "dataset_sha256": training["dataset_sha256"],
        "scope": "cooperative work-unit scheduling; not hard GPU preemption or game validation",
    }
    report = request.output_dir / "schedule.json"
    try:
        write_file(report, encode(summary))
    except (OSError, MemoryError) as error:
        diagnostic_path = report
        report = (
            Path(latest["directory"]) / "learner.json"
            if latest is not None
            else request.output_dir / "candidate/model.json"
            if state == "completed"
            else frozen_config
        )
        summary["schedule_report"] = {
            "status": "unavailable",
            "path": str(diagnostic_path),
            "error": f"{type(error).__name__}: {error}",
            "retained_result": str(report),
        }
    if failure:
        raise failure
    return RunResult({}, [], [], {"learning_schedule": summary}, report)
