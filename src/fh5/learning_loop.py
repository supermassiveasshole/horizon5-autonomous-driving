"""Serial, bounded sample/learn/evaluate/retain orchestration over synthetic I/O."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

from fh5.artifact_io import read_json, sha256_file
from fh5.candidate_archive import CandidateRestore
from fh5.candidate_selection import _input, evaluation_conditions
from fh5.candidate_store import CandidateHistory, CandidateRecord
from fh5.collection_lease import CollectionLease
from fh5.collection_store import atomic_json, encode, read_bounded, write_file
from fh5.evaluation import EvaluationPrepare, EvaluationReview, read_evaluation_batch
from fh5.evaluation_completion import completed_evaluation
from fh5.evaluation_run import EvaluationEnvironment, EvaluationRun
from fh5.learning_capacity import capacity_decision, validate_storage_budget
from fh5.learning_io import (
    EvaluationLease,
    LearningUnavailable,
    RealtimeSamplingLease,
    RejectedLease,
    SamplingLease,
)
from fh5.learning_monitor import StorageMonitor, validate_monitor
from fh5.learning_recovery import (
    archive_failed_sampling,
    completed_sampling,
    retryable_sampling,
    sampling_bindings,
    verify_archived_sampling,
)
from fh5.learning_stages import StageHistory
from fh5.learning_update_history import UpdateHistory
from fh5.learning_updates import UpdateProgress, retained_update_progress
from fh5.numeric_images import PixelContract
from fh5.presentation import optional_report
from fh5.realtime import RealtimeConfig
from fh5.sac_cycle import SACCycle, SACEnvironment, SACRealtimeCycle, sampling_update_budget
from fh5.sac_learning import SACResume, validate_sac_candidate
from fh5.sac_realtime_sampler import SACRealtimeEnvironment
from fh5.sampling_evidence import verify_sampling_sources

if TYPE_CHECKING:
    from fh5.experiment import RunResult

_Lease = TypeVar("_Lease", bound=SACEnvironment | SACRealtimeEnvironment | EvaluationEnvironment)


@dataclass(frozen=True)
class LearningLoop:
    config_file: Path
    output_dir: Path


@dataclass(frozen=True)
class LearningContinue:
    run_dir: Path
    expected_state_sha256: str


class LearningEnvironment(Protocol):
    """One backend, serial leases. Review supplies independent raw evidence, or None."""

    source_kind: Literal["synthetic"]

    def sampling(self, identity: str) -> SACEnvironment | SACRealtimeEnvironment: ...
    def evaluation(self, identity: str) -> EvaluationEnvironment: ...
    def review(self, recording_dir: Path) -> Path | None: ...
    def close(self) -> dict[str, Any]: ...


def _sha(path: Path) -> str:
    return sha256_file(path)


def _configuration(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = json.loads(read_bounded(path, 1024**2))
    asynchronous = config.get("version") == 3
    if (
        set(config)
        - {"acquisition_retry", "sampling_retry"}
        - ({"storage", "storage_monitor"} if config.get("version") in (2, 3) else set())
        != {
            "version",
            "store",
            "registry",
            "recording",
            "task",
            "reward",
            "rounds",
            "sampling" if asynchronous else "steps_per_attempt",
            "evaluation_seconds",
            "seed",
        }
        or type(config["version"]) is not int
        or config["version"] not in (1, 2, 3)
    ):
        raise ValueError("Unsupported learning loop configuration")
    if config["version"] == 2 or "storage" in config:
        config["storage"] = validate_storage_budget(config.get("storage"), path.parent)
        if "storage_monitor" in config:
            config["storage_monitor"] = validate_monitor(config["storage_monitor"])
    elif "storage_monitor" in config:
        raise ValueError("Learning storage monitor requires a storage budget")
    if asynchronous:
        sampling = config["sampling"]
        if (
            not isinstance(sampling, dict)
            or set(sampling) != {"runtime", "seconds", "max_updates"}
            or type(sampling["seconds"]) not in (int, float)
            or not math.isfinite(sampling["seconds"])
            or not 0.1 <= sampling["seconds"] <= 600
            or type(sampling["max_updates"]) is not int
            or not 1 <= sampling["max_updates"] <= 1000
        ):
            raise ValueError("Learning async sampling requires finite duration and updates")
        runtime = _sampling_runtime(sampling)
        sampling["runtime"] = json.loads(
            encode({**asdict(runtime), "pixels": runtime.pixels.metadata()})
        )
    if "acquisition_retry" in config:
        retry = config["acquisition_retry"]
        if (
            not isinstance(retry, dict)
            or set(retry) != {"max_retries", "delay_seconds"}
            or type(retry["max_retries"]) is not int
            or not 0 <= retry["max_retries"] <= 3
            or type(retry["delay_seconds"]) not in (int, float)
            or not math.isfinite(retry["delay_seconds"])
            or not 0 <= retry["delay_seconds"] <= 5
        ):
            raise ValueError("Learning acquisition retries require explicit finite bounds")
    if "sampling_retry" in config:
        retry = config["sampling_retry"]
        if (
            not isinstance(retry, dict)
            or set(retry) != {"max_retries"}
            or type(retry["max_retries"]) is not int
            or not 0 <= retry["max_retries"] <= 3
        ):
            raise ValueError("Learning sampling retries require an integer bound from 0 to 3")
    for key, low, high in (
        ("rounds", 1, 10),
        ("seed", 0, 2**32 - 11),
        *(() if asynchronous else (("steps_per_attempt", 1, 1000),)),
    ):
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError("Invalid learning loop bound: " + key)
    seconds = config["evaluation_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0.1 <= seconds <= 600:
        raise ValueError("Learning evaluation requires a finite duration")
    if set(config["store"]) != {"directory", "revision"}:
        raise ValueError("Learning requires an expected persistent store revision")
    config["store"]["directory"] = str((path.parent / config["store"]["directory"]).resolve())
    for key in ("registry", "recording", "task", "reward"):
        config[key] = str((path.parent / config[key]).resolve())
    return config


def _sampling_runtime(sampling: dict[str, Any]) -> RealtimeConfig:
    settings = dict(sampling["runtime"])
    settings["pixels"] = PixelContract.from_metadata(settings["pixels"])
    settings["action_offsets_ms"] = tuple(settings["action_offsets_ms"])
    return RealtimeConfig(**settings)


def _qualification(history: dict[str, Any]) -> dict[str, Any]:
    proof = history["qualification"]
    path = Path(proof["comparison_file"])
    if _sha(path) != proof["comparison_sha256"]:
        raise ValueError("Default qualification changed")
    comparison = json.loads(read_bounded(path, 1024**2))
    binding = _input(path.parent, comparison[proof["side"]])
    return {
        "batch": str(binding.batch_dir),
        "batch_sha256": binding.batch_sha256,
        "ledger": str(binding.ledger_file),
        "ledger_sha256": binding.ledger_sha256,
    }


def _learner(path: Path, expected: str) -> dict[str, Any]:
    return {
        "directory": str(path.resolve()),
        "sha256": expected,
        **validate_sac_candidate(path, expected),
    }


class _Loop:
    def __init__(self, root: Path, config: dict[str, Any], environment: LearningEnvironment):
        self.root, self.config, self.environment = root, config, environment
        self.monitor: StorageMonitor | None = None
        self.requested_stop_reason: str | None = None
        self.stop_lock = Lock()
        self.store = Path(config["store"]["directory"])
        self.registry = Path(config["registry"])
        self.stages = StageHistory(root)
        self.state: dict[str, Any] = {
            "version": 1,
            "scope": "synthetic_development_only",
            "rounds": [],
            "rounds_completed": 0,
            "eligible_transitions": 0,
            "learner_updates": 0,
            "phase": "initializing",
            "stop_reason": "interface_error",
            "resources_released": False,
            "commands_sent_to_game": False,
            "real_driving_validated": False,
            "driving_improvement_validated": False,
            "stages": [],
            "interruptions": [],
            "child_resources_released": True,
        }

    def save(self, phase: str, *, final: bool = False) -> None:
        self.state["phase"] = phase
        self.state["stages"] = self.stages.record(
            phase, time.perf_counter_ns(), len(self.state["rounds"]) - 1, final=final
        )
        atomic_json(self.root / "state.json", self.state)

    def capacity(self, phase: str) -> bool:
        if "storage" not in self.config:
            return True
        decision = capacity_decision(
            self.root, _sha(self.root / "state.json"), self.config["storage"], phase
        )
        stopped = (self.root / "stop.request").exists()
        decision["stop_requested"] = stopped
        self.state.setdefault("storage_checks", []).append(decision)
        if stopped:
            self.state["stop_reason"] = "stop_requested"
        elif not decision["admitted"]:
            self.state["stop_reason"] = "storage_budget_exhausted"
        self.save("capacity_checked")
        return bool(decision["admitted"] and not stopped)

    def initialize(self) -> bool:
        from fh5.experiment import run_experiment

        history = run_experiment(CandidateHistory(self.store, limit=0)).summary["candidate_store"]
        if history["revision"] != self.config["store"]["revision"]:
            raise ValueError("Learning store revision changed before initialization")
        self.state["store_revision"] = history["revision"]
        self.state["incumbent"] = _qualification(history)
        self.state["source_files"] = {
            self.config[key]: _sha(Path(self.config[key]))
            for key in ("recording", "task", "reward")
        }
        self.state["config_sha256"] = _sha(self.root / "config.json")
        self.state["initialized"] = False
        self.save("initializing")
        if not self.capacity("initialize"):
            return False
        for role in ("default", "explorer"):
            saved = history[role]
            target = self.root / "initial" / role
            run_experiment(
                CandidateRestore(
                    self.store / saved["archive"],
                    target,
                    saved["archive_sha256"],
                    "Start unattended loop",
                )
            )
            self.state[role] = _learner(target, saved["model_sha256"])
        self.state["latest_learner"] = self.state["explorer"]
        self.state["initialized"] = True
        self.save("ready")
        return True

    def acquire(
        self, kind: Literal["sampling", "evaluation"], number: int, opening: Callable[[], _Lease]
    ) -> _Lease | None:
        retry = self.config.get("acquisition_retry", {"max_retries": 0, "delay_seconds": 0})
        for attempt in range(1, retry["max_retries"] + 2):
            if self.stopped():
                self.state["stop_reason"] = self.stopping_reason()
                return None
            self.save("opening_" + kind)
            try:
                source = opening()
            except LearningUnavailable as error:
                safe_to_retry = error.resources_released is True
                self.state["child_resources_released"] &= safe_to_retry
                self.state.setdefault("acquisition_failures", []).append(
                    {
                        "kind": kind,
                        "round": number,
                        "continuation": len(self.state["interruptions"]),
                        "attempt": attempt,
                        "error": str(error),
                        "resources_released": safe_to_retry,
                        "at_ns": time.perf_counter_ns(),
                    }
                )
                self.save("acquisition_failed")
                if not safe_to_retry:
                    self.state["stop_reason"] = "release_fault"
                    return None
                if self.stopped():
                    self.state["stop_reason"] = self.stopping_reason()
                    return None
                if attempt > retry["max_retries"]:
                    self.state["stop_reason"] = "acquisition_retries_exhausted"
                    return None
                self.save("waiting_for_interface")
                deadline = time.monotonic() + retry["delay_seconds"]
                while time.monotonic() < deadline:
                    if self.stopped():
                        self.state["stop_reason"] = self.stopping_reason()
                        return None
                    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
                self.verify()
                learner = self.state["latest_learner"]
                if _learner(Path(learner["directory"]), learner["sha256"]) != learner:
                    raise ValueError("Learning acquisition checkpoint changed")
                if kind == "evaluation":
                    root = self.root / f"round-{number:03d}"
                    prepared = self.state["rounds"][number]["evaluation_prepared"]
                    if _sha(root / "evaluation.json") != prepared["config_sha256"]:
                        raise ValueError("Prepared evaluation configuration changed")
                    read_evaluation_batch(root / "batch", prepared["batch_sha256"])
                if not self.capacity(kind):
                    return None
            else:
                if self.stopped():
                    previous = self.state["child_resources_released"]
                    self.state["child_resources_released"] = False
                    released = source.close()
                    self.state.setdefault("acquisition_closes", []).append(
                        {"kind": kind, "round": number, "release": released}
                    )
                    self.state["child_resources_released"] = previous and (
                        released.get("resources_released") is True
                    )
                    self.state["stop_reason"] = self.stopping_reason()
                    return None
                return source
        return None

    def stopping_reason(self) -> str | None:
        with self.stop_lock:
            if self.requested_stop_reason is not None:
                return self.requested_stop_reason
        # Query outside the lock so a delayed filesystem read cannot block
        # another observer from latching a stop or reading an existing one.
        reason = (
            "stop_requested"
            if (self.root / "stop.request").exists()
            else self.monitor.reason()
            if self.monitor is not None
            else None
        )
        with self.stop_lock:
            if self.requested_stop_reason is None and reason is not None:
                self.requested_stop_reason = reason
            return self.requested_stop_reason

    def stopped(self) -> bool:
        return self.stopping_reason() is not None

    def restore(self, expected: str) -> bool:
        path = self.root / "state.json"
        if _sha(path) != expected:
            raise ValueError("Learning continuation state changed")
        state = read_json(path, expected_sha256=expected)
        if (
            state["version"] != 1
            or state["scope"] != "synthetic_development_only"
            or _sha(self.root / "config.json") != state["config_sha256"]
        ):
            raise ValueError("Learning continuation configuration changed")
        self.state = state
        self.stages = StageHistory(self.root, state["stages"])
        if state["child_resources_released"] is not True:
            raise ValueError("Learning continuation refuses unreleased child resources")
        if any(not row["resources_released"] for row in state.get("storage_monitors", [])):
            raise ValueError("Learning continuation refuses an unreleased storage monitor")
        if state.get("initialized", True) is False:
            if (
                "storage" not in self.config
                or state["phase"] != "stopped"
                or state["stop_reason"]
                not in ("storage_budget_exhausted", "stop_requested", "interface_error")
                or state["rounds"]
                or any(role in state for role in ("default", "explorer", "latest_learner"))
                or (self.root / "initial").exists()
            ):
                raise ValueError("Incomplete learning initialization cannot be resumed")
            self.verify()
            state["interruptions"].append(
                {
                    "stop_reason": state["stop_reason"],
                    "phase": "initializing",
                    "error": state.get("error"),
                }
            )
            state.pop("error", None)
            state["resources_released"] = False
            # Authenticate first; the caller enables failure publication before new work.
            return True
        for key in ("default", "explorer", "latest_learner"):
            saved = state[key]
            if _learner(Path(saved["directory"]), saved["sha256"]) != saved:
                raise ValueError("Learning continuation checkpoint changed")
        for number, row in enumerate(state["rounds"]):
            round_dir = self.root / f"round-{number:03d}"
            if (
                not row["complete"]
                and row.get("learner_updates", 0) < self.update_budget(row)
                and (
                    any(
                        key in row
                        for key in ("evaluation_prepared", "candidate_evaluation", "evaluation_run")
                    )
                    or any(
                        (round_dir / name).exists()
                        for name in ("evaluation.json", "batch", "evaluation")
                    )
                )
            ):
                raise ValueError(
                    "Incomplete updates already have frozen evaluation; migration required"
                )
            for archived in row.get("sampling_history", []):
                verify_archived_sampling(archived)
            for binding in sampling_bindings(row):
                if _sha(Path(binding["directory"]) / "summary.json") != binding["summary_sha256"]:
                    raise ValueError("Retained sampling result changed")
                sampled = read_json(Path(binding["directory"]) / "summary.json")
                for attempt in sampled["attempts"]:
                    verify_sampling_sources(attempt.get("source_assets", {}))
            if "candidate_evaluation" in row:
                _input(self.root, row["candidate_evaluation"]).verify()
            if row.get("update_segments"):
                self.update_progress(number, row)
        self.reconcile_sampling()
        self.reconcile_updates()
        self.reconcile_evaluation()
        self.reconcile_commit()
        self.verify()
        clean_stop = state["phase"] == "stopped"
        if not clean_stop or state["stop_reason"] != "budget_completed":
            state["interruptions"].append(
                {
                    "stop_reason": state["stop_reason"] if clean_stop else "unclean_exit",
                    "error": state.get("error") if clean_stop else None,
                    "phase": self.stages.before_stop(state["phase"])
                    if clean_stop
                    else state["phase"],
                }
            )
        state.pop("error", None)
        state["resources_released"] = False
        self.save("resuming")
        return True

    def verify(self) -> None:
        from fh5.experiment import run_experiment

        if any(
            _sha(Path(path)) != expected for path, expected in self.state["source_files"].items()
        ):
            raise ValueError("Frozen learning inputs changed")
        history = run_experiment(CandidateHistory(self.store, limit=0)).summary["candidate_store"]
        if history["revision"] != self.state["store_revision"]:
            raise ValueError("Learning store changed outside this loop")
        _input(self.root, self.state["incumbent"]).verify()

    def update_budget(self, row: dict[str, Any]) -> int:
        count: int = row.get("eligible_transitions", 0)
        return sampling_update_budget(
            count,
            self.config["sampling"]["max_updates"] if self.config["version"] == 3 else None,
        )

    def sampling_request(self, number: int) -> SACCycle | SACRealtimeCycle:
        attempt = self.state["rounds"][number].get("sampling_attempt", 0)
        directory = (
            self.root
            / f"round-{number:03d}"
            / ("learning" if attempt == 0 else f"learning-{attempt:03d}")
        )
        learner = self.state["latest_learner"]
        seed = (self.config["seed"] + number + attempt * self.config["rounds"]) % 2**32
        if self.config["version"] == 3:
            sampling = self.config["sampling"]
            return SACRealtimeCycle(
                Path(learner["directory"]),
                Path(self.config["recording"]),
                Path(self.config["task"]),
                Path(self.config["reward"]),
                directory,
                runtime=_sampling_runtime(sampling),
                seconds_per_attempt=sampling["seconds"],
                max_updates_per_attempt=sampling["max_updates"],
                cycles=1,
                seed=seed,
                expected_checkpoint_sha256=learner["sha256"],
            )
        return SACCycle(
            Path(learner["directory"]),
            Path(self.config["recording"]),
            Path(self.config["task"]),
            Path(self.config["reward"]),
            directory,
            cycles=1,
            steps_per_attempt=self.config["steps_per_attempt"],
            seed=seed,
            expected_checkpoint_sha256=learner["sha256"],
        )

    def retry_sampling(self, number: int, row: dict[str, Any]) -> bool:
        maximum = self.config.get("sampling_retry", {}).get("max_retries", 0)
        if maximum == 0 or "learning" not in row:
            raise ValueError("Interrupted sampling is retained; cannot overwrite its attempts")
        request = self.sampling_request(number)
        binding = row["learning"]
        if Path(binding["directory"]) != request.output_dir or not retryable_sampling(
            request, self.state["latest_learner"], binding["summary_sha256"]
        ):
            raise ValueError("Sampling retry requires a sealed released failure without updates")
        attempt = row.get("sampling_attempt", 0)
        if attempt >= maximum:
            self.state["stop_reason"] = "sampling_retries_exhausted"
            return False
        if self.stopped():
            self.state["stop_reason"] = self.stopping_reason()
            return False
        self.verify()
        learner = self.state["latest_learner"]
        if _learner(Path(learner["directory"]), learner["sha256"]) != learner:
            raise ValueError("Sampling retry learner changed")
        row.setdefault("sampling_history", []).append(archive_failed_sampling(binding))
        row["sampling_attempt"] = attempt + 1
        self.state.setdefault("recoveries", []).append(
            {
                "kind": "sampling_retry",
                "round": number,
                "attempt": attempt + 1,
                "reason": row["sampling_stop_reason"],
                "source_directory": binding["directory"],
            }
        )
        for key in ("learning", "sampling_stop_reason", "eligible_transitions", "learner_updates"):
            row.pop(key, None)
        self.save("retrying_sampling")
        return True

    def accept_sampling(
        self,
        row: dict[str, Any],
        directory: Path,
        summary: dict[str, Any],
        verified_learner: dict[str, Any] | None = None,
    ) -> None:
        row["learning"] = {
            "directory": str(directory),
            "summary_sha256": _sha(directory / "summary.json"),
        }
        row["sampling_stop_reason"] = summary["stop_reason"]
        self.state["child_resources_released"] &= summary["resources_released"]
        row["eligible_transitions"] = sum(
            a.get("eligible_transitions", 0) for a in summary["attempts"]
        )
        row["learner_updates"] = sum(a.get("learner_updates", 0) for a in summary["attempts"])
        row["update_budget"] = self.update_budget(row)
        self.state["eligible_transitions"] += row["eligible_transitions"]
        self.state["learner_updates"] += row["learner_updates"]
        if summary.get("latest_candidate"):
            candidate = directory / summary["latest_candidate"]
            self.state["latest_learner"] = verified_learner or _learner(
                candidate, _sha(candidate / "policy.json")
            )
            row["candidate_sha256"] = self.state["latest_learner"]["sha256"]

    def reconcile_sampling(self) -> None:
        rows = self.state["rounds"]
        if not rows or rows[-1]["complete"] or "learning" in rows[-1]:
            return
        row = rows[-1]
        request = self.sampling_request(len(rows) - 1)
        if not request.output_dir.exists():
            return
        if (
            self.state["phase"] != "updating"
            or self.state["rounds_completed"] != len(rows) - 1
            or not all(prior["complete"] for prior in rows[:-1])
            or row.get("sampling_checkpoint_sha256") != self.state["latest_learner"]["sha256"]
            or not self.state["child_resources_released"]
        ):
            raise ValueError("Unsealed sampling cannot be automatically acknowledged")
        summary_path = request.output_dir / "summary.json"
        summary = read_json(summary_path)
        if (
            self.config.get("sampling_retry", {}).get("max_retries", 0)
            and summary.get("stop_reason")
            in ("sampling_fault", "no_eligible_experience", "stop_requested")
            and not summary.get("latest_candidate")
        ):
            if not retryable_sampling(
                request,
                self.state["latest_learner"],
                _sha(summary_path),
                pending=True,
            ):
                raise ValueError("Pending sampling failure is not sealed, released and update-free")
            self.accept_sampling(row, request.output_dir, summary)
            self.state.setdefault("recoveries", []).append(
                {
                    "kind": "sealed_failed_sampling",
                    "round": len(rows) - 1,
                    "reason": summary["stop_reason"],
                    "source_directory": str(request.output_dir),
                }
            )
            return
        summary, learner = completed_sampling(
            request, self.state["latest_learner"], allow_stopped_updates=True
        )
        self.accept_sampling(row, request.output_dir, summary, learner)
        self.state.setdefault("recoveries", []).append(
            {
                "kind": "sealed_sampling",
                "round": len(rows) - 1,
                "candidate_sha256": learner["sha256"],
            }
        )

    def sample(self, number: int, row: dict[str, Any]) -> bool:
        from fh5.experiment import run_experiment

        request = self.sampling_request(number)
        row["sampling_checkpoint_sha256"] = self.state["latest_learner"]["sha256"]
        row["sampling_parent"] = dict(self.state["latest_learner"])
        self.save("opening_sampler")
        source = self.acquire(
            "sampling",
            number,
            lambda: self.environment.sampling(
                f"round-{number:03d}"
                + (
                    f"-sampling-{row['sampling_attempt']:03d}"
                    if row.get("sampling_attempt", 0)
                    else ""
                )
            ),
        )
        if source is None:
            return False
        if isinstance(request, SACRealtimeCycle):
            summary = run_experiment(
                request,
                sac_realtime_environment=RealtimeSamplingLease(
                    cast(SACRealtimeEnvironment, source), self.save
                ),
                sac_stop_requested=lambda _: self.stopped(),
            ).summary["sac_cycle"]
        else:
            summary = run_experiment(
                request,
                sac_environment=SamplingLease(cast(SACEnvironment, source), self.save),
                sac_stop_requested=lambda _: self.stopped(),
            ).summary["sac_cycle"]
        self.accept_sampling(row, request.output_dir, summary)
        self.save("learned")
        if not summary["resources_released"] or summary["stop_reason"] != "budget_completed":
            self.state["stop_reason"] = (
                self.stopping_reason() or "stop_requested"
                if summary["stop_reason"] == "stop_requested"
                else "sampling_" + summary["stop_reason"]
            )
            return False
        return True

    def update_history(self, number: int, row: dict[str, Any]) -> UpdateHistory:
        return UpdateHistory(self.root / f"round-{number:03d}", row.get("update_segments", []))

    def update_progress(
        self, number: int, row: dict[str, Any], *, proposed_entry: dict[str, Any] | None = None
    ) -> UpdateProgress:
        parent = row.get("sampling_parent", self.state["explorer"])
        if (
            parent["sha256"] != row["sampling_checkpoint_sha256"]
            or _learner(Path(parent["directory"]), parent["sha256"]) != parent
        ):
            raise ValueError("Stopped updates lost their original sampling parent")
        request = replace(
            self.sampling_request(number),
            checkpoint_dir=Path(parent["directory"]),
            expected_checkpoint_sha256=parent["sha256"],
        )
        segments = iter(self.update_history(number, row))
        if proposed_entry is not None:
            segments = chain(segments, (proposed_entry,))
        progress = retained_update_progress(request, parent, segments)
        if (
            progress.earned != self.update_budget(row)
            or progress.completed != row["learner_updates"]
            or progress.learner["sha256"] != row["candidate_sha256"]
        ):
            raise ValueError("Stopped updates differ from their retained progress")
        return progress

    def accept_updates(
        self,
        number: int,
        row: dict[str, Any],
        progress: UpdateProgress,
        output: Path,
        kind: Literal["resumed_updates", "sealed_updates"],
    ) -> None:
        history = self.update_history(number, row)
        learner = _learner(output, _sha(output / "policy.json"))
        learned = json.loads(read_bounded(output / "training-report.json", 128 * 1024**2))
        proposed = {
            **row,
            "sampling_parent": dict(row.get("sampling_parent", self.state["explorer"])),
            "learner_updates": row["learner_updates"] + learned["steps_completed"],
            "candidate_sha256": learner["sha256"],
        }
        entry = {"directory": str(output), "sha256": learner["sha256"]}
        checked = self.update_progress(number, proposed, proposed_entry=entry)
        proposed["update_segments"] = history.append(entry)
        row.update(proposed)
        self.state["learner_updates"] += checked.completed - progress.completed
        self.state["latest_learner"] = checked.learner
        self.state.setdefault("recoveries", []).append(
            {
                "kind": kind,
                "round": number,
                "completed": checked.completed - progress.completed,
                "remaining": checked.earned - checked.completed,
                "directory": str(output),
            }
        )

    def reconcile_updates(self) -> None:
        rows = self.state["rounds"]
        if not rows or rows[-1]["complete"]:
            return
        row, number = rows[-1], len(rows) - 1
        count = self.update_history(number, row).count
        output = self.root / f"round-{number:03d}" / f"updates-{count:03d}"
        if not output.exists():
            return
        phase = self.state["phase"]
        if phase == "stopped":
            phase = self.stages.before_stop(phase)
        if (
            phase != "resuming_updates"
            or self.state["rounds_completed"] != number
            or not all(prior["complete"] for prior in rows[:-1])
            or not self.state["child_resources_released"]
        ):
            raise ValueError("Pending updates are not bound to an interrupted continuation")
        progress = self.update_progress(number, row)
        if progress.learner != self.state["latest_learner"]:
            raise ValueError("Pending updates differ from the current learner")
        self.accept_updates(number, row, progress, output, "sealed_updates")

    def resume_updates(self, number: int, row: dict[str, Any]) -> bool:
        from fh5.experiment import run_experiment

        progress = self.update_progress(number, row)
        if progress.learner != self.state["latest_learner"]:
            raise ValueError("Stopped updates differ from the current learner")
        count = self.update_history(number, row).count
        if not self.capacity("updating"):
            return False
        output = self.root / f"round-{number:03d}" / f"updates-{count:03d}"
        self.save("resuming_updates")
        learned = run_experiment(
            SACResume(
                Path(progress.learner["directory"]),
                output,
                steps=progress.earned - progress.completed,
                expected_checkpoint_sha256=progress.learner["sha256"],
            ),
            sac_stop_requested=lambda _: self.stopped(),
        ).summary["sac_learning"]
        self.accept_updates(number, row, progress, output, "resumed_updates")
        self.save("learned")
        if learned["stop_reason"] == "stop_requested" or self.stopped():
            self.state["stop_reason"] = self.stopping_reason() or "stop_requested"
            return False
        return True

    def reconcile_evaluation(self) -> None:
        rows = self.state["rounds"]
        if not rows or rows[-1]["complete"] or "candidate_evaluation" in rows[-1]:
            return
        row, number = rows[-1], len(rows) - 1
        root = self.root / f"round-{number:03d}"
        if not (root / "evaluation").exists():
            return
        acknowledged = row.get("evaluation_completion_sha256")
        if (
            not self.state["child_resources_released"]
            or self.state["rounds_completed"] != number
            or not all(prior["complete"] for prior in rows[:-1])
            or row.get("candidate_sha256") != self.state["latest_learner"]["sha256"]
            or not acknowledged
            and (self.state["phase"] != "evaluating" or "evaluation_run" in row)
        ):
            raise ValueError("Unsealed evaluation cannot be automatically acknowledged")
        completion = root / "evaluation/completion.json"
        if acknowledged is not None and _sha(completion) != acknowledged:
            raise ValueError("Acknowledged evaluation completion changed")
        prepared = row["evaluation_prepared"]
        if _sha(root / "evaluation.json") != prepared["config_sha256"]:
            raise ValueError("Prepared evaluation configuration changed")
        basis = _input(self.root, self.state["incumbent"])
        batch, _, _ = read_evaluation_batch(root / "batch", prepared["batch_sha256"])
        if (
            evaluation_conditions(batch) != evaluation_conditions(basis.batch)
            or batch["config"]["model"]["manifest_sha256"] != self.state["latest_learner"]["sha256"]
            or batch["config"]["model"].get("kind") != "sac"
        ):
            raise ValueError("Completed evaluation belongs to a different learner or conditions")
        execution = completed_evaluation(
            EvaluationRun(
                root / "batch",
                prepared["batch_sha256"],
                basis.batch_dir / "start/event.json",
                root / "evaluation",
                self.config["evaluation_seconds"],
                self.registry,
                initial_operation="restart_ready",
            )
        )
        if "evaluation_run" in row and row["evaluation_run"] != execution:
            raise ValueError("Acknowledged evaluation summary changed")
        row["evaluation_run"] = execution
        row["evaluation_completion_sha256"] = _sha(completion)
        if acknowledged is None:
            self.state.setdefault("recoveries", []).append(
                {"kind": "sealed_evaluation", "round": number}
            )

    def review_ledger(self, root: Path, row: dict[str, Any]) -> Path:
        child = root / "evaluation"
        if "review_input" not in row:
            ledger = json.loads(read_bounded(child / "ledger.json", 4 * 1024**2))
            for entry in ledger["entries"]:
                recording = (child / entry["recording"]).resolve()
                entry["recording"] = str(recording)
                for key in ("execution", "preparation"):
                    if entry.get(key) is not None:
                        entry[key]["directory"] = str((child / entry[key]["directory"]).resolve())
                if entry["evidence"] is not None:
                    entry["evidence"]["file"] = str((child / entry["evidence"]["file"]).resolve())
                if (recording / "packets.jsonl").is_file():
                    proof = self.environment.review(recording)
                    if proof is not None:
                        entry["evidence"] = {"file": str(proof.resolve()), "sha256": _sha(proof)}
            # Freeze the complete input before publishing its derived file or report.
            row["review_input"] = ledger
            self.save("reviewing_evaluation")
        path = root / "parent-ledger.json"
        raw = encode(row["review_input"])
        if path.exists():
            if read_bounded(path, 4 * 1024**2) != raw:
                raise ValueError("Frozen parent evaluation review changed")
        else:
            atomic_json(path, row["review_input"])
        return path

    def evaluate(self, number: int, row: dict[str, Any]) -> dict[str, Any] | None:
        from fh5.experiment import run_experiment

        root = self.root / f"round-{number:03d}"
        basis = _input(self.root, self.state["incumbent"])
        config = json.loads(json.dumps(basis.batch["config"]))
        learner = self.state["latest_learner"]
        config["model"] = {
            "kind": "sac",
            "directory": learner["directory"],
            "manifest_sha256": learner["sha256"],
        }
        config["version"] = 2
        config["task"]["file"] = str(basis.batch_dir / "task.json")
        path = root / "evaluation.json"
        batch = root / "batch"
        if "evaluation_prepared" in row:
            digest = row["evaluation_prepared"]["batch_sha256"]
            if row["evaluation_prepared"]["config_sha256"] != _sha(path):
                raise ValueError("Prepared evaluation configuration changed")
        else:
            write_file(path, encode(config))
            self.save("preparing_evaluation")
            run_experiment(EvaluationPrepare(path, batch, self.registry))
            digest = _sha(batch / "batch.json")
            row["evaluation_prepared"] = {"batch_sha256": digest, "config_sha256": _sha(path)}
        frozen, _, _ = read_evaluation_batch(batch, digest)
        if evaluation_conditions(frozen) != evaluation_conditions(basis.batch):
            raise ValueError("Candidate evaluation changed frozen comparison conditions")
        execution = row.get("evaluation_run") if row.get("evaluation_completion_sha256") else None
        row.setdefault("evaluation_interrupted_by_stop", False)
        row.setdefault("evaluation_interrupted_by_resource", False)
        if execution is None:
            self.save("evaluating")
            if (root / "evaluation").exists():
                raise ValueError(
                    "Interrupted evaluation is retained; cannot overwrite its attempts"
                )
            source = self.acquire(
                "evaluation", number, lambda: self.environment.evaluation(f"round-{number:03d}")
            )
            if source is None:
                return None

            def evaluation_stopped() -> bool:
                reason = self.stopping_reason()
                if reason is not None:
                    field = (
                        "evaluation_interrupted_by_stop"
                        if reason == "stop_requested"
                        else "evaluation_interrupted_by_resource"
                    )
                    row[field] = True
                return reason is not None

            execution = run_experiment(
                EvaluationRun(
                    batch,
                    digest,
                    basis.batch_dir / "start/event.json",
                    root / "evaluation",
                    self.config["evaluation_seconds"],
                    self.registry,
                    initial_operation="restart_ready",
                ),
                evaluation_environment=EvaluationLease(
                    source,
                    self.save,
                    evaluation_stopped,
                ),
            ).summary["evaluation_run"]
            completion = root / "evaluation/completion.json"
            if completion.is_file():
                row["evaluation_completion_sha256"] = _sha(completion)
        row["evaluation_run"] = execution
        self.state["child_resources_released"] &= execution["resources_released"]
        publication = row.get("review_publication")
        if publication is not None and (
            not isinstance(publication, dict)
            or set(publication) != {"sequence", "directory"}
            or type(publication["sequence"]) is not int
            or publication["sequence"] < 0
            or not isinstance(publication["directory"], str)
        ):
            raise ValueError("Invalid parent review publication")
        attempt = 0 if publication is None else publication["sequence"]
        review_output = root / ("reviewed" if attempt == 0 else f"reviewed-{attempt:03d}")
        if publication is not None and Path(publication["directory"]) != review_output:
            raise ValueError("Parent review publication directory changed")
        while review_output.exists() or review_output.is_symlink() or review_output.is_junction():
            attempt += 1
            review_output = root / f"reviewed-{attempt:03d}"
        # Older directories remain independently readable on disk. Preserve a
        # legacy list if present, but never grow it with later publications.
        row["review_publication"] = {"directory": str(review_output), "sequence": attempt}
        self.save("reviewing_evaluation")
        ledger_file = self.review_ledger(root, row)
        # Never derive legality from the model or a successful execution summary.
        row["evaluation"] = run_experiment(
            EvaluationReview(
                batch,
                ledger_file,
                review_output,
                self.registry,
            )
        ).summary["evaluation"]
        binding = {
            "batch": str(batch.resolve()),
            "batch_sha256": digest,
            "ledger": str(ledger_file.resolve()),
            "ledger_sha256": _sha(ledger_file),
        }
        row["candidate_evaluation"] = binding
        self.save("evaluated")
        if not execution["resources_released"]:
            raise ValueError("Evaluation lease did not release its resources")
        return binding

    def retention_files(self, number: int, binding: dict[str, Any]) -> dict[Path, bytes]:
        root = self.root / f"round-{number:03d}"
        comparison = root / "comparison.json"
        payload = encode({"version": 1, "incumbent": self.state["incumbent"], "candidate": binding})
        return {
            comparison: payload,
            root / "retain.json": encode(
                {
                    "version": 1,
                    "comparison": {
                        "file": str(comparison.resolve()),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    },
                    "checkpoints": {
                        "incumbent": self.state["default"]["directory"],
                        "candidate": self.state["latest_learner"]["directory"],
                    },
                }
            ),
        }

    def accept_selection(
        self, row: dict[str, Any], binding: dict[str, Any], saved: dict[str, Any]
    ) -> None:
        self.state["store_revision"] = saved["revision"]
        self.state["explorer"] = self.state["latest_learner"]
        if saved["selection"] == "prefer_candidate_locally":
            self.state["default"] = self.state["latest_learner"]
            self.state["incumbent"] = binding
        row.update(
            selection=saved["selection"],
            selection_reasons=saved["reasons"],
            store_revision=saved["revision"],
            complete=True,
        )
        self.state["rounds_completed"] += 1

    def reconcile_commit(self) -> None:
        from fh5.experiment import run_experiment

        history = run_experiment(CandidateHistory(self.store, limit=0)).summary["candidate_store"]
        if history["revision"] == self.state["store_revision"]:
            return
        rows = self.state["rounds"]
        if (
            self.state["phase"] != "saving_versions"
            or not rows
            or rows[-1]["complete"]
            or self.state["rounds_completed"] != len(rows) - 1
            or not all(row["complete"] for row in rows[:-1])
            or history["parent"] != self.state["store_revision"]
            or history["operation"] != "selection"
            or history["selection"] not in ("prefer_candidate_locally", "retain_incumbent")
        ):
            raise ValueError("Learning store changed outside the pending candidate commit")
        row = rows[-1]
        binding = row["candidate_evaluation"]
        number = len(rows) - 1
        files = self.retention_files(number, binding)
        root = self.root / f"round-{number:03d}"
        expected = {
            "comparison_file": str(root / "comparison.json"),
            "comparison_sha256": hashlib.sha256(files[root / "comparison.json"]).hexdigest(),
            "side": "candidate"
            if history["selection"] == "prefer_candidate_locally"
            else "incumbent",
        }
        if (
            any(read_bounded(path, 1024**2) != raw for path, raw in files.items())
            or read_bounded(self.store / history["request"], 1024**2) != files[root / "retain.json"]
            or history["qualification"] != expected
            or row["candidate_sha256"] != self.state["latest_learner"]["sha256"]
            or not self.state["child_resources_released"]
        ):
            raise ValueError("Pending candidate commit differs from this learning round")
        chosen = (
            self.state["latest_learner"]
            if history["selection"] == "prefer_candidate_locally"
            else self.state["default"]
        )
        # Authenticate retained complete archives, not just history's model names.
        with TemporaryDirectory(prefix="commit-recovery-", dir=self.root) as temporary:
            for role, learner in (("default", chosen), ("explorer", self.state["latest_learner"])):
                archive = history[role]
                restored = run_experiment(
                    CandidateRestore(
                        self.store / archive["archive"],
                        Path(temporary) / role,
                        archive["archive_sha256"],
                        "Verify committed learning selection after process exit",
                    )
                ).summary["candidate_restore"]
                if (
                    archive["model_sha256"] != learner["sha256"]
                    or restored["checkpoint_sha256"] != learner["sha256"]
                    or restored["learner_state_sha256"] != learner["learner_state_sha256"]
                ):
                    raise ValueError("Pending candidate archive differs from the saved learner")
        if _qualification(history) != (
            binding if expected["side"] == "candidate" else self.state["incumbent"]
        ):
            raise ValueError("Pending candidate qualification changed")
        self.state.setdefault("recoveries", []).append(
            {
                "kind": "candidate_commit",
                "round": number,
                "previous_revision": self.state["store_revision"],
                "committed_revision": history["revision"],
            }
        )
        self.accept_selection(row, binding, history)

    def retain(self, number: int, row: dict[str, Any], binding: dict[str, Any]) -> None:
        from fh5.experiment import run_experiment

        for path, raw in self.retention_files(number, binding).items():
            if path.exists():
                if read_bounded(path, 1024**2) != raw:
                    raise ValueError("Prepared candidate selection changed")
            else:
                write_file(path, raw)
        self.save("saving_versions")
        saved = run_experiment(
            CandidateRecord(
                self.root / f"round-{number:03d}" / "retain.json",
                self.store,
                self.state["store_revision"],
                self.registry,
            )
        ).summary["candidate_store"]
        self.accept_selection(row, binding, saved)
        self.save("ready")

    def run(self) -> None:
        for number in range(self.config["rounds"]):
            if self.stopped():
                self.state["stop_reason"] = self.stopping_reason()
                return
            self.verify()
            if number < len(self.state["rounds"]):
                row = self.state["rounds"][number]
                if row["complete"]:
                    continue
            else:
                row = {"number": number, "complete": False}
                self.state["rounds"].append(row)
            while "candidate_sha256" not in row:
                if self.sampling_request(number).output_dir.exists():
                    if not self.retry_sampling(number, row):
                        return
                if not self.capacity("sampling"):
                    return
                if not self.sample(number, row):
                    if (
                        self.stopped()
                        or row.get("sampling_stop_reason") == "stop_requested"
                        or "learning" not in row
                        or not self.config.get("sampling_retry", {}).get("max_retries", 0)
                    ):
                        return
            if self.stopped():
                self.state["stop_reason"] = self.stopping_reason()
                return
            if row["learner_updates"] < self.update_budget(row):
                if not self.resume_updates(number, row):
                    return
            binding = row.get("candidate_evaluation")
            if binding is None:
                if not self.capacity("evaluation"):
                    return
                binding = self.evaluate(number, row)
                if binding is None:
                    return
            if self.stopped():
                self.state["stop_reason"] = self.stopping_reason()
                return
            self.verify()
            if not self.capacity("retention"):
                return
            self.retain(number, row, binding)
            if self.stopped():
                self.state["stop_reason"] = self.stopping_reason()
                return
            if row["evaluation_run"]["stop_reason"] != "plan_complete" and not (
                row.get("evaluation_interrupted_by_stop")
                or row.get("evaluation_interrupted_by_resource")
            ):
                self.state["stop_reason"] = "evaluation_" + row["evaluation_run"]["stop_reason"]
                return
        self.state["stop_reason"] = "budget_completed"


def run_learning_loop(
    request: LearningLoop | LearningContinue, environment: LearningEnvironment
) -> RunResult:
    from fh5.experiment import RunResult

    continuing = isinstance(request, LearningContinue)
    if isinstance(request, LearningContinue):
        root = request.run_dir.resolve()
        config = _configuration(root / "config.json")
    else:
        root = request.output_dir.resolve()
        config = _configuration(request.config_file)
    if not continuing:
        if root.exists():
            raise FileExistsError(root)
        if root.is_relative_to(Path(config["store"]["directory"])):
            raise ValueError("Learning output must be outside its version store")
        root.mkdir(parents=True)
    loop = _Loop(root, config, environment)
    lease = CollectionLease(Path(config["store"]["directory"]) / "learning.lock")
    began = time.monotonic()
    publish = not continuing
    try:
        if environment.source_kind != "synthetic":
            raise ValueError("Learning loop currently requires synthetic external I/O")
        lease.acquire()
        if isinstance(request, LearningContinue):
            ready = loop.restore(request.expected_state_sha256)
            publish = True
            if loop.state.get("initialized", True) is False:
                ready = loop.initialize()
        else:
            write_file(root / "config.json", encode(config))
            ready = loop.initialize()
        if ready:
            if "storage_monitor" in config:
                loop.monitor = StorageMonitor(root, config["storage"], config["storage_monitor"])
                loop.monitor.start()
            loop.run()
    except (Exception, KeyboardInterrupt) as error:
        if not publish:
            raise
        if isinstance(error, RejectedLease):
            loop.state["rejected_lease"] = error.released
            loop.state["child_resources_released"] &= (
                error.released.get("resources_released") is True
            )
        loop.state.update(
            stop_reason="user_stop" if isinstance(error, KeyboardInterrupt) else "interface_error",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        try:
            loop.state["environment"] = environment.close()
            loop.state["resources_released"] = (
                loop.state["environment"].get("resources_released") is True
            ) and loop.state["child_resources_released"]
        except Exception as error:
            loop.state["release_error"] = str(error)
        if loop.monitor is not None:
            monitored = loop.monitor.close()
            loop.state.setdefault("storage_monitors", []).append(monitored)
            loop.state["resources_released"] &= monitored["resources_released"]
        lease.close()
        if not loop.state["resources_released"]:
            loop.state["stop_reason"] = "release_fault"
        try:
            if publish:
                loop.state["elapsed_seconds"] = time.monotonic() - began
                loop.save("stopped", final=True)
        finally:
            loop.stages.close()
    atomic_json(root / "summary.json", loop.state)
    report = optional_report(
        root / "report.html",
        "合成自主学习循环（循环运行不等于驾驶能力提升）",
        loop.state,
        fallback=root / "summary.json",
    )
    return RunResult({}, [], [], {"learning_loop": loop.state}, report)
