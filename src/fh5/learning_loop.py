"""Serial, bounded sample/learn/evaluate/retain orchestration over synthetic I/O."""

from __future__ import annotations

import hashlib
import html
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

from fh5.candidate_archive import CandidateRestore
from fh5.candidate_selection import _input, evaluation_conditions
from fh5.candidate_store import CandidateHistory, CandidateRecord
from fh5.collection_lease import CollectionLease
from fh5.collection_store import atomic_json, encode, read_bounded, write_file
from fh5.evaluation import EvaluationPrepare, EvaluationReview, read_evaluation_batch
from fh5.evaluation_completion import completed_evaluation
from fh5.evaluation_run import EvaluationEnvironment, EvaluationRun
from fh5.learning_capacity import capacity_decision, validate_storage_budget
from fh5.learning_io import EvaluationLease, LearningUnavailable, RejectedLease, SamplingLease
from fh5.learning_monitor import StorageMonitor, validate_monitor
from fh5.learning_recovery import completed_sampling
from fh5.sac_cycle import SACCycle, SACEnvironment
from fh5.sac_learning import validate_sac_candidate
from fh5.sampling_evidence import verify_sampling_sources

if TYPE_CHECKING:
    from fh5.experiment import RunResult

_Lease = TypeVar("_Lease", SACEnvironment, EvaluationEnvironment)


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

    def sampling(self, identity: str) -> SACEnvironment: ...
    def evaluation(self, identity: str) -> EvaluationEnvironment: ...
    def review(self, recording_dir: Path) -> Path | None: ...
    def close(self) -> dict[str, Any]: ...


def _sha(path: Path, limit: int = 4 * 1024**2) -> str:
    return hashlib.sha256(read_bounded(path, limit)).hexdigest()


def _configuration(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = json.loads(read_bounded(path, 1024**2))
    if (
        set(config)
        - {"acquisition_retry"}
        - ({"storage", "storage_monitor"} if config.get("version") == 2 else set())
        != {
            "version",
            "store",
            "registry",
            "recording",
            "task",
            "reward",
            "rounds",
            "steps_per_attempt",
            "evaluation_seconds",
            "seed",
        }
        or type(config["version"]) is not int
        or config["version"] not in (1, 2)
    ):
        raise ValueError("Unsupported learning loop configuration")
    if config["version"] == 2:
        config["storage"] = validate_storage_budget(config.get("storage"), path.parent)
        if "storage_monitor" in config:
            config["storage_monitor"] = validate_monitor(config["storage_monitor"])
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
    for key, low, high in (
        ("rounds", 1, 10),
        ("steps_per_attempt", 1, 1000),
        ("seed", 0, 2**32 - 11),
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

    def save(self, phase: str) -> None:
        if len(self.state["stages"]) >= 1000:
            raise ValueError("Learning session exceeds 1000 state transitions")
        self.state["phase"] = phase
        self.state["stages"].append(
            {
                "phase": phase,
                "at_ns": time.perf_counter_ns(),
                "round": len(self.state["rounds"]) - 1,
            }
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

        history = run_experiment(CandidateHistory(self.store)).summary["candidate_store"]
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
        state = json.loads(read_bounded(path, 4 * 1024**2))
        if (
            state["version"] != 1
            or state["scope"] != "synthetic_development_only"
            or _sha(self.root / "config.json") != state["config_sha256"]
        ):
            raise ValueError("Learning continuation configuration changed")
        self.state = state
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
        for row in state["rounds"]:
            if "learning" in row:
                binding = row["learning"]
                if _sha(Path(binding["directory"]) / "summary.json") != binding["summary_sha256"]:
                    raise ValueError("Retained sampling result changed")
                sampled = json.loads(
                    read_bounded(Path(binding["directory"]) / "summary.json", 4 * 1024**2)
                )
                for attempt in sampled["attempts"]:
                    verify_sampling_sources(attempt.get("source_assets", {}))
            if "candidate_evaluation" in row:
                _input(self.root, row["candidate_evaluation"]).verify()
        self.reconcile_sampling()
        self.reconcile_evaluation()
        self.reconcile_commit()
        self.verify()
        clean_stop = state["phase"] == "stopped"
        if not clean_stop or state["stop_reason"] != "budget_completed":
            state["interruptions"].append(
                {
                    "stop_reason": state["stop_reason"] if clean_stop else "unclean_exit",
                    "error": state.get("error") if clean_stop else None,
                    "phase": state["stages"][-2]["phase"]
                    if clean_stop and len(state["stages"]) > 1
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
        history = run_experiment(CandidateHistory(self.store)).summary["candidate_store"]
        if history["revision"] != self.state["store_revision"]:
            raise ValueError("Learning store changed outside this loop")
        _input(self.root, self.state["incumbent"]).verify()

    def sampling_request(self, number: int) -> SACCycle:
        directory = self.root / f"round-{number:03d}" / "learning"
        learner = self.state["latest_learner"]
        return SACCycle(
            Path(learner["directory"]),
            Path(self.config["recording"]),
            Path(self.config["task"]),
            Path(self.config["reward"]),
            directory,
            cycles=1,
            steps_per_attempt=self.config["steps_per_attempt"],
            seed=self.config["seed"] + number,
            expected_checkpoint_sha256=learner["sha256"],
        )

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
        summary, learner = completed_sampling(request, self.state["latest_learner"])
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
        self.save("opening_sampler")
        source = self.acquire(
            "sampling", number, lambda: self.environment.sampling(f"round-{number:03d}")
        )
        if source is None:
            return False
        summary = run_experiment(
            request,
            sac_environment=SamplingLease(source, self.save),
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
        for attempt in range(10):
            review_output = root / ("reviewed" if attempt == 0 else f"reviewed-{attempt:03d}")
            if not review_output.exists():
                break
        else:
            raise ValueError("Parent evaluation review exceeds 10 retained publications")
        directories = row.setdefault("review_directories", [])
        if str(review_output) not in directories:
            directories.append(str(review_output))
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

        history = run_experiment(CandidateHistory(self.store)).summary["candidate_store"]
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
            if "candidate_sha256" not in row:
                if (self.root / f"round-{number:03d}" / "learning").exists():
                    raise ValueError(
                        "Interrupted sampling is retained; cannot overwrite its attempts"
                    )
                if not self.capacity("sampling"):
                    return
                if not self.sample(number, row):
                    return
            if self.stopped():
                self.state["stop_reason"] = self.stopping_reason()
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
        if publish:
            loop.state["elapsed_seconds"] = time.monotonic() - began
            loop.save("stopped")
    report = root / "report.html"
    atomic_json(root / "summary.json", loop.state)
    report.write_text(
        (
            '<!doctype html><meta charset="utf-8"><h1>合成自主学习循环</h1><p>循环运行不等于驾驶能力提升。</p><pre>'
            + html.escape(json.dumps(loop.state, ensure_ascii=False, indent=2))
            + "</pre>"
        ),
        encoding="utf-8",
    )
    return RunResult({}, [], [], {"learning_loop": loop.state}, report)
