"""Finite sample/learn alternation over declared, separately qualified external I/O."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fh5.artifacts.document import replay_document
from fh5.artifacts.io import VerifiedFile, encode, read_bounded, read_json, write_file
from fh5.driving.realtime.model import RealtimeConfig, RealtimeRun
from fh5.learning.diagnostics import prediction_identity
from fh5.learning.loop.runtime import preserve_torch_state
from fh5.learning.sac.actor import FrozenSAC
from fh5.learning.sac.realtime_experience import SACRealtimePrepare
from fh5.learning.sac.realtime_sampler import (
    SACRealtimeEnvironment,
    SACRealtimeStart,
    sample_realtime_attempt,
)
from fh5.learning.sac.replay import SACReplayPrepare
from fh5.learning.sac.sampler import (
    SACEnvironment,
    SACSample,
    SACStart,
    sample_attempt,
)
from fh5.learning.sac.sampling_actor import SACSamplingActor
from fh5.learning.sac.training import SACResume
from fh5.learning.sampling_evidence import seal_sampling_attempt, verify_sampling_sources
from fh5.reporting.presentation import optional_report

if TYPE_CHECKING:
    from fh5.result import RunResult

__all__ = [
    "SACCycle",
    "SACRealtimeCycle",
    "SACEnvironment",
    "SACSample",
    "SACStart",
    "run_sac_cycle",
    "sampling_update_budget",
]


def sampling_update_budget(eligible: int, maximum: int | None = None) -> int:
    """Grant at most one update per new transition, subject to the frozen cap."""
    if type(eligible) is not int or eligible < 0:
        raise ValueError("Sampling transition count must be a nonnegative integer")
    return eligible if maximum is None else min(eligible, maximum)


@dataclass(frozen=True)
class SACCycle:
    checkpoint_dir: Path
    recording_config_file: Path
    task_file: Path
    reward_file: Path
    output_dir: Path
    cycles: int = 2
    steps_per_attempt: int = 16
    seed: int = 19
    expected_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.cycles) is not int
            or not 1 <= self.cycles <= 10
            or type(self.steps_per_attempt) is not int
            or not 1 <= self.steps_per_attempt <= 1000
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("SAC cycle requires finite attempts, steps and sampler seed")


@dataclass(frozen=True)
class SACRealtimeCycle:
    checkpoint_dir: Path
    recording_config_file: Path
    task_file: Path
    reward_file: Path
    output_dir: Path
    runtime: RealtimeConfig
    seconds_per_attempt: float = 1.0
    cycles: int = 2
    max_updates_per_attempt: int = 8
    seed: int = 19
    expected_checkpoint_sha256: str | None = None
    live: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.cycles) is not int
            or self.cycles < 1
            or type(self.max_updates_per_attempt) is not int
            or self.max_updates_per_attempt < 1
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("SAC realtime cycle requires finite attempts, updates and seed")
        RealtimeRun(self.output_dir, self.runtime, seconds=self.seconds_per_attempt, live=self.live)


def run_sac_cycle(
    request: SACCycle | SACRealtimeCycle,
    environment: SACEnvironment | SACRealtimeEnvironment,
    stop_requested: Callable[[int], bool] | None = None,
) -> RunResult:
    from fh5.experiment import run_experiment
    from fh5.result import RunResult

    root = request.output_dir

    def stopped() -> bool:
        return (root / "stop.request").exists() or bool(stop_requested and stop_requested(0))

    if root.exists() or root.resolve().is_relative_to(request.checkpoint_dir.resolve()):
        raise ValueError("SAC cycle needs a fresh output outside its frozen checkpoint")
    summary: dict[str, Any] = {
        "attempts": [],
        "source_kind": environment.source_kind,
        "commands_sent_to_game": False,
        "real_driving_validated": False,
        "default_changed": False,
        "stop_reason": "interface_error",
        "resources_released": False,
    }
    torch = importlib.import_module("torch")
    root.mkdir(parents=True)
    try:
        if isinstance(request, SACRealtimeCycle):
            if environment.source_kind not in ("synthetic", "native") or request.live != (
                environment.source_kind == "native"
            ):
                raise ValueError("SAC sampling source and explicit live opt-in disagree")
        elif environment.source_kind != "synthetic":
            raise ValueError("Synchronous SAC cycle requires synthetic external I/O")
        checkpoint = request.checkpoint_dir
        expected_sampling_sha = request.expected_checkpoint_sha256
        files = (request.recording_config_file, request.task_file, request.reward_file)
        protocol_bytes = [read_bounded(p, 1024**2) for p in files]
        if json.loads(protocol_bytes[0])["control_source"] != "policy":
            raise ValueError("SAC sampler requires policy recording attribution")
        policy = read_json(checkpoint / "policy.json")
        source = VerifiedFile(checkpoint / "experience/replay.json", policy["replay_sha256"])
        with replay_document(source) as old_replay:
            for kind, raw in zip(("task", "reward"), protocol_bytes[1:]):
                if hashlib.sha256(raw).hexdigest() != old_replay["source_hashes"][kind]:
                    raise ValueError("Cycle protocol differs from the learner: " + kind)
        write_file(
            root / "protocol.json",
            encode(
                {
                    "source_kind": environment.source_kind,
                    "cycles": request.cycles,
                    **(
                        {
                            "runtime": asdict(request.runtime),
                            "seconds_per_attempt": request.seconds_per_attempt,
                            "max_updates_per_attempt": request.max_updates_per_attempt,
                            **({"live": True} if request.live else {}),
                        }
                        if isinstance(request, SACRealtimeCycle)
                        else {"steps_per_attempt": request.steps_per_attempt}
                    ),
                    "seed": request.seed,
                    "update_ratio": "at most one critic update per newly accepted transition",
                    "protocol_files": {
                        str(p): hashlib.sha256(raw).hexdigest()
                        for p, raw in zip(files, protocol_bytes)
                    },
                }
            ),
        )
        with preserve_torch_state(torch):
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True)
            for number in range(request.cycles):
                if stopped():
                    summary["stop_reason"] = "stop_requested"
                    break
                if any(read_bounded(p, 1024**2) != raw for p, raw in zip(files, protocol_bytes)):
                    raise ValueError("Frozen cycle protocol changed")
                actor = FrozenSAC(torch, checkpoint)
                if expected_sampling_sha is not None and actor.sha != expected_sampling_sha:
                    raise ValueError("Sampling candidate changed after its verified handoff")
                if (
                    isinstance(request, SACCycle)
                    and (request.steps_per_attempt + 1)
                    * len(actor.pixels.history_offsets_ms)
                    * actor.pixels.size[0]
                    * actor.pixels.size[1]
                    * 3
                    > 512 * 1024**2
                ):
                    raise ValueError("SAC attempt exceeds 512 MiB numerical frame budget")
                if stopped():
                    summary["stop_reason"] = "stop_requested"
                    break
                attempt_dir = root / f"attempt-{number:03d}"
                if isinstance(request, SACRealtimeCycle):
                    runtime = request.runtime
                    if (
                        runtime.pixels != actor.pixels
                        or runtime.action_offsets_ms != (200, 100, 0)
                        or actor.bc.original_contract["actor_shape"]
                        != {
                            "action_count": len(runtime.action_offsets_ms),
                            "reference_count": runtime.reference_count,
                        }
                        or any(
                            getattr(runtime, key) != getattr(actor.bounds, key)
                            for key in ("max_steer", "max_throttle", "max_brake")
                        )
                    ):
                        raise ValueError("Realtime execution contract differs from the learner")
                    result, review = sample_realtime_attempt(
                        cast(SACRealtimeEnvironment, environment),
                        SACRealtimeStart(
                            f"sac-attempt-{number}",
                            RealtimeRun(
                                attempt_dir / "execution",
                                request.runtime,
                                seconds=request.seconds_per_attempt,
                                live=request.live,
                            ),
                            checkpoint,
                            actor.sha,
                            request.seed + number,
                            request.recording_config_file,
                            request.task_file,
                            stopped,
                        ),
                    )
                else:
                    result, review = sample_attempt(
                        cast(SACEnvironment, environment),
                        actor,
                        attempt_dir,
                        request.recording_config_file,
                        f"sac-attempt-{number}",
                        request.steps_per_attempt,
                        request.seed + number,
                        stopped,
                    )
                summary["attempts"].append(result)
                summary["commands_sent_to_game"] |= result.get("commands_sent_to_game", False)
                result["source_assets"] = seal_sampling_attempt(
                    attempt_dir, review, indexed=isinstance(request, SACRealtimeCycle)
                )
                if stopped() or result["stop_reason"] == "user_stop":
                    summary["stop_reason"] = "stop_requested"
                    break
                if result["error"]:
                    summary["stop_reason"] = "sampling_fault"
                    break
                if isinstance(request, SACRealtimeCycle):
                    prepared = run_experiment(
                        SACRealtimePrepare(
                            attempt_dir / "recording",
                            attempt_dir / "execution",
                            request.task_file,
                            request.reward_file,
                            attempt_dir / "prepared",
                            review,
                        ),
                        numeric_actor=SACSamplingActor(
                            checkpoint,
                            actor.pixels,
                            actor.sha,
                            exploration_seed=request.seed + number,
                        ),
                    ).summary["sac_replay"]
                else:
                    prepared = run_experiment(
                        SACReplayPrepare(
                            attempt_dir / "recording",
                            attempt_dir / "trace.json",
                            request.task_file,
                            request.reward_file,
                            attempt_dir / "prepared",
                            review,
                        )
                    ).summary["sac_replay"]
                if isinstance(request, SACRealtimeCycle):
                    result.update(
                        {
                            key: value
                            for key, value in prepared.items()
                            if key not in ("excluded", "observation_errors")
                        },
                        excluded_transitions=len(prepared["excluded"]),
                        observation_error_count=len(prepared["observation_errors"]),
                    )
                else:
                    result.update(prepared)
                verify_sampling_sources(result["source_assets"])
                replay = attempt_dir / "prepared/replay.json"
                result["replay"] = replay.relative_to(root).as_posix()
                count = prepared["eligible_transitions"]
                if not count:
                    summary["stop_reason"] = "no_eligible_experience"
                    break
                candidate = root / f"candidate-{number:03d}"
                learned = run_experiment(
                    SACResume(
                        checkpoint,
                        candidate,
                        steps=sampling_update_budget(
                            count,
                            request.max_updates_per_attempt
                            if isinstance(request, SACRealtimeCycle)
                            else None,
                        ),
                        additions=((replay, prepared["replay_sha256"]),),
                        expected_checkpoint_sha256=actor.sha,
                    ),
                    sac_stop_requested=lambda _: stopped(),
                ).summary["sac_learning"]
                # Reload a complete snapshot before making it the next sampling candidate.
                restored = FrozenSAC(torch, candidate)
                result.update(
                    candidate=candidate.relative_to(root).as_posix(),
                    candidate_sha256=restored.sha,
                    learner_updates=learned["steps_completed"],
                    total_steps=learned["total_steps"],
                )
                if learned["stop_reason"] == "training_data_unavailable":
                    # Keep the sealed learner and unused update credit. Fresh
                    # numerical validation is required by parent continuation.
                    result["inference_reload_status"] = "deferred_training_data_unavailable"
                    result["training_error"] = learned["training_error"]
                else:
                    from fh5.learning.sac.training import SACPolicyReplay

                    checked = run_experiment(
                        SACPolicyReplay(
                            candidate,
                            candidate / "experience/replay.json",
                            root / f"reload-{number}.html",
                        )
                    ).summary["sac_policy"]
                    if "presentation" in checked:
                        result["reload_presentation"] = checked["presentation"]
                    if "diagnostic_export" in checked:
                        result["reload_diagnostic_export"] = checked["diagnostic_export"]
                    if prediction_identity(checked["predictions"]) != prediction_identity(
                        learned["predictions"]
                    ):
                        raise ValueError(
                            "Candidate reload differs from the complete learner snapshot"
                        )
                    result["inference_reload_max_error"] = 0
                checkpoint = candidate
                expected_sampling_sha = restored.sha
                summary["latest_candidate"] = candidate.relative_to(root).as_posix()
                write_file(attempt_dir / "cycle-result.json", encode(result))
                if learned["stop_reason"] != "budget_completed":
                    summary["stop_reason"] = learned["stop_reason"]
                    break
            else:
                summary["stop_reason"] = "budget_completed"
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            released = environment.close()
            summary["commands_sent_to_game"] |= released.get("commands_sent_to_game", False)
            summary["resources_released"] = released.get("resources_released") is True
            if isinstance(request, SACRealtimeCycle):
                summary["resources_released"] &= all(
                    a["resources_released"] for a in summary["attempts"]
                )
        except Exception as error:
            summary["release_error"] = str(error)
        if not summary["resources_released"] and summary["stop_reason"] == "budget_completed":
            summary["stop_reason"] = "release_fault"
        write_file(root / "summary.json", encode(summary))
    report = optional_report(
        root / "report.html",
        "SAC 采样与学习循环",
        summary,
        fallback=root / "summary.json",
    )
    return RunResult({}, [], [], {"sac_cycle": summary}, report)
