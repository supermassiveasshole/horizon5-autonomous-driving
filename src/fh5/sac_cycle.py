"""Finite sample/learn alternation, with explicit synthetic external I/O only."""

from __future__ import annotations

import hashlib
import html
import importlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fh5.collection_store import encode, read_bounded, write_file
from fh5.learning_runtime import preserve_torch_state
from fh5.realtime import RealtimeConfig, RealtimeRun
from fh5.sac_actor import FrozenSAC
from fh5.sac_learning import SACResume
from fh5.sac_realtime_experience import SACRealtimePrepare
from fh5.sac_realtime_sampler import SACRealtimeEnvironment, sample_realtime_attempt
from fh5.sac_replay import SACReplayPrepare
from fh5.sac_sampler import (
    SACEnvironment,
    SACSample,
    SACStart,
    sample_attempt,
)
from fh5.sac_sampling_actor import SACSamplingActor
from fh5.sampling_evidence import seal_sampling_attempt, verify_sampling_sources

if TYPE_CHECKING:
    from fh5.experiment import RunResult

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

    def __post_init__(self) -> None:
        if (
            type(self.cycles) is not int
            or not 1 <= self.cycles <= 10
            or type(self.max_updates_per_attempt) is not int
            or not 1 <= self.max_updates_per_attempt <= 1000
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("SAC realtime cycle requires finite attempts, updates and seed")
        RealtimeRun(self.output_dir, self.runtime, seconds=self.seconds_per_attempt)


def run_sac_cycle(
    request: SACCycle | SACRealtimeCycle,
    environment: SACEnvironment | SACRealtimeEnvironment,
    stop_requested: Callable[[int], bool] | None = None,
) -> RunResult:
    from fh5.experiment import RunResult, run_experiment

    root = request.output_dir

    def stopped() -> bool:
        return (root / "stop.request").exists() or bool(stop_requested and stop_requested(0))

    if root.exists() or root.resolve().is_relative_to(request.checkpoint_dir.resolve()):
        raise ValueError("SAC cycle needs a fresh output outside its frozen checkpoint")
    summary: dict[str, Any] = {
        "attempts": [],
        "source_kind": "synthetic",
        "commands_sent_to_game": False,
        "real_driving_validated": False,
        "default_changed": False,
        "stop_reason": "interface_error",
        "resources_released": False,
    }
    torch = importlib.import_module("torch")
    root.mkdir(parents=True)
    try:
        if environment.source_kind != "synthetic":
            raise ValueError("SAC cycle currently requires synthetic external I/O")
        checkpoint = request.checkpoint_dir
        expected_sampling_sha = request.expected_checkpoint_sha256
        files = (request.recording_config_file, request.task_file, request.reward_file)
        protocol_bytes = [read_bounded(p, 1024**2) for p in files]
        if json.loads(protocol_bytes[0])["control_source"] != "policy":
            raise ValueError("SAC sampler requires policy recording attribution")
        old_replay = json.loads(read_bounded(checkpoint / "experience/replay.json", 128 * 1024**2))
        for kind, raw in zip(("task", "reward"), protocol_bytes[1:]):
            if hashlib.sha256(raw).hexdigest() != old_replay["source_hashes"][kind]:
                raise ValueError("Cycle protocol differs from the learner: " + kind)
        write_file(
            root / "protocol.json",
            encode(
                {
                    "source_kind": "synthetic",
                    "cycles": request.cycles,
                    **(
                        {
                            "runtime": asdict(request.runtime),
                            "seconds_per_attempt": request.seconds_per_attempt,
                            "max_updates_per_attempt": request.max_updates_per_attempt,
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
                frame_count = (
                    int(request.seconds_per_attempt * request.runtime.decision_hz) + 1
                    if isinstance(request, SACRealtimeCycle)
                    else request.steps_per_attempt + 1
                )
                if (
                    frame_count
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
                        checkpoint,
                        actor.sha,
                        attempt_dir,
                        request.recording_config_file,
                        request.runtime,
                        request.seconds_per_attempt,
                        f"sac-attempt-{number}",
                        request.seed + number,
                        stopped,
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
                from fh5.sac_learning import SACPolicyReplay

                checked = run_experiment(
                    SACPolicyReplay(
                        candidate,
                        candidate / "experience/replay.json",
                        root / f"reload-{number}.html",
                    )
                ).summary["sac_policy"]["predictions"]
                if checked != learned["predictions"]:
                    raise ValueError("Candidate reload differs from the complete learner snapshot")
                result["inference_reload_max_error"] = 0
                checkpoint = candidate
                expected_sampling_sha = restored.sha
                summary["latest_candidate"] = candidate.relative_to(root).as_posix()
                write_file(attempt_dir / "cycle-result.json", encode(result))
                if learned["stop_reason"] == "stop_requested":
                    summary["stop_reason"] = "stop_requested"
                    break
            else:
                summary["stop_reason"] = "budget_completed"
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            released = environment.close()
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
    report = root / "report.html"
    write_file(
        report,
        (
            '<!doctype html><meta charset="utf-8"><h1>SAC 合成采样与学习循环</h1><pre>'
            + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
            + "</pre>"
        ).encode("utf-8"),
    )
    return RunResult({}, [], [], {"sac_cycle": summary}, report)
