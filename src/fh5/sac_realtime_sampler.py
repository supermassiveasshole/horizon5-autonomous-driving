"""One asynchronous attempt, released and sealed before its learner can run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from fh5.collection_store import encode, write_file
from fh5.learning_io import LearningUnavailable, StoppingDrive
from fh5.realtime import RealtimeEnvironment, RealtimeRun
from fh5.realtime_numeric_replay import read_realtime_journal
from fh5.sac_sampling_actor import SACSamplingActor


@dataclass(frozen=True)
class SACRealtimeStart:
    identity: str
    request: RealtimeRun
    checkpoint: Path
    expected_sha256: str
    seed: int
    recording_config: Path
    task_file: Path
    stopped: Callable[[], bool]


class SACRealtimeEnvironment(Protocol):
    @property
    def source_kind(self) -> Literal["synthetic", "native"]: ...

    def start(self, start: SACRealtimeStart) -> RealtimeEnvironment: ...
    def finish(self, recording_dir: Path) -> Path | None: ...
    def close(self) -> dict[str, Any]: ...


@dataclass
class _AttemptDrive(StoppingDrive):
    released: dict[str, Any] | None = None

    def close(self) -> dict[str, Any]:
        if self.released is None:
            self.released = {"resources_released": False}
            try:
                self.released = self.source.close()
            except (Exception, KeyboardInterrupt) as error:
                self.released["error"] = f"{type(error).__name__}: {error}"
                raise
        return self.released


def sample_realtime_attempt(
    environment: SACRealtimeEnvironment, start: SACRealtimeStart
) -> tuple[dict[str, Any], Path | None]:
    from fh5.experiment import Packet, Record, run_experiment

    root = start.request.output_dir.parent
    root.mkdir(parents=True)
    result: dict[str, Any] = {
        "epoch": start.identity,
        "sampling_checkpoint_sha256": start.expected_sha256,
        "sampler_seed": start.seed,
        "commands_sent_to_game": False,
        "stop_reason": "sampling_fault",
        "error": None,
        "resources_released": False,
        "execution": "execution",
    }
    drive = None
    review = None
    try:
        if start.stopped():
            result.update(stop_reason="user_stop", resources_released=True)
            return result, None
        drive = _AttemptDrive(environment.start(start), start.stopped)
        if drive.source_kind != environment.source_kind or start.request.live != (
            drive.source_kind == "native"
        ):
            raise ValueError("Asynchronous SAC source and live opt-in disagree")
        executed = run_experiment(
            start.request,
            realtime_environment=drive,
            numeric_actor_factory=lambda: SACSamplingActor(
                start.checkpoint,
                start.request.config.pixels,
                start.expected_sha256,
                exploration_seed=start.seed,
            ),
        ).summary["realtime"]
        result.update(
            stop_reason=executed["stop_reason"],
            resources_released=executed["resources_released"],
            decision_count=len(executed["decisions"]),
            commands_sent_to_game=executed["commands_sent_to_game"],
        )
        packets = read_realtime_journal(root / "execution", executed)
        run_experiment(
            Record(
                start.recording_config,
                root / "recording",
                "udp" if start.request.live else "synthetic",
            ),
            packets=(
                Packet(
                    p["received_monotonic_ns"], p["received_utc"], bytes.fromhex(p["payload_hex"])
                )
                for p in packets
            ),
        )
        result["received_packets"] = len(packets)
        if not result["resources_released"]:
            raise ValueError("Sampling resources were not released before learning")
        review = environment.finish(root / "recording")
        if executed["stop_reason"] not in ("time_limit", "local_end", "user_stop"):
            raise ValueError("Asynchronous sampling stopped: " + executed["stop_reason"])
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, LearningUnavailable):
            result["resources_released"] = error.resources_released
        if (
            isinstance(error, (InterruptedError, KeyboardInterrupt))
            or isinstance(error.__cause__, (InterruptedError, KeyboardInterrupt))
            or start.stopped()
        ):
            result["stop_reason"] = "user_stop"
    finally:
        if drive is not None:
            try:
                released = drive.close()
                result["resources_released"] &= released.get("resources_released") is True
            except (Exception, KeyboardInterrupt) as error:
                result["resources_released"] = False
                result["release_error"] = str(error)
        write_file(root / "sampling.json", encode(result))
    return result, review
