"""One asynchronous attempt, released and sealed before its learner can run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from fh5.collection_store import encode, write_file
from fh5.learning_io import StoppingDrive
from fh5.realtime import RealtimeConfig, RealtimeEnvironment, RealtimeRun
from fh5.realtime_numeric_replay import read_realtime_journal
from fh5.sac_sampling_actor import SACSamplingActor


class SACRealtimeEnvironment(Protocol):
    source_kind: Literal["synthetic"]

    def start(self, identity: str, runtime: RealtimeConfig) -> RealtimeEnvironment: ...
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
    environment: SACRealtimeEnvironment,
    checkpoint: Path,
    expected_sha256: str,
    root: Path,
    recording_config: Path,
    runtime: RealtimeConfig,
    seconds: float,
    identity: str,
    seed: int,
    stopped: Callable[[], bool],
) -> tuple[dict[str, Any], Path | None]:
    from fh5.experiment import Packet, Record, run_experiment

    root.mkdir(parents=True)
    result: dict[str, Any] = {
        "epoch": identity,
        "sampling_checkpoint_sha256": expected_sha256,
        "sampler_seed": seed,
        "stop_reason": "sampling_fault",
        "error": None,
        "resources_released": False,
        "execution": "execution",
    }
    drive = None
    review = None
    try:
        if stopped():
            result.update(stop_reason="user_stop", resources_released=True)
            return result, None
        drive = _AttemptDrive(environment.start(identity, runtime), stopped)
        if drive.source_kind != "synthetic":
            raise ValueError("Asynchronous SAC cycle requires synthetic external I/O")
        executed = run_experiment(
            RealtimeRun(root / "execution", runtime, seconds=seconds),
            realtime_environment=drive,
            numeric_actor_factory=lambda: SACSamplingActor(
                checkpoint, runtime.pixels, expected_sha256, exploration_seed=seed
            ),
        ).summary["realtime"]
        result.update(
            stop_reason=executed["stop_reason"],
            resources_released=executed["resources_released"],
            decisions=executed["decisions"],
        )
        packets = read_realtime_journal(root / "execution", executed)
        run_experiment(
            Record(recording_config, root / "recording"),
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
