"""Bounded synchronous synthetic sampling; numerical inference precedes archival."""

from __future__ import annotations

import hashlib
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from fh5.collection_store import encode, write_file
from fh5.control import Command
from fh5.numeric_images import NumericDecision, PixelContract
from fh5.sac_actor import FrozenSAC

if TYPE_CHECKING:
    from fh5.experiment import Packet


@dataclass(frozen=True)
class SACSample:
    packet: Packet
    decision: NumericDecision
    done: bool = False


@dataclass(frozen=True)
class SACStart:
    sample: SACSample
    initial_command: Command
    initial_issued_ns: int


class SACEnvironment(Protocol):
    source_kind: Literal["synthetic"]

    def start(self, epoch: str, pixels: PixelContract) -> SACStart: ...
    def step(self, command: Command) -> SACSample: ...
    def finish(self, recording_dir: Path) -> Path | None: ...
    def close(self) -> dict[str, Any]: ...


def _with_history(sample: SACSample, receipts: list[tuple[int, list[float]]]) -> SACSample:
    now = sample.decision.decision_ns
    if now != sample.packet.received_monotonic_ns:
        raise ValueError("Synthetic SAC requires synchronous observation and packet clocks")
    actor = deepcopy(sample.decision.actor)
    history, ages = [], []
    for offset in (200, 100, 0):
        at = now - offset * 1_000_000
        prior = next((r for r in reversed(receipts) if r[0] < at), None)
        valid = prior is not None and at - prior[0] <= 200_000_000
        history.append(prior[1] if prior and valid else None)
        ages.append((now - prior[0]) / 1e6 if prior and valid else None)
    actor.update(actions=history, action_mask=[a is not None for a in history], action_age_ms=ages)
    return replace(sample, decision=replace(sample.decision, actor=actor))


def sample_attempt(
    environment: SACEnvironment,
    actor: FrozenSAC,
    root: Path,
    config: Path,
    epoch: str,
    limit: int,
    seed: int,
) -> tuple[dict[str, Any], Path | None]:
    from fh5.experiment import Record, run_experiment

    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    decisions: list[dict[str, Any]] = []
    samples: list[SACSample] = []
    packets: list[Packet] = []
    actions: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "epoch": epoch,
        "sampling_checkpoint_sha256": actor.sha,
        "decisions": decisions,
        "sampler_seed": seed,
        "stop_reason": "step_limit",
        "error": None,
    }
    started: SACStart | None = None
    try:
        started = environment.start(epoch, actor.pixels)
        packets.append(started.sample.packet)
        if (
            started.initial_command != Command(0, 0, 0)
            or type(started.initial_issued_ns) is not int
            or not 0 <= started.initial_issued_ns < started.sample.decision.decision_ns
        ):
            raise ValueError("Synthetic attempt requires an acknowledged neutral start")
        receipts = [(started.initial_issued_ns, [0.0, 0.0])]
        samples.append(_with_history(started.sample, receipts))
        for index in range(limit):
            if (root.parent / "stop.request").exists():
                result["stop_reason"] = "stop_requested"
                break
            sample = samples[-1]
            if sample.decision.epoch != epoch or sample.done:
                if sample.decision.epoch != epoch:
                    raise ValueError("Sampler cannot join a changed epoch")
                result["stop_reason"] = "environment_done"
                break
            elapsed = (sample.decision.decision_ns - receipts[-1][0]) / 1e9
            noise = [rng.gauss(0, 1), rng.gauss(0, 1)]
            before = time.perf_counter_ns()
            prediction = actor.predict(sample.decision, receipts[-1][1], elapsed, noise)
            row = {
                "packet_index": index,
                "snapshot_sha256": actor.sha,
                "noise": noise,
                "inference_s": (time.perf_counter_ns() - before) / 1e9,
                **prediction,
                "status": "proposed",
            }
            decisions.append(row)
            command = Command(**prediction["sent"])
            try:
                following = environment.step(command)
            except Exception:
                row["status"] = "failed_or_unknown"
                raise
            row["status"] = "sent"
            packets.append(following.packet)
            row["response"] = {
                "epoch": following.decision.epoch,
                "decision_ns": following.decision.decision_ns,
                "packet_received_ns": following.packet.received_monotonic_ns,
            }
            if (
                following.decision.epoch != epoch
                or following.decision.decision_ns <= sample.decision.decision_ns
            ):
                raise ValueError("Sampler response must advance within the same epoch")
            receipts.append((sample.decision.decision_ns, prediction["command"]))
            samples.append(_with_history(following, receipts))
            actions.append(
                {
                    "from_packet_index": index,
                    "to_packet_index": index + 1,
                    "epoch": epoch,
                    "owner": "policy",
                    "status": "sent",
                    "sent": asdict(command),
                }
            )
        if samples and samples[-1].done:
            result["stop_reason"] = "environment_done"
    except Exception as error:
        result.update(stop_reason="sampling_fault", error=f"{type(error).__name__}: {error}")
    finally:
        try:
            # Archive after decisions; no image codec or disk read lies on the inference path.
            observations = []
            for index, sample in enumerate(samples):
                frames = []
                for frame in sample.decision.frames:
                    raw = bytes(frame.pixels)
                    sha = hashlib.sha256(raw).hexdigest()
                    path = "frames/" + sha + ".rgb"
                    dest = root / path
                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        write_file(dest, raw)
                    frames.append({**frame.metadata(), "path": path, "sha256": sha})
                observations.append(
                    {
                        "packet_index": index,
                        "decision_id": sample.decision.decision_id,
                        "epoch": sample.decision.epoch,
                        "decision_ns": sample.decision.decision_ns,
                        "actor": sample.decision.actor,
                        "frames": frames,
                    }
                )
            if started:
                write_file(
                    root / "trace.json",
                    encode(
                        {
                            "version": 1,
                            "kind": "synthetic-synchronous-action-trace-v1",
                            "pixel_contract": actor.pixels.metadata(),
                            "action_offsets_ms": [200, 100, 0],
                            "initial_command": asdict(started.initial_command),
                            "initial_issued_ns": started.initial_issued_ns,
                            "observations": observations,
                            "actions": actions,
                        }
                    ),
                )
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            result.update(stop_reason="sampling_fault", archive_error=detail)
            result["error"] = result["error"] or detail
        result["received_packets"] = len(packets)
        _save_diagnostics(root / "sampling.json", result)
    recording = root / "recording"
    review = None
    try:
        run_experiment(Record(config, recording), packets=packets)
        review = environment.finish(recording)
    except Exception as error:
        result.update(stop_reason="sampling_fault", error=f"{type(error).__name__}: {error}")
        _save_diagnostics(root / "finish-fault.json", result)
    return result, review


def _save_diagnostics(path: Path, result: dict[str, Any]) -> None:
    try:
        write_file(path, encode(result))
    except Exception as error:
        result["diagnostic_write_error"] = f"{type(error).__name__}: {error}"
