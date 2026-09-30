"""Bounded synchronous synthetic sampling; numerical inference precedes archival."""

from __future__ import annotations

import hashlib
import json
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from fh5.collection_store import encode, read_bounded, write_file
from fh5.control import Command
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import NumericDecision, PixelContract, validate_decision
from fh5.sac_actions import ActionBounds
from fh5.sac_checkpoint import read_checkpoint
from fh5.sac_policy import encode_history, make_policy
from fh5.sac_replay import command_action

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


class FrozenSAC:
    """Complete immutable per-attempt policy, distinct from the learner modules."""

    def __init__(self, torch: Any, checkpoint: Path) -> None:
        self.torch = torch
        manifest, saved, raw = read_checkpoint(torch, checkpoint)
        if manifest["version"] != 2:
            raise ValueError("SAC sampling requires a sealed version 2 checkpoint")
        self.sha = hashlib.sha256(raw).hexdigest()
        self.manifest = manifest
        model_dir = checkpoint / "bc"
        bc_metadata = json.loads(read_bounded(model_dir / "model.json", 1024**2))
        self.pixels = PixelContract.from_metadata(bc_metadata["numeric_contract"])
        self.bc = FrozenNumericActor(
            model_dir, self.pixels, expected_manifest_sha256=manifest["bc_manifest_sha256"]
        )
        self.bounds = ActionBounds(**manifest["bounds"])
        self.encoder = torch.nn.ModuleDict(
            {"images": deepcopy(self.bc.model.encoder), "state": deepcopy(self.bc.model.state)}
        )
        self.policy = make_policy(
            torch,
            self.bc.model.fusion,
            64 * (self.bc.original_contract["image_count"] + 1),
            manifest["configuration"]["initial_log_std"],
        )
        self.encoder.load_state_dict(saved["encoder"], strict=True)
        self.policy.load_state_dict(saved["policy"], strict=True)
        for module in (self.encoder, self.policy):
            module.eval()
            for parameter in module.parameters():
                if not torch.isfinite(parameter).all():
                    raise ValueError("Non-finite frozen SAC parameter")
                parameter.requires_grad_(False)

    def predict(
        self, decision: NumericDecision, previous: list[float], elapsed: float, noise: list[float]
    ) -> dict[str, Any]:
        reason = validate_decision(decision, self.pixels)
        if reason:
            raise ValueError("Invalid SAC sampler observation: " + reason)
        torch = self.torch
        values = self.bc.input_features(decision.actor, decision.frames)
        images = [
            torch.frombuffer(bytearray(f.pixels), dtype=torch.uint8)
            .reshape(f.size[1], f.size[0], 3)
            .permute(2, 0, 1)
            for f in decision.frames
        ]
        context = self.bounds.context(previous, elapsed)
        with torch.inference_mode():
            features = encode_history(
                torch,
                self.encoder,
                torch.stack(images).unsqueeze(0).float() / 255,
                torch.tensor([values], dtype=torch.float32),
            )
            output = self.policy(features, torch.tensor([context]), features.new_tensor([noise]))
        result = {k: v[0].tolist() for k, v in output.items()}
        steer, longitudinal = result["command"]
        command = Command(
            round(steer * 32767),
            max(0, round(longitudinal * 255)),
            max(0, round(-longitudinal * 255)),
        )
        result.update(
            sent=asdict(command), command=command_action(asdict(command)), context=context
        )
        return result


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
        write_file(root / "sampling.json", encode(result))
    recording = root / "recording"
    review = None
    try:
        run_experiment(Record(config, recording), packets=[s.packet for s in samples])
        review = environment.finish(recording)
    except Exception as error:
        result.update(stop_reason="sampling_fault", error=f"{type(error).__name__}: {error}")
        write_file(root / "finish-fault.json", encode(result))
    return result, review
