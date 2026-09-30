"""Finite double-Q warm-up with an entirely frozen BC actor; no game adapter."""

from __future__ import annotations

import hashlib
import html
import importlib
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.collection_store import encode, read_bounded, write_file
from fh5.learning_runtime import preserve_torch_state
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import PixelContract
from fh5.numeric_recording import read_numeric_frame
from fh5.sac_actions import ActionBounds

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class SACCriticWarmup:
    model_dir: Path
    replay_file: Path
    replay_sha256: str
    output_dir: Path
    steps: int = 100
    batch_size: int = 32
    learning_rate: float = 0.0001
    seed: int = 7
    bounds: ActionBounds = field(default_factory=ActionBounds)


@dataclass(frozen=True)
class SACCriticReplay:
    checkpoint_dir: Path
    replay_file: Path
    report_path: Path


def _q_heads(torch: Any, width: int) -> Any:
    nn = torch.nn
    return nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(width + 2, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            for _ in range(2)
        ]
    )


def _values(torch: Any, heads: Any, state: Any, action: Any) -> Any:
    inputs = torch.cat([state, action], dim=1)
    return torch.cat([head(inputs) for head in heads], dim=1)


def _task_features(state: dict[str, Any], context: dict[str, Any]) -> list[float]:
    gates = context["checkpoint_ids"]
    checkpoint = state["next_checkpoint"]
    result = [
        state["farthest_confirmed_m"] / context["route_length_m"],
        state["start_progress_m"] / context["route_length_m"],
        gates.index(checkpoint) / max(1, len(gates)) if checkpoint is not None else 1.0,
        state["remaining_s"] / context["max_duration_s"],
        state["no_progress_remaining_s"] / context["no_progress_timeout_s"],
    ]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in result):
        raise ValueError("Invalid independent task-state features")
    return result


def run_critic(request: SACCriticWarmup | SACCriticReplay) -> RunResult:
    torch = importlib.import_module("torch")
    with preserve_torch_state(torch):
        torch.set_num_threads(2)
        return _run(request, torch)


def _run(request: SACCriticWarmup | SACCriticReplay, torch: Any) -> RunResult:
    from fh5.experiment import RunResult

    training = isinstance(request, SACCriticWarmup)
    if isinstance(request, SACCriticWarmup):
        if request.output_dir.exists():
            raise FileExistsError(request.output_dir)
        if (
            type(request.steps) is not int
            or not 1 <= request.steps <= 10_000
            or type(request.batch_size) is not int
            or not 1 <= request.batch_size <= 256
            or not 0 < request.learning_rate <= 0.01
        ):
            raise ValueError("Critic warm-up needs bounded steps, batch and learning rate")
        expected, model_dir, bounds = request.replay_sha256, request.model_dir, request.bounds
        torch.manual_seed(request.seed)
    else:
        manifest = json.loads(read_bounded(request.checkpoint_dir / "critic.json", 1024**2))
        if manifest.get("version") != 1 or manifest.get("stage") != "critic_warmup":
            raise ValueError("Unsupported critic checkpoint version or stage")
        expected, model_dir = manifest["replay_sha256"], request.checkpoint_dir / "actor"
        bounds = ActionBounds(**manifest["bounds"])
    raw = read_bounded(request.replay_file, 128 * 1024**2)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("SAC replay changed from its frozen digest")
    replay = json.loads(raw)
    if (
        replay.get("version") != 1
        or replay.get("kind") != "sac-numeric-replay-v1"
        or replay.get("source_kind") != "synthetic"
        or not 1 <= len(replay["transitions"]) <= 10_000
    ):
        raise ValueError("Critic warm-up requires a bounded prepared synthetic replay")
    pixels = PixelContract.from_metadata(replay["pixel_contract"])
    model_payload = {
        name: read_bounded(model_dir / name, 256 * 1024**2) for name in ("model.json", "actor.pt")
    }
    model_digest = hashlib.sha256(model_payload["model.json"]).hexdigest()
    if not training and model_digest != manifest["actor_manifest_sha256"]:
        raise ValueError("Critic's frozen BC manifest changed")
    actor = FrozenNumericActor(model_dir, pixels, expected_manifest_sha256=model_digest)
    if hashlib.sha256(model_payload["actor.pt"]).hexdigest() != actor.manifest["weights_sha256"]:
        raise ValueError("BC snapshot changed while loading critic warm-up")
    for parameter in actor.model.parameters():
        parameter.requires_grad_(False)
    original = deepcopy(actor.model.state_dict())
    frame_cache: dict[str, Any] = {}
    encoded: dict[str, tuple[Any, list[float]]] = {}
    reload_inputs: dict[str, Any] = {}
    total_bytes = 0

    def observation(row: dict[str, Any]) -> tuple[Any, list[float]]:
        nonlocal total_bytes
        identity = hashlib.sha256(encode(row)).hexdigest()
        if identity not in encoded:
            frames = []
            for metadata in row["frames"]:
                key = json.dumps(metadata, sort_keys=True)
                if key not in frame_cache:
                    frame = read_numeric_frame(request.replay_file.parent, metadata)
                    total_bytes += frame.pixels.nbytes
                    if total_bytes > 512 * 1024**2:
                        raise ValueError("SAC numerical replay exceeds 512 MiB frame budget")
                    frame_cache[key] = frame
                frames.append(frame_cache[key])
            sequence = tuple(frames)
            prediction = actor.predict(row["actor"], sequence)
            values = actor.input_features(row["actor"], sequence)
            width, height = pixels.size
            rgb = (
                torch.stack(
                    [
                        torch.frombuffer(bytearray(f.pixels), dtype=torch.uint8)
                        .reshape(height, width, 3)
                        .permute(2, 0, 1)
                        for f in sequence
                    ]
                ).float()
                / 255
            )
            with torch.no_grad():
                image = actor.model.encoder(rgb).reshape(1, -1)
                state = actor.model.state(torch.tensor([values], dtype=torch.float32))
                encoded[identity] = (torch.cat([image, state], dim=1)[0], prediction)
                reload_inputs[identity] = (row["actor"], sequence, prediction)
        return encoded[identity]

    current, following, actions, next_actions, rewards, discounts, predictions = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for row in replay["transitions"]:
        state, prediction = observation(row["current"])
        previous, elapsed = row["previous_action"], row["action_elapsed_s"]
        lower, upper = bounds.interval(previous, elapsed)
        if any(
            not lo - 1e-9 <= value <= hi + 1e-9
            for lo, hi, value in zip(lower, upper, row["action"])
        ):
            raise ValueError(
                "Recorded SAC action is outside executable support; never clip replay labels"
            )
        task_features = _task_features(row["task_state"], replay["task_context"])
        current.append(
            torch.cat([state, torch.tensor([*bounds.context(previous, elapsed), *task_features])])
        )
        actions.append(row["action"])
        predictions.append(
            {"id": row["id"], "bc_action": prediction, "critic_task_features": task_features}
        )
        if row["bootstrap"]:
            if row["next"] is None or row["terminated"]:
                raise ValueError("SAC bootstrap requires a nonterminal real final observation")
            next_state, next_prediction = observation(row["next"])
            following.append(
                torch.cat(
                    [
                        next_state,
                        torch.tensor(
                            [
                                *bounds.context(row["action"], row["hold_dt_s"]),
                                *_task_features(row["next_task_state"], replay["task_context"]),
                            ]
                        ),
                    ]
                )
            )
            next_actions.append(
                bounds.deterministic(next_prediction, row["action"], row["hold_dt_s"])
            )
            discounts.append(row["discount"])
        else:
            if not row["terminated"]:
                raise ValueError("Missing bootstrap cannot be converted to a zero-value terminal")
            following.append(torch.zeros_like(current[-1]))
            next_actions.append([0.0, 0.0])
            discounts.append(0.0)
        rewards.append(row["reward"])
    state_tensor, next_tensor = torch.stack(current), torch.stack(following)
    command_tensor = torch.tensor(actions, dtype=torch.float32)
    next_command_tensor = torch.tensor(next_actions, dtype=torch.float32)
    reward_tensor, discount_tensor = (
        torch.tensor(v, dtype=torch.float32) for v in (rewards, discounts)
    )
    critic = _q_heads(torch, state_tensor.shape[1])
    target = deepcopy(critic)
    with torch.no_grad():
        before_q = _values(torch, critic, state_tensor, command_tensor)
    losses, target_values = [], []
    started = time.monotonic()
    if isinstance(request, SACCriticWarmup):
        optimizer = torch.optim.Adam(critic.parameters(), lr=request.learning_rate)
        for _ in range(request.steps):
            indices = torch.randperm(len(current))[: request.batch_size]
            with torch.no_grad():
                boot = (
                    _values(torch, target, next_tensor[indices], next_command_tensor[indices])
                    .min(dim=1)
                    .values
                )
                expected_value = reward_tensor[indices] + discount_tensor[indices] * boot
            optimizer.zero_grad(set_to_none=True)
            predicted = _values(torch, critic, state_tensor[indices], command_tensor[indices])
            loss = ((predicted - expected_value[:, None]) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite critic warm-up loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for destination, source in zip(target.parameters(), critic.parameters()):
                    destination.lerp_(source, 0.005)
            losses.append(float(loss.item()))
            target_values.append(expected_value.tolist())
    else:
        checkpoint = read_bounded(request.checkpoint_dir / "critic.pt", 256 * 1024**2)
        if hashlib.sha256(checkpoint).hexdigest() != manifest["weights_sha256"]:
            raise ValueError("Critic checkpoint hash mismatch")
        saved = torch.load(BytesIO(checkpoint), map_location="cpu", weights_only=True)
        if set(saved["target_encoder"]) != set(original) or any(
            not torch.equal(original[k], saved["target_encoder"][k]) for k in original
        ):
            raise ValueError("Warm-up target encoder differs from the frozen BC")
        critic.load_state_dict(saved["critic"], strict=True)
        target.load_state_dict(saved["target"], strict=True)
    with torch.no_grad():
        after_q = _values(torch, critic, state_tensor, command_tensor)
    if not torch.isfinite(after_q).all():
        raise ValueError("Non-finite critic checkpoint predictions")
    for entry, values in zip(predictions, after_q.tolist()):
        entry["q"] = values
    actor_change = max(
        float((original[k] - v).abs().max()) for k, v in actor.model.state_dict().items()
    )
    if actor_change:
        raise ValueError("Critic warm-up changed the frozen actor")
    summary = {
        "stage": "critic_warmup",
        "steps_completed": request.steps if isinstance(request, SACCriticWarmup) else 0,
        "actor_optimizer_steps": 0,
        "actor_change_max": actor_change,
        "q_change_max": float((after_q - before_q).abs().max()) if training else None,
        "predictions": predictions,
        "losses": losses,
        "target_values": target_values,
        "target_actions": next_actions,
        "target_policy": "deterministic bounded BC; no SAC entropy update yet",
        "replay_sha256": expected,
        "transitions": len(current),
        "frozen_encoder_sha256": actor.manifest["weights_sha256"],
        "raw_frame_bytes": total_bytes,
        "duration_s": time.monotonic() - started,
        "commands_sent": False,
        "real_driving_validated": False,
    }
    if isinstance(request, SACCriticWarmup):
        output = request.output_dir
        output.mkdir(parents=True)
        (output / "actor").mkdir()
        for name in ("actor.pt", "model.json"):
            write_file(output / "actor" / name, model_payload[name])
        torch.save(
            {
                "critic": critic.state_dict(),
                "target": target.state_dict(),
                "target_encoder": original,
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
            },
            output / "critic.pt",
        )
        reloaded = FrozenNumericActor(
            output / "actor", pixels, expected_manifest_sha256=model_digest
        )
        reload_error = max(
            abs(a - b)
            for state, frames, prediction in reload_inputs.values()
            for a, b in zip(prediction, reloaded.predict(state, frames))
        )
        if reloaded.manifest != actor.manifest or reload_error > 1e-6:
            raise ValueError("Frozen BC changed during checkpoint publication")
        summary["reload_max_abs_error"] = reload_error
        manifest = {
            "version": 1,
            "stage": "critic_warmup",
            "bounds": asdict(bounds),
            "replay_sha256": expected,
            "actor_manifest_sha256": model_digest,
            "configuration": {
                "steps": request.steps,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "seed": request.seed,
                "target_tau": 0.005,
                "device": "cpu",
            },
            "weights_sha256": hashlib.sha256((output / "critic.pt").read_bytes()).hexdigest(),
            "summary": summary,
        }
        write_file(output / "critic.json", encode(manifest))
        report = output / "report.html"
    else:
        report = request.report_path
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        '<!doctype html><meta charset="utf-8"><h1>BC 冻结与双 Q 预热</h1><pre>'
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>",
        encoding="utf-8",
    )
    return RunResult({}, [], [], {"sac": summary}, report)
