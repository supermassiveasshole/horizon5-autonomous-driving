"""Finite double-Q warm-up with an entirely frozen BC actor; no game adapter."""

from __future__ import annotations

import hashlib
import html
import importlib
import json
import math
import time
from collections.abc import Callable
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.artifact_io import VerifiedFile, sha256_file
from fh5.checkpoint_history import HistorySource, empty_history
from fh5.collection_store import encode, read_bounded, write_file
from fh5.learning_diagnostics import RecordJournal
from fh5.learning_runtime import preserve_torch_state
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import PixelContract
from fh5.numeric_recording import read_numeric_frame
from fh5.presentation import optional_report
from fh5.replay_document import replay_document
from fh5.sac_actions import ActionBounds
from fh5.sac_checkpoint import (
    continuation_history,
    critic_resume_contract,
    publish_checkpoint,
    read_critic_checkpoint,
    seal_experience,
    state_digest,
)
from fh5.sac_timing import next_action_elapsed

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


@dataclass(frozen=True)
class SACCriticResume:
    checkpoint_dir: Path
    output_dir: Path
    steps: int | None = None
    expected_checkpoint_sha256: str | None = None


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


def run_critic(
    request: SACCriticWarmup | SACCriticReplay | SACCriticResume,
    stop_requested: Callable[[int], bool] | None = None,
) -> RunResult:
    if isinstance(request, SACCriticReplay):
        if request.report_path.exists() or request.report_path.is_symlink():
            raise FileExistsError(request.report_path)
        if request.report_path.suffix.lower() != ".html":
            raise ValueError("Critic replay requires a new HTML report")
    torch = importlib.import_module("torch")
    with preserve_torch_state(torch), ExitStack() as stack:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        return _run(request, torch, stop_requested, stack)


def _run(
    operation: SACCriticWarmup | SACCriticReplay | SACCriticResume,
    torch: Any,
    stop_requested: Callable[[int], bool] | None,
    stack: ExitStack,
) -> RunResult:
    from fh5.experiment import RunResult

    request: SACCriticWarmup | SACCriticReplay
    saved = None
    start_step = 0
    history_source: HistorySource | None = None
    continuation = None
    if isinstance(operation, SACCriticResume):
        manifest, saved, parent_bytes = read_critic_checkpoint(torch, operation.checkpoint_dir)
        parent_sha = hashlib.sha256(parent_bytes).hexdigest()
        if operation.expected_checkpoint_sha256 is not None and (
            operation.expected_checkpoint_sha256 != parent_sha
        ):
            raise ValueError("Critic continuation differs from its expected parent checkpoint")
        if manifest["version"] != 2:
            raise ValueError("Critic continuation requires a sealed version 2 checkpoint")
        history_source = continuation_history(operation.checkpoint_dir, manifest, parent_bytes)
        configuration = manifest["configuration"]
        start_step, phase_budget = saved["step"], configuration["steps"]
        remaining = phase_budget - start_step
        steps = remaining if operation.steps is None else operation.steps
        if type(steps) is not int or not 0 <= steps <= remaining:
            raise ValueError("Critic continuation exceeds remaining warm-up budget")
        request = SACCriticWarmup(
            operation.checkpoint_dir / "actor",
            operation.checkpoint_dir / "experience/replay.json",
            manifest["replay_sha256"],
            operation.output_dir,
            steps,
            configuration["batch_size"],
            configuration["learning_rate"],
            configuration["seed"],
            ActionBounds(**manifest["bounds"]),
        )
        continuation = {"parent_checkpoint_sha256": parent_sha, "parent_step": start_step}
    else:
        request = operation
        if isinstance(request, SACCriticReplay):
            manifest, saved, _ = read_critic_checkpoint(torch, request.checkpoint_dir)
            phase_budget = manifest["configuration"]["steps"]
            start_step = saved.get("step", phase_budget)
        else:
            phase_budget = request.steps
    training = isinstance(request, SACCriticWarmup)
    if isinstance(request, SACCriticWarmup):
        if request.output_dir.exists():
            raise FileExistsError(request.output_dir)
        if (
            type(request.steps) is not int
            or request.steps < 0
            or phase_budget < 1
            or type(request.batch_size) is not int
            or not 1 <= request.batch_size <= 256
            or not 0 < request.learning_rate <= 0.01
        ):
            raise ValueError("Critic warm-up needs bounded steps, batch and learning rate")
        sources = [request.model_dir, request.replay_file.parent]
        if isinstance(operation, SACCriticResume):
            sources.append(operation.checkpoint_dir)
        if any(request.output_dir.resolve().is_relative_to(p.resolve()) for p in sources):
            raise ValueError("Critic output must be outside its frozen sources")
        expected, model_dir, bounds = request.replay_sha256, request.model_dir, request.bounds
        torch.manual_seed(request.seed)
    else:
        expected, model_dir = manifest["replay_sha256"], request.checkpoint_dir / "actor"
        bounds = ActionBounds(**manifest["bounds"])
    replay_file = VerifiedFile(request.replay_file, expected)
    replay = stack.enter_context(replay_document(replay_file))
    from fh5.sac_sources import replay_roles

    replay_roles(request.replay_file.parent, replay)
    pixels = PixelContract.from_metadata(replay["pixel_contract"])
    model_payload = {
        name: read_bounded(model_dir / name, 256 * 1024**2) for name in ("model.json", "actor.pt")
    }
    model_digest = hashlib.sha256(model_payload["model.json"]).hexdigest()
    if saved is not None and model_digest != manifest["actor_manifest_sha256"]:
        raise ValueError("Critic's frozen BC manifest changed")
    actor = FrozenNumericActor(model_dir, pixels, expected_manifest_sha256=model_digest)
    if hashlib.sha256(model_payload["actor.pt"]).hexdigest() != actor.manifest["weights_sha256"]:
        raise ValueError("BC snapshot changed while loading critic warm-up")
    for parameter in actor.model.parameters():
        parameter.requires_grad_(False)
    original = deepcopy(actor.model.state_dict())
    encoded: dict[str, tuple[Any, list[float]]] = {}
    reload_inputs: dict[str, dict[str, Any]] = {}
    seen_frames: set[str] = set()
    total_bytes = 0

    def predict_observation(
        predictor: FrozenNumericActor, root: Path, row: dict[str, Any]
    ) -> tuple[Any, list[float]]:
        frame_bytes = pixels.size[0] * pixels.size[1] * 3
        try:
            frames = tuple(read_numeric_frame(root, item, frame_bytes) for item in row["frames"])
            return predictor.predict_with_features(row["actor"], frames)
        finally:
            # Only frozen features are reused. Neither the corpus nor a reload
            # check retains image buffers after this observation is encoded.
            predictor.clear_input_cache()

    def observation(row: dict[str, Any]) -> tuple[Any, list[float]]:
        nonlocal total_bytes
        identity = hashlib.sha256(encode(row)).hexdigest()
        if identity not in encoded:
            encoded[identity] = predict_observation(actor, request.replay_file.parent, row)
            reload_inputs[identity] = row
            for metadata in row["frames"]:
                key = json.dumps(metadata, sort_keys=True)
                if key not in seen_frames:
                    total_bytes += pixels.size[0] * pixels.size[1] * 3
                    seen_frames.add(key)
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
                                *bounds.context(row["action"], next_action_elapsed(row)),
                                *_task_features(row["next_task_state"], replay["task_context"]),
                            ]
                        ),
                    ]
                )
            )
            next_actions.append(
                bounds.deterministic(next_prediction, row["action"], next_action_elapsed(row))
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
    if saved is not None:
        if set(saved["target_encoder"]) != set(original) or any(
            not torch.equal(original[k], saved["target_encoder"][k]) for k in original
        ):
            raise ValueError("Warm-up target encoder differs from the frozen BC")
        critic.load_state_dict(saved["critic"], strict=True)
        target.load_state_dict(saved["target"], strict=True)
    with torch.no_grad():
        before_q = _values(torch, critic, state_tensor, command_tensor)
    updates = 0
    journal = None
    reload_error = None
    started = time.monotonic()
    stop_reason = "budget_completed" if training else "frozen_replay"
    if isinstance(request, SACCriticWarmup):
        output = request.output_dir
        output.mkdir(parents=True)
        history = history_source.retain(output) if history_source is not None else empty_history()
        experience = seal_experience(replay, replay_file, output / "experience")
        (output / "actor").mkdir()
        for name, value in model_payload.items():
            write_file(output / "actor" / name, value)
        # Validate the sealed actor/experience before spending update budget.
        # Constructing a reload model must not alter the learner sampling RNG.
        with preserve_torch_state(torch):
            reloaded = FrozenNumericActor(
                output / "actor", pixels, expected_manifest_sha256=model_digest
            )
            reload_error = max(
                abs(a - b)
                for identity, row in reload_inputs.items()
                for a, b in zip(
                    encoded[identity][1],
                    predict_observation(reloaded, output / "experience", row)[1],
                )
            )
            if reloaded.manifest != actor.manifest or reload_error > 1e-6:
                raise ValueError("Frozen BC changed during checkpoint publication")
            del reloaded
        optimizer = torch.optim.Adam(critic.parameters(), lr=request.learning_rate)
        if saved is not None:
            optimizer.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["rng"])
        journal = RecordJournal(output, "diagnostics/updates.jsonl", "critic-update-jsonl-v1")
        stack.callback(journal.close)
        for step in range(start_step, start_step + request.steps):
            if (output / "stop.request").exists() or (
                stop_requested is not None and stop_requested(step)
            ):
                stop_reason = "stop_requested"
                break
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
            updates += 1
            try:
                journal.append(
                    {
                        "step": step + 1,
                        "transition_ids": [
                            replay["transitions"][i]["id"] for i in indices.tolist()
                        ],
                        "loss": float(loss.item()),
                        "targets": expected_value.tolist(),
                    }
                )
            except MemoryError as error:
                journal.unavailable(error)
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
        "steps_requested": request.steps if isinstance(request, SACCriticWarmup) else 0,
        "steps_completed": updates,
        "total_steps": start_step + updates,
        "warmup_budget": phase_budget,
        "warmup_remaining_steps": phase_budget - start_step - updates,
        "phase_status": "complete" if start_step + updates == phase_budget else "warming",
        "stop_reason": stop_reason,
        "updates": journal.finish() if journal is not None else None,
        "actor_optimizer_steps": 0,
        "actor_change_max": actor_change,
        "q_change_max": float((after_q - before_q).abs().max()) if training else None,
        "predictions": predictions,
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
        state = {
            "critic": critic.state_dict(),
            "target": target.state_dict(),
            "target_encoder": original,
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "step": start_step + updates,
        }
        for name, value in model_payload.items():
            if sha256_file(output / "actor" / name) != hashlib.sha256(value).hexdigest():
                raise ValueError("Frozen BC changed during checkpoint publication")
        summary["reload_max_abs_error"] = reload_error
        summary["learner_state_sha256"] = state_digest(torch, state)
        # Training diagnostics grow with steps/transitions. Keep them outside
        # the small checkpoint manifest consumed by reload and later resume.
        diagnostic_payload = encode(summary)
        manifest = {
            "version": 2,
            "stage": "critic_warmup",
            "bounds": asdict(bounds),
            "command_quantization": "clamp-then-round-nearest-even-v1",
            "replay_sha256": expected,
            "actor_manifest_sha256": model_digest,
            "configuration": {
                "steps": phase_budget,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "seed": request.seed,
                "target_tau": 0.005,
                "device": "cpu",
            },
            "training_report_sha256": hashlib.sha256(diagnostic_payload).hexdigest(),
            "learner_state_sha256": summary["learner_state_sha256"],
            "experience": experience,
            "continuation": continuation,
            "history": history,
            "resume_contract": critic_resume_contract(torch),
        }
        publish_checkpoint(torch, output, manifest, state, diagnostic_payload)
        report = optional_report(
            output / "report.html",
            "BC 冻结与双 Q 预热",
            summary,
            fallback=output / "training-report.json",
        )
    else:
        report = request.report_path
        report.parent.mkdir(parents=True, exist_ok=True)
        write_file(
            report,
            (
                '<!doctype html><meta charset="utf-8"><h1>BC 冻结与双 Q 预热</h1><pre>'
                + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
                + "</pre>"
            ).encode("utf-8"),
        )
    return RunResult({}, [], [], {"sac": summary}, report)
