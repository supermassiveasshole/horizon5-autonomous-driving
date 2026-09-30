"""Finite CPU SAC updates initialized by audited frozen-BC critic warm-up."""

from __future__ import annotations

import hashlib
import html
import importlib
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.collection_store import encode, read_bounded, write_file
from fh5.learning_runtime import preserve_torch_state
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import PixelContract
from fh5.sac import _q_heads, _values
from fh5.sac_actions import ActionBounds
from fh5.sac_data import LearningReplay
from fh5.sac_policy import encode_history, make_policy, soft_update

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class SACTrain:
    warmup_dir: Path
    replay_file: Path
    output_dir: Path
    steps: int = 100
    batch_size: int = 32
    critic_lr: float = 0.0001
    actor_lr: float = 0.00003
    encoder_lr: float = 0.00001
    alpha_lr: float = 0.0001
    initial_alpha: float = 0.01
    initial_log_std: float = -3.0
    normalized_target_entropy: float = -2.0
    actor_interval: int = 2
    tau: float = 0.005
    seed: int = 7


@dataclass(frozen=True)
class SACPolicyReplay:
    checkpoint_dir: Path
    replay_file: Path
    report_path: Path
    noise: tuple[float, float] = (0.0, 0.0)


def _difference(torch: Any, before: dict[str, Any], module: Any) -> float:
    return max(float((before[k] - v).abs().max()) for k, v in module.state_dict().items())


def _snapshot(module: Any) -> dict[str, Any]:
    return deepcopy(module.state_dict())


def policy_predictions(
    torch: Any,
    encoder: Any,
    policy: Any,
    data: LearningReplay,
    normalized_entropy: float,
    noise: tuple[float, float] = (0.0, 0.0),
) -> list[dict[str, Any]]:
    if len(noise) != 2 or any(not math.isfinite(v) or abs(v) > 10 for v in noise):
        raise ValueError("SAC replay noise must be a finite two-axis standard-normal diagnostic")
    result = []
    with torch.no_grad():
        for i, row in enumerate(data.rows):
            features = encode_history(torch, encoder, *data.inputs([i]))
            context = data.context[i : i + 1]
            output = policy(features, context, features.new_tensor([noise]))
            result.append(
                {
                    "id": row["id"],
                    "context": context[0].tolist(),
                    **{k: v[0].tolist() for k, v in output.items()},
                    "target_entropy": normalized_entropy + float(output["log_scale"][0]),
                }
            )
    return result


def run_sac_training(request: SACTrain) -> RunResult:
    torch = importlib.import_module("torch")
    with preserve_torch_state(torch):
        torch.set_num_threads(2)
        return _train(request, torch)


def _train(request: SACTrain, torch: Any) -> RunResult:
    from fh5.experiment import RunResult

    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    if (
        type(request.steps) is not int
        or not 0 <= request.steps <= 10_000
        or type(request.batch_size) is not int
        or not 1 <= request.batch_size <= 256
        or type(request.actor_interval) is not int
        or not 1 <= request.actor_interval <= 10
        or any(
            not 0 < v <= 0.01
            for v in (request.critic_lr, request.actor_lr, request.encoder_lr, request.alpha_lr)
        )
        or not 0 < request.initial_alpha <= 1
        or not -5 <= request.initial_log_std <= -1
        or not -10 <= request.normalized_target_entropy <= 0
        or not 0 < request.tau <= 1
    ):
        raise ValueError("Invalid bounded SAC learning configuration")
    torch.manual_seed(request.seed)
    warm_bytes = read_bounded(request.warmup_dir / "critic.json", 1024**2)
    warm = json.loads(warm_bytes)
    if (warm.get("version"), warm.get("stage")) != (1, "critic_warmup"):
        raise ValueError("SAC updates require a frozen-BC critic warm-up")
    bounds = ActionBounds(**warm["bounds"])
    bc_bytes = {
        n: read_bounded(request.warmup_dir / "actor" / n, 256 * 1024**2)
        for n in ("model.json", "actor.pt")
    }
    bc_manifest = json.loads(bc_bytes["model.json"])
    pixels = PixelContract.from_metadata(bc_manifest["numeric_contract"])
    bc = FrozenNumericActor(
        request.warmup_dir / "actor", pixels, expected_manifest_sha256=warm["actor_manifest_sha256"]
    )
    if (
        hashlib.sha256(bc_bytes["model.json"]).hexdigest() != warm["actor_manifest_sha256"]
        or hashlib.sha256(bc_bytes["actor.pt"]).hexdigest() != bc.manifest["weights_sha256"]
    ):
        raise ValueError("BC bytes changed while loading SAC initialization")
    payload = read_bounded(request.warmup_dir / "critic.pt", 256 * 1024**2)
    if hashlib.sha256(payload).hexdigest() != warm["weights_sha256"]:
        raise ValueError("Warm-up checkpoint changed")
    saved = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    if any(
        not torch.equal(v, saved["target_encoder"][k]) for k, v in bc.model.state_dict().items()
    ):
        raise ValueError("Warm-up target encoder is not the frozen BC")
    data = LearningReplay(torch, request.replay_file, warm["replay_sha256"], bc, bounds)
    feature_width = 64 * (bc.original_contract["image_count"] + 1)
    encoder = torch.nn.ModuleDict(
        {"images": deepcopy(bc.model.encoder), "state": deepcopy(bc.model.state)}
    )
    policy = make_policy(torch, bc.model.fusion, feature_width, request.initial_log_std)
    critic = _q_heads(torch, feature_width + 12)
    critic.load_state_dict(saved["critic"], strict=True)
    target_critic = deepcopy(critic)
    target_critic.load_state_dict(saved["target"], strict=True)
    target_encoder = deepcopy(encoder)
    for module in (target_critic, target_encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    critic_parameters = list(encoder.parameters()) + list(critic.parameters())
    actor_parameters = list(policy.parameters())
    if set(map(id, critic_parameters)) & set(map(id, actor_parameters)):
        raise ValueError("SAC optimizers have duplicate parameter ownership")
    critic_optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": request.encoder_lr},
            {"params": critic.parameters(), "lr": request.critic_lr},
        ]
    )
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=request.actor_lr)
    log_alpha = torch.tensor(math.log(request.initial_alpha), requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=request.alpha_lr)
    initial_encoder, initial_policy, initial_critic = map(_snapshot, (encoder, policy, critic))
    initial_target = _snapshot(target_encoder)

    before = policy_predictions(torch, encoder, policy, data, request.normalized_target_entropy)
    transfer_error = 0.0
    with torch.no_grad():
        for i, row in enumerate(data.rows):
            teacher = bc.model(*data.inputs([i]))[0].tolist()
            expected = bounds.deterministic(
                teacher, row["previous_action"], row["action_elapsed_s"]
            )
            transfer_error = max(
                transfer_error,
                max(
                    abs(round(a * scale) - round(b * scale))
                    for a, b, scale in zip(expected, before[i]["deterministic"], (32767, 255))
                ),
            )
    if transfer_error:
        raise ValueError("SAC handoff changes deterministic BC commands")
    updates, encoder_actor_change, actor_critic_change = 0, 0.0, 0.0
    started = time.monotonic()
    metrics = []
    for step in range(request.steps):
        indices = torch.randperm(len(data.rows))[: request.batch_size].tolist()
        context, task = data.context[indices], data.task[indices]
        next_context, next_task = data.next_context[indices], data.next_task[indices]
        inputs, next_inputs = data.inputs(indices), data.inputs(indices, following=True)
        before_actor = _snapshot(policy)
        with torch.no_grad():
            next_features = encode_history(torch, encoder, *next_inputs)
            sampled = policy(next_features, next_context)
            target_features = encode_history(torch, target_encoder, *next_inputs)
            target_state = torch.cat([target_features, next_context, next_task], dim=1)
            boot = _values(torch, target_critic, target_state, sampled["command"]).min(dim=1).values
            expected = data.rewards[indices] + data.discounts[indices] * (
                boot - log_alpha.exp() * sampled["log_probability"]
            )
        features = encode_history(torch, encoder, *inputs)
        states = torch.cat([features, context, task], dim=1)
        predicted = _values(torch, critic, states, data.actions[indices])
        critic_loss = (predicted - expected[:, None]).square().mean()
        if not torch.isfinite(critic_loss):
            raise ValueError("Non-finite SAC critic loss")
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_optimizer.step()
        actor_critic_change = max(actor_critic_change, _difference(torch, before_actor, policy))
        entry = {
            "step": step + 1,
            "transition_ids": [data.rows[i]["id"] for i in indices],
            "critic_loss": float(critic_loss.detach()),
            "targets": expected.tolist(),
        }
        if (step + 1) % request.actor_interval == 0:
            before_encoder = _snapshot(encoder)
            critic_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features = encode_history(torch, encoder, *inputs)
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
            sampled = policy(features.detach(), context)
            q = (
                _values(
                    torch,
                    critic,
                    torch.cat([features.detach(), context, task], dim=1),
                    sampled["command"],
                )
                .min(dim=1)
                .values
            )
            actor_loss = (log_alpha.exp().detach() * sampled["log_probability"] - q).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optimizer.step()
            for parameter in critic.parameters():
                parameter.requires_grad_(True)
            target_entropy = request.normalized_target_entropy + sampled["log_scale"]
            alpha_loss = -(
                log_alpha * (sampled["log_probability"] + target_entropy).detach()
            ).mean()
            alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            alpha_optimizer.step()
            if any(p.grad is not None for p in encoder.parameters()):
                raise ValueError("Actor gradient leaked into shared encoder")
            encoder_actor_change = max(
                encoder_actor_change, _difference(torch, before_encoder, encoder)
            )
            entry.update(
                actor_loss=float(actor_loss.detach()), alpha_loss=float(alpha_loss.detach())
            )
            updates += 1
        soft_update(torch, target_critic, critic, request.tau)
        soft_update(torch, target_encoder, encoder, request.tau)
        if any(
            not torch.isfinite(p).all() for m in (encoder, policy, critic) for p in m.parameters()
        ) or not torch.isfinite(log_alpha):
            raise ValueError("Non-finite SAC update")
        metrics.append(entry)
    summary = {
        "stage": "sac_updates",
        "source_kind": "synthetic",
        "device": "cpu",
        "update_duration_s": time.monotonic() - started,
        "steps_completed": request.steps,
        "actor_updates": updates,
        "encoder_change_max": _difference(torch, initial_encoder, encoder),
        "actor_change_max": _difference(torch, initial_policy, policy),
        "critic_change_max": _difference(torch, initial_critic, critic),
        "target_encoder_change_max": _difference(torch, initial_target, target_encoder),
        "alpha_before": request.initial_alpha,
        "alpha_after": float(log_alpha.exp().detach()),
        "optimizer_parameters_disjoint": True,
        "encoder_change_during_actor_max": encoder_actor_change,
        "actor_change_during_critic_max": actor_critic_change,
        "bc_transfer_command_error": transfer_error,
        "before_predictions": before,
        "predictions": policy_predictions(
            torch, encoder, policy, data, request.normalized_target_entropy
        ),
        "updates": metrics,
        "raw_frame_bytes": data.raw_bytes,
        "feature_cache": "none; re-encode after updates",
        "density_coordinates": "continuous command before integer quantization",
        "q_action_coordinates": "actual rounded command; straight-through actor gradient surrogate",
        "commands_sent": False,
        "real_driving_validated": False,
    }
    output = request.output_dir
    output.mkdir(parents=True)
    (output / "bc").mkdir()
    for name, value in bc_bytes.items():
        write_file(output / "bc" / name, value)
    config = {k: v for k, v in asdict(request).items() if not isinstance(v, Path)}
    metadata = {
        "version": 1,
        "stage": "sac_updates",
        "architecture": "conditional-temporal-sac-v1",
        "configuration": config,
        "bounds": asdict(bounds),
        "replay_sha256": warm["replay_sha256"],
        "warmup_manifest_sha256": hashlib.sha256(warm_bytes).hexdigest(),
        "bc_manifest_sha256": hashlib.sha256(bc_bytes["model.json"]).hexdigest(),
        "density_coordinates": summary["density_coordinates"],
        "q_action_coordinates": summary["q_action_coordinates"],
        "device": "cpu",
        "source_kind": "synthetic",
        "real_driving_validated": False,
    }
    torch.save(
        {
            "metadata": metadata,
            "encoder": encoder.state_dict(),
            "policy": policy.state_dict(),
            "critic": critic.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "target_critic": target_critic.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "log_alpha": log_alpha.detach(),
            "alpha_optimizer": alpha_optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "step": request.steps,
        },
        output / "policy.pt",
    )
    manifest = {
        **metadata,
        "weights_sha256": hashlib.sha256((output / "policy.pt").read_bytes()).hexdigest(),
    }
    write_file(output / "policy.json", encode(manifest))
    write_file(output / "training-report.json", encode(summary))
    report = output / "report.html"
    report.write_text(
        '<!doctype html><meta charset="utf-8"><h1>SAC 软件更新</h1><pre>'
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>",
        encoding="utf-8",
    )
    return RunResult({}, [], [], {"sac_learning": summary}, report)


def run_sac_policy_replay(request: SACPolicyReplay) -> RunResult:
    from fh5.experiment import RunResult

    torch = importlib.import_module("torch")
    with preserve_torch_state(torch):
        torch.set_num_threads(2)
        manifest = json.loads(read_bounded(request.checkpoint_dir / "policy.json", 1024**2))
        if (manifest.get("version"), manifest.get("architecture"), manifest.get("stage")) != (
            1,
            "conditional-temporal-sac-v1",
            "sac_updates",
        ):
            raise ValueError("Unsupported SAC policy checkpoint")
        payload = read_bounded(request.checkpoint_dir / "policy.pt", 256 * 1024**2)
        if hashlib.sha256(payload).hexdigest() != manifest["weights_sha256"]:
            raise ValueError("SAC policy checkpoint changed")
        saved = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
        if saved["metadata"] != {k: v for k, v in manifest.items() if k != "weights_sha256"}:
            raise ValueError("SAC policy metadata mismatch")
        model_dir = request.checkpoint_dir / "bc"
        bc_manifest = json.loads(read_bounded(model_dir / "model.json", 1024**2))
        pixels = PixelContract.from_metadata(bc_manifest["numeric_contract"])
        bc = FrozenNumericActor(
            model_dir, pixels, expected_manifest_sha256=manifest["bc_manifest_sha256"]
        )
        data = LearningReplay(
            torch,
            request.replay_file,
            manifest["replay_sha256"],
            bc,
            ActionBounds(**manifest["bounds"]),
        )
        encoder = torch.nn.ModuleDict(
            {"images": deepcopy(bc.model.encoder), "state": deepcopy(bc.model.state)}
        )
        policy = make_policy(
            torch,
            bc.model.fusion,
            64 * (bc.original_contract["image_count"] + 1),
            manifest["configuration"]["initial_log_std"],
        )
        encoder.load_state_dict(saved["encoder"], strict=True)
        policy.load_state_dict(saved["policy"], strict=True)
        if any(not torch.isfinite(p).all() for m in (encoder, policy) for p in m.parameters()):
            raise ValueError("Non-finite SAC checkpoint")
        summary = {
            "predictions": policy_predictions(
                torch,
                encoder,
                policy,
                data,
                manifest["configuration"]["normalized_target_entropy"],
                request.noise,
            ),
            "commands_sent": False,
            "real_driving_validated": False,
            "density_coordinates": manifest["density_coordinates"],
            "q_action_coordinates": manifest["q_action_coordinates"],
        }
    request.report_path.parent.mkdir(parents=True, exist_ok=True)
    request.report_path.write_text(
        '<!doctype html><meta charset="utf-8"><h1>SAC 冻结策略回放</h1><pre>'
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>",
        encoding="utf-8",
    )
    return RunResult({}, [], [], {"sac_policy": summary}, request.report_path)
