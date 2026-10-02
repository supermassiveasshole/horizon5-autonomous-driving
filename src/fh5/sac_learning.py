"""Finite CPU SAC updates initialized by audited frozen-BC critic warm-up."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, closing
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, NamedTuple

from fh5.artifact_io import VerifiedFile, copy_evidence
from fh5.checkpoint_history import HistorySource, empty_history
from fh5.collection_store import encode, read_bounded
from fh5.learning_diagnostics import PredictionRecorder, RecordJournal
from fh5.learning_runtime import preserve_torch_state
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import PixelContract
from fh5.presentation import optional_report
from fh5.sac import _q_heads, _values
from fh5.sac_actions import ActionBounds
from fh5.sac_actor import FrozenSAC
from fh5.sac_checkpoint import (
    checkpoint_history,
    continuation_history,
    experience_frames,
    publish_checkpoint,
    read_checkpoint,
    read_critic_checkpoint,
    resume_contract,
    seal_experience,
    state_digest,
)
from fh5.sac_data import LearningReplay, validate_cache_budget
from fh5.sac_experience import expand_experience
from fh5.sac_imitation import (
    checkpoint_imitation,
    freeze_imitation_protocol,
    guidance_loss,
    imitation_evidence,
    initial_imitation,
)
from fh5.sac_policy import encode_history, make_policy, soft_update
from fh5.sac_source_files import recording_origins, source_replays
from fh5.sac_sources import ReplaySampling

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
    demonstration_fraction: float | None = None
    imitation_weights: tuple[float, ...] | None = None
    imitation_protocol_batch: Path | None = None
    raw_cache_bytes: int = 0


@dataclass(frozen=True)
class SACPolicyReplay:
    checkpoint_dir: Path
    replay_file: Path
    report_path: Path
    noise: tuple[float, float] = (0.0, 0.0)
    raw_cache_bytes: int | None = None


@dataclass(frozen=True)
class SACResume:
    checkpoint_dir: Path
    output_dir: Path
    steps: int = 100
    additions: tuple[tuple[Path, str], ...] = ()
    expected_checkpoint_sha256: str | None = None
    demonstration_fraction: float | None = None
    imitation_comparison: Path | None = None
    imitation_registry: Path | None = None
    raw_cache_bytes: int | None = None


class _Learner(NamedTuple):
    encoder: Any
    policy: Any
    critic: Any
    target_critic: Any
    target_encoder: Any
    critic_optimizer: Any
    actor_optimizer: Any
    log_alpha: Any
    alpha_optimizer: Any
    start_step: int

    def state(self, torch: Any, step: int) -> dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "step": step,
        }


def _validate_configuration(request: SACTrain) -> None:
    validate_cache_budget(request.raw_cache_bytes)
    if (
        type(request.steps) is not int
        or request.steps < 0
        or type(request.batch_size) is not int
        or request.batch_size < 1
        or type(request.actor_interval) is not int
        or request.actor_interval < 1
        or any(
            not 0 < v <= 0.01
            for v in (request.critic_lr, request.actor_lr, request.encoder_lr, request.alpha_lr)
        )
        or not 0 < request.initial_alpha <= 1
        or not -5 <= request.initial_log_std <= -1
        or not -10 <= request.normalized_target_entropy <= 0
        or not 0 < request.tau <= 1
    ):
        raise ValueError("Invalid SAC learning configuration")


def _make_learner(
    torch: Any, bc: FrozenNumericActor, saved: dict[str, Any], request: SACTrain, *, resume: bool
) -> _Learner:
    feature_width = 64 * (bc.original_contract["image_count"] + 1)
    encoder = torch.nn.ModuleDict(
        {"images": deepcopy(bc.model.encoder), "state": deepcopy(bc.model.state)}
    )
    policy = make_policy(torch, bc.model.fusion, feature_width, request.initial_log_std)
    critic = _q_heads(torch, feature_width + 12)
    critic.load_state_dict(saved["critic"], strict=True)
    target_critic = deepcopy(critic)
    target_critic.load_state_dict(saved["target_critic" if resume else "target"], strict=True)
    target_encoder = deepcopy(encoder)
    if resume:
        encoder.load_state_dict(saved["encoder"], strict=True)
        policy.load_state_dict(saved["policy"], strict=True)
        target_encoder.load_state_dict(saved["target_encoder"], strict=True)
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
    start_step = 0
    if resume:
        critic_optimizer.load_state_dict(saved["critic_optimizer"])
        actor_optimizer.load_state_dict(saved["actor_optimizer"])
        with torch.no_grad():
            log_alpha.copy_(saved["log_alpha"])
        alpha_optimizer.load_state_dict(saved["alpha_optimizer"])
        start_step = saved["step"]
        if type(start_step) is not int or start_step < 0:
            raise ValueError("Invalid SAC continuation step")
        torch.set_rng_state(saved["rng"])
    return _Learner(
        encoder,
        policy,
        critic,
        target_critic,
        target_encoder,
        critic_optimizer,
        actor_optimizer,
        log_alpha,
        alpha_optimizer,
        start_step,
    )


def validate_sac_candidate(root: Path, expected_sha256: str) -> dict[str, Any]:
    """Restore and inspect all learning state without publishing or adding a stage."""
    torch = importlib.import_module("torch")
    with preserve_torch_state(torch), ExitStack() as resources:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        manifest, saved, raw = read_checkpoint(torch, root)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("Candidate checkpoint changed during validation")
        if manifest["version"] not in (2, 3, 4):
            raise ValueError("Candidate requires a sealed SAC continuation checkpoint")
        checkpoint_history(root, manifest)
        replay_file = root / "experience/replay.json"
        configuration = dict(manifest["configuration"], steps=0)
        # Missing in legacy checkpoints: preserve their historical cache configuration.
        configuration.setdefault("raw_cache_bytes", 512 * 1024**2)
        request = SACTrain(root, replay_file, root, **configuration)
        _validate_configuration(request)
        torch.manual_seed(request.seed)
        bc_manifest = json.loads(read_bounded(root / "bc/model.json", 1024**2))
        bc = FrozenNumericActor(
            root / "bc",
            PixelContract.from_metadata(bc_manifest["numeric_contract"]),
            expected_manifest_sha256=manifest["bc_manifest_sha256"],
        )
        data = LearningReplay(
            torch,
            replay_file,
            manifest["replay_sha256"],
            bc,
            ActionBounds(**manifest["bounds"]),
            resources=resources,
            cache_bytes=request.raw_cache_bytes,
        )
        replay = data.replay
        with closing(source_replays(replay_file.parent, replay)) as sources:
            for _ in sources:
                pass
        with closing(experience_frames(replay_file.parent, replay)) as frames:
            for _ in frames:
                pass
        ReplaySampling(data.roles, request.batch_size, request.demonstration_fraction)
        learner = _make_learner(torch, bc, saved, request, resume=True)
        # Exercise the actual numerical observation contract and reject non-finite output.
        predictions = PredictionRecorder()
        for row in policy_predictions(
            torch, learner.encoder, learner.policy, data, request.normalized_target_entropy
        ):
            predictions.add(row)
        digest = state_digest(torch, learner.state(torch, learner.start_step))
        if digest != manifest["learner_state_sha256"]:
            raise ValueError("Restoring candidate changed its learner state")
        return {"learner_state_sha256": digest, "total_steps": learner.start_step}


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
) -> Iterator[dict[str, Any]]:
    if len(noise) != 2 or any(not math.isfinite(v) or abs(v) > 10 for v in noise):
        raise ValueError("SAC replay noise must be a finite two-axis standard-normal diagnostic")
    for i, row in enumerate(data.rows):
        with torch.no_grad():
            features = encode_history(torch, encoder, *data.inputs([i]))
            context = data.batch([i]).context
            output = policy(features, context, features.new_tensor([noise]))
            prediction = {
                "id": row["id"],
                "context": context[0].tolist(),
                **{k: v[0].tolist() for k, v in output.items()},
                "target_entropy": normalized_entropy + float(output["log_scale"][0]),
            }
        yield prediction


def run_sac_training(
    request: SACTrain | SACResume, stop_requested: Callable[[int], bool] | None = None
) -> RunResult:
    torch = importlib.import_module("torch")
    with preserve_torch_state(torch), ExitStack() as resources:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        return _train(request, torch, stop_requested, resources)


def _train(
    operation: SACTrain | SACResume,
    torch: Any,
    stop_requested: Callable[[int], bool] | None,
    resources: ExitStack,
) -> RunResult:
    from fh5.experiment import RunResult

    continuation = None
    restored = None
    history_source: HistorySource | None = None
    history_blobs: dict[str, bytes | VerifiedFile] = {}
    if isinstance(operation, SACResume):
        manifest, restored, parent_bytes = read_checkpoint(torch, operation.checkpoint_dir)
        if (
            operation.expected_checkpoint_sha256 is not None
            and hashlib.sha256(parent_bytes).hexdigest() != operation.expected_checkpoint_sha256
        ):
            raise ValueError("SAC continuation differs from its expected parent checkpoint")
        if manifest["version"] not in (2, 3, 4):
            raise ValueError("SAC continuation requires a sealed version 2, 3 or 4 checkpoint")
        history_source = continuation_history(operation.checkpoint_dir, manifest, parent_bytes)
        configuration = dict(manifest["configuration"], steps=operation.steps)
        configuration.setdefault("raw_cache_bytes", 512 * 1024**2)
        if operation.demonstration_fraction is not None:
            configuration["demonstration_fraction"] = operation.demonstration_fraction
        if operation.raw_cache_bytes is not None:
            configuration["raw_cache_bytes"] = operation.raw_cache_bytes
        request = SACTrain(
            operation.checkpoint_dir,
            operation.checkpoint_dir / "experience/replay.json",
            operation.output_dir,
            **configuration,
        )
        continuation = {
            "parent_checkpoint_sha256": hashlib.sha256(parent_bytes).hexdigest(),
            "parent_step": restored["step"],
        }
    else:
        request = operation
    if request.imitation_protocol_batch is not None and request.imitation_weights is None:
        raise ValueError("Imitation protocol requires explicit imitation weights")
    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    if any(
        request.output_dir.resolve().is_relative_to(source.resolve())
        for source in (request.warmup_dir, request.replay_file.parent)
    ):
        raise ValueError("SAC output must be outside its frozen sources")
    _validate_configuration(request)
    torch.manual_seed(request.seed)
    if restored is None:
        warm, initial_critic, warm_bytes = read_critic_checkpoint(torch, request.warmup_dir)
        if warm["version"] == 2 and initial_critic["step"] != warm["configuration"]["steps"]:
            raise ValueError("Finish finite critic warm-up before starting SAC updates")
        if warm["version"] == 2:
            history_source = continuation_history(request.warmup_dir, warm, warm_bytes)
        warm_sha = hashlib.sha256(warm_bytes).hexdigest()
        bc_dir = request.warmup_dir / "actor"
    else:
        warm = {
            "bounds": manifest["bounds"],
            "replay_sha256": manifest["replay_sha256"],
            "actor_manifest_sha256": manifest["bc_manifest_sha256"],
        }
        warm_sha = manifest["warmup_manifest_sha256"]
        bc_dir = request.warmup_dir / "bc"
    bounds = ActionBounds(**warm["bounds"])
    bc_bytes = read_bounded(bc_dir / "model.json", 256 * 1024**2)
    bc_manifest = json.loads(bc_bytes)
    bc_digest = hashlib.sha256(bc_bytes).hexdigest()
    if bc_digest != warm["actor_manifest_sha256"]:
        raise ValueError("BC bytes changed while loading SAC initialization")
    imitation = (
        initial_imitation(
            request.imitation_weights,
            warm["actor_manifest_sha256"],
            freeze_imitation_protocol(request.imitation_protocol_batch)
            if request.imitation_protocol_batch is not None
            else None,
        )
        if request.imitation_weights is not None
        else None
    )
    if restored is not None:
        imitation = checkpoint_imitation(manifest)
        assert isinstance(operation, SACResume)
        history_blobs.update(imitation_evidence(operation.checkpoint_dir, imitation))
    imitation_review = None
    pixels = PixelContract.from_metadata(bc_manifest["numeric_contract"])
    bc = FrozenNumericActor(bc_dir, pixels, expected_manifest_sha256=warm["actor_manifest_sha256"])
    bc_weights = VerifiedFile(bc_dir / "actor.pt", bc.manifest["weights_sha256"])
    if restored is None:
        saved = initial_critic
        if any(
            not torch.equal(v, saved["target_encoder"][k]) for k, v in bc.model.state_dict().items()
        ):
            raise ValueError("Warm-up target encoder is not the frozen BC")
    else:
        saved = restored
    added = 0
    expansion = None
    if isinstance(operation, SACResume) and operation.additions:
        if any(
            operation.output_dir.resolve().is_relative_to(p.parent.resolve())
            for p, _ in operation.additions
        ):
            raise ValueError("SAC output must be outside its experience additions")
        temporary = Path(resources.enter_context(TemporaryDirectory(prefix="fh5-sac-")))
        replay_file, replay_sha, added, expansion = expand_experience(
            torch,
            request.replay_file,
            warm["replay_sha256"],
            operation.additions,
            temporary / "expanded",
            bc,
            bounds,
        )
        if request.steps > added:
            raise ValueError("SAC expansion permits at most one update per new transition")
        request = replace(request, replay_file=replay_file)
        warm["replay_sha256"] = replay_sha
        assert continuation is not None
        continuation["experience_additions"] = [sha for _, sha in operation.additions]
        continuation["new_transition_credit"] = added
    data = LearningReplay(
        torch,
        request.replay_file,
        warm["replay_sha256"],
        bc,
        bounds,
        resources=resources,
        cache_bytes=request.raw_cache_bytes,
    )
    if isinstance(operation, SACResume) and operation.imitation_comparison is not None:
        if imitation is None or imitation["protocol_sha256"] is None:
            raise ValueError("Imitation exit requires a protocol frozen before training")
        from fh5.sac_imitation_review import review_imitation

        root = Path(resources.enter_context(TemporaryDirectory(prefix="fh5-imitation-")))
        replay = data.replay
        imitation, imitation_review, proof_blobs = review_imitation(
            imitation,
            operation.imitation_comparison,
            operation.imitation_registry,
            root / "comparison",
            parent_sha256=hashlib.sha256(parent_bytes).hexdigest(),
            bc_manifest=bc_manifest,
            learning_origins=recording_origins(request.replay_file.parent, replay),
        )
        history_blobs.update(proof_blobs)
    sampling = ReplaySampling(data.roles, request.batch_size, request.demonstration_fraction)
    learner = _make_learner(torch, bc, saved, request, resume=restored is not None)
    (
        encoder,
        policy,
        critic,
        target_critic,
        target_encoder,
        critic_optimizer,
        actor_optimizer,
        log_alpha,
        alpha_optimizer,
        start_step,
    ) = learner
    teacher_initial = _snapshot(bc.model) if imitation is not None else None
    teacher_evaluations = 0
    if imitation is not None:
        bc.model.requires_grad_(False)
    alpha_before = float(log_alpha.exp().detach())
    initial_encoder, initial_policy, initial_critic = map(_snapshot, (encoder, policy, critic))
    initial_target = _snapshot(target_encoder)

    output = request.output_dir
    output.mkdir(parents=True)
    before_journal = RecordJournal(
        output, "diagnostics/before-predictions.jsonl", "sac-prediction-jsonl-v1"
    )
    resources.callback(before_journal.close)
    before = PredictionRecorder(before_journal)
    transfer_error: float | None = None
    if restored is None:
        transfer_error = 0.0
    for i, prediction in enumerate(
        policy_predictions(torch, encoder, policy, data, request.normalized_target_entropy)
    ):
        before.add(prediction)
        if restored is None:
            row = data.rows[i]
            with torch.no_grad():
                teacher = bc.model(*data.inputs([i]))[0].tolist()
            expected = bounds.deterministic(
                teacher, row["previous_action"], row["action_elapsed_s"]
            )
            transfer_error = max(
                transfer_error or 0.0,
                max(
                    abs(round(a * scale) - round(b * scale))
                    for a, b, scale in zip(expected, prediction["deterministic"], (32767, 255))
                ),
            )
    if transfer_error:
        raise ValueError("SAC handoff changes deterministic BC commands")
    before_summary = before.finish()
    history = history_source.retain(output) if history_source is not None else empty_history()
    experience = seal_experience(data.replay, data.file, output / "experience")
    copy_evidence(output, history_blobs)
    copy_evidence(output / "bc", {"model.json": bc_bytes, "actor.pt": bc_weights})
    updates, encoder_actor_change, actor_critic_change = 0, 0.0, 0.0
    started = time.monotonic()
    journal = RecordJournal(output, "diagnostics/updates.jsonl", "sac-update-jsonl-v1")
    resources.callback(journal.close)
    steps_completed = 0
    stop_reason = "budget_completed"
    training_error = None
    for step in range(start_step, start_step + request.steps):
        if (
            (output / "stop.request").exists()
            or stop_requested is not None
            and stop_requested(step)
        ):
            stop_reason = "stop_requested"
            break
        sampling_rng = torch.get_rng_state()
        sampled_counts = dict(sampling.sampled)
        try:
            indices = sampling.sample(torch)
            batch = data.batch(indices)
            inputs, next_inputs = data.inputs(indices), data.inputs(indices, following=True)
        except (OSError, MemoryError) as error:
            # No optimizer has run for this batch. Preserve the exact next
            # sample as well as all completed updates when input I/O fails.
            torch.set_rng_state(sampling_rng)
            sampling.sampled = sampled_counts
            stop_reason = "training_data_unavailable"
            training_error = f"{type(error).__name__}: {error}"
            break
        context, task = batch.context, batch.task
        next_context, next_task = batch.next_context, batch.next_task
        before_actor = _snapshot(policy)
        with torch.no_grad():
            next_features = encode_history(torch, encoder, *next_inputs)
            sampled = policy(next_features, next_context)
            target_features = encode_history(torch, target_encoder, *next_inputs)
            target_state = torch.cat([target_features, next_context, next_task], dim=1)
            boot = _values(torch, target_critic, target_state, sampled["command"]).min(dim=1).values
            expected = batch.rewards + batch.discounts * (
                boot - log_alpha.exp() * sampled["log_probability"]
            )
        features = encode_history(torch, encoder, *inputs)
        states = torch.cat([features, context, task], dim=1)
        predicted = _values(torch, critic, states, batch.actions)
        critic_loss = (predicted - expected[:, None]).square().mean()
        if not torch.isfinite(critic_loss):
            raise ValueError("Non-finite SAC critic loss")
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_optimizer.step()
        actor_critic_change = max(actor_critic_change, _difference(torch, before_actor, policy))
        entry = {
            "step": step + 1,
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
            if imitation is not None:
                entry["sac_actor_loss"] = float(actor_loss.detach())
                entry["imitation_weight"] = imitation["weight"]
                entry["imitation_loss"] = 0.0
                if imitation["weight"]:
                    prior_loss = guidance_loss(torch, bc.model, inputs, context, sampled["mean"])
                    teacher_evaluations += 1
                    entry["imitation_loss"] = float(prior_loss.detach())
                    actor_loss = actor_loss + imitation["weight"] * prior_loss
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
        steps_completed += 1
        try:
            if journal.error is None:
                entry["transition_ids"] = [data.rows[i]["id"] for i in indices]
                entry["source_roles"] = [data.roles[i] for i in indices]
                journal.append(entry)
        except (OSError, MemoryError) as error:
            journal.unavailable(error)
    update_duration_s = time.monotonic() - started
    predictions_journal = RecordJournal(
        output, "diagnostics/predictions.jsonl", "sac-prediction-jsonl-v1"
    )
    resources.callback(predictions_journal.close)
    predictions = PredictionRecorder(predictions_journal)
    try:
        for row in policy_predictions(
            torch, encoder, policy, data, request.normalized_target_entropy
        ):
            predictions.add(row)
    except (OSError, MemoryError) as error:
        predictions.unavailable(error)
    summary = {
        "stage": "sac_updates",
        "source_kind": "synthetic",
        "device": "cpu",
        "update_duration_s": update_duration_s,
        "steps_requested": request.steps,
        "experience_added_transitions": added,
        "experience_expansion": expansion,
        "steps_completed": steps_completed,
        "total_steps": start_step + steps_completed,
        "stop_reason": stop_reason,
        "actor_updates": updates,
        "actor_updates_total": (start_step + steps_completed) // request.actor_interval,
        "encoder_change_max": _difference(torch, initial_encoder, encoder),
        "actor_change_max": _difference(torch, initial_policy, policy),
        "critic_change_max": _difference(torch, initial_critic, critic),
        "target_encoder_change_max": _difference(torch, initial_target, target_encoder),
        "alpha_before": alpha_before,
        "alpha_after": float(log_alpha.exp().detach()),
        "optimizer_parameters_disjoint": True,
        "encoder_change_during_actor_max": encoder_actor_change,
        "actor_change_during_critic_max": actor_critic_change,
        "bc_transfer_command_error": transfer_error,
        "before_predictions": before_summary,
        "predictions": predictions.finish(),
        "updates": journal.finish(),
        "sampling": sampling.report(),
        "raw_frame_bytes": data.source_bytes,
        "raw_frame_cache": data.cache_summary(),
        "feature_cache": "none; re-encode after updates",
        "density_coordinates": "continuous command before integer quantization",
        "q_action_coordinates": "actual rounded command; straight-through actor gradient surrogate",
        "commands_sent": False,
        "real_driving_validated": False,
    }
    if training_error is not None:
        summary["training_error"] = training_error
    config = {k: v for k, v in asdict(request).items() if not isinstance(v, Path)}
    config.pop("imitation_protocol_batch", None)
    if imitation_review is not None:
        summary["imitation_review"] = imitation_review
    if imitation is None:
        config.pop("imitation_weights")
    else:
        config["imitation_weights"] = imitation["weights"]
        assert teacher_initial is not None
        teacher_change = _difference(torch, teacher_initial, bc.model)
        if teacher_change:
            raise ValueError("Temporary imitation changed its frozen teacher")
        summary["imitation"] = {
            **imitation,
            "teacher_evaluations": teacher_evaluations,
            "teacher_change_max": teacher_change,
        }
    version = 4 if imitation is not None else 3 if request.demonstration_fraction is not None else 2
    metadata = {
        "version": version,
        "stage": "sac_updates",
        "architecture": "conditional-temporal-sac-v1",
        "configuration": config,
        "bounds": asdict(bounds),
        "replay_sha256": warm["replay_sha256"],
        "warmup_manifest_sha256": warm_sha,
        "bc_manifest_sha256": bc_digest,
        "density_coordinates": summary["density_coordinates"],
        "q_action_coordinates": summary["q_action_coordinates"],
        "device": "cpu",
        "source_kind": "synthetic",
        "real_driving_validated": False,
        "experience": experience,
        "continuation": continuation,
        "history": history,
        "resume_contract": resume_contract(torch, version),
    }
    if imitation is not None:
        metadata["imitation"] = imitation
    state = learner.state(torch, start_step + steps_completed)
    summary["learner_state_sha256"] = state_digest(torch, state)
    report_bytes = encode(summary)
    metadata["training_report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    metadata["learner_state_sha256"] = summary["learner_state_sha256"]
    imitation_evidence(output, imitation)
    publish_checkpoint(torch, output, metadata, state, report_bytes)
    report = optional_report(
        output / "report.html", "SAC 软件更新", summary, fallback=output / "training-report.json"
    )
    return RunResult({}, [], [], {"sac_learning": summary}, report)


def run_sac_policy_replay(request: SACPolicyReplay) -> RunResult:
    from fh5.experiment import RunResult

    if (
        request.report_path.exists()
        or request.report_path.is_symlink()
        or request.report_path.with_suffix(".json").exists()
        or request.report_path.with_suffix(".json").is_symlink()
    ):
        raise FileExistsError(request.report_path)
    if request.report_path.suffix.lower() != ".html":
        raise ValueError("SAC policy replay requires a new HTML report")
    torch = importlib.import_module("torch")
    with preserve_torch_state(torch), ExitStack() as resources:
        torch.set_num_threads(2)
        frozen = FrozenSAC(torch, request.checkpoint_dir, allow_legacy=True)
        manifest = frozen.manifest
        data = LearningReplay(
            torch,
            request.replay_file,
            manifest["replay_sha256"],
            frozen.bc,
            frozen.bounds,
            resources=resources,
            cache_bytes=request.raw_cache_bytes
            if request.raw_cache_bytes is not None
            else manifest["configuration"].get("raw_cache_bytes", 512 * 1024**2),
        )
        journal = RecordJournal(
            request.report_path.parent,
            request.report_path.with_suffix(".predictions.jsonl").name,
            "sac-prediction-jsonl-v1",
        )
        resources.callback(journal.close)
        predictions = PredictionRecorder(journal)
        for row in policy_predictions(
            torch,
            frozen.encoder,
            frozen.policy,
            data,
            manifest["configuration"]["normalized_target_entropy"],
            request.noise,
        ):
            predictions.add(row)
        summary = {
            "predictions": predictions.finish(),
            "commands_sent": False,
            "real_driving_validated": False,
            "density_coordinates": manifest["density_coordinates"],
            "q_action_coordinates": manifest["q_action_coordinates"],
            "raw_frame_cache": data.cache_summary(),
        }
    report = optional_report(
        request.report_path,
        "SAC 冻结策略回放",
        summary,
        fallback=request.checkpoint_dir / "policy.json",
        diagnostic=request.report_path.with_suffix(".json"),
    )
    return RunResult({}, [], [], {"sac_policy": summary}, report)
