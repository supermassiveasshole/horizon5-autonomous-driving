"""Frozen numeric snapshots, bounded temporal BC and offline prediction reports."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import time
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.artifact_io import VerifiedFile, sha256_file
from fh5.bc_learning import VIEWS, _checked_config, _write
from fh5.bc_losses import BCLossHistory, read_bc_manifest
from fh5.bc_network import make_actor
from fh5.collection_store import read_bounded
from fh5.learning_runtime import TrainingBudget, move_learning_state, preserve_torch_state
from fh5.numeric_images import (
    NumericDecision,
    NumericFrame,
    asset,
)
from fh5.numeric_recording import _result
from fh5.numeric_report import preview_png
from fh5.prediction_metrics import temporal_metrics
from fh5.prediction_records import PredictionRecords, prediction_spool
from fh5.replay_document import replay_document
from fh5.temporal_data import temporal_snapshot
from fh5.temporal_features import (
    TEMPORAL_ARCHITECTURE,
    TEMPORAL_METADATA_KEYS,
    actor_shape,
    describe_time,
    temporal_features,
    time_contract,
)

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class TemporalBCTrain:
    config_file: Path
    output_dir: Path


@dataclass(frozen=True)
class TemporalBCReplay:
    model_dir: Path
    dataset_file: Path
    report_path: Path
    device: str = "cpu"


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = read_bounded(path, 128 * 1024**2)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Expected temporal BC JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _configuration(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    value, digest = _read(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Frozen training configuration changed")
    if set(value) != {
        "version",
        "dataset",
        "dataset_sha256",
        "seed",
        "steps",
        "batch_size",
        "learning_rate",
        "device",
        "time_mode",
    }:
        raise ValueError("Unsupported temporal training config")
    legacy = {k: v for k, v in value.items() if k not in ("time_mode", "dataset_sha256")}
    _checked_config(dict(legacy, image_size=[64, 36]))
    if (
        value["time_mode"] not in ("actual", "fixed")
        or not isinstance(value["dataset_sha256"], str)
        or len(value["dataset_sha256"]) != 64
    ):
        raise ValueError("Invalid time mode or frozen dataset hash")
    return value


def run_temporal_bc(
    request: TemporalBCTrain | TemporalBCReplay,
    budget: TrainingBudget | None = None,
    cpu_threads: int = 2,
) -> RunResult:
    torch = importlib.import_module("torch")
    with preserve_torch_state(torch), ExitStack() as resources, prediction_spool() as records:
        torch.set_num_threads(cpu_threads)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        return _run(request, torch, resources, records, budget)


def _run(
    request: TemporalBCTrain | TemporalBCReplay,
    torch: Any,
    resources: ExitStack,
    records: PredictionRecords,
    budget: TrainingBudget | None = None,
) -> RunResult:
    from fh5.numeric_actor import FrozenNumericActor

    manifest_digest = None
    if isinstance(request, TemporalBCTrain):
        if request.output_dir.exists():
            raise FileExistsError(request.output_dir)
        config = _configuration(request.config_file)
        dataset = request.config_file.parent / config["dataset"]
        output, report, device = (
            request.output_dir,
            request.output_dir / "report.html",
            config["device"],
        )
    else:
        manifest, manifest_digest = read_bc_manifest(request.model_dir / "model.json")
        config = manifest["config"]
        dataset, output, report, device = (
            request.dataset_file,
            request.model_dir,
            request.report_path,
            request.device,
        )
    digest = sha256_file(dataset)
    if digest != config["dataset_sha256"]:
        raise ValueError("Frozen numerical dataset hash mismatch")
    data, pixels, rows = resources.enter_context(temporal_snapshot(dataset, expected_sha256=digest))
    if budget:
        budget.checkpoint("snapshot_validated", 0)
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested temporal BC device unavailable")
    timing = time_contract(config["time_mode"], pixels)
    first: NumericDecision = rows[0]["decision"]
    contract = {
        "image_size": list(pixels.size),
        "image_count": len(first.frames),
        "numeric_size": len(temporal_features(first.actor, first.frames, timing)),
        "actor_fields": sorted(first.actor),
        "actor_shape": actor_shape(first.actor, len(first.frames)),
        "observation": {"history_offsets_ms": list(pixels.history_offsets_ms)},
        "normalization": "fixed-v1:speed/100,velocity/100,angular/5,age/1000,waypoints/100",
        "temporal": timing,
        "action": data["action_contract"],
    }
    training = isinstance(request, TemporalBCTrain)
    models: list[Any] = []
    optimizer = None

    def checkpoint(phase: str, completed: int) -> None:
        if budget:
            budget.checkpoint(
                phase,
                completed,
                lambda: move_learning_state(torch, models, optimizer, "cpu"),
                lambda: move_learning_state(torch, models, optimizer, device),
            )

    if training:
        torch.manual_seed(config["seed"])
        model = make_actor(contract).to(device)
        models.append(model)
        training_rows = rows.select("train", eligible=True)
        if not training_rows or not any(
            rows.select(split, eligible=True) for split in ("development", "evaluation")
        ):
            raise ValueError(
                "Temporal BC requires eligible train and independent held-out attempts"
            )

        def tensor(frame: NumericFrame) -> Any:
            width, height = frame.size
            return (
                torch.frombuffer(bytearray(frame.pixels), dtype=torch.uint8)
                .reshape(height, width, 3)
                .permute(2, 0, 1)
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        losses = BCLossHistory()
        resources.callback(losses.close)
        started, completed, gradient = time.monotonic(), 0, 0.0
        for step in range(config["steps"]):
            checkpoint("update", step)
            indices = torch.randperm(len(training_rows) // 2)[: config["batch_size"] // 2].tolist()
            batch = [training_rows[i * 2 + v] for i in indices for v in (0, 1)]
            rgb = (
                torch.stack([torch.stack([tensor(f) for f in r["decision"].frames]) for r in batch])
                .to(device)
                .float()
                / 255
            )
            features = torch.tensor(
                [
                    temporal_features(r["decision"].actor, r["decision"].frames, timing)
                    for r in batch
                ],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            target = torch.tensor(
                [r["entry"]["supervision"]["action"] for r in batch],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(rgb, features), target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite temporal BC loss")
            loss.backward()
            gradient += float(
                features.grad[:, -2 * (len(first.frames) - 1) :: 2].abs().sum().item()
            )
            optimizer.step()
            completed = step + 1
            losses.record(completed, loss)
            del rgb, features, target, loss
        checkpoint("freeze", completed)
        stats = {
            "steps_completed": completed,
            "duration_s": time.monotonic() - started,
            "time_gradient_l1": gradient,
            "selection": "fixed final update; no held-out model selection",
            "train_by_view": {view: len(training_rows) // 2 for view in VIEWS},
        }
        manifest = {
            "version": 2,
            "architecture": TEMPORAL_ARCHITECTURE,
            "contract": contract,
            "config": config,
            "numeric_contract": pixels.metadata(),
            "preprocessing": {
                "version": 1,
                "rgb": "full frame bilinear resize, float32 / 255; no crop",
                "size": list(pixels.size),
            },
            "future_supervision": {"enabled": False, "weight": 0},
            "dataset_sha256": digest,
            "training": stats,
            "provenance": data["provenance"],
            "groups": data["groups"],
            "torch_version": str(torch.__version__),
            "closed_loop_validated": False,
            "critic_trained": False,
        }
        output.mkdir(parents=True)
        torch.save(
            {
                "actor": model.state_dict(),
                "metadata": {k: manifest[k] for k in TEMPORAL_METADATA_KEYS},
            },
            output / "actor.pt",
        )
        manifest["weights_sha256"] = sha256_file(output / "actor.pt")
        stats["loss_history"] = losses.publish(output)
        _write(output / "model.json", manifest)
    actor = FrozenNumericActor(output, pixels, device, expected_manifest_sha256=manifest_digest)
    models.append(actor.model)
    if actor.original_contract != contract:
        raise ValueError("Snapshot model contract mismatch")
    max_error = 0.0
    preview_export: dict[str, Any] = {"status": "complete"}
    for row in rows:
        checkpoint("prediction", config["steps"] if training else 0)
        decision = row["decision"]
        features_list = actor.input_features(decision.actor, decision.frames)
        prediction = actor.predict(decision.actor, decision.frames)
        masked_actions = deepcopy(decision.actor)
        count = len(masked_actions["actions"])
        masked_actions.update(
            actions=[None] * count, action_mask=[False] * count, action_age_ms=[None] * count
        )
        without_history = actor.predict(masked_actions, decision.frames)
        previews: list[str | None] = []
        for frame in decision.frames:
            relative: str | None = None
            temporary: Path | None = None
            try:
                if preview_export["status"] == "complete":
                    pixel_digest = hashlib.sha256(frame.pixels).hexdigest()
                    relative = f"previews/{pixel_digest}.png"
                    target = output / relative
                    target.parent.mkdir(exist_ok=True)
                    if not target.exists():
                        # Publish only a complete image; a failed write must not
                        # become an apparently reusable preview on later replay.
                        temporary = target.with_name(".pending-" + target.name)
                        temporary.write_bytes(preview_png(bytes(frame.pixels), frame.size))
                        temporary.replace(target)
            except (OSError, MemoryError) as error:
                relative = None
                preview_export = {
                    "status": "unavailable",
                    "error": f"{type(error).__name__}: {error}",
                    "remaining": "deferred; rebuild with temporal-bc-replay",
                }
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except (OSError, MemoryError) as cleanup_error:
                        preview_export["cleanup_error"] = (
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
            previews.append(relative)
        if training:
            with torch.inference_mode():
                model.eval()
                before = (
                    model(
                        torch.stack([tensor(f) for f in decision.frames])
                        .unsqueeze(0)
                        .to(device)
                        .float()
                        / 255,
                        torch.tensor([features_list], dtype=torch.float32, device=device),
                    )[0]
                    .cpu()
                    .tolist()
                )
            max_error = max(max_error, max(abs(a - b) for a, b in zip(before, prediction)))
        actor.clear_input_cache()
        records.append(
            {
                "decision_id": decision.decision_id,
                "decision_ns": decision.decision_ns,
                "epoch": decision.epoch,
                "frames": [f.metadata() for f in decision.frames],
                "actor": decision.actor,
                "features": features_list,
                "timing": describe_time(decision.actor, decision.frames),
                "prediction": prediction,
                "without_action_history_prediction": without_history,
                "previous_action": next(
                    (
                        a
                        for a, m in reversed(
                            list(zip(decision.actor["actions"], decision.actor["action_mask"]))
                        )
                        if m
                    ),
                    None,
                ),
                "target": decision.supervision["action"],
                "supervision": decision.supervision,
                "split": row["split"],
                "group": row["entry"]["group"],
                "view": row["view"],
                "scored": row["entry"]["bc_eligible"],
                "status": "predicted",
                "previews": previews,
                "variant": row["entry"].get("variant", 0),
            }
        )
    actor.clear_input_cache()
    records.freeze()
    if training:
        if max_error > 1e-6:
            raise ValueError("Temporal frozen reload prediction drift exceeds tolerance")
        manifest["training"]["reload_max_abs_error"] = max_error
        _write(output / "model.json", manifest)
        verification = {
            "status": "training_reload",
            "compared_decisions": len(records),
            "max_abs_error": max_error,
            "tolerance": 1e-6,
        }
    else:
        evidence = manifest.get("verification")
        if not evidence or evidence["tolerance"] != 1e-6:
            raise ValueError("Temporal model lacks frozen prediction verification evidence")
        prior_path = asset(output, evidence["path"])
        if sha256_file(prior_path) != evidence["sha256"]:
            raise ValueError("Frozen temporal prediction evidence changed")
        prior_report = resources.enter_context(
            replay_document(VerifiedFile(prior_path, evidence["sha256"]))
        )
        if len(prior_report["decisions"]) != len(records):
            raise ValueError("Frozen temporal prediction evidence changed")
        replay_error = 0.0
        for previous, current in zip(prior_report["decisions"], records):
            for field in (
                "decision_id",
                "decision_ns",
                "epoch",
                "frames",
                "actor",
                "features",
                "timing",
            ):
                if previous[field] != current[field]:
                    raise ValueError("Temporal replay input drift: " + field)
            replay_error = max(
                replay_error,
                max(abs(a - b) for a, b in zip(previous["prediction"], current["prediction"])),
            )
        if replay_error > evidence["tolerance"]:
            raise ValueError("Temporal replay prediction drift exceeds tolerance")
        verification = {
            "status": "verified",
            "compared_decisions": len(records),
            "max_abs_error": replay_error,
            "tolerance": evidence["tolerance"],
        }
    try:
        metrics = temporal_metrics(records)
    except (OSError, MemoryError) as error:
        metrics = {
            "status": "unavailable",
            "error": f"{type(error).__name__}: {error}",
            "remaining": "deferred; rebuild with temporal-bc-replay",
        }
    summary = {
        "version": 1,
        "contract": pixels.metadata(),
        "model": actor.manifest,
        "training": manifest["training"],
        "metrics": metrics,
        "decisions": records,
        "commands_sent": False,
        "dataset_sha256": digest,
        "groups": data["groups"],
        "closed_loop_validated": False,
        "verification": verification,
        "preview_export": preview_export,
    }
    result = _result(report, summary, section="temporal_bc", root=output)
    if training:
        manifest["verification"] = {
            "path": "report.json",
            "sha256": sha256_file(report.with_suffix(".json")),
            "tolerance": 1e-6,
        }
        _write(output / "model.json", manifest)
    return result
