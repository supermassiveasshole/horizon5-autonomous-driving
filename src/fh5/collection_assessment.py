"""Offline frozen-candidate comparisons with a separately bound final holdout."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.artifact_io import VerifiedFile, sha256_file
from fh5.collection_store import read_bounded, write_file
from fh5.learning_runtime import preserve_torch_state
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_recording import _result
from fh5.numeric_report import preview_png
from fh5.prediction_metrics import assessment_metrics
from fh5.prediction_records import prediction_spool
from fh5.replay_document import read_document_fields
from fh5.temporal_data import temporal_snapshot
from fh5.temporal_features import describe_time

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class CollectionBCAssess:
    config_file: Path
    output_dir: Path


def _json(path: Path, limit: int = 128 * 1024**2) -> tuple[dict[str, Any], str]:
    raw = read_bounded(path, limit)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _model(entry: dict[str, Any], base: Path) -> tuple[Path, dict[str, Any]]:
    if not isinstance(entry, dict) or set(entry) != {"directory", "manifest_sha256"}:
        raise ValueError("Assessment requires a predeclared frozen model manifest")
    root = base / entry["directory"]
    manifest, digest = _json(root / "model.json")
    if digest != entry["manifest_sha256"]:
        raise ValueError("Frozen assessment model manifest changed")
    return root, manifest


def _compatibility(model: dict[str, Any], data: dict[str, Any]) -> list[str]:
    original, observed = model.get("provenance", {}), data["provenance"]
    reasons = []
    if model.get("version") != 2 or model.get("numeric_contract") != data["pixel_contract"]:
        reasons.append("architecture_or_pixel_contract")
    if original.get("kind") != "continuous_numeric_collection":
        reasons.append("legacy_or_unknown_collection_source")
    for key in ("input_conditions", "vehicle", "input_mapping", "diagnostic_only"):
        if key not in original or original[key] != observed[key]:
            reasons.append(key)
    for key in ("action_history_offsets_ms", "max_action_age_ms", "waypoint_distances_m"):
        if original.get("config", {}).get(key) != observed["config"][key]:
            reasons.append(key)
    if original.get("reference", {}).get("route_sha256") != observed["reference"].get(
        "route_sha256"
    ):
        reasons.append("reference_prior")
    for key in ("speed_range_mps", "steering_limit", "longitudinal_limit"):
        if original.get("envelope", {}).get(key) != observed["envelope"][key]:
            reasons.append(key)
    return reasons


def _overlap(model: dict[str, Any], heldout: list[dict[str, Any]]) -> bool:
    for prior in model["groups"]:
        if "source_ranges" not in prior:
            raise ValueError("Baseline lacks independently bound source ranges")
        for target in heldout:
            if prior["id"] == target["id"]:
                return True
            for left in prior["source_ranges"]:
                for right in target["source_ranges"]:
                    if left["session_sha256"] == right["session_sha256"] and max(
                        left["start_sequence"], right["start_sequence"]
                    ) < min(left["end_sequence"], right["end_sequence"]):
                        return True
                    if any(k not in r for r in (left, right) for k in ("first_ns", "last_ns")):
                        raise ValueError("Unknown source clock windows")
                    if max(left["first_ns"], right["first_ns"]) <= min(
                        left["last_ns"], right["last_ns"]
                    ):
                        return True
    return False


def assess_collection_bc(request: CollectionBCAssess) -> RunResult:
    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    config, config_digest = _json(request.config_file, 1024**2)
    if (
        set(config)
        != {"version", "mode", "dataset", "dataset_sha256", "candidate", "baseline", "device"}
        or config["version"] != 1
        or config["mode"] not in ("development", "final")
        or config["device"] not in ("cpu", "cuda")
    ):
        raise ValueError("Unsupported frozen collection assessment configuration")
    base = request.config_file.parent
    candidate_path, candidate = _model(config["candidate"], base)
    dataset_path = base / config["dataset"]
    digest = sha256_file(dataset_path)
    data = read_document_fields(
        VerifiedFile(dataset_path, digest), {"provenance", "groups", "pixel_contract"}
    )
    if digest != config["dataset_sha256"]:
        raise ValueError("Assessment dataset hash changed")
    final = config["mode"] == "final"
    binding = (
        candidate.get("provenance", {}).get("final_dataset_sha256")
        if final
        else candidate.get("dataset_sha256")
    )
    if binding != digest:
        raise ValueError("Assessment dataset was not bound before candidate training")
    expected_partition = "evaluation" if final else "development"
    if data["provenance"].get("partition") != expected_partition:
        raise ValueError("Assessment dataset partition differs from declared purpose")
    reasons = _compatibility(candidate, data)
    if reasons:
        raise ValueError("Candidate incompatible with assessment inputs: " + ",".join(reasons))
    split = "evaluation" if final else "development"
    heldout = [g for g in data["groups"] if g["split"] == split]
    if final and (len(heldout) != len(data["groups"]) or _overlap(candidate, heldout)):
        raise ValueError("Final holdout overlaps candidate training or development")
    baseline_path = None
    baseline_info: dict[str, Any] = {"status": "not_provided", "reasons": []}
    if config["baseline"] is not None:
        baseline_path, baseline = _model(config["baseline"], base)
        reasons = _compatibility(baseline, data)
        if not reasons:
            # Development may be a baseline's previous development set, but
            # never its training set. Final data must be wholly unseen.
            exposure = (
                baseline
                if final
                else dict(baseline, groups=[g for g in baseline["groups"] if g["split"] == "train"])
            )
            try:
                if _overlap(exposure, heldout):
                    reasons.append("baseline_seen_heldout")
            except ValueError:
                reasons.append("unverified_heldout_independence")
        baseline_info = {
            "status": "incompatible" if reasons else "comparable",
            "reasons": reasons,
            "manifest_sha256": config["baseline"]["manifest_sha256"],
        }
        if reasons:
            baseline_path = None
    # Refuse outputs beneath model/data folders even if the new leaf is absent.
    roots = [candidate_path, dataset_path.parent]
    if config["baseline"] is not None:
        roots.append(base / config["baseline"]["directory"])
    if any(request.output_dir.resolve().is_relative_to(p.resolve()) for p in roots):
        raise ValueError("Assessment output must be separate from frozen inputs")
    torch = importlib.import_module("torch")
    with preserve_torch_state(torch), ExitStack() as resources, prediction_spool() as records:
        torch.set_num_threads(2)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        data, pixels, all_rows = resources.enter_context(
            temporal_snapshot(dataset_path, expected_sha256=digest)
        )
        rows = all_rows.select(split)
        if not all_rows.select(split, eligible=True):
            raise ValueError("No eligible independent observations for assessment")
        actor = FrozenNumericActor(candidate_path, pixels, config["device"])
        challenger = (
            FrozenNumericActor(baseline_path, pixels, config["device"]) if baseline_path else None
        )
        if challenger and challenger.original_contract != actor.original_contract:
            baseline_info.update(status="incompatible", reasons=["actor_contract"])
            challenger = None
        request.output_dir.mkdir(parents=True)
        (request.output_dir / "previews").mkdir()
        for row in rows:
            decision = row["decision"]
            prediction = actor.predict(decision.actor, decision.frames)
            comparison = challenger.predict(decision.actor, decision.frames) if challenger else None
            images = []
            for frame in decision.frames:
                pixel_hash = hashlib.sha256(frame.pixels).hexdigest()
                path = "previews/" + pixel_hash + ".png"
                target = request.output_dir / path
                if not target.exists():
                    write_file(target, preview_png(bytes(frame.pixels), frame.size))
                images.append(path)
            actor.clear_input_cache()
            if challenger:
                challenger.clear_input_cache()
            records.append(
                {
                    "decision_id": decision.decision_id,
                    "decision_ns": decision.decision_ns,
                    "epoch": decision.epoch,
                    "frames": [f.metadata() for f in decision.frames],
                    "actor": decision.actor,
                    "timing": describe_time(decision.actor, decision.frames),
                    "prediction": prediction,
                    "baseline_prediction": comparison,
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
                    "split": split,
                    "group": row["entry"]["group"],
                    "view": row["view"],
                    "scored": row["entry"]["bc_eligible"],
                    "status": "predicted",
                    "previews": images,
                }
            )
        records.freeze()
        actor.clear_input_cache()
        if challenger:
            challenger.clear_input_cache()
        # Reload independently; no optimizer or model update exists in this path.
        verification: dict[str, float | None] = {
            "candidate_reload_max_abs_error": None,
            "baseline_reload_max_abs_error": None,
            "tolerance": 1e-6,
        }
        for name, model_path, field in (
            ("candidate", candidate_path, "prediction"),
            ("baseline", baseline_path if challenger else None, "baseline_prediction"),
        ):
            if model_path is None:
                continue
            reloaded = FrozenNumericActor(model_path, pixels, config["device"])
            max_error = 0.0
            for row, recorded in zip(rows, records):
                d = row["decision"]
                predicted = reloaded.predict(d.actor, d.frames)
                reloaded.clear_input_cache()
                max_error = max(
                    max_error, max(abs(a - b) for a, b in zip(predicted, recorded[field]))
                )
            reloaded.clear_input_cache()
            if max_error > 1e-6:
                raise ValueError("Frozen assessment reload drift exceeds 1e-6")
            verification[name + "_reload_max_abs_error"] = max_error
        _model(config["candidate"], base)
        if config["baseline"] is not None:
            _model(config["baseline"], base)
        try:
            metrics = assessment_metrics(records)
        except (OSError, MemoryError) as error:
            metrics = {
                "status": "unavailable",
                "error": f"{type(error).__name__}: {error}",
                "remaining": "deferred; rebuild with collection-bc-assess",
            }
        summary = {
            "version": 1,
            "mode": config["mode"],
            "selection_allowed": not final,
            "config_sha256": config_digest,
            "config": config,
            "dataset_sha256": digest,
            "contract": pixels.metadata(),
            "model": actor.manifest,
            "baseline": baseline_info,
            "diagnostic_only": actor.manifest["diagnostic_only"],
            "commands_sent": False,
            "closed_loop_validated": False,
            "decisions": records,
            "groups": heldout,
            "metrics": metrics,
            "verification": verification,
            "scope": "offline action errors; not driving ability, task reward or promotion",
        }
        return _result(request.output_dir / "report.html", summary, section="collection_assessment")
