"""Archived BC replay and shared observation/configuration helpers."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.learning.bc.legacy import BCReplay
from fh5.learning.bc.losses import read_bc_manifest
from fh5.learning.bc.network import make_actor
from fh5.learning.loop.runtime import preserve_torch_state

if TYPE_CHECKING:
    from fh5.result import RunResult

VIEWS = ("no_reference", "reference_assisted")
ARCHITECTURE = "rgb-history-conv4-64-state64-fusion128-tanh2-v1"
MODEL_METADATA_KEYS = (
    "version",
    "architecture",
    "contract",
    "config",
    "preprocessing",
    "future_supervision",
)


def _json(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _numeric(actor: dict[str, Any]) -> list[float]:
    ego = actor["ego"]
    values = [
        ego["speed_mps"] / 100,
        *[v / 100 for v in ego["velocity_car_mps"]],
        *[v / 5 for v in ego["angular_velocity_car_radps"]],
        actor["ego_age_ms"] / 1000,
        float(actor["ego_mask"]),
    ]
    for mask, age in zip(actor["image_mask"], actor["image_age_ms"]):
        values.extend([float(mask), (age or 0) / 1000])
    for action, mask, age in zip(actor["actions"], actor["action_mask"], actor["action_age_ms"]):
        values.extend([*(action if mask else [0, 0]), float(mask), (age or 0) / 1000])
    for point, mask in zip(actor["reference"]["waypoints_m"], actor["reference"]["mask"]):
        values.extend([point[0] / 100 if mask else 0, point[1] / 100 if mask else 0, float(mask)])
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite actor state")
    return values


def _read_dataset(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _json(path)
    if data.get("version") != 1:
        raise ValueError("Unsupported demonstration dataset version")
    configs = []
    for source in data["sources"]:
        directory = Path(source["directory"])
        config = _json(directory / "observation-config.json")
        if config["version"] != 2 or source["profile"]["mapping"] != "xinput-lx-rt-lt-v1":
            raise ValueError("Incompatible observation/action version")
        configs.append(config)
    if not configs or any(c != configs[0] for c in configs):
        raise ValueError("Incompatible observation histories")
    _validate_export(data)
    config = configs[0]
    camera = _json(Path(data["sources"][0]["directory"]) / "vision-session.json")
    example = next(e for e in data["examples"] if e["bc_eligible"])
    contract = {
        "observation": config,
        "action": "xinput-lx-rt-lt-v1",
        "image_count": len(config["history_offsets_ms"]),
        "numeric_size": len(_numeric(example["views"]["no_reference"])),
        "normalization": "fixed-v1:speed/100,velocity/100,angular/5,age/1000,waypoints/100",
        "actor_fields": list(example["views"]["no_reference"]),
        "camera": {k: camera[k] for k in ("camera_mode", "camera_pose")},
        "conditions": data["sources"][0]["snapshot"],
        "observed_vehicle": data["sources"][0]["observed_vehicle"],
    }
    return data, contract


def _validate_export(data: dict[str, Any]) -> None:
    """Rebuild from bound raw evidence instead of trusting editable eligibility/actor JSON."""
    from fh5.collection.demonstration_dataset import DemonstrationDataset, export_demonstrations

    with tempfile.TemporaryDirectory(prefix="fh5-bc-check-") as temporary:
        root = Path(temporary)
        config = dict(data["config"], runs=[])
        for index, source in enumerate(data["sources"]):
            directory = Path(source["directory"]).resolve()
            if (
                _hash(directory / "demonstration-session.json")
                != source["demonstration_manifest_sha256"]
            ):
                raise ValueError("Demonstration source manifest changed")
            review = root / f"review-{index}.json"
            _write(review, source["review"])
            config["runs"].append(
                {"directory": str(directory), "split": source["split"], "review": str(review)}
            )
        config_file = root / "config.json"
        _write(config_file, config)
        regenerated = export_demonstrations(DemonstrationDataset(config_file, root / "checked"))
        canonical = regenerated.summary["demonstration_dataset"]
        if canonical["examples"] != data["examples"]:
            raise ValueError("Dataset differs from canonical causal source reconstruction")
        for actual, expected in zip(data["sources"], canonical["sources"]):
            for key in (
                "packets_sha256",
                "profile",
                "snapshot",
                "source_kind",
                "observed_vehicle",
                "navigation",
                "review",
            ):
                if actual[key] != expected[key]:
                    raise ValueError("Dataset source mismatch: " + key)


def _rows(data: dict[str, Any], size: list[int], torch: Any) -> list[dict[str, Any]]:
    from PIL import Image

    cache: dict[str, Any] = {}
    rows = []
    for index, example in enumerate(data["examples"]):
        directory = Path(data["sources"][example["run_index"]]["directory"])
        for view in VIEWS:
            actor = example["views"][view]
            usable = bool(actor["ego_mask"] and all(actor["image_mask"]))
            images = []
            if usable:
                for image in actor["images"]:
                    path = directory / image["path"]
                    key = str(path)
                    if key not in cache:
                        if _hash(path) != image["sha256"]:
                            raise ValueError("Image integrity mismatch")
                        with Image.open(path) as source:
                            pixels = source.convert("RGB").resize(
                                (size[0], size[1]), Image.Resampling.BILINEAR
                            )
                        cache[key] = (
                            torch.frombuffer(bytearray(pixels.tobytes()), dtype=torch.uint8)
                            .reshape(size[1], size[0], 3)
                            .permute(2, 0, 1)
                        )
                    images.append(cache[key])
            rows.append(
                {
                    "index": index,
                    "example": example,
                    "view": view,
                    "rgb": images if usable else None,
                    "state": torch.tensor(_numeric(actor)) if usable else None,
                }
            )
    return rows


def _predict(
    model: Any, rows: list[dict[str, Any]], torch: Any, device: str, intervention: bool = False
) -> list[Any]:
    result: list[Any] = [None] * len(rows)
    model.eval()
    valid = [i for i, row in enumerate(rows) if row["rgb"] is not None]
    with torch.inference_mode():
        for start in range(0, len(valid), 32):
            indices = valid[start : start + 32]
            rgb = (
                torch.stack([torch.stack(rows[i]["rgb"]) for i in indices]).to(device).float() / 255
            )
            state = torch.stack([rows[i]["state"] for i in indices]).to(device)
            if intervention:
                rgb = torch.zeros_like(rgb)
            output = model(rgb, state)
            if not torch.isfinite(output).all():
                raise ValueError("Non-finite model prediction")
            predictions = output.cpu().tolist()
            for i, prediction in zip(indices, predictions):
                result[i] = prediction
    return result


def _error(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "mae": None, "rmse": None, "p90_absolute_error": None}
    errors = [[abs(a - b) for a, b in zip(row["prediction"], row["target"])] for row in rows]
    return {
        "count": len(rows),
        "mae": [sum(e[j] for e in errors) / len(errors) for j in range(2)],
        "rmse": [math.sqrt(sum(e[j] ** 2 for e in errors) / len(errors)) for j in range(2)],
        "p90_absolute_error": [
            sorted(e[j] for e in errors)[math.ceil(0.9 * len(errors)) - 1] for j in range(2)
        ],
    }


def _diagnostics(
    rows: list[dict[str, Any]], predictions: list[Any]
) -> tuple[list[Any], dict[str, Any]]:
    records = []
    starts: dict[int, int] = {}
    for row, prediction in zip(rows, predictions):
        example = row["example"]
        run = example["run_index"]
        starts.setdefault(run, example["decision_ns"])
        actor = example["views"][row["view"]]
        target = example["supervision"]["action"]
        scored = example["bc_eligible"] and prediction is not None
        records.append(
            {
                "example_index": row["index"],
                "run_index": run,
                "split": example["split"],
                "view": row["view"],
                "decision_ns": example["decision_ns"],
                "elapsed_s": (example["decision_ns"] - starts[run]) / 1e9,
                "prediction": prediction,
                "target": target,
                "scored": scored,
                "quality": example["supervision"]["quality"],
                "reasons": example["supervision"]["reasons"],
                "speed_kmh": actor["ego"]["speed_mps"] * 3.6 if actor["ego"] else None,
                "images": actor["images"],
                "previous_action": actor["actions"][-1] if actor["action_mask"][-1] else None,
            }
        )
    metrics: dict[str, Any] = {}
    for split in ("train", "holdout"):
        metrics[split] = {}
        for view in VIEWS:
            group = [
                r for r in records if r["split"] == split and r["view"] == view and r["scored"]
            ]
            strata: dict[str, list[dict[str, Any]]] = {}
            for r in group:
                kind = (
                    "brake"
                    if r["target"][1] < -0.05
                    else "throttle"
                    if r["target"][1] > 0.05
                    else "coast"
                )
                turn = "turn" if abs(r["target"][0]) > 0.2 else "near_straight"
                speed = (
                    "below100"
                    if r["speed_kmh"] < 100
                    else "100to250"
                    if r["speed_kmh"] < 250
                    else "above250"
                )
                for key in (
                    f"run:{r['run_index']}",
                    f"time:{r['run_index']}:{int(r['elapsed_s'] // 5) * 5}s",
                    kind,
                    turn,
                    speed,
                ):
                    strata.setdefault(key, []).append(r)
            prior = [
                dict(r, prediction=r["previous_action"])
                for r in group
                if r["previous_action"] is not None
            ]
            metrics[split][view] = {
                **_error(group),
                "strata": {k: _error(v) for k, v in strata.items()},
                "previous_action_baseline": _error(prior),
            }
    return records, metrics


def _report(path: Path, summary: dict[str, Any]) -> Path:
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    if path.suffix != ".html":
        raise ValueError("Report must be .html")
    path.parent.mkdir(parents=True, exist_ok=True)
    numerical = path.with_suffix(".json")
    _write(numerical, summary)
    try:
        payload = (
            json.dumps(summary, ensure_ascii=False, allow_nan=False)
            .replace("<", "\\u003c")
            .replace("&", "\\u0026")
        )
        template = Path(__file__).with_name("bc_report.html").read_text(encoding="utf-8")
        path.write_text(template.replace("<!--BC_DATA-->", payload), encoding="utf-8")
    except (OSError, MemoryError) as error:
        presentation = {
            "status": "unavailable",
            "path": str(path),
            "error": f"{type(error).__name__}: {error}",
            "retained_result": str(numerical),
        }
        summary["presentation"] = presentation
        try:
            path.unlink(missing_ok=True)
        except (OSError, MemoryError) as cleanup_error:
            presentation["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        return numerical
    return path


def _checked_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("BC config must be an object")
    if value.get("version") != 1 or set(value) != {
        "version",
        "dataset",
        "seed",
        "steps",
        "batch_size",
        "learning_rate",
        "image_size",
        "device",
    }:
        raise ValueError("Unsupported BC config")
    # torch.manual_seed accepts this signed/unsigned 64-bit union. Keep the
    # original configured integer; remapping belongs to the Torch generator.
    if type(value["seed"]) is not int or not -(2**63) <= value["seed"] <= 2**64 - 1:
        raise ValueError("BC seed is outside the torch.manual_seed integer range")
    if type(value["steps"]) is not int or value["steps"] < 1:
        raise ValueError("Invalid BC steps budget")
    if type(value["batch_size"]) is not int or value["batch_size"] < 1:
        raise ValueError("Invalid BC batch budget: batch_size")
    rate = value["learning_rate"]
    try:
        valid_rate = type(rate) in (int, float) and rate > 0 and math.isfinite(rate)
    except OverflowError:
        valid_rate = False
    if not valid_rate:
        raise ValueError("Invalid learning rate")
    size = value["image_size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(v) is not int or v < 1 for v in size)
    ):
        raise ValueError("Invalid image size")
    if value["device"] not in ("cpu", "cuda"):
        raise ValueError("Invalid device")
    if value["batch_size"] % 2:
        raise ValueError("Batch size must be even for paired reference views")
    return dict(value)


def run_offline(request: BCReplay) -> RunResult:
    torch = importlib.import_module("torch")
    with preserve_torch_state(torch):
        return _run_offline(request, torch)


def _run_offline(request: BCReplay, torch: Any) -> RunResult:
    from fh5.result import RunResult

    output = request.model_dir
    manifest, _ = read_bc_manifest(output / "model.json")
    if (
        not {
            "version",
            "architecture",
            "weights_sha256",
            "contract",
            "config",
            "preprocessing",
            "future_supervision",
            "training",
        }
        <= manifest.keys()
    ):
        raise ValueError("Incomplete model manifest")
    if (
        manifest["architecture"] != ARCHITECTURE
        or _hash(output / "actor.pt") != manifest["weights_sha256"]
    ):
        raise ValueError("Bad model artifact")
    config = _checked_config(manifest["config"])
    dataset_path, report, device = request.dataset_file, request.report_path, request.device
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested BC device unavailable")
    data, contract = _read_dataset(dataset_path)
    contract["image_size"] = config["image_size"]
    if contract != manifest["contract"]:
        raise ValueError("Model observation/action contract mismatch")
    rows = _rows(data, config["image_size"], torch)
    torch.set_num_threads(2)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = make_actor(contract).to(device)
    try:
        saved = torch.load(output / "actor.pt", map_location=device, weights_only=True)
        if (
            saved["metadata"] != {k: manifest[k] for k in MODEL_METADATA_KEYS}
            or manifest["version"] != 1
        ):
            raise ValueError("Model metadata mismatch")
        model.load_state_dict(saved["actor"], strict=True)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Non-finite model weights")
    except Exception as error:
        raise ValueError("Bad model weights") from error
    predictions = _predict(model, rows, torch, device)
    records, metrics = _diagnostics(rows, predictions)
    zeros = _predict(model, rows, torch, device, intervention=True)
    change = [abs(a - b) for x, y in zip(predictions, zeros) if x is not None for a, b in zip(x, y)]
    summary = {
        **manifest,
        "predictions": records,
        "metrics": metrics,
        "evaluation_dataset_sha256": _hash(dataset_path),
        "sources": data["sources"],
        "commands_sent": False,
        "visual_intervention": {
            "method": "all other inputs fixed; black RGB",
            "mean_action_change": sum(change) / len(change) if change else None,
            "proves_driving_benefit": False,
        },
    }
    report = _report(report, summary)
    return RunResult(
        {"source_kind": "offline_bc", "game_validation": "unverified"},
        [],
        [],
        {"bc": summary},
        report,
    )
