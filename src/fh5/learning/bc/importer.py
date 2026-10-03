"""Import verified historical demonstrations once; reselect actual causal frames."""

from __future__ import annotations

import hashlib
import json
import tempfile
from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.artifacts.io import asset
from fh5.collection.demonstrations import validate_demonstration
from fh5.learning.bc.legacy_training import VIEWS, _read_dataset, _write
from fh5.observation.multimodal import history_timing
from fh5.observation.numeric import PixelContract
from fh5.observation.recording import _result

if TYPE_CHECKING:
    from fh5.result import RunResult


@dataclass(frozen=True)
class TemporalBCPrepare:
    config_file: Path
    output_dir: Path


def prepare_temporal(request: TemporalBCPrepare) -> RunResult:
    from PIL import Image

    from fh5.experiment import run_experiment
    from fh5.telemetry.packet import Replay

    config = json.loads(request.config_file.read_text(encoding="utf-8"))
    if (
        set(config)
        != {
            "version",
            "dataset",
            "image_size",
            "history_patterns_ms",
            "max_selection_error_ms",
            "max_image_age_ms",
        }
        or config["version"] != 1
    ):
        raise ValueError("Invalid temporal preparation config")
    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    path = request.config_file.parent / config["dataset"]
    data, original_contract = _read_dataset(path)
    nominal = tuple(original_contract["observation"]["history_offsets_ms"])
    pixels = PixelContract(
        size=tuple(config["image_size"]), origin="legacy_offline", history_offsets_ms=nominal
    )
    patterns = config["history_patterns_ms"]
    if not 1 <= len(patterns) <= 8 or list(nominal) != patterns[0]:
        raise ValueError("First history pattern must preserve original nominal offsets")
    for pattern in patterns:
        PixelContract(size=pixels.size, history_offsets_ms=tuple(pattern))
        if len(pattern) != len(nominal):
            raise ValueError("History augmentation cannot change frame count")
    for key in ("max_selection_error_ms", "max_image_age_ms"):
        if type(config[key]) is not int or not 1 <= config[key] <= 2000:
            raise ValueError("Invalid bounded temporal selection budget")
    sources = {}
    groups = []
    for index, source in enumerate(data["sources"]):
        directory = Path(source["directory"])
        validate_demonstration(directory)
        with tempfile.TemporaryDirectory(prefix="fh5-temporal-boundaries-") as temp:
            result = run_experiment(Replay(directory, Path(temp) / "report.html"))
            boundaries, _ = history_timing(
                result.samples, result.events, result.summary["vision"]["events"], 2
            )
        journal = [
            json.loads(line)
            for line in (directory / "vision.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        frames = {f["path"]: f for f in journal if f.get("kind") == "frame"}
        sources[index] = (directory, frames, boundaries)
        groups.append(
            {
                "id": f"source-{index}",
                "split": "train" if source["split"] == "train" else "development",
                "evidence_id": source["packets_sha256"],
            }
        )
    request.output_dir.mkdir(parents=True)
    (request.output_dir / "pixels").mkdir()
    converted: dict[tuple[int, str], dict[str, Any]] = {}
    rows, exclusions = [], []
    jitter_variants = 0
    for number, example in enumerate(data["examples"]):
        if not example["bc_eligible"]:
            exclusions.append(
                {
                    "example_index": number,
                    "run_index": example["run_index"],
                    "reasons": example["supervision"]["reasons"],
                }
            )
            continue
        index, tick = example["run_index"], example["decision_ns"]
        directory, frames, boundaries = sources[index]
        group = f"source-{index}"
        segment = bisect_right(boundaries, tick)
        epoch = f"{group}:segment-{segment}"
        available = sorted(
            (
                f
                for f in frames.values()
                if f["delivered_ns"] <= tick
                and bisect_right(boundaries, f["capture_start_ns"]) == segment
            ),
            key=lambda f: f["capture_start_ns"],
        )
        seen_selections = set()
        for variant, pattern in enumerate(patterns):
            selected = []
            if variant == 0:
                selected = [frames[f["path"]] for f in example["views"]["no_reference"]["images"]]
            elif available:
                newest = available[-1]["capture_start_ns"]
                for offset in pattern:
                    target = newest - offset * 1_000_000
                    candidates = [f for f in available if f["capture_start_ns"] <= target]
                    if (
                        not candidates
                        or target - candidates[-1]["capture_start_ns"]
                        > config["max_selection_error_ms"] * 1_000_000
                    ):
                        break
                    selected.append(candidates[-1])
            ids = tuple(f["path"] for f in selected)
            if (
                len(selected) != len(nominal)
                or len(set(ids)) != len(nominal)
                or ids in seen_selections
                or any(
                    f["delivered_ns"] > tick
                    or bisect_right(boundaries, f["capture_start_ns"]) != segment
                    or tick - f["capture_start_ns"] > config["max_image_age_ms"] * 1_000_000
                    for f in selected
                )
            ):
                exclusions.append(
                    {
                        "example_index": number,
                        "variant": variant,
                        "reasons": ["unavailable_or_duplicate_temporal_history"],
                    }
                )
                continue
            seen_selections.add(ids)
            chosen = []
            for frame in selected:
                frame_key = index, frame["path"]
                if frame_key not in converted:
                    encoded = asset(directory, frame["path"])
                    if hashlib.sha256(encoded.read_bytes()).hexdigest() != frame["sha256"]:
                        raise ValueError("Historical frame hash mismatch")
                    with Image.open(encoded) as image:
                        if image.mode != "RGB" or list(image.size) != frame["size"]:
                            raise ValueError("Historical image layout mismatch")
                        rgb = image.resize(pixels.size, Image.Resampling.BILINEAR).tobytes()
                    digest = hashlib.sha256(rgb).hexdigest()
                    relative = f"pixels/{digest}.rgb"
                    (request.output_dir / relative).write_bytes(rgb)
                    converted[frame_key] = {
                        "epoch": epoch,
                        "frame_id": f"{group}:{frame['path']}",
                        "source_time_ns": frame["capture_start_ns"],
                        "capture_received_ns": frame["capture_end_ns"],
                        "preprocess_ready_ns": frame["delivered_ns"],
                        "time_quality": "capture_start_proxy",
                        "uncertainty_ns": None,
                        "availability_kind": "legacy_encoded_delivery_proxy",
                        "preprocess_version": pixels.resize,
                        "size": list(pixels.size),
                        "path": relative,
                        "sha256": digest,
                        "source_layout": {
                            "size": frame["size"],
                            "client_size": frame["client_size"],
                            "format": "RGB",
                            "stride_bytes": frame["size"][0] * 3,
                            "codec": frame["codec"],
                            "resize_method": frame["resize_method"],
                            "encoded_sha256": frame["sha256"],
                            "source_clock": "recorded_perf_counter_ns",
                            "color_space": "unknown",
                        },
                    }
                chosen.append(converted[frame_key])
            views = deepcopy(example["views"])
            for view in VIEWS:
                views[view]["images"] = [None] * len(chosen)
                views[view]["image_age_ms"] = [(tick - f["source_time_ns"]) / 1e6 for f in chosen]
            rows.append(
                {
                    "decision_id": f"legacy:{number}:variant:{variant}",
                    "group": group,
                    "epoch": epoch,
                    "decision_ns": tick,
                    "frames": chosen,
                    "views": views,
                    "bc_eligible": True,
                    "supervision": example["supervision"],
                    "variant": variant,
                    "example_index": number,
                }
            )
            jitter_variants += variant > 0
    snapshot = {
        "version": 1,
        "kind": "numeric-bc-snapshot-v1",
        "pixel_contract": pixels.metadata(),
        "action_contract": "xinput-lx-rt-lt-v1",
        "groups": groups,
        "decisions": rows,
        "provenance": {
            "kind": "historical_encoded_demonstrations",
            "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sources": data["sources"],
            "config": config,
            "final_evaluation_available": False,
            "note": "Prior holdout remains development data; proxy timestamps are not DXGI presentation times. Jitter reselects actual available frames relative to newest source time; nominal variant preserves original selection.",
        },
        "excluded": exclusions,
    }
    _write(request.output_dir / "dataset.json", snapshot)
    digest = hashlib.sha256((request.output_dir / "dataset.json").read_bytes()).hexdigest()
    summary = {
        "version": 1,
        "contract": pixels.metadata(),
        "decisions": [],
        "commands_sent": False,
        "dataset_sha256": digest,
        "decoded_unique_frames": len(converted),
        "selected_observations": len(rows),
        "jitter_variants": jitter_variants,
        "selected_by_group": dict(Counter(r["group"] for r in rows)),
        "exclusions": exclusions,
        "diagnostic_only": True,
    }
    return _result(request.output_dir / "report.html", summary, section="temporal_import")
