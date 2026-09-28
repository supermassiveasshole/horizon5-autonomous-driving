"""Frozen pixel observations and independent labels; never driving or route truth."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote

from fh5.road_evaluation import evaluate

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class Perception:
    dataset_file: Path
    protocol_file: Path
    output_dir: Path


@dataclass(frozen=True)
class PerceptionReplay:
    result_dir: Path
    report_path: Path
    labels_file: Path | None = None


@dataclass(frozen=True)
class PixelPrediction:
    size: tuple[int, int]
    classes: bytes  # Cityscapes train IDs 0..18, row major
    scores: bytes  # Maximum softmax, quantized to 0..255; uncalibrated


class RoadModel(Protocol):
    metadata: dict[str, Any]

    def predict(self, image: Path) -> PixelPrediction: ...


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _url(path: Path, report: Path) -> str:
    try:
        return quote(os.path.relpath(path, report.parent).replace("\\", "/"), safe="/")
    except ValueError:
        return path.resolve().as_uri()


def _json(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _validate_dataset(dataset: dict[str, Any], protocol: dict[str, Any]) -> None:
    try:
        if dataset["version"] != 1 or protocol["version"] != 1:
            raise ValueError("Unsupported perception dataset/protocol version")
        for key in ("confidence_threshold", "high_confidence_threshold"):
            if not 0 <= protocol[key] <= 1:
                raise ValueError("Confidence thresholds must be between zero and one")
        if protocol["confidence_threshold"] > protocol["high_confidence_threshold"]:
            raise ValueError("High-confidence threshold cannot be lower than candidate threshold")
        rows = protocol["boundary_rows"]
        if not rows or len(rows) > 20 or any(not 0 <= y <= 1 for y in rows):
            raise ValueError("Boundary rows must be image-height fractions")
        for key in ("max_result_age_ms", "max_pair_gap_ms"):
            if not math.isfinite(protocol[key]) or protocol[key] <= 0:
                raise ValueError("Timing limits must be finite and positive")
        recordings: dict[str, str] = {}
        identifiers: set[str] = set()
        selected: set[tuple[str, int]] = set()
        for clip in dataset["clips"]:
            if clip["split"] not in ("development", "holdout") or clip["id"] in identifiers:
                raise ValueError("Clips require unique identifiers and a known split")
            identifiers.add(clip["id"])
            recording = clip["vision_sha256"]
            if recording in recordings and recordings[recording] != clip["split"]:
                raise ValueError("Development and holdout recordings must be independent")
            recordings[recording] = clip["split"]
            first, last = clip["first_frame"], clip["last_frame"]
            if type(first) is not int or type(last) is not int or not 0 <= first <= last < 100000:
                raise ValueError("Invalid frame range")
            if last - first >= 2000:
                raise ValueError("Perception runs are limited to 2000 frames")
            for i in range(first, last + 1):
                if (recording, i) in selected:
                    raise ValueError("Overlapping clips double-count an observation")
                selected.add((recording, i))
            if any(type(i) is not int or not first <= i <= last for i in clip["annotation_frames"]):
                raise ValueError("Annotation frame is outside its clip")
        if not selected or len(selected) > 2000:
            raise ValueError("Select between one and 2000 frames")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid dataset/protocol structure: {error}") from error


def _boundaries(mask: bytes, size: tuple[int, int], rows: list[float]) -> list[dict[str, Any]]:
    width, height = size
    found = []
    for y in sorted({round(fraction * (height - 1)) for fraction in rows}):
        longest: tuple[int, int] | None = None
        start = None
        for x in range(width + 1):
            road = x < width and mask[y * width + x] == 1
            if road and start is None:
                start = x
            if not road and start is not None:
                if longest is None or x - start > longest[1] - longest[0] + 1:
                    longest = (start, x - 1)
                start = None
        found.append(
            {
                "y": y,
                "left": longest[0] if longest else None,
                "right": longest[1] if longest else None,
            }
        )
    return found


def _render_prediction(
    prediction: PixelPrediction, image: Path, output: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    from PIL import Image

    width, height = prediction.size
    count = width * height
    if (
        len(prediction.classes) != count
        or len(prediction.scores) != count
        or max(prediction.classes, default=255) > 18
    ):
        raise ValueError("Invalid model pixel output")
    threshold = math.ceil(protocol["confidence_threshold"] * 255)
    raw = Image.frombytes("L", prediction.size, prediction.classes)
    scores = Image.frombytes("L", prediction.size, prediction.scores)
    candidates = raw.point(
        [0 if cls in (8, 9, 10) else 1 if cls == 0 else 2 if cls == 1 else 3 for cls in range(256)]
    )
    candidates = Image.composite(
        candidates,
        Image.new("L", prediction.size),
        scores.point([255 if score >= threshold else 0 for score in range(256)]),
    )
    classes = candidates.tobytes()
    mask_path = output.with_suffix(".png")
    # Raw class ID and score stay available independently of the display palette.
    Image.merge("RGB", (candidates, raw, scores)).save(mask_path)
    palette = [(135, 95, 180, 65), (35, 220, 110, 95), (255, 195, 45, 100), (240, 65, 75, 95)]
    overlay = Image.frombytes("P", prediction.size, classes)
    overlay.putpalette([channel for color in palette for channel in color[:3]] + [0] * (252 * 3))
    overlay = overlay.convert("RGBA")
    overlay.putalpha(candidates.point([palette[c][3] if c < 4 else 0 for c in range(256)]))
    with Image.open(image) as original:
        Image.alpha_composite(original.convert("RGBA"), overlay).convert("RGB").save(
            output.with_suffix(".jpeg"), quality=90
        )
    return {
        "mask_path": str(mask_path.resolve()),
        "mask_sha256": file_hash(mask_path),
        "overlay_path": str(output.with_suffix(".jpeg").resolve()),
        "overlay_sha256": file_hash(output.with_suffix(".jpeg")),
        "coverage": {
            key: classes.count(i) / count
            for i, key in enumerate(("unknown", "road", "shoulder_candidate", "obstacle_candidate"))
        },
        "boundaries": _boundaries(classes, prediction.size, protocol["boundary_rows"]),
    }


def _write_page(path: Path, name: str, data: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    template = Path(__file__).with_name(name).read_text(encoding="utf-8")
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    with path.open("x", encoding="utf-8") as destination:
        destination.write(template.replace("<!--PERCEPTION_DATA-->", encoded))


def replay_perception(request: PerceptionReplay) -> RunResult:
    from PIL import Image

    from fh5.experiment import RunResult

    if request.report_path.suffix.lower() != ".html":
        raise ValueError("Perception report output must have an .html extension")
    for destination in (request.report_path, request.report_path.with_suffix(".json")):
        if destination.exists():
            raise FileExistsError(destination)
    request.report_path.parent.mkdir(parents=True, exist_ok=True)
    directory = request.result_dir.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name in ("perception.json", "protocol.json", "dataset.json"):
        if file_hash(directory / name) != manifest["hashes"][name]:
            raise ValueError(f"Perception integrity failure: {name}")
    road = json.loads((directory / "perception.json").read_text(encoding="utf-8"))
    for row in road["frames"]:
        row["image_url"] = None
        row["overlay_url"] = None
        try:
            if file_hash(Path(row["image_path"])) != row["image_sha256"]:
                raise ValueError("Source image integrity failure")
            row["image_url"] = _url(Path(row["image_path"]), request.report_path)
            if row["status"] != "ok":
                continue
            for key in ("mask", "overlay"):
                path = Path(row[f"{key}_path"])
                if file_hash(path) != row[f"{key}_sha256"]:
                    raise ValueError(f"{key} integrity failure")
                with Image.open(path) as decoded:
                    decoded.load()
                    if list(decoded.size) != row["size"]:
                        raise ValueError(f"{key} dimension mismatch")
            if not math.isfinite(row["inference_ms"]) or row["inference_ms"] < 0:
                raise ValueError("Invalid inference timing")
            row["stale"] = row["projected_age_ms"] > road["protocol"]["max_result_age_ms"]
            row["overlay_url"] = _url(Path(row["overlay_path"]), request.report_path)
        except (OSError, ValueError, TypeError) as error:
            row.update(status="invalid", error=str(error), overlay_url=None)
    labels = (
        json.loads(request.labels_file.read_text(encoding="utf-8-sig"))
        if request.labels_file
        else None
    )
    road["evaluation"] = evaluate(road, labels)
    road["geometry_ready"] = road["evaluation"]["geometry_ready"]
    if request.labels_file:
        road["labels_sha256"] = file_hash(request.labels_file)
    _write_page(request.report_path, "perception.html", road)
    _json(request.report_path.with_suffix(".json"), road)
    return RunResult(
        {"game_validation": "pixel_estimates_only"},
        [],
        [],
        {"capture_status": "complete", "perception": road},
        request.report_path,
    )


def run_perception(request: Perception, model: RoadModel) -> RunResult:
    from PIL import Image

    from fh5.experiment import Replay, run_experiment

    dataset = json.loads(request.dataset_file.read_text(encoding="utf-8-sig"))
    protocol = json.loads(request.protocol_file.read_text(encoding="utf-8-sig"))
    _validate_dataset(dataset, protocol)
    if model.metadata.get("protocol_file_sha256", file_hash(request.protocol_file)) != file_hash(
        request.protocol_file
    ):
        raise ValueError("Model was initialized with a different protocol")
    source_hashes = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    output = request.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "pixels").mkdir()
    (output / "sources").mkdir()
    _json(output / "dataset.json", dataset)
    _json(output / "protocol.json", protocol)
    frames: list[dict[str, Any]] = []
    recordings: dict[Path, RunResult] = {}
    for clip in dataset["clips"]:
        source = (request.dataset_file.parent / clip["recording_dir"]).resolve()
        if file_hash(source / "vision.jsonl") != clip["vision_sha256"]:
            raise ValueError("Recording does not match frozen dataset")
        if source not in recordings:
            recordings[source] = run_experiment(
                Replay(source, output / "sources" / f"{len(recordings)}.html")
            )
        visual = recordings[source].summary["vision"]["frames"]
        if clip["last_frame"] >= len(visual):
            raise ValueError("Clip extends beyond the recorded frames")
        boundaries = sorted(
            [
                e["received_monotonic_ns"]
                for e in recordings[source].events
                if type(e.get("received_monotonic_ns")) is int
            ]
            + [
                e["observed_ns"]
                for e in recordings[source].summary["vision"]["events"]
                if e.get("kind")
                in ("focus_lost", "focus_restored", "capture_discarded", "telemetry_overflow")
                and type(e.get("observed_ns")) is int
            ]
        )
        for number in range(clip["first_frame"], clip["last_frame"] + 1):
            observed = visual[number]
            image_path = source / observed["path"]
            sample = observed["online"]["sample"]
            row = {
                "id": f"{clip['id']}:{number}",
                "split": clip["split"],
                "clip": clip["id"],
                "source_recording": clip["vision_sha256"],
                "frame_number": number,
                "image_path": str(image_path),
                "image_sha256": observed["sha256"],
                "size": observed["size"],
                "capture_start_ns": observed["capture_start_ns"],
                "source_observation": observed["online"]["reason"],
                "source_age_ms": observed["online"]["handoff_age_ms"],
                "segment": sample["segment"] if sample else None,
                "continuity": bisect_right(boundaries, observed["capture_start_ns"]),
                "speed_kmh": sample["speed_kmh"] if sample else None,
                "tags": clip["tags"],
                "annotate": number in clip["annotation_frames"],
                "online_usable": False,
                "status": "ok",
                "boundaries": [],
            }
            try:
                if (
                    observed["image_url"] is None
                    or observed["online"]["reason"] == "artifact_integrity"
                ):
                    raise ValueError("Missing or hash-mismatched image")
                with Image.open(image_path) as image:
                    image.load()
                    if list(image.size) != observed["size"] or image.mode != "RGB":
                        raise ValueError("Image dimensions or RGB mode disagree with record")
                started = time.perf_counter_ns()
                prediction = model.predict(image_path)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if list(prediction.size) != observed["size"]:
                    raise ValueError("Model output size disagrees with source image")
                row.update(
                    _render_prediction(
                        prediction, image_path, output / "pixels" / f"{len(frames):06d}", protocol
                    )
                )
                row["inference_ms"] = elapsed
                row["processing_ms"] = (time.perf_counter_ns() - started) / 1e6
                row["projected_age_ms"] = row["source_age_ms"] + row["processing_ms"]
                row["stale"] = row["projected_age_ms"] > protocol["max_result_age_ms"]
            except (OSError, ValueError, RuntimeError) as error:
                row.update(status="invalid", error=str(error))
            frames.append(row)
    road = {
        "version": 1,
        "model": model.metadata,
        "protocol": protocol,
        "protocol_sha256": file_hash(output / "protocol.json"),
        "dataset_sha256": file_hash(output / "dataset.json"),
        "frames": frames,
        "geometry_ready": False,
        "evaluation": {"status": "awaiting_independent_labels"},
    }
    _json(output / "perception.json", road)
    _json(
        output / "manifest.json",
        {
            "hashes": {
                name: file_hash(output / name)
                for name in ("perception.json", "protocol.json", "dataset.json")
            },
            "source_hashes": source_hashes,
        },
    )
    report = output / "report.html"
    display = {
        **road,
        "frames": [
            {
                **f,
                "image_url": _url(Path(f["image_path"]), report),
                "overlay_url": _url(Path(f["overlay_path"]), report)
                if f.get("overlay_path")
                else None,
            }
            for f in frames
        ],
    }
    _write_page(
        output / "annotate.html",
        "annotate.html",
        {
            "protocol_sha256": road["protocol_sha256"],
            "dataset_sha256": road["dataset_sha256"],
            "frames": [
                {
                    **{key: f[key] for key in ("id", "image_url", "image_sha256", "size", "tags")},
                    "rows": sorted(
                        {round(y * (f["size"][1] - 1)) for y in protocol["boundary_rows"]}
                    ),
                }
                for f in display["frames"]
                if f["annotate"]
            ],
        },
    )
    return replay_perception(PerceptionReplay(output, report))
