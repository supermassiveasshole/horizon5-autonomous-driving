"""Independent pixel labels and explicitly limited temporal error measurements."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

REQUIRED_TAGS = {
    "straight",
    "left_turn",
    "right_turn",
    "shoulder",
    "guardrail",
    "low_speed",
    "medium_speed",
    "high_speed",
    "ego_occlusion",
    "blue_arrows",
    "hud",
    "shadow",
    "motion_blur",
}


def distribution(values: list[float]) -> dict[str, Any]:
    values = sorted(values)
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p95": values[math.ceil(len(values) * 0.95) - 1] if values else None,
        "max": max(values) if values else None,
    }


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _label_pixels(label: dict[str, Any], size: tuple[int, int]) -> tuple[bytes, bytes]:
    from PIL import Image, ImageDraw

    planes = []
    for key in ("road_polygons", "ignore_polygons"):
        plane = Image.new("L", size)
        draw = ImageDraw.Draw(plane)
        if not isinstance(label.get(key), list):
            raise ValueError(f"Label requires {key}")
        for polygon in label[key]:
            if (
                not isinstance(polygon, list)
                or len(polygon) < 3
                or any(
                    not isinstance(p, list)
                    or len(p) != 2
                    or any(type(v) is not int for v in p)
                    or not 0 <= p[0] < size[0]
                    or not 0 <= p[1] < size[1]
                    for p in polygon
                )
            ):
                raise ValueError("Label polygon requires in-image integer vertices")
            draw.polygon([tuple(p) for p in polygon], fill=1)
        planes.append(plane.tobytes())
    return planes[0], planes[1]


def _score_frame(frame: dict[str, Any], label: dict[str, Any], threshold: int) -> dict[str, Any]:
    from PIL import Image

    size = tuple(frame["size"])
    truth, ignored = _label_pixels(label, (size[0], size[1]))
    with Image.open(frame["mask_path"]) as mask:
        candidates, raw_classes, scores = (p.tobytes() for p in mask.split())
    counts: dict[str, Any] = dict(
        tp=0,
        fp=0,
        fn=0,
        high_errors=0,
        high_count=0,
        unknown=0,
        observed=0,
        high_road_fp=0,
        high_road_count=0,
    )
    for gt, skip, pred, cls, score in zip(
        truth, ignored, candidates, raw_classes, scores, strict=True
    ):
        if skip:
            continue
        road = pred == 1
        counts["observed"] += 1
        counts["tp"] += bool(gt and road)
        counts["fp"] += bool(not gt and road)
        counts["fn"] += bool(gt and not road)
        counts["unknown"] += pred == 0
        if pred != 0 and score >= threshold:
            counts["high_count"] += 1
            counts["high_errors"] += (cls == 0) != bool(gt)
            counts["high_road_count"] += cls == 0
            counts["high_road_fp"] += cls == 0 and not gt
    errors: dict[str, list[float]] = {"left": [], "right": []}
    residuals: dict[str, float] = {}
    targets = misses = 0
    predictions = {b["y"]: b for b in frame["boundaries"]}
    seen = set()
    for boundary in label["boundaries"]:
        y = boundary["y"]
        if type(y) is not int or y not in predictions or y in seen:
            raise ValueError("Independent boundary row is duplicate or outside protocol")
        seen.add(y)
        left, right = boundary.get("left"), boundary.get("right")
        if any(x is not None and type(x) is not int for x in (left, right)):
            raise ValueError("Independent boundary coordinates must be integers or null")
        if left is not None and right is not None and left > right:
            raise ValueError("Left boundary must precede right boundary")
        for side in ("left", "right"):
            x = boundary.get(side)
            if x is None:
                continue
            if type(x) is not int or not 0 <= x < size[0] or ignored[y * size[0] + x]:
                raise ValueError("Independent boundary must be visible and in the image")
            targets += 1
            predicted = predictions[y][side]
            if predicted is None:
                misses += 1
            else:
                errors[side].append(abs(predicted - x))
                residuals[f"{y}:{side}"] = predicted - x
    return {
        **counts,
        "left_errors": errors["left"],
        "right_errors": errors["right"],
        "boundary_targets": targets,
        "boundary_misses": misses,
        "residuals": residuals,
    }


def _aggregate(scored: list[dict[str, Any]]) -> dict[str, Any]:
    sums = {
        key: sum(s[key] for s in scored)
        for key in (
            "tp",
            "fp",
            "fn",
            "high_errors",
            "high_count",
            "observed",
            "unknown",
            "boundary_targets",
            "boundary_misses",
            "high_road_fp",
            "high_road_count",
        )
    }
    left = [v for s in scored for v in s["left_errors"]]
    right = [v for s in scored for v in s["right_errors"]]
    return {
        "annotated_frames": len(scored),
        "pixel_counts": sums,
        "road_iou": _ratio(sums["tp"], sums["tp"] + sums["fp"] + sums["fn"]),
        "road_miss_fraction": _ratio(sums["fn"], sums["tp"] + sums["fn"]),
        "high_confidence_error_fraction": _ratio(sums["high_errors"], sums["high_count"]),
        "high_confidence_pixels": sums["high_count"],
        "high_confidence_false_road_fraction": _ratio(
            sums["high_road_fp"], sums["high_road_count"]
        ),
        "unknown_fraction": _ratio(sums["unknown"], sums["observed"]),
        "left_mae_px": statistics.mean(left) if left else None,
        "right_mae_px": statistics.mean(right) if right else None,
        "left_count": len(left),
        "right_count": len(right),
        "missed_boundary_fraction": _ratio(sums["boundary_misses"], sums["boundary_targets"]),
    }


def evaluate(road: dict[str, Any], labels: dict[str, Any] | None) -> dict[str, Any]:
    protocol = road["protocol"]
    frames = road["frames"]
    result: dict[str, Any] = {
        "status": "awaiting_independent_labels",
        "geometry_ready": False,
        "inference_ms": distribution([f["inference_ms"] for f in frames if f["status"] == "ok"]),
        "processing_ms": distribution(
            [f["processing_ms"] for f in frames if f["status"] == "ok" and "processing_ms" in f]
        ),
        "invalid_frames": sum(f["status"] != "ok" for f in frames),
        "stale_frames": sum(f.get("stale", False) for f in frames),
    }
    by_id = {f["id"]: f for f in frames}
    scored: dict[str, dict[str, Any]] = {}
    labelled_tags: dict[str, set[str]] = defaultdict(set)
    if labels is not None:
        if not isinstance(labels, dict) or not isinstance(labels.get("annotator"), dict):
            raise ValueError("Independent labels and annotator must be objects")
        if (
            labels.get("version") != 1
            or labels.get("protocol_sha256") != road["protocol_sha256"]
            or labels.get("dataset_sha256") != road["dataset_sha256"]
            or labels.get("annotator", {}).get("kind") != "human"
            or not isinstance(labels["annotator"].get("name"), str)
            or not labels["annotator"]["name"].strip()
        ):
            raise ValueError("Independent labels require matching provenance and a human annotator")
        if not isinstance(labels.get("frames"), list):
            raise ValueError("Independent labels require a frames list")
        seen = set()
        for label in labels["frames"]:
            if (
                not isinstance(label, dict)
                or not isinstance(label.get("id"), str)
                or not isinstance(label.get("boundaries"), list)
                or any(not isinstance(b, dict) or "y" not in b for b in label["boundaries"])
                or not isinstance(label.get("tags", []), list)
                or any(not isinstance(tag, str) for tag in label.get("tags", []))
            ):
                raise ValueError("Invalid independent label structure")
            identifier = label["id"]
            if identifier not in by_id or identifier in seen:
                raise ValueError("Unknown or duplicate label frame")
            seen.add(identifier)
            frame = by_id[identifier]
            if label.get("image_sha256") != frame["image_sha256"]:
                raise ValueError("Label source image hash mismatch")
            if label.get("reviewed") is not True or frame["status"] != "ok":
                continue
            scored[identifier] = _score_frame(
                frame, label, math.ceil(protocol["high_confidence_threshold"] * 255)
            )
            scored[identifier]["tags"] = label.get("tags", frame["tags"])
            labelled_tags[frame["split"]].update(scored[identifier]["tags"])
        result["status"] = "evaluated"
    result["splits"] = {}
    result["per_frame"] = scored
    for split in ("development", "holdout"):
        subset = [f for f in frames if f["split"] == split]
        metrics = _aggregate([scored[f["id"]] for f in subset if f["id"] in scored])
        motion: list[float] = []
        residual_motion: list[float] = []
        pairs = 0
        for a, b in zip(subset, subset[1:]):
            gap = (b["capture_start_ns"] - a["capture_start_ns"]) / 1e6
            if (
                a["clip"] != b["clip"]
                or a.get("continuity") != b.get("continuity")
                or a["segment"] != b["segment"]
                or a["status"] != "ok"
                or b["status"] != "ok"
                or not 0 < gap <= protocol["max_pair_gap_ms"]
                or a["source_observation"] not in (None, "stale_observation")
                or b["source_observation"] not in (None, "stale_observation")
            ):
                continue
            pairs += 1
            for ba, bb in zip(a["boundaries"], b["boundaries"], strict=True):
                for side in ("left", "right"):
                    if ba[side] is not None and bb[side] is not None:
                        motion.append(abs(bb[side] - ba[side]))
            if a["id"] in scored and b["id"] in scored:
                ra, rb = scored[a["id"]]["residuals"], scored[b["id"]]["residuals"]
                residual_motion.extend(abs(rb[k] - ra[k]) for k in ra.keys() & rb.keys())
        metrics["raw_boundary_motion_px"] = distribution(motion)
        metrics["residual_jitter_px"] = distribution(residual_motion)
        metrics["continuous_pairs"] = pairs
        metrics["coverage_gaps"] = sorted(REQUIRED_TAGS - labelled_tags[split])
        metrics["inference_ms"] = distribution(
            [f["inference_ms"] for f in subset if f["status"] == "ok"]
        )
        result["splits"][split] = metrics
    result["strata"] = {
        split: {
            tag: _aggregate(
                [
                    scored[f["id"]]
                    for f in frames
                    if f["id"] in scored and f["split"] == split and tag in scored[f["id"]]["tags"]
                ]
            )
            for tag in sorted(labelled_tags[split])
        }
        for split in ("development", "holdout")
    }
    gates = protocol["gates"]
    holdout = result["splits"]["holdout"]
    checks = {
        "labels_and_coverage": all(
            m["annotated_frames"] >= gates["min_annotated_frames_per_split"]
            and not m["coverage_gaps"]
            for m in result["splits"].values()
        ),
        "valid_artifacts": result["invalid_frames"] == 0,
    }
    for name, value, limit, minimum in (
        ("road_iou", holdout["road_iou"], gates["min_road_iou"], True),
        ("left_boundary", holdout["left_mae_px"], gates["max_boundary_mae_px"], False),
        ("right_boundary", holdout["right_mae_px"], gates["max_boundary_mae_px"], False),
        (
            "boundary_misses",
            holdout["missed_boundary_fraction"],
            gates["max_missed_boundary_fraction"],
            False,
        ),
        (
            "high_confidence_errors",
            holdout["high_confidence_error_fraction"],
            gates["max_high_confidence_error_fraction"],
            False,
        ),
        ("inference_p95", holdout["inference_ms"]["p95"], gates["max_inference_p95_ms"], False),
        (
            "residual_jitter",
            holdout["residual_jitter_px"]["p95"],
            gates["max_boundary_residual_jitter_px"],
            False,
        ),
    ):
        checks[name] = value is not None and (value >= limit if minimum else value <= limit)
    result["gates"] = checks
    result["geometry_ready"] = all(checks.values())
    return result
