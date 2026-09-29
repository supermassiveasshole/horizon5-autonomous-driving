"""Versioned, causal multimodal observations; no policy or game input."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ObservationReplay:
    recording_dir: Path
    report_path: Path
    route_file: Path
    config_file: Path


def _hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_settings(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = {
        "version",
        "period_ms",
        "history_offsets_ms",
        "max_image_age_ms",
        "max_telemetry_age_ms",
        "waypoint_distances_m",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        raise ValueError("Unsupported observation settings")
    for name in ("period_ms", "max_image_age_ms", "max_telemetry_age_ms"):
        v = value[name]
        if type(v) not in (int, float) or not math.isfinite(v) or not 10 <= v <= 1000:
            raise ValueError(f"Invalid observation {name}")
    for name, maximum in (("history_offsets_ms", 2000), ("waypoint_distances_m", 500)):
        seq = value[name]
        if (
            not isinstance(seq, list)
            or not 1 <= len(seq) <= 16
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= maximum
                for v in seq
            )
        ):
            raise ValueError(f"Invalid observation {name}")
        if len(set(seq)) != len(seq):
            raise ValueError(f"Duplicate observation {name}")
    if value["history_offsets_ms"] != sorted(value["history_offsets_ms"], reverse=True) or (
        value["history_offsets_ms"][-1] != 0
    ):
        raise ValueError("History offsets must descend to zero")
    if value["waypoint_distances_m"] != sorted(value["waypoint_distances_m"]):
        raise ValueError("Waypoint distances must increase")
    return value


def freeze_inputs(config_file: Path | None, route_file: Path | None) -> dict[str, bytes]:
    """Bind passive collection to the route and settings selected before it began."""
    if config_file is None or route_file is None:
        return {}
    from fh5.routes import load_route

    config = read_settings(config_file)
    load_route(route_file)
    manifest = json.loads(route_file.read_text(encoding="utf-8"))
    files = {
        "observation-config.json": json.dumps(config, allow_nan=False).encode(),
        "observation-route/route.json": route_file.read_bytes(),
    }
    for item in [*manifest["assets"].values(), *manifest["evidence"]]:
        content = (route_file.parent / item["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("Reference changed while freezing observation inputs")
        files["observation-route/" + item["path"]] = content
    return files


def _preview(
    sample: dict[str, Any], route: dict[str, Any], distances: list[float]
) -> dict[str, Any]:
    # Heading-aligned ground plane, not pitch/roll corrected 3D or camera geometry.
    position = sample["position_m"]
    points = route["points"]
    match = sample["route"]
    distance, station = match["distance_m"], match["reference_s_m"]
    # Reuse the causal route matcher, but do not require legal-progress annotations
    # to consume a navigation prior. "unanchored" only forbids formal progress.
    status = (
        match["status"]
        if match["status"] in {"discontinuity", "ambiguous", "outside_reference", "inactive"}
        else "located"
    )
    result: dict[str, Any] = {
        "status": status,
        "reference_s_m": station if status == "located" else None,
        "distance_m": distance,
        "waypoints_m": [],
        "waypoint_mask": [],
        "meaning": "navigation_reference_only_not_legal_progress",
    }
    motion = sample["motion"]
    if motion is None:
        result["status"] = "unknown_heading"
    stations = [p["s_m"] for p in points]
    for ahead in distances:
        target = station + ahead
        valid = result["status"] == "located" and target <= stations[-1]
        result["waypoint_mask"].append(valid)
        if not valid:
            result["waypoints_m"].append(None)
            continue
        index = min(bisect_right(stations, target), len(points) - 1)
        a, b = points[index - 1], points[index]
        fraction = (target - a["s_m"]) / (b["s_m"] - a["s_m"])
        target_pos = [x + fraction * (y - x) for x, y in zip(a["position_m"], b["position_m"])]
        dx, dz = target_pos[0] - position[0], target_pos[2] - position[2]
        yaw = motion["yaw_rad"]
        result["waypoints_m"].append(
            [
                math.cos(yaw) * dx - math.sin(yaw) * dz,
                math.sin(yaw) * dx + math.cos(yaw) * dz,
            ]
        )
    return result


def build_observations(
    request: ObservationReplay,
    samples: list[dict[str, Any]],
    events: list[dict[str, Any]],
    vision: dict[str, Any] | None,
    route: dict[str, Any],
) -> dict[str, Any]:
    from PIL import Image

    from fh5.vision import _distribution

    config = read_settings(request.config_file)
    source_hash = _hash(request.recording_dir / "packets.jsonl")
    if source_hash == route["source"]["packets_sha256"]:
        raise ValueError("Reference must come from an independent historical recording")
    ordered = sorted(samples, key=lambda s: s["received_monotonic_ns"])
    times = [s["received_monotonic_ns"] for s in ordered]
    if ordered != samples or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Observation telemetry clock must increase in recording order")
    frames = sorted(
        (vision or {}).get("frames", []), key=lambda f: f.get("delivered_ns", f["available_ns"])
    )
    deliveries = [f.get("delivered_ns", f["available_ns"]) for f in frames]
    image_errors: list[list[str]] = []
    for frame in frames:
        errors = []
        if "delivered_ns" not in frame:
            errors.append("unknown_delivery")
        if frame.get("color") != "RGB":
            errors.append("unsupported_color")
        try:
            with Image.open(request.recording_dir / frame["path"]) as pixels:
                if pixels.mode != "RGB" or list(pixels.size) != frame.get("size"):
                    errors.append("pixel_metadata_mismatch")
                pixels.load()
        except (OSError, ValueError, Image.DecompressionBombError):
            errors.append("invalid_pixels")
        image_errors.append(errors)
    boundaries = [
        e["received_monotonic_ns"] for e in events if type(e.get("received_monotonic_ns")) is int
    ]
    boundaries += [
        s["received_monotonic_ns"]
        for a, s in zip(ordered, ordered[1:])
        if a["segment"] != s["segment"]
    ]
    boundaries += [
        e["observed_ns"]
        for e in (vision or {}).get("events", [])
        if e.get("kind")
        in (
            "focus_lost",
            "focus_restored",
            "capture_discarded",
            "telemetry_overflow",
            "rewind",
            "restart",
        )
        and type(e.get("observed_ns")) is int
    ]
    advanced = []
    last_advance = 0
    for index, sample in enumerate(ordered):
        if sample["route"]["status"] == "discontinuity" or sample["motion"] is None:
            boundaries.append(sample["received_monotonic_ns"])
        if (
            index == 0
            or sample["segment"] != ordered[index - 1]["segment"]
            or sample["game_timestamp_ms"] > ordered[index - 1]["game_timestamp_ms"]
        ):
            last_advance = sample["received_monotonic_ns"]
        advanced.append(last_advance)
    boundaries = sorted(set(boundaries))
    decisions: list[dict[str, Any]] = []
    step = int(config["period_ms"] * 1e6)
    tick_rows = [e for e in (vision or {}).get("events", []) if e.get("kind") == "observation_tick"]
    recorded = [e.get("observed_ns") for e in tick_rows]
    if any(type(t) is not int or t < 0 for t in recorded) or any(
        b <= a for a, b in zip(recorded, recorded[1:])
    ):
        raise ValueError("Invalid recorded observation clock")
    ticks = recorded if tick_rows else (range(times[0], times[-1] + 1, step) if times else [])
    if len(ticks) > 100_000:
        raise ValueError("Observation replay exceeds 100000 decisions; use a shorter recording")
    for tick in ticks:
        sample_index = bisect_right(times, tick) - 1
        if sample_index < 0:
            decisions.append(
                {
                    "decision_ns": tick,
                    "segment": None,
                    "usable": False,
                    "reasons": ["missing_telemetry"],
                    "telemetry_age_ms": None,
                    "telemetry": None,
                    "images": [None] * len(config["history_offsets_ms"]),
                    "history_mask": [False] * len(config["history_offsets_ms"]),
                    "route": {
                        "status": "missing_telemetry",
                        "waypoints_m": [],
                        "waypoint_mask": [],
                    },
                }
            )
            continue
        sample = ordered[sample_index]
        age = (tick - sample["received_monotonic_ns"]) / 1e6
        images: list[dict[str, Any] | None] = []
        seen = set()
        for offset in config["history_offsets_ms"]:
            index = bisect_right(deliveries, tick - int(offset * 1e6)) - 1
            if index < 0 or index in seen:
                images.append(None)
                continue
            seen.add(index)
            frame = frames[index]
            frame_age = (tick - frame["capture_start_ns"]) / 1e6
            image_reasons = list(image_errors[index])
            if frame["image_url"] is None:
                image_reasons.append("artifact_integrity")
            if frame_age > offset + config["max_image_age_ms"]:
                image_reasons.append("stale_image")
            if bisect_right(boundaries, tick) > bisect_right(boundaries, frame["capture_start_ns"]):
                image_reasons.append("history_discontinuity")
            images.append(
                {
                    **{
                        k: frame.get(k)
                        for k in (
                            "path",
                            "sha256",
                            "image_url",
                            "capture_start_ns",
                            "capture_end_ns",
                            "available_ns",
                            "size",
                            "client_size",
                            "color",
                            "resize_method",
                        )
                    },
                    "delivered_ns": deliveries[index],
                    "age_ms": frame_age,
                    "valid": not image_reasons,
                    "reasons": image_reasons,
                }
            )
        preview = _preview(sample, route, config["waypoint_distances_m"])
        mask = [bool(f and f["valid"]) for f in images]
        reasons = []
        if not sample["is_race_on"]:
            reasons.append("inactive_telemetry")
        if age > config["max_telemetry_age_ms"]:
            reasons.append("stale_telemetry")
        if sample["motion"] is None:
            reasons.append("invalid_motion")
        if tick - advanced[sample_index] > 250_000_000:
            reasons.append("stalled_game_clock")
        if (vision or {}).get("integrity_errors"):
            reasons.append("artifact_integrity")
        if not all(mask):
            reasons.append("incomplete_image_history")
        if preview["status"] != "located":
            reasons.append(preview["status"])
        if not all(preview["waypoint_mask"]):
            reasons.append("incomplete_route_preview")
        decisions.append(
            {
                "decision_ns": tick,
                "segment": bisect_right(boundaries, tick),
                "telemetry_segment": sample["segment"],
                "usable": not reasons,
                "reasons": reasons,
                "telemetry_age_ms": age,
                "telemetry": {
                    k: sample[k]
                    for k in (
                        "packet_index",
                        "received_monotonic_ns",
                        "game_timestamp_ms",
                        "position_m",
                        "speed_mps",
                        "motion",
                        "car_ordinal",
                        "car_performance_index",
                    )
                },
                "images": images,
                "history_mask": mask,
                "route": preview,
            }
        )
    return {
        "version": 1,
        "config": config,
        "decisions": decisions,
        "decision_count": len(decisions),
        "usable_decisions": sum(o["usable"] for o in decisions),
        "usable_fraction": sum(o["usable"] for o in decisions) / len(decisions)
        if decisions
        else 0.0,
        "ages_ms": {
            "telemetry": _distribution(
                [o["telemetry_age_ms"] for o in decisions if o["telemetry_age_ms"] is not None]
            ),
            "images_by_history_slot": [
                _distribution(
                    [o["images"][i]["age_ms"] for o in decisions if o["images"][i] is not None]
                )
                for i in range(len(config["history_offsets_ms"]))
            ],
        },
        "clock": "recorded_passive_checks_not_policy_calls"
        if tick_rows
        else "reconstructed_receipt_grid_not_actual_policy_calls",
        "preprocessing": {
            "version": "source-rgb-v1",
            "normalization": "none_rgb_uint8",
            "resize": "source_frame_metadata",
            "history": "causal_unique_frames",
        },
        "coordinate_frame": "heading_plane_right_forward_metres_v1",
        "camera": (vision or {}).get("session", {}).get("camera_mode", "unknown"),
        "camera_pose": "dynamic_unknown",
        "commands_sent": False,
        "reason_counts": dict(Counter(r for o in decisions for r in o["reasons"])),
        "integrity_errors": (vision or {}).get("integrity_errors", []),
        "source": {
            "recording": str(request.recording_dir.resolve()),
            "packets_sha256": source_hash,
            "session_sha256": _hash(request.recording_dir / "session.json"),
            "route_sha256": _hash(request.route_file),
            "route_source": route["source"],
            "config_sha256": _hash(request.config_file),
        },
    }
