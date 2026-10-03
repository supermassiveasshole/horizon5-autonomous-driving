"""Frozen, whole-recording demonstration splits and supervision-only future geometry."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left, bisect_right
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.collection.demonstrations import DemonstrationReplay, _hash, validate_demonstration

if TYPE_CHECKING:
    from fh5.result import RunResult


@dataclass(frozen=True)
class DemonstrationDataset:
    config_file: Path
    output_dir: Path


def _review(path: Path, directory: Path) -> dict[str, Any]:
    review: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if review.get("version") != 1 or any(
        review.get(key + "_sha256") != _hash(directory / (key + ".jsonl"))
        for key in ("packets", "vision")
    ):
        raise ValueError("Review is not bound to this recording")
    previous_end = -1
    for section in review["intervals"]:
        start, end = section["start_ns"], section["end_ns"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end
            or start < previous_end
            or section["quality"] not in ("trusted", "failed", "unknown")
            or not section.get("evidence")
        ):
            raise ValueError("Review intervals must be ordered, disjoint, evidenced and half-open")
        previous_end = end
    changes = review["intent_changes"]
    if any(
        type(e.get("observed_ns")) is not int or e["observed_ns"] < 0 or not e.get("evidence")
        for e in changes
    ) or any(a["observed_ns"] >= b["observed_ns"] for a, b in zip(changes, changes[1:])):
        raise ValueError("Intent changes need ordered timestamps and evidence")
    return review


def _preflight(config_file: Path, config: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    if config.get("version") != 1 or {entry["split"] for entry in config["runs"]} != {
        "train",
        "holdout",
    }:
        raise ValueError("Dataset requires whole train and holdout recordings")
    offsets = config["future_offsets_ms"]
    if (
        not offsets
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in [config["max_label_delay_ms"], *offsets]
        )
        or any(a >= b for a, b in zip(offsets, offsets[1:]))
    ):
        raise ValueError("Label delay and ordered future offsets must be positive finite numbers")
    prepared = []
    seen: set[str] = set()
    conditions = None
    windows: list[tuple[int, int]] = []
    for entry in config["runs"]:
        directory = (config_file.parent / entry["directory"]).resolve()
        profile = validate_demonstration(directory)
        review = _review(config_file.parent / entry["review"], directory)
        digest = _hash(directory / "packets.jsonl")
        if digest in seen:
            raise ValueError("Duplicate recordings cannot form independent splits")
        seen.add(digest)
        metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        vision = json.loads((directory / "vision-session.json").read_text(encoding="utf-8"))
        observations = json.loads(
            (directory / "observation-config.json").read_text(encoding="utf-8")
        )
        if observations.get("version") != 2 or observations["reference_mode"] != "optional":
            raise ValueError("Dual-view demonstrations require optional-reference observation v2")
        current = (
            metadata["snapshot"],
            metadata["source_kind"],
            profile,
            vision["camera_mode"],
            observations,
        )
        calibrated_conditions = profile["calibration"].get("conditions")
        if calibrated_conditions is not None and calibrated_conditions != metadata["snapshot"]:
            raise ValueError("Calibration conditions do not match demonstration")
        if conditions is not None and conditions != current:
            raise ValueError(
                "Demonstration conditions, device, mapping or observation versions differ"
            )
        conditions = current
        start, end = vision["started_ns"], vision["ended_ns"]
        if any(start < other_end and end > other_start for other_start, other_end in windows):
            raise ValueError("Overlapping recording windows are not independent runs")
        windows.append((start, end))
        prepared.append((directory, review))
    for directory, _ in prepared:
        reference = directory / "observation-route/route.json"
        if reference.exists():
            source = json.loads(reference.read_text(encoding="utf-8"))["source"]["packets_sha256"]
            if source in seen:
                raise ValueError("Frozen reference must be independent of all demonstration splits")
    return prepared


def _trusted_span(review: dict[str, Any], start: int, end: int) -> bool:
    return any(
        s["quality"] == "trusted" and s["start_ns"] <= start <= end < s["end_ns"]
        for s in review["intervals"]
    ) and not any(start < c["observed_ns"] <= end for c in review["intent_changes"])


def _future(
    decision: dict[str, Any],
    samples: list[dict[str, Any]],
    times: list[int],
    boundaries: list[int],
    offsets: list[float],
    review: dict[str, Any],
) -> tuple[list[Any], list[bool]]:
    points, mask = [], []
    origin = decision["telemetry"]
    for offset in offsets:
        target = decision["decision_ns"] + int(offset * 1e6)
        index = bisect_left(times, target)
        valid = (
            origin is not None
            and origin["motion"] is not None
            and bisect_right(boundaries, origin["received_monotonic_ns"]) == decision["segment"]
            and 0 < index < len(samples)
            and bisect_right(boundaries, target) == decision["segment"]
            and _trusted_span(review, origin["received_monotonic_ns"], target)
        )
        point = None
        if valid:
            a, b = samples[index - 1], samples[index]
            valid = (
                a["segment"] == b["segment"] == decision["telemetry_segment"]
                and a["is_race_on"]
                and b["is_race_on"]
                and 0 < times[index] - times[index - 1] <= 250_000_000
                and b["game_timestamp_ms"] > a["game_timestamp_ms"]
                and bisect_right(boundaries, times[index]) == decision["segment"]
                and bisect_right(boundaries, times[index - 1]) == decision["segment"]
                and _trusted_span(
                    review, min(origin["received_monotonic_ns"], times[index - 1]), times[index]
                )
            )
            if valid:
                fraction = (target - times[index - 1]) / (times[index] - times[index - 1])
                position = [
                    x + fraction * (y - x) for x, y in zip(a["position_m"], b["position_m"])
                ]
                dx, dz = (
                    position[0] - origin["position_m"][0],
                    position[2] - origin["position_m"][2],
                )
                yaw = origin["motion"]["yaw_rad"]
                point = [
                    math.cos(yaw) * dx - math.sin(yaw) * dz,
                    math.sin(yaw) * dx + math.cos(yaw) * dz,
                ]
        points.append(point)
        mask.append(bool(valid))
    return points, mask


def export_demonstrations(request: DemonstrationDataset) -> RunResult:
    from fh5.experiment import run_experiment
    from fh5.reporting.telemetry import write_report

    config = json.loads(request.config_file.read_text(encoding="utf-8"))
    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    prepared = _preflight(request.config_file, config)
    examples = []
    sources: list[dict[str, Any]] = []
    first_result = None
    for run_index, entry in enumerate(config["runs"]):
        directory, review = prepared[run_index]
        result = run_experiment(
            DemonstrationReplay(directory, request.output_dir / f"run-{run_index}.html")
        )
        if first_result is None:
            first_result = result
        observations = result.summary["observations"]
        demo = result.summary["demonstration"]
        if demo["integrity_errors"] or observations["action_source"].get("errors"):
            raise ValueError("Demonstration pixels or actions failed integrity checks")
        vehicles = {
            (s["car_ordinal"], s["car_class"], s["car_performance_index"])
            for s in result.samples
            if s["is_race_on"]
        }
        if len(vehicles) != 1:
            raise ValueError("Demonstration must contain one active vehicle configuration")
        vehicle = list(next(iter(vehicles)))
        if sources and sources[0]["observed_vehicle"] != vehicle:
            raise ValueError("Observed vehicle differs between demonstrations")
        inputs = demo["inputs"]
        input_times = [row["poll_ns"] for row in inputs]
        times = [s["received_monotonic_ns"] for s in result.samples]
        boundaries = observations["history_boundaries_ns"]
        sources.append(
            {
                "directory": str(directory),
                "split": entry["split"],
                "packets_sha256": observations["source"]["packets_sha256"],
                "profile": demo["profile"],
                "snapshot": result.metadata["snapshot"],
                "source_kind": result.metadata["source_kind"],
                "observed_vehicle": vehicle,
                "navigation": observations["navigation"],
                "review": review,
                "review_sha256": _hash(request.config_file.parent / entry["review"]),
                "demonstration_manifest_sha256": _hash(directory / "demonstration-session.json"),
            }
        )
        for decision in observations["decisions"]:
            tick = decision["decision_ns"]
            index = bisect_left(input_times, tick)
            label = inputs[index] if index < len(inputs) else None
            delay = (label["poll_ns"] - tick) / 1e6 if label else None
            reasons = list(decision["reasons"])
            quality = next(
                (
                    section["quality"]
                    for section in review["intervals"]
                    if section["start_ns"] <= tick < section["end_ns"]
                ),
                "unknown",
            )
            if quality != "trusted":
                reasons.append("demonstration_" + quality)
            if not demo["calibrated"]:
                reasons.append("unverified_mapping")
            if (
                label is None
                or delay is None
                or (label["available_ns"] - tick) / 1e6 > config["max_label_delay_ms"]
            ):
                reasons.append("missing_or_late_label")
            elif not label["mapping_valid"]:
                reasons.extend(label["reasons"])
            elif bisect_right(boundaries, label["available_ns"]) != decision[
                "segment"
            ] or not _trusted_span(review, tick, label["available_ns"]):
                reasons.append("label_discontinuity")
            future, mask = _future(
                decision, result.samples, times, boundaries, config["future_offsets_ms"], review
            )
            supervision = {
                "action_mask": not reasons,
                "action": label["mapped"] if label else None,
                "label_poll_ns": label["poll_ns"] if label else None,
                "label_available_ns": label["available_ns"] if label else None,
                "label_delay_ms": delay,
                "future_waypoints_m": future,
                "future_mask": mask,
                "future_offsets_ms": config["future_offsets_ms"],
                "quality": quality,
                "reasons": reasons,
            }
            no_reference = deepcopy(decision["actor"])
            count = len(no_reference["reference"]["mask"])
            no_reference["reference"] = {"waypoints_m": [None] * count, "mask": [False] * count}
            example = {
                "run_index": run_index,
                "split": entry["split"],
                "decision_ns": tick,
                "bc_eligible": not reasons,
                "views": {"no_reference": no_reference, "reference_assisted": decision["actor"]},
                "view_reasons": {
                    "no_reference": "explicit_training_reference_mask",
                    "reference_assisted": decision["route"]["status"],
                },
                "supervision": supervision,
            }
            examples.append(example)
            decision["supervision"] = supervision
        reviewed = request.output_dir / f"run-{run_index}-targets.html"
        write_report(
            reviewed,
            {
                "metadata": result.metadata,
                "samples": result.samples,
                "events": result.events,
                "summary": result.summary,
            },
        )
    assert first_result is not None
    dataset = {
        "version": 1,
        "config_sha256": hashlib.sha256(request.config_file.read_bytes()).hexdigest(),
        "config": config,
        "sources": sources,
        "examples": examples,
        "eligible_by_split": dict(Counter(e["split"] for e in examples if e["bc_eligible"])),
        "commands_sent": False,
    }
    (request.output_dir / "dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    first_result.summary["demonstration_dataset"] = dataset
    first_result = replace(first_result, report_path=request.output_dir / "report.html")
    write_report(
        first_result.report_path,
        {
            "metadata": first_result.metadata,
            "samples": first_result.samples,
            "events": first_result.events,
            "summary": first_result.summary,
        },
    )
    return first_result
