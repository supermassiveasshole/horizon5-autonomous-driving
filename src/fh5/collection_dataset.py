"""Immutable selections of sealed continuous recordings, with reviewed attempt groups."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.collection_review import _references, collection_result
from fh5.collection_store import encode, read_bounded, write_file
from fh5.demonstrations import _profile
from fh5.numeric_images import PixelContract

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class CollectionDataset:
    config_file: Path
    output_dir: Path


@dataclass(frozen=True)
class CollectionDatasetReview:
    dataset_file: Path
    report_path: Path


def _evidence(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, str) and 0 < len(v.strip()) <= 2000 for v in value)
    )


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value) is not None


def _bounds(item: dict[str, Any], lower: int, upper: int) -> tuple[int, int]:
    start, end = item["start_sequence"], item["end_sequence"]
    if type(start) is not int or type(end) is not int or not lower <= start < end <= upper:
        raise ValueError("Review intervals must be ordered, disjoint and half-open")
    return start, end


def validate_review(value: dict[str, Any], binding: str) -> None:
    if (
        value.get("version") != 1
        or value.get("session_sha256") != binding
        or type(value.get("conditions_verified")) is not bool
        or not _evidence(value.get("evidence"))
        or not isinstance(value.get("attempts"), list)
        or not 1 <= len(value["attempts"]) <= 1000
    ):
        raise ValueError("Collection review needs bound session, conditions and attempt evidence")
    previous = 0
    for attempt in value["attempts"]:
        if (
            not _identifier(attempt.get("id"))
            or not _identifier(attempt.get("group"))
            or attempt.get("split") not in ("train", "development", "evaluation")
            or not isinstance(attempt.get("related_attempts"), list)
            or not all(_identifier(v) for v in attempt["related_attempts"])
            or not isinstance(attempt.get("intervals"), list)
            or len(attempt["intervals"]) > 1000
        ):
            raise ValueError("Invalid attempt identity, split or reviewed intervals")
        start, end = _bounds(attempt, previous, 10_000_000)
        previous = end
        position = start
        for interval in attempt["intervals"]:
            _, position = _bounds(interval, position, end)
            if (
                interval.get("quality") not in ("trusted", "failed", "unknown")
                or not _evidence(interval.get("evidence"))
                or not isinstance(interval.get("reasons"), list)
                or any(not isinstance(r, str) or not r for r in interval["reasons"])
                or (interval["quality"] == "trusted" and interval["reasons"])
                or interval.get("road_kind")
                not in ("straight", "left_curve", "right_curve", "unknown")
            ):
                raise ValueError("Reviewed quality needs evidence and explicit road knowledge")
    if len({a["group"] for a in value["attempts"]}) > 1 and not _evidence(
        value.get("independence_evidence")
    ):
        raise ValueError(
            "Independent attempts in one session require explicit independence evidence"
        )


def _config(value: dict[str, Any]) -> None:
    if (
        set(value) != {"version", "seed", "sources", "rules"}
        or value["version"] != 1
        or type(value["seed"]) is not int
        or not 0 <= value["seed"] < 2**32
        or not isinstance(value["sources"], list)
        or not 1 <= len(value["sources"]) <= 100
    ):
        raise ValueError("Invalid bounded collection dataset configuration")
    rules = value["rules"]
    if set(rules) != {
        "max_samples_per_attempt",
        "speed_range_mps",
        "steering_limit",
        "longitudinal_limit",
        "max_label_delay_ms",
    }:
        raise ValueError("Unsupported collection dataset selection rules")
    if (
        type(rules["max_samples_per_attempt"]) is not int
        or not 1 <= rules["max_samples_per_attempt"] <= 5000
    ):
        raise ValueError("Invalid per-attempt reservoir size")
    speed = rules["speed_range_mps"]
    if not isinstance(speed, list) or len(speed) != 2:
        raise ValueError("Invalid speed envelope")
    if any(
        type(v) not in (int, float) or not math.isfinite(v)
        for v in [
            *speed,
            rules["steering_limit"],
            rules["longitudinal_limit"],
            rules["max_label_delay_ms"],
        ]
    ):
        raise ValueError("Selection limits must be finite numbers")
    if not (
        0 <= speed[0] < speed[1] <= 150
        and 0 < rules["steering_limit"] <= 1
        and 0 < rules["longitudinal_limit"] <= 1
        and 1 <= rules["max_label_delay_ms"] <= 250
    ):
        raise ValueError("Invalid action, speed or label-age envelope")


def _freeze_source(entry: dict[str, Any], base: Path) -> dict[str, Any]:
    if set(entry) != {"recording", "review"}:
        raise ValueError("Source requires recording and review paths")
    root = (base / entry["recording"]).resolve()
    payload = read_bounded(root / "session.json", 1024**2)
    binding = hashlib.sha256(payload).hexdigest()
    index = json.loads(read_bounded(root / "index.json", 4 * 1024**2))
    references = _references(index, binding)
    if not references:
        raise ValueError("No published sealed blocks are available")
    review = json.loads(read_bounded(base / entry["review"], 4 * 1024**2))
    validate_review(review, binding)
    return {
        "recording": str(root),
        "session_sha256": binding,
        "blocks": list(references.values()),
        "review": review,
        "review_sha256": hashlib.sha256(encode(review)).hexdigest(),
    }


def _groups(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts: dict[str, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    for source in sources:
        review = source["review"]
        validate_review(review, source["session_sha256"])
        if hashlib.sha256(encode(review)).hexdigest() != source["review_sha256"]:
            raise ValueError("Frozen review changed")
        for attempt in review["attempts"]:
            identifier, group = attempt["id"], attempt["group"]
            if identifier in attempts:
                raise ValueError("Duplicate attempt identity")
            attempts[identifier] = attempt
            value = groups.setdefault(
                group, {"id": group, "split": attempt["split"], "attempts": []}
            )
            if value["split"] != attempt["split"]:
                raise ValueError("Related attempt group cannot cross dataset splits")
            value["attempts"].append(identifier)
    for attempt in attempts.values():
        for related in attempt["related_attempts"]:
            if related not in attempts or attempts[related]["group"] != attempt["group"]:
                raise ValueError("Related attempts must exist and share one group")
    return sorted(groups.values(), key=lambda g: g["id"])


def build_snapshot(config: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    from fh5.collection_selection import select_source

    _config(config)
    groups = _groups(sources)
    seen = set()
    compatibility = None
    samples, details = [], []
    windows: list[tuple[int, int]] = []
    diagnostic = False
    for number, source in enumerate(sources):
        binding = source["session_sha256"]
        if binding in seen:
            raise ValueError("Duplicate source cannot form independent dataset groups")
        seen.add(binding)
        root = Path(source["recording"])
        payload = read_bounded(root / "session.json", 1024**2)
        if hashlib.sha256(payload).hexdigest() != binding:
            raise ValueError("Frozen collection session changed")
        session = json.loads(payload)
        if (
            session.get("kind") != "continuous-numeric-collection-v1"
            or session.get("version") != 1
            or session.get("commands_sent") is not False
        ):
            raise ValueError("Unsupported passive collection source")
        profile = _profile(encode(session["profile"]))
        if profile["calibration"]["status"] != "verified":
            raise ValueError("Uncalibrated input cannot enter dataset")
        contract = PixelContract.from_metadata(session["configuration"]["pixels"])
        current = (
            contract.metadata(),
            profile["mapping"],
            session["input_conditions"],
            session["configuration"]["expected_car_ordinal"],
            session["configuration"]["expected_pi"],
            session["source_kind"],
        )
        if compatibility is not None and compatibility != current:
            raise ValueError("Dataset sources have incompatible input conditions")
        compatibility = current
        diagnostic |= (
            session["source_kind"] != "live_passive"
            or session["software_snapshot"].get("verified") is not True
        )
        rows, stats = select_source(
            source, session, config["rules"], random.Random(config["seed"] + number)
        )
        for detail in stats:
            first, last = detail["first_ns"], detail["last_ns"]
            if any(first <= end and begin <= last for begin, end in windows):
                raise ValueError("Overlapping source windows cannot establish independent attempts")
        windows.extend((d["first_ns"], d["last_ns"]) for d in stats)
        samples.extend(rows)
        details.extend(stats)
        if len(samples) > 50_000:
            raise ValueError("Snapshot exceeds 50000 selected observations; use smaller reservoirs")
    assert compatibility is not None
    return {
        "version": 1,
        "kind": "collection-dataset-snapshot-v1",
        "config": config,
        "sources": sources,
        "groups": groups,
        "samples": samples,
        "coverage": details,
        "pixel_contract": compatibility[0],
        "action_contract": compatibility[1],
        "diagnostic_only": diagnostic,
        "closed_loop_validated": False,
        "scope": "frozen selection; temporal training adapter pending",
    }


def _summary(data: dict[str, Any], digest: str) -> dict[str, Any]:
    splits = {g["id"]: g["split"] for g in data["groups"]}
    counts = Counter(splits[s["group"]] for s in data["samples"] if s["bc_eligible"])
    development = [d for d in data["coverage"] if d["split"] != "evaluation"]
    events: Counter[str] = Counter()
    for entry in development:
        events.update(entry["trusted_events"])
    prompts = {
        "startup": "正常转向起步并驶入道路，不必反复直踩 RT。",
        "left": "补充正常左弯驾驶。",
        "right": "补充正常右弯驾驶。",
        "release_rt": "在正常驾驶中完全松开 RT。",
        "brake": "在适当路段松 RT 后用 LT 减速。",
    }
    return {
        "version": 1,
        "verified": True,
        "dataset_sha256": digest,
        "commands_sent": False,
        "diagnostic_only": data["diagnostic_only"],
        "ready_for_software_training": counts["train"] > 0 and counts["development"] > 0,
        "real_candidate_ready": False,
        "bc_samples_by_split": dict(counts),
        "development_coverage": development,
        "supplement_suggestions": [
            message for name, message in prompts.items() if not events[name]
        ],
        "evaluation": {
            "groups": sum(g["split"] == "evaluation" for g in data["groups"]),
            "coverage": "withheld",
        },
        "training_adapter": "collection-bc-prepare",
        "closed_loop_validated": False,
    }


def run_collection_dataset(request: CollectionDataset | CollectionDatasetReview) -> RunResult:
    from fh5.experiment import RunResult

    if isinstance(request, CollectionDataset):
        if request.output_dir.exists():
            raise FileExistsError(request.output_dir)
        config = json.loads(read_bounded(request.config_file, 4 * 1024**2))
        _config(config)
        sources = [_freeze_source(s, request.config_file.parent) for s in config["sources"]]
        _report_destination(request.output_dir / "report.html", sources)
        data = build_snapshot(config, sources)
        payload = encode(data)
        if len(payload) > 128 * 1024**2:
            raise ValueError("Dataset exceeds 128 MiB snapshot budget; reduce sources or reviews")
        request.output_dir.mkdir(parents=True)
        path = request.output_dir / "dataset.json"
        temporary = path.with_suffix(".tmp")
        write_file(temporary, payload)
        temporary.rename(path)
        report = request.output_dir / "report.html"
    else:
        path, report = request.dataset_file, request.report_path
        data = json.loads(read_bounded(path, 128 * 1024**2))
        if data.get("kind") != "collection-dataset-snapshot-v1" or data.get("version") != 1:
            raise ValueError("Unsupported collection dataset snapshot")
        _report_destination(report, data["sources"])
        if build_snapshot(data["config"], data["sources"]) != data:
            raise ValueError("Dataset differs from canonical frozen source reconstruction")
    summary = _summary(data, hashlib.sha256(read_bounded(path, 128 * 1024**2)).hexdigest())
    collection_result(report, summary, title="持续采集数据快照")
    write_file(report.with_suffix(".json"), encode(summary))
    return RunResult({}, [], [], {"collection_dataset": summary}, report)


def _report_destination(report: Path, sources: list[dict[str, Any]]) -> None:
    if report.suffix.lower() != ".html":
        raise ValueError("Dataset report must use an .html path distinct from JSON evidence")
    for path in (report, report.with_suffix(".json")):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        if any(path.resolve().is_relative_to(Path(s["recording"]).resolve()) for s in sources):
            raise ValueError("Dataset outputs cannot be written inside source recordings")
