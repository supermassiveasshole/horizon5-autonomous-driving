"""Immutable selections of sealed continuous recordings, with reviewed attempt groups."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import sqlite3
from collections import Counter
from collections.abc import Generator, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from fh5.artifacts.document import ReplayArray, replay_document, write_replay_document
from fh5.artifacts.io import VerifiedFile, encode, read_bounded, sha256_file, write_file
from fh5.artifacts.json_view import JsonArray, write_json
from fh5.collection.demonstrations import _profile
from fh5.collection.index import read_references
from fh5.observation.numeric import PixelContract
from fh5.reporting.presentation import optional_report

if TYPE_CHECKING:
    from fh5.result import RunResult


@dataclass(frozen=True)
class CollectionDatasetReview:
    dataset_file: Path
    report_path: Path


REVIEW_ARRAYS: set[tuple[str, ...]] = {
    ("evidence",),
    ("independence_evidence",),
    ("attempts", "*", "intervals"),
    ("attempts", "*", "related_attempts"),
    ("attempts", "*", "intervals", "*", "evidence"),
    ("attempts", "*", "intervals", "*", "reasons"),
}
SNAPSHOT_ARRAYS: set[tuple[str, ...]] = {
    ("config", "sources"),
    ("sources", "*", "blocks"),
    ("sources", "*", "review", "attempts"),
    *(("sources", "*", "review", *path) for path in REVIEW_ARRAYS),
}


class _CollectionIndex:
    """Run-owned selection and export rows; no persistent format or recovery state."""

    def __init__(self, database: sqlite3.Connection) -> None:
        self.database = database
        self.counts: Counter[str] = Counter()
        self.unique_frames = 0
        database.execute(
            "CREATE TABLE records (section TEXT, position INTEGER, data TEXT NOT NULL, "
            "source TEXT, sequence INTEGER, PRIMARY KEY (section, position), "
            "UNIQUE (source, sequence))"
        )
        database.execute(
            "CREATE TABLE identities (kind TEXT, identity TEXT, data TEXT, "
            "PRIMARY KEY (kind, identity)) WITHOUT ROWID"
        )
        database.execute(
            "CREATE TABLE relations (owner TEXT, related TEXT, "
            "PRIMARY KEY (owner, related)) WITHOUT ROWID"
        )

    def add_attempt(self, attempt: dict[str, Any]) -> None:
        try:
            self.database.execute(
                "INSERT INTO identities VALUES ('attempt', ?, ?)",
                (attempt["id"], encode(attempt["group"]).decode()),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("Duplicate attempt identity") from error
        self.database.executemany(
            "INSERT OR IGNORE INTO relations VALUES (?, ?)",
            ((attempt["id"], related) for related in attempt["related_attempts"]),
        )

    def check_related_attempts(self) -> None:
        invalid = self.database.execute(
            "SELECT 1 FROM relations r "
            "JOIN identities a ON a.kind = 'attempt' AND a.identity = r.owner "
            "LEFT JOIN identities b ON b.kind = 'attempt' AND b.identity = r.related "
            "WHERE b.identity IS NULL OR a.data != b.data LIMIT 1"
        ).fetchone()
        if invalid is not None:
            raise ValueError("Related attempts must exist and share one group")

    def append(self, section: str, row: dict[str, Any]) -> None:
        self.database.execute(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
            (
                section,
                self.counts[section],
                encode(row).decode(),
                row["source_sha256"] if section == "samples" else None,
                row["sequence"] if section == "samples" else None,
            ),
        )
        self.counts[section] += 1

    def array(self, section: str) -> ReplayArray:
        return ReplayArray(self.database, section, self.counts[section])

    def sample(self, source: str, sequence: int) -> dict[str, Any] | None:
        found = self.database.execute(
            "SELECT data FROM records WHERE source = ? AND sequence = ?", (source, sequence)
        ).fetchone()
        return None if found is None else json.loads(found[0])

    def frame_is_new(self, frame: dict[str, Any]) -> bool:
        key = json.dumps([frame["epoch"], frame["frame_id"]])
        prior = self.database.execute(
            "SELECT data FROM identities WHERE kind = 'frame' AND identity = ?", (key,)
        ).fetchone()
        if prior is not None:
            if json.loads(prior[0]) != frame:
                raise ValueError("Captured frame identity changed between source blocks")
            return False
        self.database.execute(
            "INSERT INTO identities VALUES ('frame', ?, ?)", (key, encode(frame).decode())
        )
        self.unique_frames += 1
        return True

    def pixel_is_new(self, digest: str) -> bool:
        return bool(
            self.database.execute(
                "INSERT OR IGNORE INTO identities VALUES ('pixel', ?, 'true')", (digest,)
            ).rowcount
        )


def _evidence(value: Any) -> bool:
    return (
        _sequence(value)
        and bool(value)
        and all(isinstance(v, str) and 0 < len(v.strip()) <= 2000 for v in value)
    )


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value) is not None


def _bounds(item: dict[str, Any], lower: int, upper: int | None) -> tuple[int, int]:
    start, end = item["start_sequence"], item["end_sequence"]
    if (
        type(start) is not int
        or type(end) is not int
        or not lower <= start < end
        or (upper is not None and end > upper)
    ):
        raise ValueError("Review intervals must be ordered, disjoint and half-open")
    return start, end


def validate_review(value: dict[str, Any], binding: str) -> None:
    if (
        value.get("version") != 1
        or value.get("session_sha256") != binding
        or type(value.get("conditions_verified")) is not bool
        or not _evidence(value.get("evidence"))
        or not _sequence(value.get("attempts"))
        or not value["attempts"]
    ):
        raise ValueError("Collection review needs bound session, conditions and attempt evidence")
    previous = 0
    first_group = None
    independent_groups = False
    for attempt in value["attempts"]:
        if (
            not _identifier(attempt.get("id"))
            or not _identifier(attempt.get("group"))
            or attempt.get("split") not in ("train", "development", "evaluation")
            or not _sequence(attempt.get("related_attempts"))
            or not all(_identifier(v) for v in attempt["related_attempts"])
            or not _sequence(attempt.get("intervals"))
        ):
            raise ValueError("Invalid attempt identity, split or reviewed intervals")
        if first_group is None:
            first_group = attempt["group"]
        independent_groups |= attempt["group"] != first_group
        start, end = _bounds(attempt, previous, None)
        previous = end
        position = start
        for interval in attempt["intervals"]:
            _, position = _bounds(interval, position, end)
            if (
                interval.get("quality") not in ("trusted", "failed", "unknown")
                or not _evidence(interval.get("evidence"))
                or not _sequence(interval.get("reasons"))
                or any(not isinstance(r, str) or not r for r in interval["reasons"])
                or (interval["quality"] == "trusted" and interval["reasons"])
                or interval.get("road_kind")
                not in ("straight", "left_curve", "right_curve", "unknown")
            ):
                raise ValueError("Reviewed quality needs evidence and explicit road knowledge")
    if independent_groups and not _evidence(value.get("independence_evidence")):
        raise ValueError(
            "Independent attempts in one session require explicit independence evidence"
        )


def _config(value: dict[str, Any]) -> None:
    if (
        set(value) != {"version", "seed", "sources", "rules"}
        or value["version"] != 1
        or type(value["seed"]) is not int
        or not 0 <= value["seed"] < 2**32
        or not _sequence(value["sources"])
        or not value["sources"]
    ):
        raise ValueError("Invalid collection dataset configuration")
    rules = value["rules"]
    if set(rules) != {
        "max_samples_per_attempt",
        "speed_range_mps",
        "steering_limit",
        "longitudinal_limit",
        "max_label_delay_ms",
    }:
        raise ValueError("Unsupported collection dataset selection rules")
    if type(rules["max_samples_per_attempt"]) is not int or rules["max_samples_per_attempt"] < 1:
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


def _review_digest(review: dict[str, Any]) -> str:
    with TemporaryDirectory(prefix="fh5-collection-review-") as temporary:
        path = Path(temporary) / "review.json"
        write_replay_document(path, review)
        return sha256_file(path)


@contextmanager
def _freeze_source(entry: dict[str, Any], base: Path) -> Iterator[dict[str, Any]]:
    if set(entry) != {"recording", "review"}:
        raise ValueError("Source requires recording and review paths")
    root = (base / entry["recording"]).resolve()
    payload = read_bounded(root / "session.json", 1024**2)
    binding = hashlib.sha256(payload).hexdigest()
    with TemporaryDirectory(prefix="fh5-collection-source-") as temporary:
        # The producer replaces index.json atomically while collection continues.
        # Copy one opened version before hashing it, rather than opening two versions.
        frozen_index = Path(temporary) / "index.json"
        with (root / "index.json").open("rb") as source, frozen_index.open("xb") as target:
            shutil.copyfileobj(source, target)
        review_path = base / entry["review"]
        with (
            replay_document(VerifiedFile(frozen_index, sha256_file(frozen_index))) as document,
            read_references(root, binding, document) as references,
            replay_document(
                VerifiedFile(review_path, sha256_file(review_path)), nested_arrays=REVIEW_ARRAYS
            ) as review,
        ):
            if not references:
                raise ValueError("No published sealed blocks are available")
            validate_review(review, binding)
            yield {
                "recording": str(root),
                "session_sha256": binding,
                "blocks": references,
                "review": review,
                "review_sha256": _review_digest(review),
            }


@contextmanager
def freeze_sources(entries: Sequence[dict[str, Any]], base: Path) -> Iterator[ReplayArray]:
    """Freeze one source at a time, then share one scoped index for all selections."""
    with TemporaryDirectory(prefix="fh5-collection-sources-") as temporary:
        path = Path(temporary) / "frozen-sources.json"

        def sources() -> Generator[dict[str, Any]]:
            for entry in entries:
                with _freeze_source(entry, base) as source:
                    yield source

        with (
            closing(sources()) as values,
            path.open("x", encoding="utf-8", newline="\n") as stream,
        ):
            write_json(stream, {"sources": JsonArray(values)})
        with replay_document(
            VerifiedFile(path, sha256_file(path)), nested_arrays=SNAPSHOT_ARRAYS
        ) as frozen:
            yield frozen["sources"]


def _groups(sources: Sequence[dict[str, Any]], index: _CollectionIndex) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for source in sources:
        review = source["review"]
        validate_review(review, source["session_sha256"])
        if _review_digest(review) != source["review_sha256"]:
            raise ValueError("Frozen review changed")
        for attempt in review["attempts"]:
            identifier, group = attempt["id"], attempt["group"]
            index.add_attempt(attempt)
            value = groups.setdefault(
                group, {"id": group, "split": attempt["split"], "attempts": []}
            )
            if value["split"] != attempt["split"]:
                raise ValueError("Related attempt group cannot cross dataset splits")
            value["attempts"].append(identifier)
    index.check_related_attempts()
    return sorted(groups.values(), key=lambda g: g["id"])


@contextmanager
def build_snapshot(
    config: dict[str, Any], sources: Sequence[dict[str, Any]]
) -> Iterator[tuple[dict[str, Any], _CollectionIndex]]:
    with TemporaryDirectory(prefix="fh5-collection-dataset-") as temporary:
        try:
            with closing(sqlite3.connect(Path(temporary) / "dataset.sqlite3")) as database:
                index = _CollectionIndex(database)
                yield _snapshot(config, sources, index), index
        except sqlite3.Error as error:
            raise OSError("Cannot index collection dataset: " + str(error)) from error


def _snapshot(
    config: dict[str, Any], sources: Sequence[dict[str, Any]], index: _CollectionIndex
) -> dict[str, Any]:
    from fh5.collection.selection import select_source

    _config(config)
    groups = _groups(sources, index)
    seen = set()
    compatibility = None
    details = []
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
        stats = select_source(
            source, session, config["rules"], random.Random(config["seed"] + number), index
        )
        for detail in stats:
            first, last = detail["first_ns"], detail["last_ns"]
            if any(first <= end and begin <= last for begin, end in windows):
                raise ValueError("Overlapping source windows cannot establish independent attempts")
        windows.extend((d["first_ns"], d["last_ns"]) for d in stats)
        details.extend(stats)
    assert compatibility is not None
    return {
        "version": 1,
        "kind": "collection-dataset-snapshot-v1",
        "config": config,
        "sources": sources,
        "groups": groups,
        "samples": index.array("samples"),
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


def _same(first: Any, second: Any) -> bool:
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(_same(first[k], second[k]) for k in first)
    if _sequence(first) and _sequence(second):
        return len(first) == len(second) and all(_same(a, b) for a, b in zip(first, second))
    return bool(first == second)


def review_collection_dataset(request: CollectionDatasetReview) -> RunResult:
    from fh5.result import RunResult

    path, report = request.dataset_file, request.report_path
    digest = sha256_file(path)
    with replay_document(VerifiedFile(path, digest), nested_arrays=SNAPSHOT_ARRAYS) as data:
        if data.get("kind") != "collection-dataset-snapshot-v1" or data.get("version") != 1:
            raise ValueError("Unsupported collection dataset snapshot")
        _report_destination(report, data["sources"])
        with build_snapshot(data["config"], data["sources"]) as (expected, _):
            if not _same(data, expected):
                raise ValueError("Dataset differs from canonical frozen source reconstruction")
        summary = _summary(data, digest)
    write_file(report.with_suffix(".json"), encode(summary))
    report = optional_report(
        report,
        "持续采集数据快照（完整封存不等于优质示范）",
        summary,
        fallback=report.with_suffix(".json"),
        exclusive=True,
    )
    return RunResult({}, [], [], {"collection_dataset": summary}, report)


def _report_destination(report: Path, sources: Sequence[dict[str, Any]]) -> None:
    if report.suffix.lower() != ".html":
        raise ValueError("Dataset report must use an .html path distinct from JSON evidence")
    for path in (report, report.with_suffix(".json")):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        if any(path.resolve().is_relative_to(Path(s["recording"]).resolve()) for s in sources):
            raise ValueError("Dataset outputs cannot be written inside source recordings")
