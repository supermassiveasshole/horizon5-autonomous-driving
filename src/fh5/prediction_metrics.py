"""Exact temporal prediction metrics with growing errors and groups stored on disk."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fh5.bc_learning import VIEWS

_SPLITS = ("train", "development", "evaluation")
_VARIANTS = ("nominal", "reselected")
_ASSESSMENT_STRATA = ("startup", "left", "right", "release_rt", "coast", "brake")
_ASSESSMENT_PREDICTIONS = (
    ("candidate", "prediction"),
    ("baseline", "baseline_prediction"),
    ("copy_recent_action", "previous_action"),
)


def _initialize(index: sqlite3.Connection) -> None:
    # SQLite's transient sorting/grouping work also belongs on disk.
    index.execute("PRAGMA temp_store = FILE")
    index.execute(
        "CREATE TABLE errors (bucket TEXT, position INTEGER, group_id TEXT NOT NULL, "
        "error0 REAL NOT NULL, error1 REAL NOT NULL, "
        "PRIMARY KEY (bucket, position)) WITHOUT ROWID"
    )
    for field in ("error0", "error1", "group_id"):
        index.execute(f"CREATE INDEX by_{field} ON errors (bucket, {field})")


def _values(index: sqlite3.Connection, bucket: str, axis: int) -> Iterator[float]:
    # The axis is one of the two fixed action coordinates, never input SQL.
    with closing(
        index.execute(
            f"SELECT error{axis} FROM errors WHERE bucket = ? ORDER BY position", (bucket,)
        )
    ) as rows:
        for (value,) in rows:
            yield value


def _score(index: sqlite3.Connection, bucket: str) -> dict[str, Any]:
    count = int(
        index.execute("SELECT COUNT(*) FROM errors WHERE bucket = ?", (bucket,)).fetchone()[0]
    )
    if not count:
        return {"count": 0, "mae": None, "rmse": None, "p90_absolute_error": None}
    # Use Python's source-order sum, including its float summation semantics.
    # SQL SUM or a hand-written accumulator can change the published numbers.
    return {
        "count": count,
        "mae": [sum(_values(index, bucket, axis)) / count for axis in range(2)],
        "rmse": [
            math.sqrt(sum(value**2 for value in _values(index, bucket, axis)) / count)
            for axis in range(2)
        ],
        "p90_absolute_error": [
            index.execute(
                f"SELECT error{axis} FROM errors WHERE bucket = ? "
                f"ORDER BY error{axis} LIMIT 1 OFFSET ?",
                (bucket, math.ceil(0.9 * count) - 1),
            ).fetchone()[0]
            for axis in range(2)
        ],
    }


def _groups(index: sqlite3.Connection, bucket: str) -> int:
    return int(
        index.execute(
            "SELECT COUNT(DISTINCT group_id) FROM errors WHERE bucket = ?", (bucket,)
        ).fetchone()[0]
    )


def _store(
    index: sqlite3.Connection,
    bucket: str,
    position: int,
    group: str,
    errors: tuple[float, float],
) -> None:
    index.execute("INSERT INTO errors VALUES (?, ?, ?, ?, ?)", (bucket, position, group, *errors))


def _errors(prediction: list[float], target: list[float]) -> tuple[float, float]:
    return abs(prediction[0] - target[0]), abs(prediction[1] - target[1])


def temporal_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Consume records once, preserving the temporal BC metric values and layout.

    Only fixed metric names remain in Python containers. Sample errors and
    attempt-group membership grow in a temporary SQLite database; its indexes
    supply exact nearest-rank percentiles rather than an approximation.
    """
    try:
        with (
            TemporaryDirectory(prefix="fh5-prediction-metrics-") as temporary,
            closing(sqlite3.connect(Path(temporary) / "metrics.sqlite3")) as index,
        ):
            _initialize(index)
            # Each family has at most the seven named strata. Preserve their
            # first-observed order as in the original report dictionaries.
            strata: dict[str, dict[str, None]] = {}
            for position, row in enumerate(records):
                if row["split"] not in _SPLITS or row["view"] not in VIEWS or not row["scored"]:
                    continue
                variant = "nominal" if row["variant"] == 0 else "reselected"
                family = f"{row['split']}:{row['view']}:{variant}"
                group = row["group"]
                target = row["target"]
                errors = _errors(row["prediction"], target)
                _store(index, family, position, group, errors)
                if row["previous_action"] is not None:
                    _store(
                        index,
                        family + ":copy_recent_action",
                        position,
                        group,
                        _errors(row["previous_action"], target),
                    )
                _store(
                    index,
                    family + ":without_action_history",
                    position,
                    group,
                    _errors(row["without_action_history_prediction"], target),
                )
                speed = row["actor"]["ego"]["speed_mps"] * 3.6
                steer, longitudinal = target
                present = strata.setdefault(family, {})
                for name, applies in (
                    ("stationary", speed < 1),
                    ("startup", speed < 15 and longitudinal > 0.05),
                    ("left", steer < -0.2),
                    ("right", steer > 0.2),
                    ("throttle", longitudinal > 0.05),
                    ("brake", longitudinal < -0.05),
                    ("coast", abs(longitudinal) <= 0.05),
                ):
                    if applies:
                        present[name] = None
                        _store(index, family + ":" + name, position, group, errors)
            index.commit()
            metrics: dict[str, Any] = {}
            for split in _SPLITS:
                metrics[split] = {}
                for view in VIEWS:
                    variants: dict[str, Any] = {}
                    for variant in _VARIANTS:
                        family = f"{split}:{view}:{variant}"
                        variants[variant] = {
                            **_score(index, family),
                            "attempt_groups": _groups(index, family),
                            "strata": {
                                name: {
                                    **_score(index, family + ":" + name),
                                    "attempt_groups": _groups(index, family + ":" + name),
                                }
                                for name in strata.get(family, {})
                            },
                            "copy_recent_action": _score(index, family + ":copy_recent_action"),
                            "without_action_history": _score(
                                index, family + ":without_action_history"
                            ),
                        }
                    metrics[split][view] = variants
            return metrics
    except sqlite3.Error as error:
        raise OSError("Cannot compute temporal prediction metrics: " + str(error)) from error


def assessment_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Consume assessment records once and retain every predefined empty stratum."""
    try:
        with (
            TemporaryDirectory(prefix="fh5-assessment-metrics-") as temporary,
            closing(sqlite3.connect(Path(temporary) / "metrics.sqlite3")) as index,
        ):
            _initialize(index)
            index.execute(
                "CREATE TABLE members (bucket TEXT, group_id TEXT, "
                "PRIMARY KEY (bucket, group_id)) WITHOUT ROWID"
            )
            for position, row in enumerate(records):
                if row["view"] not in VIEWS or not row["scored"]:
                    continue
                steer, pedal = row["target"]
                prior = row["previous_action"]
                buckets = [row["view"]]
                for name, applies in (
                    ("startup", row["actor"]["ego"]["speed_mps"] * 3.6 < 15 and pedal > 0.05),
                    ("left", steer < -0.2),
                    ("right", steer > 0.2),
                    ("release_rt", prior is not None and prior[1] > 0 and pedal <= 0),
                    ("coast", abs(pedal) <= 0.05),
                    ("brake", pedal < -0.05),
                ):
                    if applies:
                        buckets.append(row["view"] + ":" + name)
                for bucket in buckets:
                    # Group evidence exists even when a comparator is absent.
                    index.execute(
                        "INSERT OR IGNORE INTO members VALUES (?, ?)", (bucket, row["group"])
                    )
                    for name, field in _ASSESSMENT_PREDICTIONS:
                        if row[field] is not None:
                            _store(
                                index,
                                bucket + ":" + name,
                                position,
                                row["group"],
                                _errors(row[field], row["target"]),
                            )
            index.commit()

            def scores(bucket: str) -> dict[str, Any]:
                return {
                    **{
                        name: _score(index, bucket + ":" + name)
                        for name, _ in _ASSESSMENT_PREDICTIONS
                    },
                    "independent_groups": int(
                        index.execute(
                            "SELECT COUNT(*) FROM members WHERE bucket = ?", (bucket,)
                        ).fetchone()[0]
                    ),
                }

            return {
                view: {
                    **scores(view),
                    "strata": {name: scores(view + ":" + name) for name in _ASSESSMENT_STRATA},
                }
                for view in VIEWS
            }
    except sqlite3.Error as error:
        raise OSError("Cannot compute assessment prediction metrics: " + str(error)) from error
