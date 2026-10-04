"""Reproducible synthetic counterexamples through the same recording/reward seam."""

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.evaluation.rewards import RewardReplay
from fh5.observation.routes import BuildRoute
from fh5.reporting.telemetry import write_report

if TYPE_CHECKING:
    from fh5.result import RunResult


@dataclass(frozen=True)
class RewardAudit:
    reward_file: Path
    output_dir: Path


def _save(path: Path, value: Any) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def _record(root: Path, name: str, positions: list[tuple[float, float]], speed: float = 4) -> Path:
    from fh5.reporting.recording import run_recording_report
    from fh5.telemetry.packet import Packet, Record

    config = _save(
        root / (name + "-record.json"),
        {
            "schema_version": 1,
            "control_source": "human",
            "snapshot": {
                k: {"value": None, "status": "unverified"}
                for k in ("vehicle", "variant", "tune", "assists", "event", "environment")
            },
        },
    )
    packets = []
    for i, (x, z) in enumerate(positions):
        raw = bytearray(324)
        struct.pack_into("<iI", raw, 0, 1, 1000 + i * 100)
        struct.pack_into("<iii", raw, 212, 2941, 6, 999)
        struct.pack_into("<ffff", raw, 244, x, 2, z, speed)
        packets.append(
            Packet(1_000_000_000 + i * 100_000_000, "2026-09-30T00:00:00+00:00", bytes(raw))
        )
    source = root / name
    run_recording_report(Record(config, source), packets=packets)
    return source


def audit_rewards(request: RewardAudit) -> "RunResult":
    from fh5.evaluation.rewards import settle_rewards
    from fh5.reporting.recording import run_recording_report
    from fh5.result import RunResult

    root = request.output_dir
    root.mkdir(parents=True, exist_ok=False)
    reference = _record(root, "reference", [(0, 0), (1, 0), (2, 0), (3, 0)])
    proof = root / "fixture-evidence.md"
    proof.write_text(
        "Synthetic authored geometry/events. No real-game recognition or driving evidence.\n",
        encoding="utf-8",
    )
    verified = {"status": "verified", "evidence": [proof.name]}
    geometry = {
        "version": 1,
        "reference_review": verified,
        "checkpoints_review": verified,
        "checkpoints": [],
        "corridors": [
            {
                "id": "straight",
                "s_start_m": 0,
                "s_end_m": 3,
                "polygon_xz": [[-1, -1], [4, -1], [4, 1], [-1, 1]],
                "y_min_m": 1,
                "y_max_m": 3,
                **verified,
            }
        ],
    }
    annotations = _save(root / "geometry.json", geometry)
    run_recording_report(BuildRoute(reference, root / "route", 0, 3, annotations_file=annotations))
    geometry["checkpoints"] = [
        {
            "id": "gate",
            "s_m": 1.5,
            "left_xz": [1.5, -0.1],
            "right_xz": [1.5, 0.1],
            "y_min_m": 1,
            "y_max_m": 3,
            **verified,
        }
    ]
    gated = _save(root / "gated-geometry.json", geometry)
    run_recording_report(BuildRoute(reference, root / "gated-route", 0, 3, annotations_file=gated))
    normal = [(0.0, 0.2), (1.0, 0.2), (2.0, 0.2), (3.0, 0.2)]
    cases: dict[str, tuple[list[tuple[float, float]], str | None, int, str]] = {
        "complete": (normal, None, 0, "success"),
        "early_crash": (normal[:1], "driving_failure", 0, "failure"),
        "late_abandon": ([(0, 0.2), (1, 0.2), (2.8, 0.2), (2.8, 0.2)], "stop", 3, "failure"),
        "park": ([(0, 0.2)] * 8, None, 0, "failure"),
        "backtrack": (
            [(0, 0.2), (1, 0.2), (0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)],
            None,
            0,
            "success",
        ),
        "wrong_exit": ([(0, 0.2), (1, 2), (3, 0.2)], None, 0, "quarantined"),
        "terrain_shortcut": ([(0, 0.2), (1, 2), (3, 0.2)], "grass_shortcut", 1, "failure"),
        "route_jump": ([(0, 0.2), (3, 0.2)], None, 0, "quarantined"),
        "missed_checkpoint": (normal, None, 0, "failure"),
        "navigation_recomputed": (
            [(0, 0.2), (1, 0.2), (0, 0.2), (1, 0.2)],
            "navigation_recomputed",
            2,
            "truncated",
        ),
        "navigation_hidden": ([(0, 0.2)] * 3, "navigation_hidden", 1, "truncated"),
        "destination_changed": (
            [(0, 0.2), (1, 0.2), (0, 0.2), (1, 0.2)],
            "destination_changed",
            2,
            "truncated",
        ),
    }
    rows = []
    base = None
    for name, (positions, kind, event_index, expected) in cases.items():
        source = _record(root, name, positions, speed=0 if name == "route_jump" else 4)
        route = root / ("gated-route" if name == "missed_checkpoint" else "route") / "route.json"
        task = _save(
            root / (name + "-task.json"),
            {
                "version": 1,
                "task_id": "counterexample-" + name,
                "scope": "local",
                "route_file": str(route.resolve()),
                "route_sha256": hashlib.sha256(route.read_bytes()).hexdigest(),
                "geometry_source_sha256": [],
                "start_mode": "manual_placement",
                "control_owner": "human",
                "expected_car_ordinal": 2941,
                "expected_pi": 999,
                "max_speed_kmh": 20,
                "max_duration_s": 1,
                "no_progress_timeout_s": 0.5,
            },
        )
        provenance = {
            "source": "independent_review",
            "reviewer": "synthetic fixture author",
            "evidence": ["fixture"],
        }
        evidence = _save(
            root / (name + "-evidence.json"),
            {
                "version": 1,
                "recording_sha256": hashlib.sha256(
                    (source / "packets.jsonl").read_bytes()
                ).hexdigest(),
                "items": [
                    {
                        "id": "fixture",
                        "path": proof.name,
                        "sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
                    }
                ],
                "coverage": [
                    {
                        "packet_range": [0, len(positions) - 1],
                        "checks": [
                            "wall_riding",
                            "reset_boost",
                            "grass_shortcut",
                            "interventions",
                            "conditions",
                        ],
                        **provenance,
                    }
                ],
                "events": [
                    {
                        "packet_index": event_index,
                        "kind": kind,
                        "status": "confirmed",
                        "destination_distance_before_m": 1000,
                        "destination_distance_after_m": 0,
                        **provenance,
                    }
                ]
                if kind
                else [],
            },
        )
        result = settle_rewards(
            RewardReplay(source, root / (name + "-settlement"), task, request.reward_file, evidence)
        )
        if base is None:
            base = result
        segments = result.summary["rewards"]["segments"]
        first = segments[0]
        progress = sum(
            step["progress_delta_m"] for segment in segments for step in segment["steps"]
        )
        passed = first["outcome"] == expected
        if name in {"wrong_exit", "terrain_shortcut", "route_jump", "navigation_hidden", "park"}:
            passed = passed and progress == 0
        if name in {"navigation_recomputed", "destination_changed"}:
            passed = passed and progress == 1
        if name == "backtrack":
            passed = passed and progress == 3
        rows.append(
            {
                "case": name,
                "expected": expected,
                "outcome": first["outcome"],
                "return": first["discounted_return"],
                "progress_m": progress,
                "passed": passed,
                "report": result.report_path.relative_to(root).as_posix(),
            }
        )
    assert base is not None
    returns = {row["case"]: row["return"] for row in rows}
    ordering = all(
        returns["complete"] > returns[n]
        for n in ("early_crash", "late_abandon", "park", "backtrack")
    )
    audit = {
        "version": 1,
        "source_kind": "synthetic",
        "cases": rows,
        "complete_return_ordering_passed": ordering,
        "passed": ordering and all(row["passed"] for row in rows),
        "real_recognition_verified": False,
    }
    _save(root / "audit.json", audit)
    summary = {**base.summary, "reward_audit": audit}
    report = root / "report.html"
    write_report(
        report,
        {
            "metadata": base.metadata,
            "samples": base.samples,
            "events": base.events,
            "summary": summary,
        },
    )
    return RunResult(base.metadata, base.samples, base.events, summary, report)
