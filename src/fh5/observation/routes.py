"""Evidence-linked local reference routes and conservative offline localization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

from fh5.artifacts.document import read_document_fields
from fh5.artifacts.io import VerifiedFile, sha256_file

UNLOCATED_ROUTE_STATUSES = frozenset(
    {"ambiguous", "outside_reference", "inactive", "discontinuity"}
)


@dataclass(frozen=True)
class BuildRoute:
    recording_dir: Path
    output_dir: Path
    first_packet: int
    last_packet: int
    spacing_m: float = 2.0
    annotations_file: Path | None = None

    @property
    def report_path(self) -> Path:
        return self.output_dir / "report.html"


@dataclass(frozen=True)
class RouteCheck:
    """Audit a declared local window against an independently frozen route."""

    recording_dir: Path
    report_path: Path
    route_file: Path
    first_packet: int
    last_packet: int
    max_speed_kmh: float = 20.0

    def __post_init__(self) -> None:
        if (
            type(self.first_packet) is not int
            or type(self.last_packet) is not int
            or not 0 <= self.first_packet < self.last_packet
        ):
            raise ValueError("Invalid local check packet range")
        if not math.isfinite(self.max_speed_kmh) or not 0 < self.max_speed_kmh <= 40:
            raise ValueError("Local check speed limit must be above zero and at most 40 km/h")


def check_route_recording(
    request: RouteCheck,
    samples: list[dict[str, Any]],
    route: dict[str, Any],
    metadata: dict[str, Any],
    recording_packet_count: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    recording_sha256 = hashlib.sha256(
        (request.recording_dir / "packets.jsonl").read_bytes()
    ).hexdigest()
    speed = max(s["speed_kmh"] for s in samples)
    progress = max(s["route"]["confirmed_progress_m"] for s in samples)
    checks = {
        "reference_source_reused": recording_sha256 != route["source"]["packets_sha256"],
        "incomplete_recording": metadata["capture_status"] == "completed",
        "missing_packets": len(samples) == request.last_packet - request.first_packet + 1,
        "multiple_segments": len({s["segment"] for s in samples}) == 1,
        "vehicle_changed": len({(s["car_ordinal"], s["car_performance_index"]) for s in samples})
        == 1,
        "route_not_reviewed": route["low_speed_ready"],
        "speed_limit_exceeded": speed <= request.max_speed_kmh,
        # A slanted gate may be crossed after its reference-line station. The
        # locator defers progress while waiting; completing the route below is
        # still required, so an unresolved gate cannot pass this data check.
        "unconfirmed_path": all(
            s["route"]["status"] in ("matched", "awaiting_checkpoint") for s in samples
        ),
        "route_start_missing": samples[0]["route"]["reference_s_m"] <= 0.25,
        "route_end_missing": progress >= route["length_m"] - 1e-6,
    }
    reasons.extend(reason for reason, passed in checks.items() if not passed)
    return {
        "version": 1,
        "passed": not reasons,
        "reasons": reasons,
        "packet_range": [request.first_packet, request.last_packet],
        "recording_packet_count": recording_packet_count,
        "selected_packet_count": request.last_packet - request.first_packet + 1,
        "selected_valid_packets": len(samples),
        "source_kind": metadata["source_kind"],
        "recording_sha256": recording_sha256,
        "session_sha256": hashlib.sha256(
            (request.recording_dir / "session.json").read_bytes()
        ).hexdigest(),
        "route_sha256": hashlib.sha256(request.route_file.read_bytes()).hexdigest(),
        "speed_limit_kmh": request.max_speed_kmh,
        "max_speed_kmh": speed,
        "confirmed_progress_m": progress,
        "formal_validity": "not_evaluated",
        "meaning": "Local telemetry and geometry check only; review images and conditions separately",
    }


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _vector(value: Any, size: int) -> bool:
    return isinstance(value, list) and len(value) == size and all(_number(v) for v in value)


def _validate_notes(notes: dict[str, Any], length: float) -> None:
    try:
        if type(notes["version"]) is not int or notes["version"] != 1:
            raise ValueError("Unsupported annotation version")
        if not isinstance(notes["corridors"], list) or not isinstance(notes["checkpoints"], list):
            raise ValueError("Annotations require corridor and checkpoint lists")
        items = [
            notes["reference_review"],
            notes["checkpoints_review"],
            *notes["corridors"],
            *notes["checkpoints"],
        ]
        for item in items:
            if item["status"] not in ("verified", "unverified"):
                raise ValueError("Unknown annotation status")
            if (
                not isinstance(item["evidence"], list)
                or any(not isinstance(e, str) or not e for e in item["evidence"])
                or (item["status"] == "verified" and not item["evidence"])
            ):
                raise ValueError("Verified annotations require evidence files")
        identities = set()
        last_station = 0.0
        for kind in ("corridors", "checkpoints"):
            for item in notes[kind]:
                if not isinstance(item["id"], str) or not item["id"] or item["id"] in identities:
                    raise ValueError("Annotation identifiers must be unique")
                identities.add(item["id"])
                if (
                    not _number(item["y_min_m"])
                    or not _number(item["y_max_m"])
                    or item["y_min_m"] >= item["y_max_m"]
                ):
                    raise ValueError("Annotation requires a finite height band")
                if kind == "corridors":
                    if (
                        not _number(item["s_start_m"])
                        or not _number(item["s_end_m"])
                        or not 0 <= item["s_start_m"] < item["s_end_m"] <= length
                    ):
                        raise ValueError("Invalid corridor station range")
                    polygon = item["polygon_xz"]
                    if (
                        not isinstance(polygon, list)
                        or len(polygon) < 3
                        or not all(_vector(p, 2) for p in polygon)
                    ):
                        raise ValueError("Invalid corridor polygon")
                    turns = [
                        _cross(polygon[i - 2], polygon[i - 1], p) for i, p in enumerate(polygon)
                    ]
                    if not (all(t > 1e-8 for t in turns) or all(t < -1e-8 for t in turns)):
                        raise ValueError("Corridor cells must be strictly convex")
                    # Every other vertex must lie on the inside of every edge (also rejects stars).
                    direction = 1 if turns[0] > 0 else -1
                    if any(
                        direction * _cross(a, b, p) < -1e-8
                        for a, b in zip(polygon, polygon[1:] + polygon[:1])
                        for p in polygon
                    ):
                        raise ValueError("Corridor polygon must not intersect itself")
                else:
                    if not _number(item["s_m"]) or not last_station < item["s_m"] <= length:
                        raise ValueError("Checkpoint stations must be in route order")
                    last_station = item["s_m"]
                    if (
                        not _vector(item["left_xz"], 2)
                        or not _vector(item["right_xz"], 2)
                        or math.dist(item["left_xz"], item["right_xz"]) < 0.01
                    ):
                        raise ValueError("Checkpoint requires distinct gate endpoints")
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid route annotations") from error


def build_route(request: BuildRoute, samples: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        type(request.first_packet) is not int
        or type(request.last_packet) is not int
        or not 0 <= request.first_packet < request.last_packet
    ):
        raise ValueError("Invalid reference packet range")
    if not math.isfinite(request.spacing_m) or not 0.25 <= request.spacing_m <= 10:
        raise ValueError("Reference spacing must be between 0.25 and 10 metres")
    selected = [
        s for s in samples if request.first_packet <= s["packet_index"] <= request.last_packet
    ]
    if (
        len(selected) != request.last_packet - request.first_packet + 1
        or len({s["segment"] for s in selected}) != 1
        or not all(s["is_race_on"] for s in selected)
        or len({(s["car_ordinal"], s["car_performance_index"]) for s in selected}) != 1
        or any(
            b["game_timestamp_ms"] < a["game_timestamp_ms"] for a, b in zip(selected, selected[1:])
        )
    ):
        raise ValueError("A reference needs one continuous active segment with one vehicle")
    clock_advanced_ns = selected[0]["received_monotonic_ns"]
    for a, b in zip(selected, selected[1:]):
        if b["game_timestamp_ms"] > a["game_timestamp_ms"]:
            clock_advanced_ns = b["received_monotonic_ns"]
        elif b["received_monotonic_ns"] - clock_advanced_ns > 250_000_000:
            raise ValueError("Reference game clock stalled")
    points = [selected[0]]
    for sample in selected[1:-1]:
        if math.dist(points[-1]["position_m"], sample["position_m"]) >= request.spacing_m:
            points.append(sample)
    if math.dist(selected[-1]["position_m"], points[-1]["position_m"]) > 0.001:
        points.append(selected[-1])
    if len(points) < 2:
        raise ValueError("A reference must contain movement")
    station = 0.0
    vertices = []
    for i, sample in enumerate(points):
        if i:
            station += math.dist(points[i - 1]["position_m"], sample["position_m"])
        vertices.append(
            {
                "s_m": station,
                "position_m": sample["position_m"],
                "source_packet": sample["packet_index"],
            }
        )
    reference = {
        "version": 1,
        "points": vertices,
        "review": {"status": "unverified", "evidence": []},
    }
    corridor = {"version": 1, "sections": []}
    checkpoints = {"version": 1, "gates": [], "review": {"status": "unverified", "evidence": []}}
    evidence: dict[str, bytes] = {}
    if request.annotations_file is not None:
        notes = json.loads(request.annotations_file.read_text(encoding="utf-8"))
        _validate_notes(notes, station)
        reference["review"] = notes["reference_review"]
        corridor["sections"] = notes["corridors"]
        checkpoints.update(gates=notes["checkpoints"], review=notes["checkpoints_review"])
        for item in [
            reference["review"],
            checkpoints["review"],
            *notes["corridors"],
            *notes["checkpoints"],
        ]:
            frozen = []
            for name in item["evidence"]:
                data = (request.annotations_file.parent / name).read_bytes()
                target = f"evidence/{hashlib.sha256(data).hexdigest()}{Path(name).suffix}"
                evidence[target] = data
                frozen.append(target)
            item["evidence"] = frozen
    request.output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in evidence.items():
        dest = request.output_dir / name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(content)
    assets = {}
    for name, data in (
        ("reference", reference),
        ("corridor", corridor),
        ("checkpoints", checkpoints),
    ):
        path = request.output_dir / f"{name}.json"
        _write(path, data)
        assets[name] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {
        "version": 1,
        "assets": assets,
        "evidence": [
            {"path": name, "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in evidence.items()
        ],
        "source": {
            "recording": request.recording_dir.name,
            "packets_sha256": hashlib.sha256(
                (request.recording_dir / "packets.jsonl").read_bytes()
            ).hexdigest(),
            "session_sha256": hashlib.sha256(
                (request.recording_dir / "session.json").read_bytes()
            ).hexdigest(),
            "first_packet": request.first_packet,
            "last_packet": request.last_packet,
            "spacing_m": request.spacing_m,
        },
    }
    _write(request.output_dir / "route.json", manifest)
    return load_route(request.output_dir / "route.json")


def load_route(path: Path) -> dict[str, Any]:
    manifest = read_document_fields(
        VerifiedFile(path, sha256_file(path)), {"version", "assets", "evidence", "source"}
    )
    try:
        if type(manifest["version"]) is not int or manifest["version"] != 1:
            raise ValueError("Unsupported route version")
        if set(manifest["assets"]) != {"reference", "corridor", "checkpoints"}:
            raise ValueError("Missing route assets")
        evidence_paths = set()
        for asset in chain(manifest["assets"].values(), manifest["evidence"]):
            file = (path.parent / asset["path"]).resolve()
            if not file.is_relative_to(path.parent.resolve()):
                raise ValueError("Route asset outside bundle")
            if sha256_file(file) != asset["sha256"]:
                raise ValueError("Route asset hash mismatch")
            evidence_paths.add(asset["path"])
        fields = {
            "reference": {"version", "points", "review"},
            "corridor": {"version", "sections"},
            "checkpoints": {"version", "gates", "review"},
        }
        data = {
            name: read_document_fields(
                VerifiedFile(path.parent / asset["path"], asset["sha256"]), fields[name]
            )
            for name, asset in manifest["assets"].items()
        }
        if any(type(d["version"]) is not int or d["version"] != 1 for d in data.values()):
            raise ValueError("Unsupported route asset version")
        points = data["reference"]["points"]
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("Route needs at least two reference points")
        station = 0.0
        for i, point in enumerate(points):
            if not _vector(point["position_m"], 3) or not _number(point["s_m"]):
                raise ValueError("Invalid reference vertex")
            if i:
                gap = math.dist(points[i - 1]["position_m"], point["position_m"])
                if gap <= 0.001:
                    raise ValueError("Reference vertices must be distinct")
                station += gap
            if abs(point["s_m"] - station) > 1e-5:
                raise ValueError("Reference stations disagree with geometry")
        notes = {
            "version": 1,
            "reference_review": data["reference"]["review"],
            "checkpoints_review": data["checkpoints"]["review"],
            "corridors": data["corridor"]["sections"],
            "checkpoints": data["checkpoints"]["gates"],
        }
        _validate_notes(notes, station)
        for item in [
            notes["reference_review"],
            notes["checkpoints_review"],
            *notes["corridors"],
            *notes["checkpoints"],
        ]:
            if any(e not in evidence_paths for e in item["evidence"]):
                raise ValueError("Unbound annotation evidence")
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid route bundle") from error
    sections = data["corridor"]["sections"]
    covered = sum(
        (b["s_m"] - a["s_m"])
        * _covered_fraction(a["position_m"], b["position_m"], a["s_m"], b["s_m"], sections)
        for a, b in zip(points, points[1:])
    )
    gates_fit = True
    for gate in data["checkpoints"]["gates"]:
        fits = False
        for a, b in zip(points, points[1:]):
            if a["s_m"] < gate["s_m"] <= b["s_m"]:
                crossing = _gate_crossing(a["position_m"], b["position_m"], gate)
                fits = (
                    crossing is not None
                    and abs(a["s_m"] + crossing * (b["s_m"] - a["s_m"]) - gate["s_m"]) <= 0.25
                )
                direction = _cross(
                    gate["left_xz"], gate["right_xz"], [b["position_m"][0], b["position_m"][2]]
                ) - _cross(
                    gate["left_xz"], gate["right_xz"], [a["position_m"][0], a["position_m"][2]]
                )
                gate["forward_sign"] = 1 if direction > 0 else -1
                break
        gate["deadline_s_m"] = max(
            min(_projections([p[0], (gate["y_min_m"] + gate["y_max_m"]) / 2, p[1]], points))[1]
            for p in (gate["left_xz"], gate["right_xz"])
        )
        gates_fit = gates_fit and fits
    return {
        "version": 1,
        "source": manifest["source"],
        "points": points,
        "length_m": points[-1]["s_m"],
        "reference_points": len(points),
        "corridors": data["corridor"]["sections"],
        "checkpoints": data["checkpoints"]["gates"],
        "reference_review": data["reference"]["review"],
        "checkpoints_review": data["checkpoints"]["review"],
        "verified_corridor_length_m": covered,
        "checkpoint_geometry_consistent": gates_fit,
        "low_speed_ready": gates_fit
        and covered >= points[-1]["s_m"] - 1e-6
        and data["reference"]["review"]["status"] == "verified"
        and data["checkpoints"]["review"]["status"] == "verified"
        and all(g["status"] == "verified" for g in data["checkpoints"]["gates"]),
    }


def _cross(a: list[float], b: list[float], p: list[float]) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _covered_fraction(
    a: list[float], b: list[float], sa: float, sb: float, sections: list[dict[str, Any]]
) -> float:
    """Clip a movement segment against surveyed convex cells and union its intervals."""
    intervals = []
    for cell in sections:
        if cell["status"] != "verified":
            continue
        polygon = cell["polygon_xz"]
        edges = list(zip(polygon, polygon[1:] + polygon[:1]))
        orientation = 1 if sum(p[0] * q[1] - q[0] * p[1] for p, q in edges) > 0 else -1
        constraints = [
            (sa - cell["s_start_m"], sb - cell["s_start_m"]),
            (cell["s_end_m"] - sa, cell["s_end_m"] - sb),
            (a[1] - cell["y_min_m"], b[1] - cell["y_min_m"]),
            (cell["y_max_m"] - a[1], cell["y_max_m"] - b[1]),
        ]
        constraints += [
            (orientation * _cross(p, q, [a[0], a[2]]), orientation * _cross(p, q, [b[0], b[2]]))
            for p, q in edges
        ]
        lo, hi = 0.0, 1.0
        for u, v in constraints:
            if u < -1e-8 and v < -1e-8:
                hi = -1
                break
            if abs(v - u) > 1e-12:
                crossing = -u / (v - u)
                if u < 0:
                    lo = max(lo, crossing)
                elif v < 0:
                    hi = min(hi, crossing)
        if hi >= lo:
            intervals.append((lo, hi))
    end = total = 0.0
    for lo, hi in sorted(intervals):
        total += max(0, hi - max(end, lo))
        end = max(end, hi)
    return total


def _gate_crossing(a: list[float], b: list[float], gate: dict[str, Any]) -> float | None:
    left, right = gate["left_xz"], gate["right_xz"]
    u = _cross(left, right, [a[0], a[2]])
    v = _cross(left, right, [b[0], b[2]])
    if abs(v - u) < 1e-12:
        return None
    if "forward_sign" in gate and (v - u) * gate["forward_sign"] <= 0:
        return None
    t = -u / (v - u)
    if not 0 <= t <= 1:
        return None
    p = [x + t * (y - x) for x, y in zip(a, b)]
    length2 = sum((y - x) ** 2 for x, y in zip(left, right))
    along = (
        (p[0] - left[0]) * (right[0] - left[0]) + (p[2] - left[1]) * (right[1] - left[1])
    ) / length2
    return t if 0 <= along <= 1 and gate["y_min_m"] <= p[1] <= gate["y_max_m"] else None


def _projections(position: list[float], points: list[dict[str, Any]]) -> list[tuple[float, float]]:
    candidates = []
    for a, b in zip(points, points[1:]):
        delta = [y - x for x, y in zip(a["position_m"], b["position_m"])]
        length2 = sum(v * v for v in delta)
        fraction = max(
            0.0,
            min(
                1.0, sum((p - x) * d for p, x, d in zip(position, a["position_m"], delta)) / length2
            ),
        )
        projected = [x + fraction * d for x, d in zip(a["position_m"], delta)]
        candidates.append(
            (math.dist(position, projected), a["s_m"] + fraction * math.sqrt(length2))
        )
    return candidates


def locate_route(
    samples: list[dict[str, Any]],
    route: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Locate sequential samples; optional state supports the same checks online."""
    points = route["points"]
    events = []
    saved = state or {}
    progress = saved.get("progress", 0.0)
    next_gate = saved.get("next_gate", 0)
    previous = saved.get("previous")
    blocked = saved.get("blocked", False)
    clock_advanced_ns = saved.get("clock_advanced_ns", 0)
    for sample in samples:
        if previous is not None and sample["segment"] != previous["segment"]:
            previous = None
            progress = 0.0
            next_gate = 0
            blocked = False
        if previous is None or sample["game_timestamp_ms"] > previous["game_timestamp_ms"]:
            clock_advanced_ns = sample["received_monotonic_ns"]
        candidates = _projections(sample["position_m"], points)
        temporal_ok = True
        if previous is not None:
            dt = (sample["received_monotonic_ns"] - previous["received_monotonic_ns"]) / 1e9
            game_dt = (sample["game_timestamp_ms"] - previous["game_timestamp_ms"]) / 1000
            bound = 2 + 1.5 * max(sample["speed_mps"], previous["speed_mps"]) * max(
                0, min(dt, game_dt) if game_dt > 0 else dt
            )
            temporal_ok = (
                0 < dt <= 0.5
                and 0 <= game_dt <= 0.5
                and sample["received_monotonic_ns"] - clock_advanced_ns <= 250_000_000
                and math.dist(sample["position_m"], previous["position_m"]) <= bound
            )
            # An unresolved projection is not a trustworthy station prior. Keeping
            # its arbitrary branch would turn repeated ambiguity into false certainty.
            if previous["route"]["status"] not in UNLOCATED_ROUTE_STATUSES:
                candidates = [
                    c for c in candidates if abs(c[1] - previous["route"]["reference_s_m"]) <= bound
                ]
        if candidates:
            distance, station = min(candidates)
        else:
            distance, station = 0.0, previous["route"]["reference_s_m"] if previous else 0.0
            temporal_ok = False
        ambiguous = any(abs(s - station) > 5 and d <= distance + 0.5 for d, s in candidates)
        gates = route["checkpoints"]
        a = previous["position_m"] if previous else sample["position_m"]
        sa = previous["route"]["reference_s_m"] if previous else station
        covered = (
            _covered_fraction(a, sample["position_m"], sa, station, route["corridors"]) >= 1 - 1e-8
        )
        status = "matched"
        if not sample["is_race_on"]:
            status = "inactive"
        elif not temporal_ok:
            status = "discontinuity"
        elif ambiguous:
            status = "ambiguous"
        elif distance > 10:
            status = "outside_reference"
        elif previous is None and (abs(station - saved.get("anchor_s_m", 0)) > 1 or distance > 2):
            status = "unanchored"
        elif not covered:
            status = "unverified_corridor"
        elif route["reference_review"]["status"] != "verified":
            status = "unverified_reference"
        elif (
            route["checkpoints_review"]["status"] != "verified"
            or not route["checkpoint_geometry_consistent"]
        ):
            status = "unverified_checkpoints"
        elif blocked:
            status = "continuity_unverified"
        if status == "matched":
            crossings: list[tuple[float, int]] = []
            if previous is not None:
                for i in range(next_gate, len(gates)):
                    crossing = _gate_crossing(a, sample["position_m"], gates[i])
                    if crossing is not None:
                        crossings.append((crossing, i))
            crossings.sort()
            if any(
                i != next_gate + j or gates[i]["status"] != "verified"
                for j, (_, i) in enumerate(crossings)
            ):
                status = "checkpoint_order"
            else:
                for _, i in crossings:
                    next_gate += 1
                    events.append(
                        {
                            "kind": "route_checkpoint_passed",
                            "checkpoint": gates[i]["id"],
                            "packet_index": sample["packet_index"],
                        }
                    )
                if next_gate < len(gates) and station >= gates[next_gate]["s_m"]:
                    status = (
                        "missing_checkpoint"
                        if station >= gates[next_gate]["deadline_s_m"] + 1e-6
                        else "awaiting_checkpoint"
                    )
        old_progress = progress
        if status == "matched":
            progress = max(progress, station)
        elif status != "awaiting_checkpoint":
            blocked = True
        sample["route"] = {
            "reference_s_m": station,
            "distance_m": distance,
            "confirmed_progress_m": progress,
            "new_progress_m": progress - old_progress,
            "status": status,
            "next_checkpoint": gates[next_gate]["id"] if next_gate < len(gates) else None,
        }
        previous = sample
    if state is not None:
        state.update(
            progress=progress,
            next_gate=next_gate,
            previous=previous,
            blocked=blocked,
            clock_advanced_ns=clock_advanced_ns,
        )
    return events
