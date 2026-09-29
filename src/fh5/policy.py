"""Bounded visual policy execution over the external experiment boundary."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from fh5.action_history import action_slots
from fh5.control import Command
from fh5.observations import _preview, actor_fields
from fh5.routes import load_route
from fh5.vision import ColorFrame

if TYPE_CHECKING:
    from fh5.experiment import Packet, RunResult

NEUTRAL = Command(0, 0, 0)


@dataclass(frozen=True)
class PolicyDrive:
    config_file: Path
    output_dir: Path


@dataclass(frozen=True)
class PolicyInput:
    packets: tuple[Packet, ...] = ()
    frame: ColorFrame | None = None
    focused: bool = False
    stop_requested: bool = False
    fault: str | None = None
    events: tuple[dict[str, Any], ...] = ()


class PolicyEnvironment(Protocol):
    source_kind: Literal["udp", "synthetic"]
    events: list[dict[str, Any]]

    def now_ns(self) -> int: ...
    def read(self, period_s: float) -> PolicyInput: ...
    def send(self, command: Command) -> None: ...
    def close(self) -> bool: ...


class PolicyActor(Protocol):
    kind: str
    manifest: dict[str, Any]

    def predict(self, actor: dict[str, Any], images: list[bytes]) -> list[float]: ...


def validate_policy_file(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from fh5.experiment import _validate_config

    root = _validate_config(json.loads(path.read_text(encoding="utf-8-sig")))
    p = root.get("policy")
    if (
        root["control_source"] != "policy"
        or not isinstance(p, dict)
        or type(p.get("version")) is not int
        or p["version"] != 1
    ):
        raise ValueError("Policy execution requires a version 1 policy configuration")
    for key, low, high in (
        ("max_speed_kmh", 1, 15),
        ("start_speed_kmh", 0, 1),
        ("max_steer", 0, 0.2),
        ("max_throttle", 0, 0.25),
        ("max_brake", 0.1, 0.5),
        ("max_duration_s", 0.1, 30),
        ("ready_timeout_s", 1, 300),
        ("inference_timeout_ms", 1, 100),
        ("max_command_age_ms", 10, 150),
        ("braking_s", 0.3, 3),
    ):
        value = p.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError("Invalid bounded policy setting: " + key)
    for key in ("expected_car_ordinal", "expected_pi"):
        if type(p.get(key)) is not int or p[key] < 1:
            raise ValueError("Invalid policy vehicle identity")
    if (
        p.get("reference_mode") not in ("required", "optional", "disabled")
        or p.get("camera_mode") != "chase_far"
        or p.get("navigation") != "visible_no_junction"
    ):
        raise ValueError(
            "Declare reference, chase_far camera and visible navigation without junctions"
        )
    if p.get("device") not in ("cpu", "cuda"):
        raise ValueError("Invalid policy device")
    for key in ("model_dir", "evaluation_route_file"):
        if not isinstance(p.get(key), str) or not p[key]:
            raise ValueError("Missing policy asset: " + key)
    route = load_route(path.parent / p["evaluation_route_file"])
    if not route["low_speed_ready"] or route["length_m"] > 60:
        raise ValueError("Policy requires a verified local route of at most 60 metres")
    for key, maximum in (
        ("end_margin_m", route["length_m"] / 2),
        ("start_tolerance_m", 1),
        ("start_station_m", route["length_m"] - p.get("end_margin_m", 0) - 0.5),
    ):
        value = p.get(key, 0.25 if key == "start_tolerance_m" else 0)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError("Invalid local task bound: " + key)
    expected = p.get("expected_route_sha256")
    if (
        expected
        and hashlib.sha256((path.parent / p["evaluation_route_file"]).read_bytes()).hexdigest()
        != expected
    ):
        raise ValueError("Evaluation route changed")
    return root, route


def _observation(
    settings: dict[str, Any],
    tick: int,
    sample: dict[str, Any],
    frames: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    deliveries = [f["delivered_ns"] for f in frames]
    images: list[dict[str, Any] | None] = []
    seen = set()
    for offset in reversed(settings["history_offsets_ms"]):
        index = bisect_right(deliveries, tick - int(offset * 1e6)) - 1
        if index < 0 or index in seen:
            images.append(None)
            continue
        seen.add(index)
        f = frames[index]
        age = (tick - f["capture_start_ns"]) / 1e6
        images.append({**f, "age_ms": age, "valid": age <= offset + settings["max_image_age_ms"]})
    images.reverse()
    age = (tick - sample["received_monotonic_ns"]) / 1e6
    mask = [bool(f and f["valid"]) for f in images]
    slots = action_slots(
        {"rows": actions, "times": [a["available_ns"] for a in actions]},
        tick,
        [],
        0,
        settings["action_history_offsets_ms"],
        settings["max_action_age_ms"],
    )
    preview = (
        _preview(sample, reference, settings["waypoint_distances_m"])
        if reference
        else {
            "status": "disabled" if settings["reference_mode"] == "disabled" else "absent",
            "waypoints_m": [None] * len(settings["waypoint_distances_m"]),
            "waypoint_mask": [False] * len(settings["waypoint_distances_m"]),
        }
    )
    reasons = []
    if not all(mask):
        reasons.append("incomplete_image_history")
    if age > settings["max_telemetry_age_ms"]:
        reasons.append("stale_telemetry")
    if not sample["motion"]:
        reasons.append("invalid_motion")
    if settings["reference_mode"] == "required" and (
        preview["status"] != "located" or not all(preview["waypoint_mask"])
    ):
        reasons.append("required_reference_unavailable")
    actor = actor_fields(
        {
            "telemetry": sample,
            "telemetry_age_ms": age,
            "images": images,
            "history_mask": mask,
            "route": preview,
            "usable": not reasons,
            "reasons": reasons,
        },
        slots,
    )
    return {
        "decision_ns": tick,
        "telemetry_packet_index": sample["packet_index"],
        "telemetry_received_ns": sample["received_monotonic_ns"],
        "images": images,
        "action_history": slots,
        "route": preview,
        "actor": actor,
        "usable": not reasons,
        "reasons": reasons,
    }


def run_policy(
    request: PolicyDrive, environment: PolicyEnvironment, actor: PolicyActor | None
) -> RunResult:
    from fh5.experiment import Record, run_experiment
    from fh5.policy_runtime import PolicySession

    try:
        root, route = validate_policy_file(request.config_file)
        config = root["policy"]
        if actor is None:
            from fh5.policy_actor import FrozenActor

            actor = FrozenActor(request.config_file)
        if environment.source_kind == "udp":
            from fh5.policy_actor import FrozenActor

            if not isinstance(actor, FrozenActor) or actor.bound_config != root:
                raise ValueError(
                    "Live execution requires the validated frozen actor and unchanged config"
                )
            if not config.get("expected_route_sha256") or config.get("end_margin_m", 0) < 2:
                raise ValueError("Live task requires a bound route hash and braking reserve")
        settings = {
            **actor.manifest["contract"]["observation"],
            "reference_mode": config["reference_mode"],
        }
        reference = None
        reference_error = None
        if config["reference_mode"] != "disabled" and config.get("reference_file"):
            try:
                reference = load_route(request.config_file.parent / config["reference_file"])
            except (OSError, ValueError, TypeError, KeyError) as error:
                if config["reference_mode"] == "required":
                    raise
                reference_error = str(error)
        if config["reference_mode"] == "required" and reference is None:
            raise ValueError("Required reference is missing")
        session = PolicySession(
            request.output_dir, config, route, reference, settings, environment, actor
        )
        session.result.update(
            evaluation_route=route, reference=reference, reference_error=reference_error
        )
        recorded = run_experiment(
            Record(request.config_file, request.output_dir, environment.source_kind),
            packets=session.stream(),
        )
        # Standard Record/Replay attaches the finalized policy, RGB and commands.
        return recorded
    finally:
        environment.close()


def read_policy(directory: Path) -> dict[str, Any]:
    value = json.loads((directory / "policy.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Unsupported policy session")
    expected = {
        "packets.jsonl",
        "commands.jsonl",
        "control.json",
        "vision.jsonl",
        "vision-session.json",
        "policy-decisions.jsonl",
    }
    errors = []
    hashes = value.get("hashes", {})
    if set(hashes) != expected:
        errors.append("Incomplete policy evidence hashes")
    for name in expected:
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(name):
            errors.append("Policy evidence changed: " + name)
    value["artifact_errors"] = errors
    value["formal_validity"] = "pending_independent_review"
    return value
