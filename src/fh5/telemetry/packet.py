"""FH5 telemetry packets and recording requests, independent of experiment workflows."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

FORMAT_VERSION = 1
DECODER_VERSION = "fh5-dash-324-v2"
DIAGNOSTICS = {
    "version": 1,
    "receive_gap_seconds": 0.5,
    "jump_slack_metres": 20.0,
    "jump_speed_metres_per_second": 200.0,
}
SNAPSHOT_FIELDS = ("vehicle", "variant", "tune", "assists", "event", "environment")


@dataclass(frozen=True)
class Packet:
    received_monotonic_ns: int
    received_utc: str
    payload: bytes


@dataclass(frozen=True)
class Record:
    config_file: Path
    output_dir: Path
    source_kind: Literal["udp", "synthetic"] = "synthetic"


@dataclass(frozen=True)
class Replay:
    recording_dir: Path
    report_path: Path
    route_file: Path | None = None


def decode_packet(packet: Packet) -> dict[str, Any]:
    if len(packet.payload) != 324:
        raise ValueError(f"Unsupported FH5 packet length: {len(packet.payload)} (expected 324)")
    data = packet.payload
    sample = {
        "received_monotonic_ns": packet.received_monotonic_ns,
        "received_utc": packet.received_utc,
        "game_timestamp_ms": struct.unpack_from("<I", data, 4)[0],
        "is_race_on": struct.unpack_from("<i", data, 0)[0],
        "position_m": list(struct.unpack_from("<fff", data, 244)),
        "speed_mps": struct.unpack_from("<f", data, 256)[0],
        "speed_kmh": struct.unpack_from("<f", data, 256)[0] * 3.6,
        "car_ordinal": struct.unpack_from("<i", data, 212)[0],
        "car_class": struct.unpack_from("<i", data, 216)[0],
        "car_performance_index": struct.unpack_from("<i", data, 220)[0],
        "telemetry_controls": {
            "accel": data[315],
            "brake": data[316],
            "steer": struct.unpack_from("<b", data, 320)[0],
        },
        "command": None,
    }
    values = [*sample["position_m"], sample["speed_mps"]]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite position or speed")
    if sample["is_race_on"] not in (0, 1):
        raise ValueError("IsRaceOn must be 0 or 1")
    velocity = list(struct.unpack_from("<fff", data, 32))
    angular = list(struct.unpack_from("<fff", data, 44))
    yaw = struct.unpack_from("<f", data, 56)[0]
    motion_valid = (
        all(math.isfinite(v) for v in [*velocity, *angular, yaw]) and abs(yaw) <= math.pi + 1e-5
    )
    sample["motion"] = (
        {"yaw_rad": yaw, "velocity_car_mps": velocity, "angular_velocity_car_radps": angular}
        if motion_valid
        else None
    )
    sample["motion_status"] = "decoded" if motion_valid else "invalid_motion"
    return sample


def validate_record_config(config: object) -> dict[str, Any]:
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int:
        raise ValueError("Config must be an object with integer schema_version")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported config schema_version")
    if config.get("control_source") not in ("human", "unknown", "calibration", "policy"):
        raise ValueError("Unknown control_source")
    snapshot = config.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("Config requires a snapshot object")
    for name in set(SNAPSHOT_FIELDS) | set(snapshot):
        fact = snapshot.get(name)
        if not isinstance(fact, dict) or fact.get("status") not in (
            "unverified",
            "user_reported",
            "verified",
        ):
            raise ValueError(f"snapshot.{name} requires value and verification status")
        if "value" not in fact or (
            fact["value"] is not None and not isinstance(fact["value"], str)
        ):
            raise ValueError(f"snapshot.{name}.value must be text or null")
        if fact["status"] == "verified" and (
            not fact["value"]
            or not isinstance(fact.get("evidence"), str)
            or not fact["evidence"].strip()
        ):
            raise ValueError(f"snapshot.{name}: verified facts require a value and evidence")
    json.dumps(config, allow_nan=False)
    return config
