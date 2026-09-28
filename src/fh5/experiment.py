"""The experiment-run interface for recording and replaying external telemetry."""

from __future__ import annotations

import json
import math
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fh5.control import Control, ControlEnvironment, read_control, run_control
from fh5.events import EventEnvironment, EventRun, read_event, run_event
from fh5.report import write_report
from fh5.routes import BuildRoute, build_route, load_route, locate_route
from fh5.vision import VisionEnvironment, VisionRecord, read_vision, run_vision

FORMAT_VERSION = 1
DECODER_VERSION = "fh5-dash-324-v1"
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


@dataclass(frozen=True)
class RunResult:
    metadata: dict[str, Any]
    samples: list[dict[str, Any]]
    events: list[dict[str, Any]]
    summary: dict[str, Any]
    report_path: Path


def _decode(packet: Packet) -> dict[str, Any]:
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
    return sample


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _read_packet(line: bytes) -> Packet:
    row = json.loads(line)
    if not isinstance(row, dict):
        raise ValueError("Record must be a JSON object")
    if type(row.get("received_monotonic_ns")) is not int or row["received_monotonic_ns"] < 0:
        raise ValueError("Record requires a nonnegative integer receive time")
    if not isinstance(row.get("received_utc"), str) or not isinstance(row.get("payload_hex"), str):
        raise ValueError("Record requires received_utc and payload_hex strings")
    timestamp = datetime.fromisoformat(row["received_utc"])
    offset = timestamp.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("received_utc must include a UTC offset of zero")
    return Packet(
        row["received_monotonic_ns"], row["received_utc"], bytes.fromhex(row["payload_hex"])
    )


def _validate_config(config: object) -> dict[str, Any]:
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int:
        raise ValueError("Config must be an object with integer schema_version")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported config schema_version")
    if config.get("control_source") not in ("human", "unknown", "calibration"):
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


def run_experiment(
    request: Record | Replay | Control | EventRun | BuildRoute | VisionRecord,
    *,
    packets: Iterable[Packet] | None = None,
    environment: ControlEnvironment | None = None,
    event_environment: EventEnvironment | None = None,
    vision_environment: VisionEnvironment | None = None,
) -> RunResult:
    """Run one record/replay operation; injected packets are the environment seam."""
    if isinstance(request, VisionRecord):
        if vision_environment is None:
            raise ValueError("VisionRecord requires a passive observation environment")
        return run_vision(request, vision_environment)
    if isinstance(request, EventRun):
        if event_environment is None:
            raise ValueError("EventRun requires an external game environment")
        return run_event(request, event_environment)
    if isinstance(request, Control):
        if environment is None:
            raise ValueError("Control requires an external game environment")
        return run_control(request, environment)
    if isinstance(request, Record):
        if packets is None:
            raise ValueError("A record run requires an external packet source")
        config = _validate_config(json.loads(request.config_file.read_text(encoding="utf-8-sig")))
        if request.source_kind not in ("udp", "synthetic"):
            raise ValueError("Unknown packet source_kind")
        metadata = {
            "format_version": FORMAT_VERSION,
            "decoder_version": DECODER_VERSION,
            "diagnostics": DIAGNOSTICS.copy(),
            "created_utc": datetime.now(UTC).isoformat(),
            "source_kind": request.source_kind,
            "game_validation": "unverified",
            "control_source": config["control_source"],
            "snapshot": config["snapshot"],
            "capture_status": "recording",
        }
        request.output_dir.mkdir(parents=True, exist_ok=False)
        if request.source_kind == "udp":
            metadata["capture_start_monotonic_ns"] = time.perf_counter_ns()
        _write_json(request.output_dir / "session.json", metadata)
        try:
            with (request.output_dir / "packets.jsonl").open("w", encoding="utf-8") as target:
                for packet in packets:
                    target.write(
                        json.dumps(
                            {
                                "received_monotonic_ns": packet.received_monotonic_ns,
                                "received_utc": packet.received_utc,
                                "payload_hex": packet.payload.hex(),
                            }
                        )
                        + "\n"
                    )
                    target.flush()
        except KeyboardInterrupt:
            metadata["capture_status"] = "interrupted"
        except OSError as error:
            metadata["capture_status"] = "source_error"
            metadata["capture_error"] = str(error)
        else:
            metadata["capture_status"] = "completed"
        finally:
            close = getattr(packets, "close", None)
            if close is not None:
                close()
        metadata["ended_utc"] = datetime.now(UTC).isoformat()
        if request.source_kind == "udp":
            metadata["capture_end_monotonic_ns"] = time.perf_counter_ns()
        _write_json(request.output_dir / "session.json", metadata)
        directory = request.output_dir
        report_path = directory / "report.html"
    else:
        directory = request.recording_dir
        report_path = request.report_path
        metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("session.json must be an object")
        if (
            type(metadata.get("format_version")) is not int
            or metadata["format_version"] != FORMAT_VERSION
        ):
            raise ValueError("Unsupported session format_version")
        if metadata.get("decoder_version") != DECODER_VERSION:
            raise ValueError("Unsupported session decoder_version")
        if metadata.get("diagnostics") != DIAGNOSTICS:
            raise ValueError("Unsupported session diagnostics")
        _validate_config(
            {
                "schema_version": 1,
                "control_source": metadata.get("control_source"),
                "snapshot": metadata.get("snapshot"),
            }
        )
        if metadata.get("source_kind") not in ("udp", "synthetic"):
            raise ValueError("Unknown session source_kind")
        if metadata.get("capture_status") not in (
            "recording",
            "completed",
            "interrupted",
            "source_error",
        ):
            raise ValueError("Unknown session capture_status")
        if metadata.get("game_validation") != "unverified" or not isinstance(
            metadata.get("created_utc"), str
        ):
            raise ValueError("Session requires created_utc and unverified game_validation")

    samples: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    capture_status = metadata.get("capture_status", "recording")
    if capture_status != "completed":
        events.append(
            {
                "kind": "capture_" + capture_status,
                "detail": metadata.get("capture_error", "Capture did not reach its time limit"),
                "packet_index": None,
            }
        )
    segment = 0
    break_pending = False
    previous_receive: int | None = None
    first_receive: int | None = None
    packet_count = 0
    lines = (directory / "packets.jsonl").read_bytes().splitlines(keepends=True)
    for packet_index, line in enumerate(lines):
        packet_count += 1
        try:
            packet = _read_packet(line)
        except (ValueError, UnicodeError) as error:
            tail = packet_index == len(lines) - 1 and not line.endswith(b"\n")
            events.append(
                {
                    "kind": "incomplete_tail" if tail else "corrupt_record",
                    "packet_index": packet_index,
                    "detail": str(error),
                }
            )
            break_pending = True
            continue
        reasons: list[tuple[str, str]] = []
        if first_receive is None:
            first_receive = packet.received_monotonic_ns
        if previous_receive is not None:
            receive_delta = (packet.received_monotonic_ns - previous_receive) / 1e9
            if receive_delta > DIAGNOSTICS["receive_gap_seconds"]:
                reasons.append(("receive_gap", f"No datagram for {receive_delta:.3f} seconds"))
            elif receive_delta <= 0:
                reasons.append(("receive_clock_discontinuity", "Receive clock did not advance"))
        previous_receive = packet.received_monotonic_ns
        try:
            sample = _decode(packet)
        except ValueError as error:
            kind = "unsupported_packet" if len(packet.payload) != 324 else "invalid_value"
            reasons.append((kind, str(error)))
            sample = None
        if sample is not None and samples:
            previous = samples[-1]
            game_delta = sample["game_timestamp_ms"] - previous["game_timestamp_ms"]
            if game_delta < 0:
                wrapped = (
                    previous["game_timestamp_ms"] > 0xFFFF0000
                    and sample["game_timestamp_ms"] < 0x10000
                )
                if wrapped:
                    reasons.append(("game_clock_wrap", "Game uint32 timer crossed its boundary"))
                else:
                    reasons.append(("game_time_discontinuity", "Game time moved backwards"))
            if sample["is_race_on"] != previous["is_race_on"]:
                kind = "resumed" if sample["is_race_on"] else "paused"
                reasons.append((kind, "IsRaceOn changed; this is not a finish or rewind signal"))
            dt = (packet.received_monotonic_ns - previous["received_monotonic_ns"]) / 1e9
            displacement = math.dist(sample["position_m"], previous["position_m"])
            if (
                sample["is_race_on"]
                and previous["is_race_on"]
                and displacement
                > DIAGNOSTICS["jump_slack_metres"]
                + DIAGNOSTICS["jump_speed_metres_per_second"] * max(0.0, dt)
            ):
                reasons.append(("position_jump", f"Position changed by {displacement:.1f} metres"))
        for kind, detail in reasons:
            events.append(
                {
                    "kind": kind,
                    "detail": detail,
                    "packet_index": packet_index,
                    "received_monotonic_ns": packet.received_monotonic_ns,
                }
            )
        if sample is None:
            break_pending = True
            continue
        if samples and (break_pending or reasons):
            segment += 1
        sample["segment"] = segment
        sample["packet_index"] = packet_index
        samples.append(sample)
        break_pending = False
    for boundary, clock, received in (
        ("start", metadata.get("capture_start_monotonic_ns"), first_receive),
        ("end", metadata.get("capture_end_monotonic_ns"), previous_receive),
    ):
        if type(clock) is not int or received is None:
            continue
        duration = (received - clock if boundary == "start" else clock - received) / 1e9
        if duration > DIAGNOSTICS["receive_gap_seconds"]:
            events.append(
                {
                    "kind": "receive_gap",
                    "packet_index": None,
                    "boundary": boundary,
                    "duration_seconds": duration,
                    "detail": f"No datagram for {duration:.3f} seconds at capture {boundary}",
                }
            )
    if not samples:
        events.append(
            {
                "kind": "no_telemetry",
                "packet_index": None,
                "detail": "No supported, valid telemetry packets were received",
            }
        )
    receive_span = sum(
        max(0, b["received_monotonic_ns"] - a["received_monotonic_ns"]) / 1e9
        for a, b in zip(samples, samples[1:])
    )
    summary = {
        "capture_status": capture_status,
        "receive_span_seconds": receive_span,
        "packet_count": packet_count,
        "valid_packets": len(samples),
        "active_packets": sum(s["is_race_on"] for s in samples),
        "invalid_packets": packet_count - len(samples),
        "segments": segment + bool(samples),
    }
    if metadata.get("control_source") == "calibration" or (directory / "control.json").exists():
        summary["control"] = read_control(directory, samples)
    if any(
        (directory / name).exists()
        for name in ("event-run.json", "event-config.json", "event-journal.jsonl")
    ):
        event_run = read_event(directory)
        summary["event_run"] = event_run["summary"]
        events.extend(event_run["events"])
    if isinstance(request, BuildRoute):
        route = build_route(request, samples)
        samples = [
            s for s in samples if request.first_packet <= s["packet_index"] <= request.last_packet
        ]
        events = [
            e
            for e in events
            if e.get("packet_index") is not None
            and request.first_packet <= e["packet_index"] <= request.last_packet
        ]
        metadata["analysis_packet_range"] = [request.first_packet, request.last_packet]
        summary.update(
            packet_count=len(samples),
            valid_packets=len(samples),
            active_packets=sum(s["is_race_on"] for s in samples),
            invalid_packets=0,
            segments=len({s["segment"] for s in samples}),
            receive_span_seconds=(
                samples[-1]["received_monotonic_ns"] - samples[0]["received_monotonic_ns"]
            )
            / 1e9,
        )
        events.extend(locate_route(samples, route))
        summary["route"] = route
    elif isinstance(request, Replay) and request.route_file is not None:
        route = load_route(request.route_file)
        events.extend(locate_route(samples, route))
        summary["route"] = route
    if (directory / "vision-session.json").exists():
        summary["vision"] = read_vision(directory, report_path, samples, events)
    write_report(
        report_path,
        {"metadata": metadata, "samples": samples, "events": events, "summary": summary},
    )
    return RunResult(metadata, samples, events, summary, report_path)
