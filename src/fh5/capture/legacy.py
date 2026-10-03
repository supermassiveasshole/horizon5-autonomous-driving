"""Passive color observations, receipt-causal pairing, and offline evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol
from urllib.parse import quote

from fh5.artifacts.io import source_hashes as package_source_hashes

if TYPE_CHECKING:
    from fh5.result import RunResult
    from fh5.telemetry.packet import Packet


@dataclass(frozen=True)
class ColorFrame:
    capture_start_ns: int
    capture_end_ns: int
    available_ns: int
    encoded: bytes
    codec: Literal["png", "jpeg"]
    size: tuple[int, int]
    client_size: tuple[int, int]
    captured_utc: str | None = None
    color: Literal["RGB"] = "RGB"
    resize_method: str = "unspecified"
    jpeg_quality: int | None = None
    chroma_subsampling: int | None = None


@dataclass(frozen=True)
class VisionInput:
    packets: tuple[Packet, ...] = ()
    frame: ColorFrame | None = None
    events: tuple[dict[str, Any], ...] = ()
    stop_requested: bool = False
    fault: str | None = None


class VisionEnvironment(Protocol):
    source_kind: Literal["udp", "synthetic"]

    def now_ns(self) -> int: ...
    def read(self, period_s: float) -> VisionInput: ...
    def close(self) -> bool: ...


@dataclass(frozen=True)
class VisionRecord:
    config_file: Path
    output_dir: Path
    seconds: float = 60.0
    period_s: float = 0.1
    max_age_ms: float = 100.0
    max_bytes: int = 256 * 1024**2
    observation_config: Path | None = None
    route_file: Path | None = None

    def __post_init__(self) -> None:
        if self.observation_config is None and self.route_file is not None:
            raise ValueError("Observation capture requires a config for the reference")
        for name, value, low, high in (
            ("seconds", self.seconds, 0.1, 600),
            ("period_s", self.period_s, 0.05, 5),
            ("max_age_ms", self.max_age_ms, 1, 5000),
            ("max_bytes", self.max_bytes, 4096, 1024**3),
        ):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and between {low} and {high}")


def _hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run_vision(request: VisionRecord, environment: VisionEnvironment) -> RunResult:
    # Compose the captured images with telemetry through the experiment interface.
    from fh5.experiment import run_experiment
    from fh5.observation.multimodal import ObservationReplay, freeze_inputs
    from fh5.telemetry.packet import Record
    from fh5.telemetry.recording import write_json as _write_json

    directory = request.output_dir

    def observations() -> Iterator[Packet]:
        (directory / "frames").mkdir()
        for name, contents in frozen.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        (directory / "vision-config.json").write_bytes(request.config_file.read_bytes())
        session: dict[str, Any] = {
            "version": 1,
            "camera_mode": "chase_far",
            "camera_status": "user_reported",
            "camera_pose": "dynamic_unknown",
            "settings": {
                k: v
                for k, v in asdict(request).items()
                if k not in ("config_file", "output_dir", "observation_config", "route_file")
            },
            "source_hashes": package_source_hashes(),
            "stop_reason": "recording",
        }
        _write_json(directory / "vision-session.json", session)
        started = environment.now_ns()
        count = 0
        written = 0
        last_check = started - 1_000_000_000
        observation_period = (
            json.loads(frozen["observation-config.json"])["period_ms"] * 1e6 if frozen else 0
        )
        try:
            with (directory / "vision.jsonl").open("x", encoding="utf-8") as journal:
                while environment.now_ns() - started < request.seconds * 1e9:
                    if (directory / "STOP").exists():
                        session["stop_reason"] = "stop_file"
                        break
                    batch = environment.read(min(0.02, request.period_s))
                    for packet in batch.packets:
                        # Conservative allowance for raw JSONL envelope plus hex payload.
                        cost = len(packet.payload) * 2 + len(packet.received_utc) + 256
                        if written + cost > request.max_bytes:
                            session["stop_reason"] = "byte_limit"
                            break
                        written += cost
                        yield packet
                    if session["stop_reason"] == "byte_limit":
                        break
                    for event in batch.events:
                        line = json.dumps(event, allow_nan=False) + "\n"
                        if written + len(line.encode("utf-8")) > request.max_bytes:
                            session["stop_reason"] = "byte_limit"
                            break
                        written += len(line.encode("utf-8"))
                        journal.write(line)
                    if session["stop_reason"] == "byte_limit":
                        break
                    if batch.stop_requested or batch.fault:
                        session["stop_reason"] = batch.fault or "user_stop"
                        break
                    if batch.frame is not None:
                        frame = batch.frame
                        delivered = environment.now_ns()
                        if written + len(frame.encoded) + 2048 > request.max_bytes:
                            session["stop_reason"] = "byte_limit"
                            break
                        written += len(frame.encoded) + 2048
                        relative = f"frames/{count:06d}.{frame.codec}"
                        path = directory / relative
                        path.write_bytes(frame.encoded)
                        row = {k: v for k, v in asdict(frame).items() if k != "encoded"}
                        row.update(
                            kind="frame",
                            path=relative,
                            sha256=_hash(path),
                            stored_ns=environment.now_ns(),
                            delivered_ns=delivered,
                        )
                        journal.write(json.dumps(row, allow_nan=False) + "\n")
                        count += 1
                    if frozen and environment.now_ns() - last_check >= observation_period:
                        last_check = environment.now_ns()
                        tick_line = (
                            json.dumps({"kind": "observation_tick", "observed_ns": last_check})
                            + "\n"
                        )
                        if written + len(tick_line) > request.max_bytes:
                            session["stop_reason"] = "byte_limit"
                            break
                        written += len(tick_line)
                        journal.write(tick_line)
                    journal.flush()
                else:
                    session["stop_reason"] = "time_limit"
        except (KeyboardInterrupt, GeneratorExit):
            session["stop_reason"] = "interrupted"
            raise
        except OSError as error:
            session.update(stop_reason="source_error", error=str(error))
            raise
        finally:
            session["resources_released"] = environment.close()
            session["budget_bytes_used"] = written
            session["started_ns"] = started
            session["ended_ns"] = environment.now_ns()
            session["hashes"] = {
                name: _hash(directory / name)
                for name in ("vision-config.json", "vision.jsonl", "packets.jsonl")
            }
            _write_json(directory / "vision-session.json", session)

    try:
        frozen = freeze_inputs(request.observation_config, request.route_file)
        result = run_experiment(
            Record(request.config_file, directory, environment.source_kind), packets=observations()
        )
        if frozen:
            return run_experiment(
                ObservationReplay(
                    directory,
                    directory / "observations.html",
                    directory / "observation-route/route.json"
                    if "observation-route/route.json" in frozen
                    else None,
                    directory / "observation-config.json",
                )
            )
        return result
    finally:
        environment.close()


def read_vision(
    directory: Path,
    report_path: Path,
    samples: list[dict[str, Any]],
    telemetry_events: list[dict[str, Any]],
) -> dict[str, Any]:
    session = json.loads((directory / "vision-session.json").read_text(encoding="utf-8"))
    if session.get("version") != 1:
        raise ValueError("Unsupported vision session version")
    errors = []
    for name, digest in session.get("hashes", {}).items():
        if name not in ("vision-config.json", "vision.jsonl", "packets.jsonl"):
            errors.append("Unexpected hashed artifact")
        elif not (directory / name).is_file() or _hash(directory / name) != digest:
            errors.append(f"Hash mismatch or missing file: {name}")
    rows = []
    journal = directory / "vision.jsonl"
    if not journal.is_file():
        errors.append("Missing vision journal")
    else:
        for number, line in enumerate(journal.read_bytes().splitlines()):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("kind"), str):
                    raise ValueError("Expected an observation object")
                if row["kind"] == "frame":
                    clocks: list[Any] = [
                        row.get(k)
                        for k in ("capture_start_ns", "capture_end_ns", "available_ns", "stored_ns")
                    ]
                    if not all(type(c) is int and c >= 0 for c in clocks) or clocks != sorted(
                        clocks
                    ):
                        raise ValueError("Invalid frame timestamps")
                    delivered = row.get("delivered_ns", row["available_ns"])
                    if (
                        type(delivered) is not int
                        or not row["available_ns"] <= delivered <= row["stored_ns"]
                    ):
                        raise ValueError("Invalid image handoff timestamp")
                    if not isinstance(row.get("path"), str) or not isinstance(
                        row.get("sha256"), str
                    ):
                        raise ValueError("Frame requires a path and hash")
                rows.append(row)
            except (ValueError, UnicodeError) as error:
                errors.append(f"Invalid vision journal row {number}: {error}")
    if not session.get("hashes"):
        errors.append("Recording did not finalize its artifact hashes")
    ordered = sorted(samples, key=lambda s: s["received_monotonic_ns"])
    times = [s["received_monotonic_ns"] for s in ordered]
    boundaries = [
        e["received_monotonic_ns"]
        for e in telemetry_events
        if type(e.get("received_monotonic_ns")) is int
    ] + [
        r["observed_ns"]
        for r in rows
        if r.get("kind")
        in ("focus_lost", "focus_restored", "capture_discarded", "telemetry_overflow")
        and type(r.get("observed_ns")) is int
    ]
    boundaries.sort()
    max_age = session["settings"]["max_age_ms"]
    frames = []
    for row in rows:
        if row.get("kind") != "frame":
            continue
        path = (directory / row["path"]).resolve()
        if not path.is_relative_to((directory / "frames").resolve()):
            raise ValueError("Frame path must stay within the recording")
        if not path.is_file() or _hash(path) != row["sha256"]:
            errors.append(f"Hash mismatch or missing frame: {row['path']}")
            row["image_url"] = None
        else:
            try:
                relative = os.path.relpath(path, report_path.parent).replace("\\", "/")
            except ValueError:  # Windows reports may live on another drive.
                row["image_url"] = path.as_uri()
            else:
                row["image_url"] = quote(relative, safe="/")
        available = row["available_ns"]
        index = bisect_right(times, available) - 1
        online = ordered[index] if index >= 0 else None
        before_index = bisect_right(times, row["capture_start_ns"]) - 1
        before = ordered[before_index] if before_index >= 0 else None
        reason = None
        if online is None or before is None:
            reason = "missing_telemetry"
        elif bisect_right(boundaries, available) > bisect_right(boundaries, times[before_index]):
            reason = "observation_discontinuity"
        elif before["segment"] != online["segment"]:
            reason = "telemetry_segment_change"
        elif not before["is_race_on"] or not online["is_race_on"]:
            reason = "inactive_telemetry"
        elif (
            max(
                available - times[index],
                row["capture_start_ns"] - times[before_index],
                available - row["capture_start_ns"],
                row.get("delivered_ns", available) - row["capture_start_ns"],
            )
            > max_age * 1e6
        ):
            reason = "stale_observation"
        if errors or row["image_url"] is None:
            reason = "artifact_integrity"
        row["online"] = {
            "sample": online,
            "telemetry_age_ms": (available - times[index]) / 1e6 if online else None,
            "frame_age_ms": (available - row["capture_start_ns"]) / 1e6,
            "handoff_age_ms": (row.get("delivered_ns", available) - row["capture_start_ns"]) / 1e6,
            "usable": reason is None,
            "reason": reason,
            "meaning": "receipt-causal at encoded image availability; not physics synchronization",
        }
        after = bisect_left(times, row["capture_end_ns"])
        candidate = ordered[after] if after < len(ordered) else None
        row["posthoc_after_capture"] = (
            candidate
            if (
                reason is None
                and candidate
                and online
                and candidate["segment"] == online["segment"]
                and candidate["received_monotonic_ns"] - row["capture_end_ns"] <= max_age * 1e6
                and bisect_right(boundaries, candidate["received_monotonic_ns"])
                == bisect_right(boundaries, row["capture_start_ns"])
            )
            else None
        )
        frames.append(row)
    intervals = [
        (b["capture_start_ns"] - a["capture_start_ns"]) / 1e6
        for a, b in zip(frames, frames[1:])
        if bisect_right(boundaries, b["capture_start_ns"])
        == bisect_right(boundaries, a["capture_start_ns"])
        and b["capture_start_ns"] > a["capture_start_ns"]
    ]
    duration = (session.get("ended_ns", 0) - session.get("started_ns", 0)) / 1e9
    timing = {
        "capture_ms": _distribution(
            [(f["capture_end_ns"] - f["capture_start_ns"]) / 1e6 for f in frames]
        ),
        "available_ms": _distribution(
            [(f["available_ns"] - f["capture_start_ns"]) / 1e6 for f in frames]
        ),
        "stored_ms": _distribution(
            [(f["stored_ns"] - f["capture_start_ns"]) / 1e6 for f in frames]
        ),
        "frame_interval_ms": _distribution(intervals),
        "continuous_hz": 1000 * len(intervals) / sum(intervals) if intervals else None,
        "wall_hz": len(frames) / duration if duration > 0 else None,
        "udp_interval_ms": _distribution([(b - a) / 1e6 for a, b in zip(times, times[1:])]),
    }
    return {
        "session": session,
        "frames": frames,
        "frame_count": len(frames),
        "usable_frames": sum(f["online"]["usable"] for f in frames),
        "timing": timing,
        "events": [r for r in rows if r.get("kind") != "frame"],
        "integrity_errors": errors,
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "median": statistics.median(ordered) if ordered else None,
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1] if ordered else None,
        "max": ordered[-1] if ordered else None,
    }
