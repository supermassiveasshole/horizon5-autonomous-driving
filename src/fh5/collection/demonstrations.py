"""Passive human input evidence and supervised examples; never drive the game."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.capture.legacy import VisionEnvironment, VisionInput, VisionRecord, run_vision

if TYPE_CHECKING:
    from fh5.result import RunResult


@dataclass(frozen=True)
class DemonstrationRecord:
    vision: VisionRecord
    input_profile: Path


@dataclass(frozen=True)
class DemonstrationReplay:
    recording_dir: Path
    report_path: Path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(contents: bytes) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(contents)
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("device"), dict)
        or not isinstance(value.get("calibration"), dict)
    ):
        raise ValueError("Input profile needs device and calibration objects")
    if value.get("version") != 1 or value.get("mapping") != "xinput-lx-rt-lt-v1":
        raise ValueError("Unsupported input profile")
    if (
        value["device"].get("api") != "xinput1_4"
        or type(value["device"].get("index")) is not int
        or not 0 <= value["device"]["index"] <= 3
    ):
        raise ValueError("Invalid XInput device identity")
    if value["calibration"].get("status") not in ("unverified", "verified"):
        raise ValueError("Invalid calibration status")
    if value["calibration"]["status"] == "verified" and not value["calibration"].get("evidence"):
        raise ValueError("Verified input mapping needs evidence")
    return value


def _mapped(row: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    raw = row["raw"]
    reasons = []
    fields = {
        "packet_number": (0, 2**32 - 1),
        "buttons": (0, 65535),
        "left_trigger": (0, 255),
        "right_trigger": (0, 255),
        **{key: (-32768, 32767) for key in ("thumb_lx", "thumb_ly", "thumb_rx", "thumb_ry")},
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != set(fields)
        or any(
            type(raw[k]) is not int or not low <= raw[k] <= high
            for k, (low, high) in fields.items()
        )
    ):
        reasons.append("invalid_raw_state")
    if row.get("connected") is not True:
        reasons.append("disconnected")
    if row.get("focused") is not True:
        reasons.append("focus_lost")
    if row.get("other_input") is not False:
        reasons.append("other_input")
    if row.get("device_index") != profile["device"]["index"]:
        reasons.append("device_mismatch")
    if not reasons:
        if raw["left_trigger"] and raw["right_trigger"]:
            reasons.append("simultaneous_pedals")
        if raw["buttons"]:
            reasons.append("button_action")
        if abs(raw["thumb_rx"]) > 8000 or abs(raw["thumb_ry"]) > 8000:
            reasons.append("camera_movement")
    if (
        type(row.get("observed_ns")) is not int
        or type(row.get("available_ns")) is not int
        or not 0 <= row["observed_ns"] <= row["available_ns"]
    ):
        raise ValueError("Invalid input clock")
    if reasons:
        return {
            "poll_ns": row["observed_ns"],
            "available_ns": row["available_ns"],
            "raw": raw,
            "device_index": row.get("device_index"),
            "mapped": None,
            "mapping_valid": False,
            "reasons": reasons,
            "game_adoption": "unverified",
        }
    lx, rt, lt = raw["thumb_lx"], raw["right_trigger"], raw["left_trigger"]
    return {
        "poll_ns": row["observed_ns"],
        "available_ns": row["available_ns"],
        "raw": raw,
        "device_index": row["device_index"],
        "mapped": [lx / (32768 if lx < 0 else 32767), (rt - lt) / 255],
        "mapping_valid": True,
        "reasons": [],
        "game_adoption": "unverified",
    }


def _bound_files(directory: Path) -> set[str]:
    files = {
        "packets.jsonl",
        "session.json",
        "vision.jsonl",
        "vision-config.json",
        "vision-session.json",
        "input-profile.json",
        "actions.jsonl",
        "action-history.json",
    }
    if (directory / "observation-config.json").exists():
        files.add("observation-config.json")
    files.update(
        p.relative_to(directory).as_posix()
        for p in (directory / "observation-route").rglob("*")
        if p.is_file()
    )
    return files


def validate_demonstration(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "demonstration-session.json").read_text(encoding="utf-8"))
    if (
        manifest.get("version") != 1
        or set(manifest.get("hashes", {})) != _bound_files(directory)
        or any(
            not (directory / name).is_file() or _hash(directory / name) != digest
            for name, digest in manifest["hashes"].items()
        )
    ):
        raise ValueError("Demonstration integrity check failed")
    return _profile((directory / "input-profile.json").read_bytes())


class _InputJournal:
    """Validate the input clock at delivery and cut history at every ownership loss."""

    def __init__(self, environment: VisionEnvironment, profile: dict[str, Any]) -> None:
        self.environment = environment
        self.profile = profile
        self.source_kind = environment.source_kind
        self.previous = environment.now_ns()

    def now_ns(self) -> int:
        return self.environment.now_ns()

    def close(self) -> bool:
        return self.environment.close()

    def read(self, period_s: float) -> VisionInput:
        batch = self.environment.read(period_s)
        events = list(batch.events)
        for event in batch.events:
            if event.get("kind") != "human_input":
                continue
            row = _mapped(event, self.profile)
            if not self.previous < row["poll_ns"] <= row["available_ns"] <= self.now_ns():
                raise ValueError("Input clock is out of order or beyond delivery")
            reasons = list(row["reasons"])
            if row["poll_ns"] - self.previous > 250_000_000:
                reasons.append("input_poll_gap")
            self.previous = row["available_ns"]
            if reasons:
                events.append(
                    {"kind": "input_boundary", "observed_ns": row["poll_ns"], "reasons": reasons}
                )
        return replace(batch, events=tuple(events))


def record_demonstration(request: DemonstrationRecord, environment: VisionEnvironment) -> RunResult:
    directory = request.vision.output_dir
    try:
        contents = request.input_profile.read_bytes()
        profile = _profile(contents)
        result = run_vision(request.vision, _InputJournal(environment, profile))
        (directory / "input-profile.json").write_bytes(contents)
        inputs = [
            _mapped(row, profile)
            for row in result.summary["vision"]["events"]
            if row.get("kind") == "human_input"
        ]
        times = [s["received_monotonic_ns"] for s in result.samples]
        actions = []
        for row in inputs:
            index = bisect_right(times, row["poll_ns"]) - 1
            if row["mapping_valid"] and index >= 0:
                actions.append(
                    {
                        "occurred_ns": row["poll_ns"],
                        "available_ns": row["available_ns"],
                        "telemetry_segment": result.samples[index]["segment"],
                        "source": "human_input",
                        "steer": row["mapped"][0],
                        "longitudinal": row["mapped"][1],
                    }
                )
        with (directory / "actions.jsonl").open("x", encoding="utf-8") as stream:
            for action in actions:
                stream.write(json.dumps(action, allow_nan=False) + "\n")
        (directory / "action-history.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "mapping_version": "dual-axis-v1",
                    "actions_sha256": _hash(directory / "actions.jsonl"),
                    "packets_sha256": _hash(directory / "packets.jsonl"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest = {
            "version": 1,
            "hashes": {name: _hash(directory / name) for name in sorted(_bound_files(directory))},
        }
        (directory / "demonstration-session.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return replay_demonstration(
            DemonstrationReplay(directory, directory / "demonstration.html")
        )
    finally:
        environment.close()


def replay_demonstration(request: DemonstrationReplay) -> RunResult:
    from fh5.experiment import run_experiment
    from fh5.observation.multimodal import ObservationReplay
    from fh5.reporting.telemetry import write_report
    from fh5.telemetry.packet import Replay

    directory = request.recording_dir
    if request.report_path.exists() or request.report_path.with_suffix(".json").exists():
        raise FileExistsError(request.report_path)
    profile = validate_demonstration(directory)
    # The base report is independently reproducible and keeps its original meaning.
    base = request.report_path.with_name(request.report_path.stem + "-observations.html")
    config = directory / "observation-config.json"
    reference = directory / "observation-route/route.json"
    result = run_experiment(
        ObservationReplay(directory, base, reference if reference.exists() else None, config)
        if config.exists()
        else Replay(directory, base)
    )
    inputs = [
        _mapped(row, profile)
        for row in result.summary["vision"]["events"]
        if row.get("kind") == "human_input"
    ]
    start = result.summary["vision"]["session"]["started_ns"]
    end = result.summary["vision"]["session"]["ended_ns"]
    if any(not start <= row["poll_ns"] <= row["available_ns"] <= end for row in inputs) or any(
        a["available_ns"] >= b["poll_ns"] for a, b in zip(inputs, inputs[1:])
    ):
        raise ValueError("Input times outside recording or not strictly ordered")
    result.summary["demonstration"] = {
        "version": 1,
        "profile": profile,
        "calibrated": profile["calibration"]["status"] == "verified",
        "inputs": inputs,
        "commands_sent": False,
        "integrity_errors": result.summary["vision"]["integrity_errors"],
    }
    result = replace(result, report_path=request.report_path)
    write_report(
        result.report_path,
        {
            "metadata": result.metadata,
            "samples": result.samples,
            "events": result.events,
            "summary": result.summary,
        },
    )
    return result
