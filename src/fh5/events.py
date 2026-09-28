"""Bounded event lifecycle; visual evidence and restart actions stay separate from driving."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import uuid4

if TYPE_CHECKING:
    from fh5.experiment import Packet, RunResult


@dataclass(frozen=True)
class EventRun:
    config_file: Path
    output_dir: Path


@dataclass(frozen=True)
class ScreenFrame:
    received_monotonic_ns: int
    width: int
    height: int
    grayscale: bytes


@dataclass(frozen=True)
class EventInput:
    packets: tuple[Packet, ...] = ()
    frame: ScreenFrame | None = None
    focused: bool = False
    stop_requested: bool = False
    fault: str | None = None


class EventEnvironment(Protocol):
    source_kind: Literal["udp", "synthetic"]

    def now_ns(self) -> int: ...
    def read(self, period_s: float) -> EventInput: ...
    def pulse(self, button: str) -> None:
        """Send a bounded button pulse, returning only after release; never drive."""
        ...

    def release(self) -> None: ...
    def close(self) -> None: ...


def _pgm(data: bytes) -> tuple[int, int, bytes]:
    header, size, maximum, pixels = data.split(b"\n", 3)
    width, height = map(int, size.split())
    if (
        header != b"P5"
        or maximum != b"255"
        or width <= 0
        or height <= 0
        or len(pixels) != width * height
    ):
        raise ValueError("Expected canonical 8-bit PGM")
    return width, height, pixels


def validate_event_file(path: Path) -> dict[str, Any]:
    """Validate finite limits and all evidence before sending a menu action."""
    from fh5.experiment import _validate_config

    root = _validate_config(json.loads(path.read_text(encoding="utf-8-sig")))
    config = root.get("event_run")
    if (
        not isinstance(config, dict)
        or type(config.get("version")) is not int
        or config["version"] != 1
    ):
        raise ValueError("Unsupported event_run version")
    if type(config.get("conditions_verified")) is not bool:
        raise ValueError("conditions_verified must be a boolean")
    purpose = config.setdefault("purpose", "event")
    if purpose not in ("event", "restart_probe"):
        raise ValueError("Unknown event purpose")
    for name, low, high in [
        ("max_attempts", 1, 10),
        ("expected_car_ordinal", 1, 1000000),
        ("expected_pi", 1, 999),
    ]:
        if type(config.get(name)) is not int or not low <= config[name] <= high:
            raise ValueError(f"Invalid {name}")
    for name, lower, upper in [
        ("ready_timeout_s", 0.5, 120),
        ("restart_timeout_s", 0.5, 120),
        ("attempt_timeout_s", 0.5, 1800),
        ("start_radius_m", 0.1, 20),
    ]:
        value = config.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not lower <= value <= upper
        ):
            raise ValueError(f"Invalid {name}")
    position = config.get("start_position_m")
    if (
        not isinstance(position, list)
        or len(position) != 3
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in position)
    ):
        raise ValueError("start_position_m requires three finite coordinates")
    size = config.get("screen_size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(v) is not int or not 1 <= v <= 4096 for v in size)
    ):
        raise ValueError("Invalid screen_size")
    signatures = config.get("signatures")
    if not isinstance(signatures, dict) or len(signatures) > 32:
        raise ValueError("signatures must contain at most 32 screens")
    for name, patches in signatures.items():
        if not isinstance(name, str) or not isinstance(patches, list) or not 1 <= len(patches) <= 8:
            raise ValueError("Each screen requires 1..8 patches")
        for patch in patches:
            if not isinstance(patch, dict):
                raise ValueError("Invalid patch")
            box = patch.get("box")
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(type(v) is not int for v in box)
                or not 0 <= box[0] < box[2] <= size[0]
                or not 0 <= box[1] < box[3] <= size[1]
            ):
                raise ValueError("Patch must lie within screen_size")
            error = patch.get("max_error")
            if (
                isinstance(error, bool)
                or not isinstance(error, (int, float))
                or not 0 <= error <= 0.1
            ):
                raise ValueError("max_error must be between 0 and 0.1")
            if not isinstance(patch.get("template"), str):
                raise ValueError("Missing patch template")
            width, height, _ = _pgm((path.parent / patch["template"]).read_bytes())
            if (width, height) != (box[2] - box[0], box[3] - box[1]):
                raise ValueError("Patch and template dimensions differ")
    for name in ("start_steps", "restart_steps", "finish_steps"):
        steps = config.setdefault(name, [])
        if not isinstance(steps, list) or len(steps) > 32:
            raise ValueError(f"Invalid {name}")
        for step in steps:
            if (
                not isinstance(step, dict)
                or step.get("screen") not in signatures
                or step.get("button")
                not in ("A", "B", "X", "Y", "START", "UP", "DOWN", "LEFT", "RIGHT")
            ):
                raise ValueError(f"Invalid step in {name}")
    evidence = config.get("verification_evidence")
    if not isinstance(evidence, list) or any(not isinstance(e, str) for e in evidence):
        raise ValueError("verification_evidence must be a list of paths")
    if purpose == "restart_probe" and (
        config["max_attempts"] > 3
        or config["attempt_timeout_s"] > 10
        or config["ready_timeout_s"] > 30
        or config["restart_timeout_s"] > 30
    ):
        raise ValueError("Restart probes allow at most three short stationary attempts")
    if config["conditions_verified"] or purpose == "restart_probe":
        if (
            not evidence
            or any(not (path.parent / e).is_file() for e in evidence)
            or (
                config["conditions_verified"]
                and any(f["status"] != "verified" for f in root["snapshot"].values())
            )
        ):
            raise ValueError("Verified conditions require snapshot and local evidence")
        if not {"ready", "driving", "finish"} <= signatures.keys() or any(
            not config.get(name) for name in ("start_steps", "restart_steps", "finish_steps")
        ):
            raise ValueError("Verified conditions require start, restart and finish recipes")
    return root


def _recognize(frame: ScreenFrame, config: dict[str, Any], templates: dict[str, bytes]) -> str:
    if [frame.width, frame.height] != config["screen_size"]:
        return "unknown"
    matches = []
    for name, patches in config["signatures"].items():
        errors = []
        for patch in patches:
            left, top, right, bottom = patch["box"]
            template = templates[patch["template"]]
            pixels = b"".join(
                frame.grayscale[y * frame.width + left : y * frame.width + right]
                for y in range(top, bottom)
            )
            error = sum(abs(a - b) for a, b in zip(pixels, template)) / (len(template) * 255)
            errors.append(error <= patch["max_error"])
        if errors and all(errors):
            matches.append(name)
    return matches[0] if len(matches) == 1 else "unknown"


def run_event(request: EventRun, environment: EventEnvironment) -> RunResult:
    try:
        root = validate_event_file(request.config_file)
        return _run_event(request, environment, root)
    finally:
        environment.close()


def _running_fault(
    sample: dict[str, Any], previous: dict[str, Any] | None, config: dict[str, Any]
) -> str | None:
    if sample["is_race_on"] and (
        sample["car_ordinal"] != config["expected_car_ordinal"]
        or sample["car_performance_index"] != config["expected_pi"]
    ):
        return "vehicle_changed"
    if previous is None:
        return None
    delta = sample["received_monotonic_ns"] - previous["received_monotonic_ns"]
    if delta <= 0:
        return "telemetry_clock_invalid"
    if (
        sample["is_race_on"]
        and previous["is_race_on"]
        and (sample["game_timestamp_ms"] - previous["game_timestamp_ms"]) % (2**32) > 60_000
    ):
        return "game_time_jump"
    if (
        sample["is_race_on"]
        and previous["is_race_on"]
        and math.dist(sample["position_m"], previous["position_m"]) > 20 + 200 * delta / 1e9
    ):
        return "position_jump"
    return None


def _run_event(request: EventRun, environment: EventEnvironment, root: dict[str, Any]) -> RunResult:
    from fh5.experiment import Record, _decode, _write_json, run_experiment

    root = deepcopy(root)
    config = root["event_run"]
    assets: dict[str, bytes] = {}
    templates: dict[str, bytes] = {}
    for patches in config["signatures"].values():
        for patch in patches:
            original = request.config_file.parent / patch["template"]
            name = f"event-assets/{len(assets):03d}.pgm"
            assets[name] = original.read_bytes()
            templates[name] = _pgm(assets[name])[2]
            patch["template"] = name
    evidence = []
    for original in config["verification_evidence"]:
        name = f"event-assets/{len(assets):03d}{Path(original).suffix}"
        assets[name] = (request.config_file.parent / original).read_bytes()
        evidence.append(name)
    config["verification_evidence"] = evidence
    event_summary: dict[str, Any] = {
        "version": 1,
        "stop_reason": "incomplete",
        "attempts": [],
        "unattended_verified": False,
        "release_sent": False,
        "purpose": config["purpose"],
        "conditions_verified": config["conditions_verified"],
    }
    events: list[dict[str, Any]] = []

    def capture() -> Iterator[Packet]:
        directory = request.output_dir / "frames"
        directory.mkdir()
        (request.output_dir / "event-assets").mkdir()
        for name, data in assets.items():
            (request.output_dir / name).write_bytes(data)
        _write_json(request.output_dir / "event-config.json", root)
        now = environment.now_ns()
        phase = "prepare"
        phase_since = now
        steps = config["start_steps"]
        step = 0
        frame_index = 0
        previous_screen = "unknown"
        previous_frame_ns = -1
        stable = 0
        latest: dict[str, Any] | None = None
        attempt: dict[str, Any] | None = None
        run_id = uuid4().hex
        journal_path = request.output_dir / "event-journal.jsonl"

        def log(kind: str, **detail: Any) -> None:
            event = {"kind": kind, "received_monotonic_ns": now, **detail}
            with journal_path.open("a", encoding="utf-8") as journal:
                journal.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
            events.append(event)

        try:
            environment.release()
            for name in [*assets, "event-config.json"]:
                log(
                    "asset",
                    path=name,
                    sha256=hashlib.sha256((request.output_dir / name).read_bytes()).hexdigest(),
                )
            log("prepare")
            while True:
                observation = environment.read(0.1)
                previous_now = now
                now = environment.now_ns()
                yield from observation.packets
                telemetry_fault = None
                for packet in observation.packets:
                    sample = _decode(packet)
                    if phase == "running":
                        telemetry_fault = telemetry_fault or _running_fault(sample, latest, config)
                    if (
                        config["purpose"] == "restart_probe"
                        and sample["is_race_on"]
                        and (sample["speed_kmh"] > 3 or any(sample["telemetry_controls"].values()))
                    ):
                        telemetry_fault = telemetry_fault or "probe_vehicle_moving_or_input"
                    latest = sample
                frame = observation.frame
                screen = "unknown"
                frame_valid = (
                    frame is not None
                    and [frame.width, frame.height] == config["screen_size"]
                    and len(frame.grayscale) == frame.width * frame.height
                )
                if frame:
                    data = f"P5\n{frame.width} {frame.height}\n255\n".encode() + frame.grayscale
                    filename = f"frames/{frame_index:06d}.pgm"
                    (request.output_dir / filename).write_bytes(data)
                    if frame_valid:
                        screen = _recognize(frame, config, templates)
                    log(
                        "screen",
                        path=filename,
                        sha256=hashlib.sha256(data).hexdigest(),
                        screen=screen,
                    )
                    frame_index += 1
                stable = stable + 1 if screen == previous_screen else 1
                previous_screen = screen
                # Classification and evidence I/O may outlast the input freshness window.
                now = environment.now_ns()
                if not config["conditions_verified"] and config["purpose"] != "restart_probe":
                    event_summary["stop_reason"] = "conditions_unverified"
                    log("conditions_unverified")
                    break
                reason = observation.fault or telemetry_fault
                if observation.stop_requested:
                    reason = "user_stop"
                elif not observation.focused:
                    reason = "focus_lost"
                elif now <= previous_now:
                    reason = "clock_stalled"
                elif frame is None or not 0 <= now - frame.received_monotonic_ns <= 500_000_000:
                    reason = "screen_stale"
                elif not frame_valid:
                    reason = "screen_malformed"
                elif frame.received_monotonic_ns <= previous_frame_ns:
                    reason = "screen_repeated"
                elif (
                    latest is None or not 0 <= now - latest["received_monotonic_ns"] <= 500_000_000
                ):
                    reason = "telemetry_stale"
                if reason:
                    event_summary["stop_reason"] = reason
                    log(reason)
                    break
                assert frame is not None
                previous_frame_ns = frame.received_monotonic_ns
                if phase in ("prepare", "restart"):
                    timeout = (
                        config["ready_timeout_s"]
                        if phase == "prepare"
                        else config["restart_timeout_s"]
                    )
                    if now - phase_since >= timeout * 1e9:
                        event_summary["stop_reason"] = f"{phase}_timeout"
                        log(event_summary["stop_reason"], step=step, screen=screen)
                        break
                    if step < len(steps):
                        action = steps[step]
                        if screen == action["screen"] and stable >= 2:
                            try:
                                environment.pulse(action["button"])
                            except Exception as error:
                                log(
                                    "menu_action",
                                    button=action["button"],
                                    status="failed",
                                    error=str(error),
                                )
                                raise
                            else:
                                log("menu_action", button=action["button"], status="sent")
                            step += 1
                            stable = 0
                        continue
                    if (
                        screen == "driving"
                        and stable >= 2
                        and latest
                        and latest["is_race_on"] == 1
                        and latest["car_ordinal"] == config["expected_car_ordinal"]
                        and latest["car_performance_index"] == config["expected_pi"]
                        and latest["speed_kmh"] <= 3
                        and math.dist(latest["position_m"], config["start_position_m"])
                        <= config["start_radius_m"]
                    ):
                        if phase == "restart":
                            log("recovery_verified")
                        attempt = {
                            "attempt_id": f"{run_id}-{len(event_summary['attempts']) + 1}",
                            "started_ns": now,
                            "outcome": "running",
                        }
                        event_summary["attempts"].append(attempt)
                        log("started", attempt_id=attempt["attempt_id"])
                        phase, phase_since = "running", now
                    continue
                if attempt is not None:
                    finished = screen == "finish" and stable >= 2
                    timed_out = now - phase_since >= config["attempt_timeout_s"] * 1e9
                    if not finished and not timed_out:
                        continue
                    attempt.update(
                        outcome="completion_observed" if finished else "failed",
                        ended_ns=now,
                        reason="finish_screen" if finished else "attempt_timeout",
                    )
                    log(
                        attempt["outcome"],
                        attempt_id=attempt["attempt_id"],
                        reason=attempt["reason"],
                    )
                    if len(event_summary["attempts"]) >= config["max_attempts"]:
                        event_summary["stop_reason"] = "attempt_limit"
                        break
                    phase, phase_since, step, stable = "restart", now, 0, 0
                    steps = config["finish_steps"] if finished else config["restart_steps"]
                    log("restart", attempt_id=attempt["attempt_id"])
        except KeyboardInterrupt:
            event_summary["stop_reason"] = "user_stop"
            log("user_stop")
        except Exception as error:
            event_summary["stop_reason"] = "interface_error"
            log("interface_error", error=str(error))
        finally:
            # Release before journal writes: disk errors must not leave input held.
            release_error = None
            try:
                environment.release()
                event_summary["release_sent"] = True
            except Exception as error:
                release_error = str(error)
            if attempt and attempt["outcome"] == "running":
                attempt.update(
                    outcome="interrupted", ended_ns=now, reason=event_summary["stop_reason"]
                )
                log("interrupted", attempt_id=attempt["attempt_id"], reason=attempt["reason"])
            if release_error is None:
                log("released")
            else:
                log("release_failed", error=release_error)
            for adapter_event in getattr(environment, "events", []):
                log("adapter_event", detail=adapter_event)
            # Record flushes each yielded datagram before resuming this generator.
            log(
                "asset",
                path="packets.jsonl",
                sha256=hashlib.sha256(
                    (request.output_dir / "packets.jsonl").read_bytes()
                ).hexdigest(),
            )
            event_summary["evidence_status"] = "complete"
            log("run_ended", summary=deepcopy(event_summary))
            _write_json(
                request.output_dir / "event-run.tmp",
                {
                    "summary": event_summary,
                    "events": events,
                    "journal_sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest(),
                },
            )
            (request.output_dir / "event-run.tmp").replace(request.output_dir / "event-run.json")

    return run_experiment(
        Record(request.config_file, request.output_dir, environment.source_kind), packets=capture()
    )


def read_event(directory: Path) -> dict[str, Any]:
    """Replay durable evidence, preserving a readable prefix after interrupted writes."""
    summary: dict[str, Any] = {
        "version": 1,
        "stop_reason": "incomplete",
        "attempts": [],
        "release_sent": False,
        "unattended_verified": False,
    }
    events: list[dict[str, Any]] = []
    errors = []
    journal_data = b""
    try:
        journal_data = (directory / "event-journal.jsonl").read_bytes()
        for line in journal_data.splitlines(keepends=True):
            event = json.loads(line)
            if (
                not line.endswith(b"\n")
                or not isinstance(event, dict)
                or not isinstance(event.get("kind"), str)
            ):
                raise ValueError("Invalid journal row")
            events.append(event)
    except (OSError, ValueError) as error:
        errors.append(f"journal: {error}")
    recovered: dict[str, dict[str, Any]] = {}
    for event in events:
        identity = event.get("attempt_id")
        if not isinstance(identity, str):
            continue
        if event["kind"] == "started":
            recovered[identity] = {
                "attempt_id": identity,
                "started_ns": event.get("received_monotonic_ns"),
                "outcome": "interrupted",
                "reason": "incomplete_evidence",
            }
        elif identity in recovered and event["kind"] in (
            "failed",
            "interrupted",
            "completion_observed",
        ):
            recovered[identity].update(
                outcome=event["kind"],
                reason=event.get("reason"),
                ended_ns=event.get("received_monotonic_ns"),
            )
    summary["attempts"] = list(recovered.values())
    try:
        saved = json.loads((directory / "event-run.json").read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or not isinstance(saved.get("summary"), dict):
            raise ValueError("Invalid event summary")
        if saved["summary"].get("version") != 1:
            raise ValueError("Unsupported event summary version")
        if (
            saved.get("events") != events
            or saved.get("journal_sha256") != hashlib.sha256(journal_data).hexdigest()
        ):
            raise ValueError("Journal does not match summary")
        if not events or events[-1]["kind"] != "run_ended":
            raise ValueError("Missing final journal outcome")
        recorded = events[-1].get("summary")
        if not isinstance(recorded, dict) or not isinstance(recorded.get("attempts"), list):
            raise ValueError("Invalid final journal outcome")
        summary = deepcopy(recorded)
        if summary != saved["summary"]:
            raise ValueError("Attempt outcomes do not match journal")
    except (OSError, ValueError) as error:
        errors.append(f"summary: {error}")
    for event in events:
        if event["kind"] not in ("screen", "asset"):
            continue
        try:
            if not isinstance(event.get("path"), str):
                raise ValueError("Missing frame path")
            path = (directory / event["path"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Frame is outside recording")
            if hashlib.sha256(path.read_bytes()).hexdigest() != event.get("sha256"):
                raise ValueError("Frame checksum mismatch")
        except (OSError, ValueError) as error:
            errors.append(f"frame: {error}")
    if errors:
        summary.update(evidence_status="incomplete", release_sent=False, unattended_verified=False)
        events.append({"kind": "event_evidence_incomplete", "detail": "; ".join(errors)})
    return {"summary": summary, "events": events}
