"""Bounded calibration runs over an external game/clock/controller boundary."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TextIO

if TYPE_CHECKING:
    from fh5.experiment import Packet, RunResult


@dataclass(frozen=True)
class Command:
    steer_i16: int
    throttle_u8: int
    brake_u8: int


@dataclass(frozen=True)
class Control:
    config_file: Path
    output_dir: Path


@dataclass(frozen=True)
class ControlInput:
    packets: tuple[Packet, ...] = ()
    focused: bool = False
    stop_requested: bool = False
    fault: str | None = None


class ControlEnvironment(Protocol):
    source_kind: Literal["udp", "synthetic"]
    events: list[dict[str, Any]]

    def now_ns(self) -> int: ...
    def read(self, period_s: float) -> ControlInput: ...
    def send(self, command: Command) -> None: ...
    def close(self) -> None: ...


def record_command(
    environment: ControlEnvironment,
    journal: TextIO,
    commands: list[dict[str, Any]],
    command: Command,
    requested: dict[str, Any] | None,
    owner: str,
) -> dict[str, Any]:
    """Record the driver call outcome, including failed or partially applied calls."""
    row: dict[str, Any] = {
        "requested": requested or {},
        "target": asdict(command),
        "sent": None,
        "owner": owner,
        "issued_ns": environment.now_ns(),
        "status": "failed",
    }
    try:
        environment.send(command)
        row.update(sent=asdict(command), status="sent")
    except Exception as error:
        row["error"] = str(error)
        raise
    finally:
        row["returned_ns"] = environment.now_ns()
        commands.append(row)
        journal.write(json.dumps(row, allow_nan=False) + "\n")
        journal.flush()
    return row


def validate_control_file(path: Path) -> dict[str, Any]:
    """Reject unbounded calibration before creating a controller or sending input."""
    from fh5.experiment import _validate_config

    root = _validate_config(json.loads(path.read_text(encoding="utf-8-sig")))
    config = root.get("control")
    if root["control_source"] != "calibration" or not isinstance(config, dict):
        raise ValueError("Control requires control_source=calibration and a control object")
    if type(config.get("version")) is not int or config["version"] != 1:
        raise ValueError("Unsupported control version")
    if config.get("action_version") != "steer-signed-longitudinal-v1":
        raise ValueError("Unsupported control action_version")
    for key, low_int, high_int in [("expected_car_ordinal", 1, 1_000_000), ("expected_pi", 1, 999)]:
        if type(config.get(key)) is not int or not low_int <= config[key] <= high_int:
            raise ValueError(f"Invalid {key}")

    def bounded(obj: dict[str, Any], key: str, low: float, high: float) -> None:
        value = obj.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"{key} must be finite and between {low} and {high}")

    for key, low, high in [
        ("rate_hz", 10, 50),
        ("max_speed_kmh", 1, 40),
        ("start_speed_kmh", 0, 5),
        ("max_steer", 0, 0.5),
        ("max_throttle", 0, 0.35),
        ("max_brake", 0, 1),
        ("telemetry_timeout_s", 0.1, 0.5),
        ("ready_timeout_s", 0.1, 60),
    ]:
        bounded(config, key, low, high)
    if config["start_speed_kmh"] >= config["max_speed_kmh"]:
        raise ValueError("start_speed_kmh must be below max_speed_kmh")
    steps = config.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 100:
        raise ValueError("Control requires 1..100 bounded steps")
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Each control step must be an object")
        bounded(step, "seconds", 0.02, 30)
        bounded(step, "steer", -1, 1)
        if "target_speed_kmh" in step:
            if "longitudinal" in step:
                raise ValueError("A step cannot request both speed feedback and fixed pedals")
            bounded(step, "target_speed_kmh", 0, config["max_speed_kmh"] - 3)
        else:
            bounded(step, "longitudinal", -1, 1)
    if sum(step["seconds"] for step in steps) > 30:
        raise ValueError("Calibration may last at most 30 seconds")
    feedback = any("target_speed_kmh" in step for step in steps)
    if not feedback and config["max_throttle"] > 0.25:
        raise ValueError("Fixed calibration throttle may not exceed 0.25")
    guard = config.get("spatial_guard")
    if feedback and not isinstance(guard, dict):
        raise ValueError("Speed feedback requires an explicit spatial_guard")
    if feedback and (config["max_steer"] != 0 or any(s["steer"] != 0 for s in steps)):
        raise ValueError("Speed collection assistance currently requires neutral steering")
    if feedback and (
        any("target_speed_kmh" not in s for s in steps) or steps[-1]["target_speed_kmh"] != 0
    ):
        raise ValueError("Speed feedback requires speed-only steps ending with a zero target")
    if feedback and steps[-1]["seconds"] < 0.3:
        # Longer than the 250 ms stall guard: a non-stalled schedule must
        # visit the braking phase instead of jumping over it at the next tick.
        raise ValueError("Final zero-speed phase must last at least 0.3 seconds")
    if guard is not None:
        if not isinstance(guard, dict):
            raise ValueError("spatial_guard must be an object")
        position = guard.get("start_position_m")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("spatial_guard requires a three-dimensional start position")
        for value in position:
            bounded({"coordinate": value}, "coordinate", -100_000, 100_000)
        for key, low, high in [
            ("start_radius_m", 0.1, 2),
            ("heading_rad", -math.pi, math.pi),
            ("max_heading_error_rad", 0.01, 0.3),
            ("max_lateral_m", 0.1, 2),
            ("max_distance_m", 1, 60),
        ]:
            bounded(guard, key, low, high)
    return config


def run_control(request: Control, environment: ControlEnvironment) -> RunResult:
    try:
        config = validate_control_file(request.config_file)
        return _run_control(request, environment, config)
    finally:
        environment.close()


def _sample_stop_reason(sample: dict[str, Any], config: dict[str, Any]) -> str | None:
    if not sample["is_race_on"]:
        return "inactive"
    if (
        sample["car_ordinal"] != config["expected_car_ordinal"]
        or sample["car_performance_index"] != config["expected_pi"]
    ):
        return "vehicle_changed"
    if sample["speed_kmh"] >= config["max_speed_kmh"]:
        return "speed_limit"
    return None


def _spatial_stop_reason(
    sample: dict[str, Any], config: dict[str, Any], *, starting: bool
) -> str | None:
    guard = config.get("spatial_guard")
    if guard is None or not sample["is_race_on"]:
        return None
    if sample["motion"] is None:
        return "invalid_motion"
    position = sample["position_m"]
    origin = guard["start_position_m"]
    distance = math.dist(position, origin)
    if starting and distance > guard["start_radius_m"]:
        return "start_position_mismatch"
    if distance > guard["max_distance_m"]:
        return "distance_limit"
    yaw = guard["heading_rad"]
    lateral = math.cos(yaw) * (position[0] - origin[0]) - math.sin(yaw) * (position[2] - origin[2])
    if abs(lateral) > guard["max_lateral_m"]:
        return "lateral_limit"
    if abs(position[1] - origin[1]) > 2:
        return "height_limit"
    error = math.atan2(
        math.sin(sample["motion"]["yaw_rad"] - yaw), math.cos(sample["motion"]["yaw_rad"] - yaw)
    )
    if abs(error) > guard["max_heading_error_rad"]:
        return "heading_limit"
    return None


def _forward_m(sample: dict[str, Any], guard: dict[str, Any]) -> float:
    position = sample["position_m"]
    origin = guard["start_position_m"]
    yaw = guard["heading_rad"]
    return float(
        math.sin(yaw) * (position[0] - origin[0]) + math.cos(yaw) * (position[2] - origin[2])
    )


def _run_control(
    request: Control, environment: ControlEnvironment, config: dict[str, Any]
) -> RunResult:
    from fh5.experiment import Record, _decode, run_experiment

    result: dict[str, Any] = {
        "version": 1,
        "config": config,
        "source_kind": environment.source_kind,
        "stop_reason": "running",
        "release_sent": False,
        "commands": [],
        "adapter_events": environment.events,
        "feedback_version": "bounded-speed-v1"
        if any("target_speed_kmh" in s for s in config["steps"])
        else None,
    }

    def stream() -> Iterator[Packet]:
        (request.output_dir / "control.json").write_text(json.dumps(result), encoding="utf-8")
        with (request.output_dir / "commands.jsonl").open("x", encoding="utf-8") as journal:

            def send(command: Command, requested: dict[str, Any] | None, owner: str) -> None:
                record_command(environment, journal, result["commands"], command, requested, owner)

            started: int | None = None
            waiting_since = environment.now_ns()
            latest: dict[str, Any] | None = None
            game_advanced_ns = waiting_since
            previous_tick = waiting_since
            guard = config.get("spatial_guard")
            max_forward: float | None = None
            try:
                send(Command(0, 0, 0), None, "stop_guard")
                while True:
                    frame = environment.read(1 / config["rate_hz"])
                    yield from frame.packets
                    now = environment.now_ns()
                    reason: str | None = None
                    if started is not None and now - previous_tick > 250_000_000:
                        reason = "control_stalled"
                    if now < previous_tick:
                        reason = "receive_time_jump"
                    previous_tick = now
                    for packet in frame.packets:
                        try:
                            sample = _decode(packet)
                        except ValueError:
                            reason = "invalid_telemetry"
                            break
                        if latest is not None:
                            game_delta = (
                                sample["game_timestamp_ms"] - latest["game_timestamp_ms"]
                            ) % (2**32)
                            if game_delta > 60_000:
                                reason = "game_time_jump"
                            if (
                                sample["is_race_on"]
                                and latest["is_race_on"]
                                and math.dist(sample["position_m"], latest["position_m"])
                                > 20
                                + 200
                                * max(
                                    0,
                                    (
                                        sample["received_monotonic_ns"]
                                        - latest["received_monotonic_ns"]
                                    )
                                    / 1e9,
                                )
                            ):
                                reason = "position_jump"
                            if game_delta:
                                game_advanced_ns = sample["received_monotonic_ns"]
                        else:
                            game_advanced_ns = sample["received_monotonic_ns"]
                        if started is not None:
                            reason = _sample_stop_reason(sample, config) or reason
                            reason = _spatial_stop_reason(sample, config, starting=False) or reason
                            if guard is not None and max_forward is not None:
                                forward = _forward_m(sample, guard)
                                if forward < max_forward - 0.5:
                                    reason = "reverse_motion"
                                max_forward = max(max_forward, forward)
                        latest = sample
                    if frame.fault:
                        reason = frame.fault
                    elif frame.stop_requested:
                        reason = "user_stop"
                    elif not frame.focused:
                        reason = "focus_lost"
                    elif (
                        latest is None
                        or now - latest["received_monotonic_ns"]
                        > config["telemetry_timeout_s"] * 1e9
                    ):
                        reason = "telemetry_stale"
                    elif sample_reason := _sample_stop_reason(latest, config):
                        reason = sample_reason
                    elif spatial_reason := _spatial_stop_reason(
                        latest, config, starting=started is None
                    ):
                        reason = spatial_reason
                    elif now - game_advanced_ns > config["telemetry_timeout_s"] * 1e9:
                        reason = "game_time_stalled"
                    if started is None:
                        if reason in ("focus_lost", "telemetry_stale", "inactive") or (
                            reason is None
                            and latest is not None
                            and latest["speed_kmh"] > config["start_speed_kmh"]
                        ):
                            if now - waiting_since > config["ready_timeout_s"] * 1e9:
                                result["stop_reason"] = "ready_timeout"
                                break
                            continue
                    if reason:
                        result["stop_reason"] = reason
                        break
                    if started is None:
                        started = now
                        if guard is not None:
                            assert latest is not None
                            max_forward = _forward_m(latest, guard)
                    elapsed = now - started
                    end = 0
                    for step in config["steps"]:
                        end += round(step["seconds"] * 1e9)
                        if elapsed < end:
                            break
                    else:
                        result["stop_reason"] = (
                            "not_stopped"
                            if result["feedback_version"]
                            and latest is not None
                            and latest["speed_kmh"] > 0.5
                            else "completed"
                        )
                        break
                    steer = max(-config["max_steer"], min(config["max_steer"], step["steer"]))
                    requested = {"steer": step["steer"]}
                    if "target_speed_kmh" in step:
                        assert latest is not None
                        target = step["target_speed_kmh"]
                        speed = latest["speed_kmh"]
                        # Never hold the brake at rest: LT can select reverse in
                        # automatic shifting. The independent hard speed stop remains.
                        pedal = (
                            -config["max_brake"]
                            if speed > target + 0.5
                            else config["max_throttle"]
                            if speed < target - 0.5
                            else 0.0
                        )
                        requested.update(target_speed_kmh=target, observed_speed_kmh=speed)
                    else:
                        pedal = step["longitudinal"]
                    requested["longitudinal"] = pedal
                    longitudinal = max(-config["max_brake"], min(config["max_throttle"], pedal))
                    command = Command(
                        round(steer * 32767),
                        round(max(0, longitudinal) * 255),
                        round(max(0, -longitudinal) * 255),
                    )
                    send(
                        command,
                        requested,
                        "calibration",
                    )
                send(Command(0, 0, 0), None, "stop_guard")
                result["release_sent"] = True
                # Observe release without issuing further driving commands.
                for _ in range(6):
                    yield from environment.read(0.05).packets
            except KeyboardInterrupt:
                result["stop_reason"] = "user_stop"
            except Exception as error:
                result.update(stop_reason="interface_error", error=str(error))
            finally:
                try:
                    if not result["release_sent"]:
                        send(Command(0, 0, 0), None, "stop_guard")
                        result["release_sent"] = True
                except Exception as error:
                    result.update(stop_reason="interface_error", release_error=str(error))
                finally:
                    try:
                        environment.close()
                    except Exception as error:
                        result.update(stop_reason="interface_error", close_error=str(error))
                    final_summary = request.output_dir / "control-final.tmp"
                    final_summary.write_text(
                        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                    )
                    final_summary.replace(request.output_dir / "control.json")

    return run_experiment(
        Record(request.config_file, request.output_dir, environment.source_kind), packets=stream()
    )


def read_control(directory: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Recover command evidence independently of a successful final summary write."""
    errors = []
    try:
        result = json.loads((directory / "control.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        result = {"version": 1, "stop_reason": "incomplete", "release_sent": False}
        errors.append(f"Control summary unavailable: {error}")
    if (
        not isinstance(result, dict)
        or type(result.get("version")) is not int
        or result["version"] != 1
    ):
        raise ValueError("Unsupported control recording version")
    if type(result.get("release_sent")) is not bool or not isinstance(
        result.get("stop_reason"), str
    ):
        raise ValueError("Invalid control outcome or release confirmation")
    commands = []
    try:
        lines = (directory / "commands.jsonl").read_bytes().splitlines()
    except OSError as error:
        lines = []
        errors.append(str(error))
    for index, line in enumerate(lines):
        try:
            command = json.loads(line)
            if not isinstance(command, dict) or command.get("status") not in ("sent", "failed"):
                raise ValueError("Invalid command entry")
            for field in ("issued_ns", "returned_ns"):
                if type(command.get(field)) is not int or command[field] < 0:
                    raise ValueError("Invalid command time")
            if command["returned_ns"] < command["issued_ns"]:
                raise ValueError("Command time moved backwards")
            for name in ("target", "sent"):
                value = command.get(name)
                if name == "sent" and command["status"] == "failed" and value is None:
                    continue
                if not isinstance(value, dict):
                    raise ValueError("Invalid command values")
                for axis, low, high in [
                    ("steer_i16", -32767, 32767),
                    ("throttle_u8", 0, 255),
                    ("brake_u8", 0, 255),
                ]:
                    if type(value.get(axis)) is not int or not low <= value[axis] <= high:
                        raise ValueError("Invalid command axis")
            commands.append(command)
        except (ValueError, UnicodeDecodeError) as error:
            errors.append(f"Command line {index + 1}: {error}")
    if result.get("stop_reason") != "running" and commands != result.get("commands"):
        errors.append("Command journal differs from the finalized record")
    if result.get("release_sent") and (
        not commands
        or commands[-1].get("owner") != "stop_guard"
        or commands[-1].get("status") != "sent"
        or commands[-1].get("sent") != asdict(Command(0, 0, 0))
    ):
        errors.append("Final release command evidence is missing")
    result["commands"] = commands
    result["artifact_errors"] = errors
    if errors or result.get("stop_reason") == "running":
        result.update(stop_reason="incomplete", release_sent=False)
    result["sustained_sampling_allowed"] = False
    result["game_response_validation"] = "unverified"
    intervals = [
        (b["issued_ns"] - a["issued_ns"]) / 1e6
        for a, b in zip(commands, commands[1:])
        if a.get("owner") == b.get("owner")
        and a.get("owner") in ("calibration", "policy", "baseline")
    ]
    durations = [(c["returned_ns"] - c["issued_ns"]) / 1e6 for c in commands]
    result["timing"] = {
        "command_count": len(commands),
        "max_interval_ms": max(intervals, default=None),
        "mean_interval_ms": sum(intervals) / len(intervals) if intervals else None,
        "max_send_duration_ms": max(durations, default=None),
    }
    observations = []
    successful = [c for c in commands if c["status"] == "sent"]
    for field, axis in [("accel", "throttle_u8"), ("brake", "brake_u8"), ("steer", "steer_i16")]:
        edges = [
            c
            for i, c in enumerate(successful)
            if not i or c["sent"][axis] != successful[i - 1]["sent"][axis]
        ]
        for i, command in enumerate(edges):
            before = [
                s
                for s in samples
                if s["is_race_on"] and s["received_monotonic_ns"] <= command["issued_ns"]
            ]
            if not before:
                continue
            baseline = before[-1]["telemetry_controls"][field]
            end_ns = min(
                command["issued_ns"] + 500_000_000,
                edges[i + 1]["issued_ns"] if i + 1 < len(edges) else 2**63,
            )
            response = next(
                (
                    s
                    for s in samples
                    if s["is_race_on"]
                    and command["returned_ns"] <= s["received_monotonic_ns"] < end_ns
                    and abs(s["telemetry_controls"][field] - baseline) >= 2
                ),
                None,
            )
            observations.append(
                {
                    "field": field,
                    "issued_ns": command["issued_ns"],
                    "target_value": command["sent"][axis],
                    "baseline_value": baseline,
                    "observed_value": response["telemetry_controls"][field] if response else None,
                    "delay_ms": (response["received_monotonic_ns"] - command["issued_ns"]) / 1e6
                    if response
                    else None,
                }
            )
    result["response_observations"] = observations
    return result
