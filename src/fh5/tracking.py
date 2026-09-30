"""Bounded traditional route feedback; independent from multimodal learning."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from fh5.control import Command, ControlEnvironment, record_command
from fh5.routes import load_route, locate_route

if TYPE_CHECKING:
    from fh5.experiment import Packet, RunResult

NEUTRAL = Command(0, 0, 0)


class _StopTracking(Exception):
    """A latched attempt boundary, distinct from an unexpected transport error."""


@dataclass(frozen=True)
class TrackingDrive:
    config_file: Path
    output_dir: Path


def validate_tracking_file(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from fh5.experiment import _validate_config

    root = _validate_config(json.loads(path.read_text(encoding="utf-8-sig")))
    config = root.get("tracking")
    if root["control_source"] != "policy" or not isinstance(config, dict):
        raise ValueError("Tracking requires control_source=policy and a tracking object")
    if type(config.get("version")) is not int or config["version"] != 1:
        raise ValueError("Unsupported tracking version")
    if config.get("action_version") != "steer-signed-longitudinal-v1":
        raise ValueError("Unsupported tracking action_version")
    for key, lower, upper in [("expected_car_ordinal", 1, 1_000_000), ("expected_pi", 1, 999)]:
        if type(config.get(key)) is not int or not lower <= config[key] <= upper:
            raise ValueError(f"Invalid {key}")
    for key, low, high in [
        ("rate_hz", 10, 50),
        ("target_speed_kmh", 3, 12),
        ("max_speed_kmh", 6, 15),
        ("max_steer", 0, 0.5),
        ("max_throttle", 0, 0.25),
        ("max_brake", 0.1, 0.5),
        ("max_duration_s", 1, 30),
        ("ready_timeout_s", 0.1, 60),
        ("braking_s", 0.3, 3),
        ("telemetry_timeout_s", 0.1, 0.25),
        ("no_progress_timeout_s", 1, 5),
        ("recovery_timeout_s", 0.5, 5),
        ("end_margin_m", 1, 5),
        ("wheelbase_m", 1, 5),
        ("steering_scale_rad", 0.1, 1),
        ("lateral_accel_mps2", 0.1, 1),
        ("braking_decel_mps2", 0.1, 2),
    ]:
        value = config.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"{key} must be finite and between {low} and {high}")
    if config["target_speed_kmh"] > config["max_speed_kmh"] - 3:
        raise ValueError("Target speed requires at least 3 km/h below the stop limit")
    if not isinstance(config.get("route_file"), str) or not config["route_file"]:
        raise ValueError("Tracking requires a route_file")
    route_file = path.parent / config["route_file"]
    if hashlib.sha256(route_file.read_bytes()).hexdigest() != config["route_sha256"]:
        raise ValueError("Tracking route changed")
    route = load_route(route_file)
    if not route["low_speed_ready"] or not 3 <= route["length_m"] <= 60:
        raise ValueError("Tracking requires verified local geometry of 3..60 metres")
    if config["end_margin_m"] >= route["length_m"] - 1:
        raise ValueError("End margin leaves no driving segment")
    return root, route


def _reference_at(route: dict[str, Any], station: float) -> tuple[list[float], float]:
    points = route["points"]
    for a, b in zip(points, points[1:]):
        if station <= b["s_m"]:
            break
    fraction = max(0.0, min(1.0, (station - a["s_m"]) / (b["s_m"] - a["s_m"])))
    position = [x + fraction * (y - x) for x, y in zip(a["position_m"], b["position_m"])]
    return position, math.atan2(
        b["position_m"][0] - a["position_m"][0], b["position_m"][2] - a["position_m"][2]
    )


def _freeze_route(request: TrackingDrive, config: dict[str, Any]) -> None:
    source = request.config_file.parent / config["route_file"]
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["route_sha256"]:
        raise ValueError("Tracking route changed before recording")
    manifest = json.loads(raw)
    destination = request.output_dir / "tracking-route"
    destination.mkdir()
    for asset in [*manifest["assets"].values(), *manifest["evidence"]]:
        file = (source.parent / asset["path"]).resolve()
        target = (destination / asset["path"]).resolve()
        if not file.is_relative_to(source.parent.resolve()) or not target.is_relative_to(
            destination.resolve()
        ):
            raise ValueError("Route asset outside bundle")
        data = file.read_bytes()
        if hashlib.sha256(data).hexdigest() != asset["sha256"]:
            raise ValueError("Tracking route asset changed before recording")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (destination / "route.json").write_bytes(raw)


def read_tracking_route(directory: Path, control: dict[str, Any]) -> dict[str, Any]:
    path = directory / "tracking-route/route.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != control["config"]["route_sha256"]:
        raise ValueError("Frozen tracking route hash mismatch")
    return load_route(path)


def _speed_profile(route: dict[str, Any], config: dict[str, Any]) -> list[tuple[float, float]]:
    limits = []
    points = route["points"]
    for a, b, c in zip(points, points[1:], points[2:]):
        p, q, r = ([n["position_m"][i] for i in (0, 2)] for n in (a, b, c))
        cross = (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0])
        product = math.dist(p, q) * math.dist(q, r) * math.dist(p, r)
        curvature = 2 * abs(cross) / product if product > 1e-9 else 1e6
        cap = min(
            config["target_speed_kmh"] / 3.6,
            math.sqrt(config["lateral_accel_mps2"] / max(curvature, 1e-9)),
        )
        limits.append((b["s_m"], cap))
    limits.append((route["length_m"] - config["end_margin_m"], 0.0))
    return limits


class _Session:
    def __init__(
        self,
        request: TrackingDrive,
        config: dict[str, Any],
        route: dict[str, Any],
        env: ControlEnvironment,
    ) -> None:
        self.request, self.config, self.route, self.env = request, config, route, env
        self.started: int | None = None
        self.braking: int | None = None
        self.recovery_started: int | None = None
        self.recovery_spent_ns = 0
        self.latest: dict[str, Any] | None = None
        self.route_state: dict[str, Any] = {}
        self.index = 0
        self.close_attempted = False
        self.speed_limits = _speed_profile(route, config)
        self.previous_now = env.now_ns()
        self.waiting_since = self.previous_now
        self.progress_ns = self.previous_now
        self.progress_m = 0.0
        self.clock_advanced_ns = self.previous_now
        self.previous_sample: dict[str, Any] | None = None
        self.result: dict[str, Any] = {
            "version": 1,
            "controller_kind": "route-feedback-v1",
            "config": config,
            "source_kind": env.source_kind,
            "stop_reason": "running",
            "release_sent": False,
            "commands": [],
            "adapter_events": env.events,
            "formal_validity": "pending_independent_review",
        }

    def accept_packet(self, packet: Packet, now: int) -> None:
        from fh5.experiment import _decode

        index = self.index
        self.index += 1
        self.result["last_packet_index"] = index
        if packet.received_monotonic_ns > now or (
            self.previous_sample
            and packet.received_monotonic_ns <= self.previous_sample["received_monotonic_ns"]
        ):
            raise _StopTracking("invalid_receive_time")
        try:
            sample = _decode(packet)
        except ValueError as error:
            raise _StopTracking("invalid_telemetry") from error
        sample.update(packet_index=index, segment=0)
        self.result["last_position_m"] = sample["position_m"]
        if not sample["is_race_on"] and self.started is None:
            self.latest = self.previous_sample = None
            self.route_state.clear()
            return
        if (
            sample["car_ordinal"] != self.config["expected_car_ordinal"]
            or sample["car_performance_index"] != self.config["expected_pi"]
        ):
            raise _StopTracking("vehicle_changed")
        if not sample["is_race_on"]:
            raise _StopTracking("inactive")
        if not 0 <= sample["speed_kmh"] < self.config["max_speed_kmh"]:
            raise _StopTracking("speed_limit")
        if sample["motion"] is None:
            raise _StopTracking("invalid_motion")
        if (
            self.previous_sample is None
            or sample["game_timestamp_ms"] > self.previous_sample["game_timestamp_ms"]
        ):
            self.clock_advanced_ns = packet.received_monotonic_ns
        if now - self.clock_advanced_ns > self.config["telemetry_timeout_s"] * 1e9:
            raise _StopTracking("game_time_stalled")
        locate_route([sample], self.route, state=self.route_state)
        if sample["route"]["status"] not in ("matched", "awaiting_checkpoint"):
            raise _StopTracking("route_" + sample["route"]["status"])
        if sample["route"]["distance_m"] > 1.5:
            raise _StopTracking("lateral_limit")
        _, heading = _reference_at(self.route, sample["route"]["reference_s_m"])
        heading_error = sample["motion"]["yaw_rad"] - heading
        sample["heading_error_rad"] = math.atan2(math.sin(heading_error), math.cos(heading_error))
        if abs(sample["heading_error_rad"]) > 0.9:
            raise _StopTracking("heading_limit")
        self.latest = self.previous_sample = sample

    def feedback(self, sample: dict[str, Any]) -> tuple[float, float]:
        station = sample["route"]["reference_s_m"]
        target = self.config["target_speed_kmh"] / 3.6
        for at, cap in self.speed_limits:
            if at >= station:
                target = min(
                    target,
                    math.sqrt(cap * cap + 2 * self.config["braking_decel_mps2"] * (at - station)),
                )
        lookahead = max(2.0, min(5.0, sample["speed_mps"]))
        aim, _ = _reference_at(self.route, station + lookahead)
        yaw = sample["motion"]["yaw_rad"]
        dx, dz = aim[0] - sample["position_m"][0], aim[2] - sample["position_m"][2]
        lateral = math.cos(yaw) * dx - math.sin(yaw) * dz
        angle = math.atan2(2 * self.config["wheelbase_m"] * lateral, max(0.01, dx * dx + dz * dz))
        steer = max(
            -self.config["max_steer"],
            min(self.config["max_steer"], angle / self.config["steering_scale_rad"]),
        )
        return steer, target

    def send(
        self, journal: TextIO, command: Command, owner: str, requested: dict[str, Any]
    ) -> None:
        if (
            requested
            and self.env.now_ns() - requested["telemetry_received_ns"]
            > self.config["telemetry_timeout_s"] * 1e9
        ):
            raise _StopTracking("decision_stale")
        row = record_command(self.env, journal, self.result["commands"], command, requested, owner)
        if command != NEUTRAL and row["returned_ns"] - row["issued_ns"] >= 250_000_000:
            raise _StopTracking("send_stalled")

    def stream(self) -> Generator[Packet]:
        directory = self.request.output_dir
        (directory / "control.json").write_text(json.dumps(self.result), encoding="utf-8")
        with (directory / "commands.jsonl").open("x", encoding="utf-8") as journal:
            try:
                _freeze_route(self.request, self.config)
                self.send(journal, NEUTRAL, "stop_guard", {})
                while True:
                    frame = self.env.read(1 / self.config["rate_hz"])
                    yield from frame.packets
                    now = self.env.now_ns()
                    if now <= self.previous_now:
                        raise _StopTracking("scheduler_clock")
                    if self.started is not None and now - self.previous_now >= 250_000_000:
                        raise _StopTracking("control_stalled")
                    self.previous_now = now
                    if frame.stop_requested:
                        raise _StopTracking("user_stop")
                    if frame.fault:
                        raise _StopTracking(frame.fault)
                    if (
                        self.started is None
                        and now - self.waiting_since >= self.config["ready_timeout_s"] * 1e9
                    ):
                        raise _StopTracking("ready_timeout")
                    if not frame.focused:
                        if self.started is not None:
                            raise _StopTracking("focus_lost")
                        self.index += len(frame.packets)
                        continue
                    for packet in frame.packets:
                        self.accept_packet(packet, now)
                    if self.latest is None and self.started is None:
                        continue
                    if (
                        self.latest is None
                        or now - self.latest["received_monotonic_ns"]
                        > self.config["telemetry_timeout_s"] * 1e9
                    ):
                        raise _StopTracking("telemetry_stale")
                    if now - self.clock_advanced_ns > self.config["telemetry_timeout_s"] * 1e9:
                        raise _StopTracking("game_time_stalled")
                    if not frame.packets:
                        continue
                    sample = self.latest
                    if self.started is None:
                        if sample["speed_kmh"] > 0.5:
                            raise _StopTracking("start_not_stopped")
                        if (
                            sample["route"]["distance_m"] > 0.5
                            or abs(sample["heading_error_rad"]) > 0.3
                        ):
                            raise _StopTracking("start_pose")
                        self.started = now
                        self.progress_ns = now
                    station = sample["route"]["reference_s_m"]
                    if station >= self.progress_m + 0.25:
                        self.progress_m, self.progress_ns = station, now
                    elif (
                        self.braking is None
                        and now - self.progress_ns >= self.config["no_progress_timeout_s"] * 1e9
                    ):
                        raise _StopTracking("no_progress")
                    steer, target = self.feedback(sample)
                    distance = sample["route"]["distance_m"]
                    heading = abs(sample["heading_error_rad"])
                    if self.recovery_started is None and (distance > 0.75 or heading > 0.5):
                        self.recovery_started = now
                    if self.recovery_started is not None:
                        elapsed = now - self.recovery_started
                        if (
                            self.recovery_spent_ns + elapsed
                            >= self.config["recovery_timeout_s"] * 1e9
                        ):
                            raise _StopTracking("recovery_timeout")
                        if distance < 0.4 and heading < 0.25:
                            self.recovery_spent_ns += elapsed
                            self.recovery_started = None
                        else:
                            target = min(target, 6 / 3.6)
                    end = self.route["length_m"] - self.config["end_margin_m"]
                    if (
                        station >= end - 0.25
                        or now - self.started >= self.config["max_duration_s"] * 1e9
                    ):
                        if self.braking is None:
                            self.braking = now
                            self.result["stop_reason"] = (
                                "local_end" if station >= end - 0.25 else "time_limit"
                            )
                    if self.braking is not None:
                        if sample["speed_kmh"] <= 0.5:
                            if self.result["stop_reason"] == "local_end" and (
                                sample["route"]["status"] != "matched"
                                or sample["route"]["confirmed_progress_m"] < end - 0.25
                            ):
                                self.result["stop_reason"] = "unconfirmed_target"
                            break
                        if now - self.braking >= self.config["braking_s"] * 1e9:
                            self.result["stop_reason"] = "not_stopped"
                            break
                        target = 0.0
                    error = target - sample["speed_mps"]
                    pedal = max(
                        -self.config["max_brake"], min(self.config["max_throttle"], error * 0.6)
                    )
                    requested = {
                        "steer": steer,
                        "longitudinal": pedal,
                        "target_speed_kmh": target * 3.6,
                        "telemetry_packet_index": sample["packet_index"],
                        "telemetry_received_ns": sample["received_monotonic_ns"],
                        "reference_s_m": station,
                        "mode": "braking"
                        if self.braking is not None
                        else "recovering"
                        if self.recovery_started is not None
                        else "tracking",
                    }
                    self.send(
                        journal,
                        Command(
                            round(steer * 32767),
                            round(max(0, pedal) * 255),
                            round(max(0, -pedal) * 255),
                        ),
                        "baseline" if self.braking is None else "stop_guard",
                        requested,
                    )
            except _StopTracking as stop:
                self.result["stop_reason"] = str(stop)
            except KeyboardInterrupt:
                self.result["stop_reason"] = "user_stop"
            except Exception as error:
                self.result.update(stop_reason="interface_error", error=str(error))
            finally:
                try:
                    self.send(journal, NEUTRAL, "stop_guard", {})
                    self.result["release_sent"] = True
                except Exception as error:
                    self.result.update(
                        prior_stop_reason=self.result["stop_reason"],
                        stop_reason="interface_error",
                        release_error=str(error),
                    )
                try:
                    self.close_attempted = True
                    self.env.close()
                except Exception as error:
                    self.result.setdefault("prior_stop_reason", self.result["stop_reason"])
                    self.result.update(stop_reason="interface_error", close_error=str(error))
                path = directory / "control-final.tmp"
                path.write_text(
                    json.dumps(self.result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
                path.replace(directory / "control.json")


def run_tracking(request: TrackingDrive, environment: ControlEnvironment) -> RunResult:
    from fh5.experiment import Record, run_experiment

    session = None
    stream = None
    try:
        root, route = validate_tracking_file(request.config_file)
        session = _Session(request, root["tracking"], route, environment)
        stream = session.stream()
        return run_experiment(
            Record(request.config_file, request.output_dir, environment.source_kind), packets=stream
        )
    finally:
        if stream is not None:
            stream.close()
        if session is None or not session.close_attempted:
            environment.close()
