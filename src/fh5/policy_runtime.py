"""One bounded attempt. Only the caller thread owns actuator writes and journals."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

from fh5.control import Command
from fh5.policy_writer import PolicyImageWriter
from fh5.routes import locate_route

if TYPE_CHECKING:
    from fh5.experiment import Packet
    from fh5.policy import PolicyActor, PolicyEnvironment, PolicyInput


class StopAttempt(Exception):
    pass


class FinishPolicy(Exception):
    """Cancel the pending inference, then let the stop guard finish braking."""


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


class PolicySession:
    def __init__(
        self,
        directory: Path,
        config: dict[str, Any],
        route: dict[str, Any],
        reference: dict[str, Any] | None,
        settings: dict[str, Any],
        environment: PolicyEnvironment,
        actor: PolicyActor,
    ) -> None:
        self.directory, self.config, self.route = directory, config, route
        self.reference, self.settings, self.env, self.actor = (
            reference,
            settings,
            environment,
            actor,
        )
        self.frames: list[dict[str, Any]] = []
        self.pixels: dict[str, bytes] = {}
        self.actions: list[dict[str, Any]] = []
        self.latest: dict[str, Any] | None = None
        self.geometry: dict[str, Any] = {}
        self.prior_sample: dict[str, Any] | None = None
        self.task_state: dict[str, Any] = {"anchor_s_m": config.get("start_station_m", 0)}
        self.prior_state: dict[str, Any] = {}
        self.started = self.env.now_ns()
        self.boundary_ns = self.started
        self.clock_advanced_ns = self.started
        self.clock_advances = 0
        self.armed: int | None = None
        self.braking: int | None = None
        self.focused = False
        self.packet_index = self.frame_index = self.written = 0
        self.worker: threading.Thread | None = None
        self.result: dict[str, Any] = {
            "version": 1,
            "stop_reason": "running",
            "release_sent": False,
            "commands": [],
            "adapter_events": environment.events,
            "decisions": [],
            "model": actor.manifest,
            "actor_kind": actor.kind,
            "config": config,
            "reference_mode": config["reference_mode"],
            "geometry_completed": False,
            "formal_validity": "pending_independent_review",
            "game_response_validation": "unverified",
            "unattended_allowed": False,
            "started_ns": self.started,
            "source_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
        }

    def send(self, command: Command, owner: str, requested: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "issued_ns": self.env.now_ns(),
            "owner": owner,
            "requested": requested,
            "target": asdict(command),
            "sent": None,
            "status": "failed",
        }
        try:
            self.env.send(command)
            row.update(sent=asdict(command), status="sent")
        except Exception as error:
            row["error"] = str(error)
            raise
        finally:
            row["returned_ns"] = self.env.now_ns()
            self.result["commands"].append(row)
            self.cj.write(json.dumps(row, allow_nan=False) + "\n")
            self.cj.flush()
        return row

    def receive(self, batch: PolicyInput, *, guard: bool = True) -> Iterator[Packet]:
        from fh5.experiment import _decode

        now = self.env.now_ns()
        if guard and self.writer.error:
            self.result["image_writer_error"] = self.writer.error
            raise StopAttempt("image_writer_error")
        self.focused = batch.focused
        if guard and (batch.stop_requested or batch.fault):
            # Keep the raw batch even when F8/fault arrives with it.
            for packet in batch.packets:
                yield packet
                self.packet_index += 1
            raise StopAttempt(batch.fault or "user_stop")
        for packet in batch.packets:
            yield packet
            index = self.packet_index
            self.packet_index += 1
            self.written += len(packet.payload) * 2 + 512
            if not guard:
                continue
            s = _decode(packet)
            s.update(packet_index=index, segment=0)
            if s["received_monotonic_ns"] > now or s["speed_mps"] < 0:
                raise StopAttempt("invalid_telemetry")
            if self.latest and s["received_monotonic_ns"] <= self.latest["received_monotonic_ns"]:
                raise StopAttempt("telemetry_clock_discontinuity")
            if self.latest and s["game_timestamp_ms"] > self.latest["game_timestamp_ms"]:
                self.clock_advanced_ns = s["received_monotonic_ns"]
                self.clock_advances += 1
            if self.armed is None and (
                not s["is_race_on"]
                or (
                    self.latest
                    and (
                        not self.latest["is_race_on"]
                        or s["game_timestamp_ms"] < self.latest["game_timestamp_ms"]
                        or s["received_monotonic_ns"] - self.latest["received_monotonic_ns"]
                        > 100_000_000
                    )
                )
            ):
                self.boundary_ns = s["received_monotonic_ns"]
                self.clock_advances = 0
                self.frames.clear()
                self.pixels.clear()
                self.prior_state.clear()
            self.latest = s
            if self.armed is not None:
                locate_route([s], self.route, state=self.task_state)
            else:
                locate_route(
                    [s], self.route, state={"anchor_s_m": self.config.get("start_station_m", 0)}
                )
            self.geometry = dict(s["route"])
            if self.reference:
                prior = dict(s)
                locate_route([prior], self.reference, state=self.prior_state)
                self.prior_sample = prior
            if self.armed is not None:
                self.check()
        for event in batch.events:
            if not self.writer.append(event):
                raise StopAttempt("image_writer_backpressure")
            if self.armed is None and event.get("kind") in (
                "focus_lost",
                "focus_restored",
                "capture_discarded",
            ):
                self.boundary_ns = now
                self.frames.clear()
                self.pixels.clear()
                self.prior_state.clear()
            if guard and self.armed is not None and event.get("kind") == "capture_discarded":
                raise StopAttempt("capture_discarded")
        if guard and self.armed is not None and not batch.focused:
            raise StopAttempt("focus_lost")
        if batch.frame and batch.focused:
            f = batch.frame
            clocks = [f.capture_start_ns, f.capture_end_ns, f.available_ns, now]
            if any(type(v) is not int or v < 0 for v in clocks) or clocks != sorted(clocks):
                raise StopAttempt("invalid_image_clock")
            if self.frames and f.capture_start_ns <= self.frames[-1]["capture_start_ns"]:
                raise StopAttempt("invalid_image_clock")
            if f.codec not in ("jpeg", "png") or f.color != "RGB":
                raise StopAttempt("invalid_image_format")
            relative = f"frames/{self.frame_index:06d}.{f.codec}"
            self.frame_index += 1
            self.written += len(f.encoded) + 2048
            if self.written > 256 * 1024**2:
                raise StopAttempt("byte_limit")
            row = {k: v for k, v in asdict(f).items() if k != "encoded"}
            row.update(
                kind="frame",
                path=relative,
                sha256=hashlib.sha256(f.encoded).hexdigest(),
                delivered_ns=now,
            )
            if f.capture_start_ns >= self.boundary_ns:
                self.frames.append(row)
                self.pixels[relative] = f.encoded
            if not self.writer.append(row, f.encoded):
                raise StopAttempt("image_writer_backpressure")
            # Keep sufficient history in memory; all frames remain on disk.
            horizon = (
                max(self.settings["history_offsets_ms"]) + self.settings["max_image_age_ms"] + 1000
            ) * 1e6
            while len(self.frames) > 1 and now - self.frames[0]["delivered_ns"] > horizon:
                self.pixels.pop(self.frames.pop(0)["path"])
        if guard and (not self.focused or not self.latest or not self.latest["is_race_on"]):
            if self.armed is not None:
                raise StopAttempt("inactive")
            self.frames.clear()
            self.pixels.clear()
            self.prior_state.clear()
        if guard and self.armed is not None:
            self.check()
            self.brake_if_due()

    def check(self) -> None:
        s, cfg = self.latest, self.config
        if s is None:
            raise StopAttempt("missing_telemetry")
        if not self.focused:
            raise StopAttempt("focus_lost")
        if not s["is_race_on"]:
            raise StopAttempt("inactive")
        if (
            s["car_ordinal"] != cfg["expected_car_ordinal"]
            or s["car_performance_index"] != cfg["expected_pi"]
        ):
            raise StopAttempt("unexpected_vehicle")
        if (
            self.env.now_ns() - s["received_monotonic_ns"]
            > self.settings["max_telemetry_age_ms"] * 1e6
        ):
            raise StopAttempt("stale_telemetry")
        if not s["motion"]:
            raise StopAttempt("invalid_motion")
        if self.env.now_ns() - self.clock_advanced_ns > 100_000_000:
            raise StopAttempt("stalled_game_clock")
        if s["speed_kmh"] >= cfg["max_speed_kmh"]:
            raise StopAttempt("speed_limit")
        if self.geometry["status"] not in ("matched", "awaiting_checkpoint"):
            raise StopAttempt("task_location_untrusted")
        if (self.directory / "STOP").exists():
            raise StopAttempt("stop_file")

    def brake_if_due(self) -> bool:
        if self.braking is not None:
            return True
        if self.armed is None:
            return False
        reached = (
            self.geometry["confirmed_progress_m"]
            >= self.route["length_m"] - self.config.get("end_margin_m", 0) - 1e-6
        )
        if not reached and self.env.now_ns() - self.armed < self.config["max_duration_s"] * 1e9:
            return False
        self.result["geometry_completed"] = reached
        self.result["stop_reason"] = "local_end" if reached else "time_limit"
        self.braking = self.env.now_ns()
        if self.latest and self.latest["speed_kmh"] > 0.5:
            self.send(Command(0, 0, round(self.config["max_brake"] * 255)), "stop_guard", {})
        else:
            self.send(Command(0, 0, 0), "stop_guard", {})
        return True

    def infer(self, decision: dict[str, Any]) -> Iterator[Packet]:
        observation = decision["observation"]
        pixels = [self.pixels[f["path"]] for f in observation["images"]]
        output: Queue[Any] = Queue(maxsize=1)

        def predict() -> None:
            try:
                output.put((True, self.actor.predict(observation["actor"], pixels)))
            except BaseException as error:
                output.put((False, str(error)))

        self.worker = threading.Thread(target=predict, daemon=True, name="fh5-policy-inference")
        self.worker.start()
        deadline = decision["decision_ns"] + self.config["inference_timeout_ms"] * 1e6
        while True:
            if self.brake_if_due():
                raise FinishPolicy(self.result["stop_reason"])
            if self.env.now_ns() > deadline:
                raise StopAttempt("inference_timeout")
            try:
                ok, value = output.get(timeout=0.001)
            except Empty:
                yield from self.receive(self.env.read(0.01))
                continue
            decision["inference_returned_ns"] = self.env.now_ns()
            if not ok:
                raise ValueError("Policy inference failed: " + value)
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1 for v in value
                )
            ):
                raise StopAttempt("invalid_prediction")
            decision["prediction"] = value
            if self.env.now_ns() > deadline:
                raise StopAttempt("inference_timeout")
            # Recheck the most recent environment after inference, before actuator send.
            yield from self.receive(self.env.read(0.005))
            if self.brake_if_due():
                raise FinishPolicy(self.result["stop_reason"])
            now = self.env.now_ns()
            if now - decision["decision_ns"] > self.config["max_command_age_ms"] * 1e6:
                raise StopAttempt("stale_command")
            if (
                now - observation["telemetry_received_ns"]
                > self.settings["max_telemetry_age_ms"] * 1e6
            ):
                raise StopAttempt("stale_decision_telemetry")
            for offset, frame in zip(self.settings["history_offsets_ms"], observation["images"]):
                if (
                    now - frame["capture_start_ns"]
                    > (offset + self.settings["max_image_age_ms"]) * 1e6
                ):
                    raise StopAttempt("stale_decision_image")
            return

    def stream(self) -> Iterator[Packet]:
        from fh5.policy import _observation

        (self.directory / "frames").mkdir()
        (self.directory / "vision-config.json").write_text(
            json.dumps(self.config), encoding="utf-8"
        )
        self.writer = PolicyImageWriter(self.directory, self.env.now_ns)
        with (
            (self.directory / "commands.jsonl").open("x", encoding="utf-8") as self.cj,
            (self.directory / "policy-decisions.jsonl").open("x", encoding="utf-8") as dj,
        ):
            try:
                self.send(Command(0, 0, 0), "stop_guard", {})
                last_decision = 0
                while True:
                    yield from self.receive(self.env.read(0.02))
                    now = self.env.now_ns()
                    if (
                        self.armed is None
                        and now - self.started > self.config["ready_timeout_s"] * 1e9
                    ):
                        raise StopAttempt("ready_timeout")
                    if not self.latest or not self.focused or not self.latest["is_race_on"]:
                        continue
                    if self.braking is not None:
                        if self.latest["speed_kmh"] <= 0.5:
                            break
                        if now - self.braking >= self.config["braking_s"] * 1e9:
                            raise StopAttempt("not_stopped")
                        self.send(
                            Command(0, 0, round(self.config["max_brake"] * 255)), "stop_guard", {}
                        )
                        continue
                    if now - last_decision < self.settings["period_ms"] * 1e6:
                        continue
                    sample = self.prior_sample if self.reference else self.latest
                    assert sample is not None
                    observation = _observation(
                        self.settings, now, sample, self.frames, self.actions, self.reference
                    )
                    if self.armed is None:
                        a, b = (
                            self.route["points"][0]["position_m"],
                            self.route["points"][1]["position_m"],
                        )
                        heading = math.atan2(b[0] - a[0], b[2] - a[2])
                        yaw = (
                            self.latest["motion"]["yaw_rad"]
                            if self.latest["motion"]
                            else heading + math.pi
                        )
                        heading_error = abs(
                            math.atan2(math.sin(yaw - heading), math.cos(yaw - heading))
                        )
                        if (
                            self.geometry["status"] != "matched"
                            or abs(
                                self.geometry["reference_s_m"]
                                - self.config.get("start_station_m", 0)
                            )
                            > self.config.get("start_tolerance_m", 0.25)
                            or self.latest["speed_kmh"] > self.config["start_speed_kmh"]
                            or not observation["usable"]
                            or heading_error > 0.35
                            or self.clock_advances < 2
                            or now - self.clock_advanced_ns > 100_000_000
                        ):
                            continue
                        self.check()
                        self.armed = now
                        self.result.update(
                            armed_ns=now,
                            start_packet=self.latest["packet_index"],
                            start_station_m=self.geometry["reference_s_m"],
                        )
                        locate_route([self.latest], self.route, state=self.task_state)
                    elif not observation["usable"]:
                        raise StopAttempt(observation["reasons"][0])
                    last_decision = now
                    decision: dict[str, Any] = {
                        "decision_ns": now,
                        "observation": observation,
                        "task_location": self.geometry,
                        "prediction": None,
                        "sent": None,
                        "status": "rejected",
                    }
                    self.result["decisions"].append(decision)
                    try:
                        yield from self.infer(decision)
                        if self.brake_if_due():
                            raise FinishPolicy(self.result["stop_reason"])
                        steer, longitudinal = decision["prediction"]
                        cfg = self.config
                        command = Command(
                            round(max(-cfg["max_steer"], min(cfg["max_steer"], steer)) * 32767),
                            round(max(0, min(cfg["max_throttle"], longitudinal)) * 255),
                            round(max(0, min(cfg["max_brake"], -longitudinal)) * 255),
                        )
                        row = self.send(
                            command, "policy", {"steer": steer, "longitudinal": longitudinal}
                        )
                        decision.update(
                            {
                                k: row[k]
                                for k in ("target", "sent", "issued_ns", "returned_ns", "status")
                            }
                        )
                        self.actions.append(
                            {
                                "occurred_ns": row["issued_ns"],
                                "available_ns": row["returned_ns"],
                                "telemetry_segment": 0,
                                "source": "controller_sent",
                                "steer": command.steer_i16 / 32767,
                                "longitudinal": (command.throttle_u8 - command.brake_u8) / 255,
                            }
                        )
                    except FinishPolicy as error:
                        decision["rejection"] = str(error)
                        continue
                    except StopAttempt as error:
                        decision["rejection"] = str(error)
                        raise
                    finally:
                        dj.write(json.dumps(decision, allow_nan=False) + "\n")
                        dj.flush()
            except StopAttempt as error:
                self.result["stop_reason"] = str(error)
            except (Exception, KeyboardInterrupt) as error:
                self.result.update(stop_reason="interface_error", error=str(error))
            finally:
                try:
                    self.send(Command(0, 0, 0), "stop_guard", {})
                    self.result["release_sent"] = True
                except Exception as error:
                    self.result.update(stop_reason="interface_error", release_error=str(error))
                # Keep a brief passive response window; no policy can regain control.
                try:
                    for _ in range(15):
                        yield from self.receive(self.env.read(0.02), guard=False)
                except (Exception, KeyboardInterrupt) as error:
                    self.result["release_observation_error"] = str(error)
                try:
                    self.result["resources_released"] = self.env.close()
                except Exception as error:
                    self.result.update(resources_released=False, close_error=str(error))
                self.result["inference_worker_released"] = (
                    self.worker is None or not self.worker.is_alive()
                )
                self.result["image_writer_released"] = self.writer.close()
                self.result["environment_released"] = self.result["resources_released"]
                self.result["resources_released"] = (
                    self.result["environment_released"]
                    and self.result["image_writer_released"]
                    and self.result["inference_worker_released"]
                )
                if self.writer.error:
                    self.result["image_writer_error"] = self.writer.error
                    if self.result["stop_reason"] in ("local_end", "time_limit"):
                        self.result["stop_reason"] = "image_writer_error"
                self.result["ended_ns"] = self.env.now_ns()
                self.result["final_task_location"] = self.geometry
                self.result["last_policy_packet"] = (
                    self.latest["packet_index"] if self.latest else None
                )
                write_json(
                    self.directory / "control.json",
                    {k: v for k, v in self.result.items() if k not in ("decisions", "model")},
                )
        self.finalize()

    def finalize(self) -> None:
        def digest(name: str) -> str:
            return hashlib.sha256((self.directory / name).read_bytes()).hexdigest()

        write_json(
            self.directory / "vision-session.json",
            {
                "version": 1,
                "camera_mode": "chase_far",
                "camera_status": "user_reported",
                "camera_pose": "dynamic_unknown",
                "settings": {"max_age_ms": self.settings["max_image_age_ms"]},
                "resources_released": self.result["resources_released"],
                "budget_bytes_used": self.written,
                "started_ns": self.started,
                "ended_ns": self.result["ended_ns"],
                "stop_reason": self.result["stop_reason"],
                "hashes": {
                    name: digest(name)
                    for name in ("vision-config.json", "vision.jsonl", "packets.jsonl")
                },
            },
        )
        self.result["hashes"] = {
            name: digest(name)
            for name in (
                "packets.jsonl",
                "commands.jsonl",
                "control.json",
                "vision.jsonl",
                "vision-session.json",
                "policy-decisions.jsonl",
            )
        }
        write_json(self.directory / "policy.json", self.result)
