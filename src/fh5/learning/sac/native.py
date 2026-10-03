"""Restart, requalify the frozen SAC candidate, then acquire its driving lease."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from fh5.artifacts.io import encode, sha256_file, write_file
from fh5.commands.numeric_drive import native_driving_environment
from fh5.driving.config import NumericDriveConfiguration
from fh5.driving.events import EventEnvironment, EventRun
from fh5.driving.realtime.model import RealtimeEnvironment
from fh5.driving.realtime.shadow import ShadowEnvironment
from fh5.evaluation.attempts import _task
from fh5.evaluation.handoff import ReadyHandoff
from fh5.evaluation.native import _EventLease
from fh5.evaluation.native_io import native_event_environment
from fh5.evaluation.start import event_payloads
from fh5.learning.loop.environments import LearningUnavailable, _StopMenu
from fh5.learning.sac.realtime_sampler import SACRealtimeStart, _AttemptDrive


class NativeSACSamplingEnvironment:
    source_kind: Literal["native"] = "native"

    def __init__(
        self,
        configuration: Path,
        event_config: Path,
        *,
        shadow_seconds: float,
        handoff_timeout_s: float,
        initial_operation: Literal["start_ready", "restart_ready"] = "restart_ready",
        review: Callable[[Path], Path | None] | None = None,
        shadow_factory: Callable[[NumericDriveConfiguration], RealtimeEnvironment] = (
            ShadowEnvironment.from_native
        ),
        menu_factory: Callable[[Path, NumericDriveConfiguration], EventEnvironment] = (
            native_event_environment
        ),
        driving_factory: Callable[[NumericDriveConfiguration], RealtimeEnvironment] = (
            native_driving_environment
        ),
    ) -> None:
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (shadow_seconds, handoff_timeout_s)
        ) or initial_operation not in ("start_ready", "restart_ready"):
            raise ValueError(
                "Native sampling needs positive shadow/handoff durations and operation"
            )
        self.template = json.loads(configuration.read_bytes())
        self.template["capture_config"] = str(
            (configuration.parent / self.template["capture_config"]).resolve()
        )
        self.template["task"]["route_file"] = str(
            (configuration.parent / self.template["task"]["route_file"]).resolve()
        )
        self.event_assets = event_payloads(event_config)
        self.event = json.loads(self.event_assets["start/event.json"])
        self.shadow_seconds, self.handoff_timeout_s = shadow_seconds, handoff_timeout_s
        self.initial_operation, self.review = initial_operation, review
        self.shadow_factory, self.menu_factory, self.driving_factory = (
            shadow_factory,
            menu_factory,
            driving_factory,
        )
        self.observations: list[_AttemptDrive] = []
        self.menus: list[_EventLease] = []
        self.errors: list[str] = []
        self.resources_released = True
        self.commands_sent = 0
        self.menu_sends = 0
        self.attempts = 0

    def _plan(
        self,
        start: SACRealtimeStart,
        path: Path,
        *,
        exploratory: bool,
        expected_conditions: dict[str, Any] | None,
    ) -> NumericDriveConfiguration:
        from fh5.telemetry.packet import validate_record_config as _validate_config

        plan = NumericDriveConfiguration(
            path, start.request.output_dir, start.request.seconds, start.request.live
        )
        record = _validate_config(json.loads(start.recording_config.read_bytes()))
        task = _task(start.task_file)
        event = self.event["event_run"]
        if (
            not start.request.live
            or plan.request != start.request
            or plan.model_hash != start.expected_sha256
            or plan.exploration_seed != (start.seed if exploratory else None)
            or record["control_source"] != "policy"
            or task["control_owner"] != "policy"
            or record["snapshot"] != self.event["snapshot"]
            or not event["conditions_verified"]
            or event["purpose"] != "event"
            or task["route_sha256"] != plan.task.expected_route_sha256
            or sha256_file(start.task_file.parent / task["route_file"]) != task["route_sha256"]
            or any(
                event[key] != task[key] or task[key] != getattr(start.request.config, key)
                for key in ("expected_car_ordinal", "expected_pi")
            )
        ):
            raise ValueError("Native sampling runtime, route or event/record conditions differ")
        if expected_conditions is not None and plan.bindings["conditions"] != expected_conditions:
            raise ValueError("Native preparation differs from frozen input conditions")
        if task["version"] == 2 and (
            sha256_file(start.task_file.parent / task["automatic_start"]["event_file"])
            != task["automatic_start"]["event_sha256"]
            or event_payloads(start.task_file.parent / task["automatic_start"]["event_file"])
            != self.event_assets
            or task["automatic_start"]["handoff_timeout_s"] != self.handoff_timeout_s
        ):
            raise ValueError("Native sampling differs from the automatic start task")
        return plan

    def prepare(
        self,
        start: SACRealtimeStart,
        *,
        exploratory: bool = True,
        expected_conditions: dict[str, Any] | None = None,
    ) -> tuple[NumericDriveConfiguration, dict[str, Any]]:
        """Restart and qualify this exact candidate before opening its driving lease."""
        from fh5.experiment import run_experiment

        root = start.request.output_dir.parent

        def check_stop() -> None:
            if start.stopped():
                raise InterruptedError("Native sampling stopped before device acquisition")

        try:
            check_stop()
            if not self.close()["resources_released"]:
                raise ValueError("Previous native sampling resources remain unreleased")
            document = json.loads(json.dumps(self.template))
            document["model"] = {
                "kind": "sac",
                "directory": str(start.checkpoint.resolve()),
                "device": "cpu",
                "manifest_sha256": start.expected_sha256,
                **({"exploration_seed": start.seed} if exploratory else {}),
            }
            document["shadow"] = None
            config_file = root / "shadow-config.json"
            write_file(config_file, encode(document))
            plan = self._plan(
                start,
                config_file,
                exploratory=exploratory,
                expected_conditions=expected_conditions,
            )
            if plan.qualification["reasons"] != ["missing_shadow_evidence"]:
                plan.require_eligible()
            for name, payload in self.event_assets.items():
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                write_file(destination, payload)
            event_file = root / "start/event.json"
            check_stop()
            try:
                menu_source = self.menu_factory(event_file, plan)
            except (Exception, KeyboardInterrupt) as error:
                self.errors.append(f"Menu acquisition cleanup unconfirmed: {error}")
                raise
            menu = _EventLease(menu_source)
            self.menus.append(menu)
            if menu.source_kind != "udp" or event_payloads(event_file) != self.event_assets:
                raise ValueError("Native sampling menu source or frozen assets changed")
            check_stop()
            prepared = run_experiment(
                EventRun(
                    event_file,
                    root / "ready",
                    operation=self.initial_operation if self.attempts == 0 else "restart_ready",
                    live=True,
                ),
                event_environment=_StopMenu(menu, start.stopped),
            ).summary["event_run"]
            menu.close()
            if prepared["stop_reason"] == "user_stop":
                raise InterruptedError("Native sampling stopped during menu preparation")
            if (
                not prepared["ready_verified"]
                or not prepared["release_sent"]
                or prepared["evidence_status"] != "complete"
            ):
                raise ValueError("Native sampling restart was not confirmed")
            check_stop()
            previous_bindings = plan.bindings
            plan = self._plan(
                start,
                config_file,
                exploratory=exploratory,
                expected_conditions=expected_conditions,
            )
            if (
                plan.bindings != previous_bindings
                or event_payloads(event_file) != self.event_assets
            ):
                raise ValueError("Native sampling conditions changed during preparation")
            shadow_plan = NumericDriveConfiguration(
                config_file, root / "shadow", self.shadow_seconds, False, mode="shadow"
            )
            check_stop()
            try:
                shadow_source = self.shadow_factory(shadow_plan)
            except (Exception, KeyboardInterrupt) as error:
                self.errors.append(f"Shadow acquisition cleanup unconfirmed: {error}")
                raise
            shadow = _AttemptDrive(shadow_source, start.stopped)
            self.observations.append(shadow)
            if shadow.source_kind != "shadow":
                raise ValueError("Native sampling requires a read-only shadow adapter")
            shadow_result = run_experiment(
                shadow_plan.request,
                realtime_environment=shadow,
                numeric_actor_factory=shadow_plan.actor,
            ).summary["realtime"]
            if not shadow_result["resources_released"]:
                raise ValueError("Shadow resources were not released before driving acquisition")
            if shadow_result["stop_reason"] == "user_stop":
                raise InterruptedError("Native sampling stopped during shadow qualification")
            check_stop()
            document["shadow"] = {"directory": str((root / "shadow").resolve())}
            config_file = root / "drive.json"
            write_file(config_file, encode(document))
            plan = self._plan(
                start,
                config_file,
                exploratory=exploratory,
                expected_conditions=expected_conditions,
            )
            plan.require_eligible()
            if not self.close()["resources_released"]:
                raise ValueError("Native preparation resources remain unreleased")
            return plan, prepared
        except (Exception, KeyboardInterrupt) as error:
            released = self.close()
            if isinstance(error, LearningUnavailable):
                error.resources_released &= released["resources_released"]
                raise
            raise LearningUnavailable(
                str(error), resources_released=released["resources_released"]
            ) from error

    def start(self, start: SACRealtimeStart) -> RealtimeEnvironment:
        def check_stop() -> None:
            if start.stopped():
                raise InterruptedError("Native sampling stopped before driving acquisition")

        try:
            plan, prepared = self.prepare(start)
            check_stop()
            try:
                driving_source = self.driving_factory(plan)
            except (Exception, KeyboardInterrupt) as error:
                self.errors.append(f"Driving acquisition cleanup unconfirmed: {error}")
                raise
            drive = _AttemptDrive(driving_source, start.stopped)
            self.observations.append(drive)
            if drive.source_kind != "native":
                raise ValueError("Native sampling requires a qualified native driving adapter")
            check_stop()
            self.attempts += 1
            return ReadyHandoff(
                drive,
                prepared["ready_state"],
                self.event["event_run"],
                start.request.config,
                deadline_ns=round(prepared["ready_observed_ns"] + self.handoff_timeout_s * 1e9),
            )
        except (Exception, KeyboardInterrupt) as error:
            released = self.close()
            if isinstance(error, LearningUnavailable):
                error.resources_released &= released["resources_released"]
                raise
            raise LearningUnavailable(
                str(error), resources_released=released["resources_released"]
            ) from error

    def finish(self, recording_dir: Path) -> Path | None:
        if not self.close()["resources_released"]:
            raise ValueError("Native sampling resources were not released before review")
        return self.review(recording_dir) if self.review else None

    def close(self) -> dict[str, Any]:
        while self.observations:
            observation = self.observations.pop()
            try:
                result = observation.close()
                self.resources_released &= result.get("resources_released") is True
                self.commands_sent += result.get("controller_sends", 0)
            except (Exception, KeyboardInterrupt) as error:
                self.errors.append(str(error))
        while self.menus:
            menu = self.menus.pop()
            try:
                menu.close()
            except (Exception, KeyboardInterrupt) as error:
                self.errors.append(str(error))
            self.commands_sent += menu.sends
            self.menu_sends += menu.sends
        return {
            "resources_released": self.resources_released and not self.errors,
            "commands_sent_to_game": self.commands_sent > 0,
            "menu_sends": self.menu_sends,
            "errors": list(self.errors),
        }
