"""Native repeated evaluation, qualified before acquiring any external resource."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fh5.collection_store import read_bounded
from fh5.events import EventEnvironment, EventInput
from fh5.numeric_drive_config import NumericDriveConfiguration
from fh5.realtime import RealtimeEnvironment

if TYPE_CHECKING:
    from fh5.evaluation_run import EvaluationRun


class _EventLease:
    """Keep menu release failures visible and make parent cleanup idempotent."""

    def __init__(self, source: EventEnvironment) -> None:
        self.source = source
        self.source_kind = source.source_kind
        self.closed = False
        self.error: str | None = None
        self.sends = 0

    @property
    def events(self) -> list[dict[str, Any]]:
        return getattr(self.source, "events", [])

    def now_ns(self) -> int:
        return self.source.now_ns()

    def read(self, period_s: float) -> EventInput:
        return self.source.read(period_s)

    def pulse(self, button: str) -> None:
        self.source.pulse(button)
        self.sends += 1

    def release(self) -> None:
        self.source.release()
        self.sends += 1

    def close(self) -> None:
        if not self.closed:
            try:
                self.source.close()
            except (Exception, KeyboardInterrupt) as error:
                self.error = f"{type(error).__name__}: {error}"
                raise
            finally:
                self.closed = True
        if self.error:
            raise OSError(self.error)


class NativeEvaluationEnvironment:
    source_kind: Literal["native"] = "native"

    def __init__(
        self,
        configuration: Path,
        *,
        menu_factory: Callable[[Path, NumericDriveConfiguration], EventEnvironment],
        driving_factory: Callable[[NumericDriveConfiguration], RealtimeEnvironment],
    ) -> None:
        self.configuration = configuration
        self.menu_factory, self.driving_factory = menu_factory, driving_factory
        self.request: EvaluationRun | None = None
        self.plan: NumericDriveConfiguration | None = None
        self.bindings: dict[str, Any] = {}
        self.slots: list[str] = []
        self.menus: list[_EventLease] = []
        self.drives: list[RealtimeEnvironment] = []
        self.acquisition_errors: list[str] = []
        self.device = "unqualified"

    def prepare(self, request: EvaluationRun, batch: dict[str, Any]) -> dict[str, Any]:
        if not request.live or not 0.1 <= request.seconds <= 30 or batch["version"] != 3:
            raise ValueError(
                "Native evaluation requires live opt-in, v3 batch and at most 30 seconds"
            )
        plan = NumericDriveConfiguration(
            self.configuration, request.output_dir, request.seconds, True
        )
        plan.require_eligible()
        if plan.exploration_seed is not None:
            raise ValueError("Native evaluation requires deterministic policy execution")
        config = batch["config"]
        task = json.loads(read_bounded(request.batch_dir / "task.json", 1024**2))
        runtime = {**asdict(plan.request.config), "pixels": plan.capture.pixels.metadata()}
        if (
            config["model"]["kind"] != plan.model_kind
            or config["model"]["manifest_sha256"] != plan.model_hash
            or config["model"]["device"] != plan.device
            or config["runtime"] != json.loads(json.dumps(runtime))
            or config["conditions"]["numeric_input_conditions"] != plan.bindings["conditions"]
            or task["version"] != 2
            or task["route_sha256"] != plan.task.expected_route_sha256
            or task["control_owner"] != "policy"
        ):
            raise ValueError("Native driving configuration differs from frozen evaluation batch")
        self.request, self.plan = request, plan
        self.device, self.bindings = plan.device, plan.bindings
        self.slots = [slot["id"] for slot in config["plan"]]
        return {"bindings": self.bindings, "qualification": plan.qualification}

    def _attempt(self, slot: str) -> NumericDriveConfiguration:
        if self.request is None or self.plan is None or slot not in self.slots:
            raise ValueError("Native evaluation is not prepared")
        if (
            hashlib.sha256(read_bounded(self.configuration, 1024**2)).hexdigest()
            != self.bindings["drive_config_sha256"]
        ):
            raise ValueError("Native evaluation configuration changed")
        plan = NumericDriveConfiguration(
            self.configuration,
            self.request.output_dir / f"attempt-{self.slots.index(slot):04d}/execution",
            self.request.seconds,
            True,
        )
        plan.require_eligible()
        if plan.bindings != self.bindings:
            raise ValueError("Native evaluation conditions changed")
        return plan

    def event(self, slot_id: str) -> EventEnvironment:
        plan = self._attempt(slot_id)
        assert self.request is not None
        try:
            source = self.menu_factory(self.request.output_dir / "event.json", plan)
        except (Exception, KeyboardInterrupt) as error:
            self.acquisition_errors.append(f"Menu acquisition cleanup unconfirmed: {error}")
            raise
        lease = _EventLease(source)
        self.menus.append(lease)
        if lease.source_kind != "udp":
            lease.close()
            raise ValueError("Native preparation requires original UDP evidence")
        return lease

    def driving(self, slot_id: str, ready_state: dict[str, Any]) -> RealtimeEnvironment:
        if not self.menus or not self.menus[-1].closed or self.menus[-1].error:
            raise ValueError("Menu resources must be released before driving")
        plan = self._attempt(slot_id)
        try:
            drive = self.driving_factory(plan)
        except (Exception, KeyboardInterrupt) as error:
            self.acquisition_errors.append(f"Driving acquisition cleanup unconfirmed: {error}")
            raise
        self.drives.append(drive)
        if drive.source_kind != "native":
            drive.close()
            raise ValueError("Native evaluation requires a qualified native driving adapter")
        return drive

    def close(self) -> dict[str, Any]:
        errors = list(self.acquisition_errors)
        released = True
        for drive in self.drives:
            try:
                released = drive.close().get("resources_released", False) and released
            except Exception as error:
                errors.append(str(error))
        for menu in self.menus:
            try:
                menu.close()
            except Exception as error:
                errors.append(str(error))
        return {
            "resources_released": released and not errors,
            "errors": errors,
            "menu_sends": sum(menu.sends for menu in self.menus),
        }
