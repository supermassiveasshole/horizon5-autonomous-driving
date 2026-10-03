"""Numerical observation adapter with an independently guarded output lease."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from fh5.driving.control import Command
from fh5.driving.realtime.model import RealtimeRun, TimelineInput
from fh5.driving.realtime.shadow import ShadowEnvironment
from fh5.driving.windows import NEUTRAL, Controller, LiveEnvironment

if TYPE_CHECKING:
    from fh5.driving.config import NumericDriveConfiguration


class NumericDrivingEnvironment:
    def __init__(
        self,
        observations: ShadowEnvironment,
        controller_factory: Callable[[], Controller],
        *,
        source_kind: Literal["synthetic", "native"] = "native",
        configuration: NumericDriveConfiguration | None = None,
    ) -> None:
        if source_kind not in ("synthetic", "native"):
            raise ValueError("Unknown numerical driving source")
        self.observations = observations
        self.controller_factory = controller_factory
        self.source_kind = source_kind
        self.configuration = configuration
        self.authorized = False
        self.control: LiveEnvironment | None = None
        self.controller: Controller | None = None
        self.latest: TimelineInput | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._result: dict[str, Any] | None = None
        self._acquiring = False
        self._prepared_ns: int | None = None
        self._preparation_fault: str | None = None
        self.sent_count = self.failed_count = 0

    def authorize(
        self, request: RealtimeRun, manifest: dict[str, Any], inference_device: str | None
    ) -> None:
        if self.configuration is None:
            raise ValueError("Native driving requires qualified input and shadow bindings")
        self.configuration.authorize(request, manifest, inference_device)
        if (
            self.observations.request != request
            or self.observations.capture_config != self.configuration.capture
            or self.observations.task != self.configuration.task
            or self.observations.input_conditions != self.configuration.bindings
        ):
            raise ValueError("Native observation adapter differs from qualified conditions")
        self.authorized = True

    def _check_ready(self, *, creating: bool = False) -> bool:
        value, cfg = self.latest, self.observations.request.config
        focused, stop = self.signals()
        if stop or not focused:
            raise OSError("user_stop" if stop else "focus_lost")
        if value is None or not self.observations.ready:
            if creating:
                return False
            raise OSError("Driving observation is not ready")
        safety = value.safety
        if self.observations.fault or safety.fault or safety.task_fault or not safety.active:
            raise OSError(
                self.observations.fault or safety.fault or safety.task_fault or "inactive"
            )
        now = time.perf_counter_ns()
        if not 0 <= now - safety.received_ns <= cfg.max_telemetry_age_ms * 1_000_000:
            if creating:
                return False
            raise OSError("stale_telemetry")
        if creating:
            if value.observation is None:
                return False
            if (
                not 0
                <= now - value.observation.frames[-1].source_time_ns
                <= cfg.max_image_age_ms * 1_000_000
            ):
                return False
        return True

    def prepare_control(self) -> None:
        """Acquire once on the runtime caller while its input worker keeps reading."""
        with self._lock:
            if self._closed:
                raise OSError("Numerical driving environment closed")
            if self.controller is not None or self._acquiring:
                return
            if self.latest is None or self.latest.observation is None:
                return
            if not self._check_ready(creating=True):
                return
            self._acquiring = True
        try:
            controller = self.controller_factory()
            with self._lock:
                # Preserve ownership even if starting the device supervisor fails.
                self.controller = controller
                self.control = LiveEnvironment(
                    None, controller, self.observations.desktop, monitor_idle=True
                )
                self._prepared_ns = time.perf_counter_ns()
        finally:
            self._acquiring = False

    def read(self, period_s: float) -> TimelineInput:
        if self.source_kind == "native" and not self.authorized:
            raise OSError("Native numerical driving has not been qualified")
        value = self.observations.read(period_s)
        with self._lock:
            if self._closed:
                raise OSError("Numerical driving environment closed")
            self.latest = value
            self.signals()
            if self._acquiring or self.controller is not None:
                if value.safety.stop_requested:
                    self._preparation_fault = "user_stop"
                elif not value.safety.focused:
                    self._preparation_fault = self._preparation_fault or "focus_lost"
            if self._preparation_fault:
                return replace(
                    value,
                    safety=replace(value.safety, fault=self._preparation_fault),
                    observation=None,
                )
            if self.control and self.control.fault:
                return replace(value, safety=replace(value.safety, fault=self.control.fault))
            if (
                self._prepared_ns is None
                or value.safety.received_ns <= self._prepared_ns
                or value.observation is not None
                and value.observation.frames[-1].source_time_ns <= self._prepared_ns
            ):
                return replace(value, observation=None)
        return value

    def signals(self) -> tuple[bool, bool]:
        focused, stop = self.observations.signals()
        if self._acquiring and (stop or not focused):
            self._preparation_fault = (
                "user_stop" if stop else self._preparation_fault or "focus_lost"
            )
        return (
            focused and self._preparation_fault != "focus_lost",
            stop or self._preparation_fault == "user_stop",
        )

    def send(self, command: Command) -> None:
        with self._lock:
            if self._closed:
                raise OSError("Numerical driving environment closed")
            if command != NEUTRAL:
                self._check_ready()
            if self.control is None:
                if command != NEUTRAL:
                    raise OSError("Controller not ready")
                return
            try:
                self.control.send(command)
                self.sent_count += 1
            except Exception:
                self.failed_count += 1
                raise

    def close(self) -> dict[str, Any]:
        with self._lock:
            if self._result is not None:
                return self._result
            self._closed = True
            release: dict[str, Any] = {
                "created": self.controller is not None,
                "resources_released": True,
                "events": [],
            }
            output = self.control or self.controller
            if output:
                for _ in range(3):
                    event: dict[str, Any] = {
                        "owner": "adapter_shutdown",
                        "issued_ns": time.perf_counter_ns(),
                        "status": "failed",
                    }
                    try:
                        output.send(NEUTRAL)
                        event["status"] = "sent"
                    except Exception as error:
                        event["error"] = str(error)
                    finally:
                        event["returned_ns"] = time.perf_counter_ns()
                        release["events"].append(event)
                    if event["status"] == "sent":
                        break
                try:
                    output.close()
                except Exception as error:
                    release.update(resources_released=False, close_error=str(error))
                release["watchdog_events"] = list(self.control.events) if self.control else []
            # Release the actuator before joining capture workers or flushing data.
            try:
                result = self.observations.close()
            except Exception as error:
                result = {"resources_released": False, "close_error": str(error)}
            release_events = release["events"] + release.get("watchdog_events", [])
            result.update(
                mode="numeric_driving",
                controller_created=self.controller is not None,
                controller=release,
                controller_sends=self.sent_count
                + sum(e["status"] == "sent" for e in release_events),
                controller_send_failures=self.failed_count
                + sum(e["status"] == "failed" for e in release_events),
                runtime_send_returns=self.sent_count,
                runtime_send_errors=self.failed_count,
                qualification=self.configuration.qualification if self.configuration else None,
                resources_released=bool(
                    result["resources_released"] and release["resources_released"]
                ),
            )
            self._result = result
            return result
