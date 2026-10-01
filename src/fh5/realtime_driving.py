"""Numerical observation adapter with an independently guarded output lease."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal

from fh5.control import Command
from fh5.live import NEUTRAL, Controller, LiveEnvironment
from fh5.realtime import TimelineInput
from fh5.realtime_shadow import ShadowEnvironment


class NumericDrivingEnvironment:
    def __init__(
        self,
        observations: ShadowEnvironment,
        controller_factory: Callable[[], Controller],
        *,
        source_kind: Literal["synthetic", "native"] = "native",
    ) -> None:
        if source_kind not in ("synthetic", "native"):
            raise ValueError("Unknown numerical driving source")
        self.observations = observations
        self.controller_factory = controller_factory
        self.source_kind = source_kind
        self.control: LiveEnvironment | None = None
        self.latest: TimelineInput | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._result: dict[str, Any] | None = None
        self.sent_count = self.failed_count = 0

    def _check_ready(self, *, creating: bool = False) -> None:
        value, cfg = self.latest, self.observations.request.config
        focused, stop = self.signals()
        if stop or not focused:
            raise OSError("user_stop" if stop else "focus_lost")
        if value is None or not self.observations.ready:
            raise OSError("Driving observation is not ready")
        safety = value.safety
        if self.observations.fault or safety.fault or safety.task_fault or not safety.active:
            raise OSError(
                self.observations.fault or safety.fault or safety.task_fault or "inactive"
            )
        now = time.perf_counter_ns()
        if not 0 <= now - safety.received_ns <= cfg.max_telemetry_age_ms * 1_000_000:
            raise OSError("stale_telemetry")
        if creating:
            if value.observation is None:
                raise OSError("Driving observation is not ready")
            if (
                not 0
                <= now - value.observation.frames[-1].source_time_ns
                <= cfg.max_image_age_ms * 1_000_000
            ):
                raise OSError("stale_image")

    def read(self, period_s: float) -> TimelineInput:
        value = self.observations.read(period_s)
        with self._lock:
            if self._closed:
                raise OSError("Numerical driving environment closed")
            self.latest = value
            if self.control is None and value.observation is not None:
                self._check_ready(creating=True)
                controller = self.controller_factory()
                try:
                    self.control = LiveEnvironment(
                        None, controller, self.observations.desktop, monitor_idle=True
                    )
                except BaseException:
                    controller.close()
                    raise
                # Device creation can be slow. No prediction precedes this call,
                # and the runner will check the returned observation's age again.
                self._check_ready(creating=True)
            if self.control and self.control.fault:
                return replace(value, safety=replace(value.safety, fault=self.control.fault))
        return value

    def signals(self) -> tuple[bool, bool]:
        return self.observations.signals()

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
                "created": self.control is not None,
                "resources_released": True,
                "events": [],
            }
            if self.control:
                for _ in range(3):
                    event: dict[str, Any] = {
                        "owner": "adapter_shutdown",
                        "issued_ns": time.perf_counter_ns(),
                        "status": "failed",
                    }
                    try:
                        self.control.send(NEUTRAL)
                        event["status"] = "sent"
                    except Exception as error:
                        event["error"] = str(error)
                    finally:
                        event["returned_ns"] = time.perf_counter_ns()
                        release["events"].append(event)
                    if event["status"] == "sent":
                        break
                try:
                    self.control.close()
                except Exception as error:
                    release.update(resources_released=False, close_error=str(error))
                release["watchdog_events"] = list(self.control.events)
            # Release the actuator before joining capture workers or flushing data.
            try:
                result = self.observations.close()
            except Exception as error:
                result = {"resources_released": False, "close_error": str(error)}
            result.update(
                mode="numeric_driving",
                controller_created=self.control is not None,
                controller=release,
                controller_sends=self.sent_count,
                controller_send_failures=self.failed_count,
                resources_released=bool(
                    result["resources_released"] and release["resources_released"]
                ),
            )
            self._result = result
            return result
