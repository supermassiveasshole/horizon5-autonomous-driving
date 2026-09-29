"""Compose a single UDP/capture receiver with the verified actuator watchdog."""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any, Literal

from fh5.control import Command
from fh5.live import Controller, DesktopState, LiveEnvironment
from fh5.live_vision import ColorSource, LiveVisionEnvironment
from fh5.policy import PolicyInput


class LivePolicyEnvironment:
    source_kind: Literal["udp", "synthetic"] = "udp"

    def __init__(
        self,
        receiver: socket.socket,
        controller: Controller,
        desktop: DesktopState,
        *,
        frame_factory: Callable[[], ColorSource],
    ) -> None:
        self.desktop = desktop
        self.control = LiveEnvironment(receiver, controller, desktop)
        try:
            self.vision = LiveVisionEnvironment(
                receiver, desktop, 0.05, frame_factory=frame_factory
            )
        except BaseException:
            self.control.close()
            raise
        self.events: list[dict[str, Any]] = self.control.events
        self.released: bool | None = None

    def now_ns(self) -> int:
        return self.vision.now_ns()

    def read(self, period_s: float) -> PolicyInput:
        batch = self.vision.read(period_s)
        return PolicyInput(
            batch.packets,
            batch.frame,
            self.desktop.focused(),
            batch.stop_requested,
            self.control.fault or batch.fault,
            batch.events,
        )

    def send(self, command: Command) -> None:
        self.control.send(command)

    def close(self) -> bool:
        if self.released is None:
            try:
                self.control.close()
            finally:
                self.released = self.vision.close()
        return self.released
