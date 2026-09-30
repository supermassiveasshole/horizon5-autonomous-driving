"""Bounded numerical decision contracts; replay never opens a game or controller."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from fh5.numeric_images import NumericFrame, PixelContract

if TYPE_CHECKING:
    from fh5.control import Command
    from fh5.experiment import Packet


MAX_REALTIME_REPORT_BYTES = 256 * 1024**2


@dataclass(frozen=True)
class RealtimeConfig:
    pixels: PixelContract = field(default_factory=PixelContract)
    decision_hz: int = 20
    inference_deadline_ms: int = 80
    action_lease_ms: int = 150
    max_image_age_ms: int = 100
    max_telemetry_age_ms: int = 100
    watchdog_ms: int = 250
    expected_car_ordinal: int = 2941
    expected_pi: int = 999
    max_speed_kmh: float = 15
    start_speed_kmh: float = 1
    max_steer: float = 0.4
    max_throttle: float = 0.25
    max_brake: float = 0.5
    action_offsets_ms: tuple[int, ...] = (200, 100, 0)
    reference_count: int = 5

    def __post_init__(self) -> None:
        for name, lo, hi in (
            ("decision_hz", 10, 20),
            ("inference_deadline_ms", 1, 100),
            ("action_lease_ms", 10, 250),
            ("max_image_age_ms", 1, 100),
            ("max_telemetry_age_ms", 1, 100),
            ("watchdog_ms", 10, 250),
            ("expected_car_ordinal", 1, 1_000_000),
            ("expected_pi", 1, 999),
            ("reference_count", 1, 256),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError("Invalid real-time bound: " + name)
        for name, low, high in (
            ("max_speed_kmh", 1, 15),
            ("start_speed_kmh", 0, 1),
            ("max_steer", 0, 0.5),
            ("max_throttle", 0, 0.25),
            ("max_brake", 0, 0.5),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError("Invalid real-time bound: " + name)
        if (
            not 1 <= len(self.action_offsets_ms) <= 64
            or self.action_offsets_ms[-1] != 0
            or any(type(v) is not int or not 0 <= v <= 2000 for v in self.action_offsets_ms)
            or any(a <= b for a, b in zip(self.action_offsets_ms, self.action_offsets_ms[1:]))
            or self.action_lease_ms > self.watchdog_ms
        ):
            raise ValueError("Invalid real-time history or lease/watchdog relation")


@dataclass(frozen=True)
class SafetyState:
    epoch: str
    received_ns: int
    game_timestamp_ms: int
    focused: bool
    active: bool
    stop_requested: bool
    car_ordinal: int
    pi: int
    speed_kmh: float
    task_fault: str | None
    fault: str | None = None
    task_location: dict[str, Any] | None = None
    telemetry_packet_index: int | None = None


@dataclass(frozen=True)
class RealtimeObservation:
    epoch: str
    frames: tuple[NumericFrame, ...]
    ego: dict[str, Any]
    telemetry_received_ns: int


@dataclass(frozen=True)
class TimelineInput:
    at_ns: int
    safety: SafetyState
    observation: RealtimeObservation | None
    raw_packets: tuple[Packet, ...] = ()
    capture_epoch: str | None = None


@dataclass(frozen=True)
class InferenceReply:
    """External inference service simulation; None means it never returns."""

    delay_ms: int | None
    prediction: tuple[float, float] = (0.2, 0.3)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.delay_ms is not None and (type(self.delay_ms) is not int or self.delay_ms < 0):
            raise ValueError("Invalid inference reply delay")


@dataclass(frozen=True)
class RealtimeReplay:
    output_dir: Path
    config: RealtimeConfig
    inputs: tuple[TimelineInput, ...]
    replies: tuple[InferenceReply, ...]
    require_command_context: bool = False

    def __post_init__(self) -> None:
        if (
            not 1 <= len(self.inputs) <= 12_000
            or any(type(p.at_ns) is not int or p.at_ns < 0 for p in self.inputs)
            or any(a.at_ns >= b.at_ns for a, b in zip(self.inputs, self.inputs[1:]))
            or self.inputs[-1].at_ns - self.inputs[0].at_ns > 600_000_000_000
            or len(self.replies) > 12_000
            or type(self.require_command_context) is not bool
        ):
            raise ValueError("Replay requires a bounded forward timeline")


@dataclass(frozen=True)
class RealtimeRun:
    output_dir: Path
    config: RealtimeConfig = field(default_factory=RealtimeConfig)
    seconds: float = 30
    startup_timeout_s: float = 30
    journal_capacity: int = 512
    archive_limit_bytes: int = 512 * 1024**2

    def __post_init__(self) -> None:
        for value, maximum in ((self.seconds, 600), (self.startup_timeout_s, 60)):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0.1 <= value <= maximum
            ):
                raise ValueError("Real-time runs require finite duration and startup bounds")
        if type(self.journal_capacity) is not int or not 1 <= self.journal_capacity <= 4096:
            raise ValueError("Invalid real-time journal capacity")
        if (
            type(self.archive_limit_bytes) is not int
            or not 1 <= self.archive_limit_bytes <= 4 * 1024**3
        ):
            raise ValueError("Invalid real-time image archive budget")


@dataclass(frozen=True)
class RealtimeNumericReplay:
    recording_dir: Path
    report_path: Path
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if (
            type(self.tolerance) not in (int, float)
            or not math.isfinite(self.tolerance)
            or not 0 <= self.tolerance <= 1e-3
            or self.report_path.suffix.lower() != ".html"
        ):
            raise ValueError("Invalid real-time numerical replay report or tolerance")


class RealtimeEnvironment(Protocol):
    """External I/O seam. Shadow send is a no-op; actual game control belongs to #9."""

    @property
    def source_kind(self) -> Literal["synthetic", "shadow"]: ...

    def read(self, period_s: float) -> TimelineInput: ...
    def signals(self) -> tuple[bool, bool]: ...
    def send(self, command: Command) -> None: ...
    def close(self) -> dict[str, Any]: ...
