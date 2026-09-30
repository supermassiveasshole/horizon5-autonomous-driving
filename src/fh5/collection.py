"""Continuous passive acquisition contracts, independent of OS transports."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from fh5.numeric_images import NumericFrame, PixelContract

if TYPE_CHECKING:
    from fh5.experiment import Packet


def metadata_budget(value: Any, limit: int = 1024**2) -> int:
    """Conservatively bound retained JSON-shaped metadata without encoding it."""
    used = nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal used, nodes
        nodes += 1
        if depth > 8 or nodes > 8192:
            raise ValueError("Collection metadata exceeds structure limit")
        kind = type(item)
        if item is None or kind is bool:
            used += 64
        elif kind is int:
            if item.bit_length() > 64:
                raise ValueError("Collection metadata integer exceeds 64 bits")
            used += 64
        elif kind is float and math.isfinite(item):
            used += 64
        elif kind is str:
            # Covers Python Unicode storage and worst-case JSON character escaping.
            used += 128 + len(item) * 6
        elif kind in (dict, list, tuple):
            used += 256 + 64 * len(item)
        else:
            raise ValueError("Unsupported collection metadata value")
        if used > limit:
            raise ValueError("Collection metadata exceeds byte limit")
        if kind is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("Collection metadata keys must be strings")
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif kind in (list, tuple):
            for child in item:
                visit(child, depth + 1)

    visit(value, 0)
    return used


@dataclass(frozen=True)
class CollectionConfig:
    pixels: PixelContract = field(default_factory=PixelContract)
    seconds: float = 4 * 3600
    poll_hz: int = 120
    observation_hz: int = 20
    block_rows: int = 3600
    block_bytes: int = 64 * 1024**2
    queue_items: int = 256
    queue_bytes: int = 128 * 1024**2
    max_blocks: int = 4096
    max_disk_bytes: int = 64 * 1024**3
    min_free_bytes: int = 2 * 1024**3
    expected_car_ordinal: int = 2941
    expected_pi: int = 999
    max_age_ms: int = 100

    def __post_init__(self) -> None:
        if (
            type(self.seconds) not in (int, float)
            or not math.isfinite(self.seconds)
            or not 0.1 <= self.seconds <= 12 * 3600
        ):
            raise ValueError("Invalid continuous collection duration")
        if (
            self.pixels.origin != "direct_numeric"
            or self.pixels.size[0] > 640
            or self.pixels.size[1] > 360
        ):
            raise ValueError(
                "Continuous collection requires direct numerical pixels, at most 640x360"
            )
        for name, lo, hi in (
            ("poll_hz", 1, 240),
            ("observation_hz", 1, 60),
            ("block_rows", 1, 4096),
            ("block_bytes", 1024, 256 * 1024**2),
            ("queue_items", 1, 4096),
            ("queue_bytes", 1, 256 * 1024**2),
            ("max_blocks", 1, 8192),
            ("max_disk_bytes", 1024, 1024**4),
            ("min_free_bytes", 0, 1024**4),
            ("expected_car_ordinal", 1, 1_000_000),
            ("expected_pi", 1, 999),
            ("max_age_ms", 1, 250),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError("Invalid collection bound: " + name)
        if self.observation_hz > self.poll_hz:
            raise ValueError("Observation rate exceeds input polling rate")


@dataclass(frozen=True)
class CollectionInput:
    at_ns: int
    packets: tuple[Packet, ...] = ()
    human_input: dict[str, Any] | None = None
    frames: tuple[NumericFrame, ...] = ()
    capture_epoch: str | None = None
    focused: bool = False
    boundary: str | None = None
    stop_requested: bool = False
    fault: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.at_ns) is not int
            or self.at_ns < 0
            or len(self.packets) > 64
            or len(self.frames) > 16
        ):
            raise ValueError("Invalid bounded collection input")
        if any(len(p.payload) > 65535 for p in self.packets):
            raise ValueError("Collection datagram exceeds UDP size")
        metadata_budget(
            {
                "human_input": self.human_input,
                "frames": [f.metadata() for f in self.frames],
                "capture_epoch": self.capture_epoch,
                "boundary": self.boundary,
                "fault": self.fault,
                "packet_times": [[p.received_monotonic_ns, p.received_utc] for p in self.packets],
            },
            limit=64 * 1024,
        )
        object.__setattr__(self, "human_input", deepcopy(self.human_input))


@dataclass(frozen=True)
class CollectionRun:
    output_dir: Path
    input_profile: Path
    config: CollectionConfig = field(default_factory=CollectionConfig)
    software_snapshot: dict[str, Any] = field(default_factory=lambda: {"status": "unfrozen"})
    input_conditions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CollectionReview:
    recording_dir: Path
    report_path: Path

    def __post_init__(self) -> None:
        if self.report_path.suffix.lower() != ".html":
            raise ValueError("Collection review requires an HTML report path")


@dataclass(frozen=True)
class CollectionControl:
    recording_dir: Path
    stop: bool = False


class CollectionEnvironment(Protocol):
    """Passive input only; there is deliberately no action-sending capability."""

    @property
    def source_kind(self) -> str: ...
    def read(self, period_s: float) -> CollectionInput | None: ...
    def close(self) -> dict[str, Any]: ...
