"""Versioned numerical RGB inputs. No capture backend, image codec or game control."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class PixelContract:
    size: tuple[int, int] = (480, 270)
    version: int = 1
    resize: str = "full-frame-pillow-bilinear-v1"
    origin: str = "direct_numeric"
    history_order: str = "oldest_first"
    history_offsets_ms: tuple[int, ...] = (200, 100, 0)

    def __post_init__(self) -> None:
        rgb_byte_count(self.size)
        offsets = self.history_offsets_ms
        if (
            self.version != 1
            or not self.resize
            or self.origin not in ("direct_numeric", "legacy_offline")
            or self.history_order != "oldest_first"
            or not 1 <= len(offsets) <= 16
            or offsets[-1] != 0
            or any(type(v) is not int or not 0 <= v <= 10_000 for v in offsets)
            or any(a <= b for a, b in zip(offsets, offsets[1:]))
        ):
            raise ValueError("Invalid numerical pixel contract")

    @classmethod
    def from_metadata(cls, value: dict[str, Any]) -> PixelContract:
        contract = cls(
            size=tuple(value["size"]),
            version=value["version"],
            resize=value["resize"],
            origin=value["origin"],
            history_order=value["history_order"],
            history_offsets_ms=tuple(value["history_offsets_ms"]),
        )
        if contract.metadata() != value:
            raise ValueError("Unsupported numerical pixel contract metadata")
        return contract

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "size": list(self.size),
            "resize": self.resize,
            "origin": self.origin,
            "history_order": self.history_order,
            "history_offsets_ms": list(self.history_offsets_ms),
            "color": "RGB",
            "dtype": "uint8",
            "layout": "HWC-contiguous",
            "normalization": "float32/255",
            "crop": "full_frame",
        }


@dataclass(frozen=True)
class NumericFrame:
    epoch: str
    frame_id: str
    source_time_ns: int
    capture_received_ns: int
    preprocess_ready_ns: int
    time_quality: str
    uncertainty_ns: int | None
    size: tuple[int, int]
    pixels: memoryview
    source_layout: dict[str, Any]
    availability_kind: str = "numeric_ready"
    preprocess_version: str = "full-frame-pillow-bilinear-v1"

    def __post_init__(self) -> None:
        # Own immutable storage before a capture producer can reuse its buffer.
        expected_bytes = rgb_byte_count(self.size)
        times = (self.source_time_ns, self.capture_received_ns, self.preprocess_ready_ns)
        if (
            not self.epoch
            or not self.frame_id
            or not self.time_quality
            or not self.preprocess_version
            or not self.availability_kind
            or any(type(t) is not int or t < 0 for t in times)
            or not times[0] <= times[1] <= times[2]
            or (
                self.uncertainty_ns is not None
                and (type(self.uncertainty_ns) is not int or self.uncertainty_ns < 0)
            )
        ):
            raise ValueError("Invalid numerical frame identity or monotonic timestamps")
        width, height = self.size
        owned = bytes(self.pixels)
        if len(owned) != expected_bytes:
            raise ValueError("Numerical RGB byte length does not match dimensions")
        object.__setattr__(self, "pixels", memoryview(owned).cast("B", (height, width, 3)))
        object.__setattr__(
            self, "source_layout", json.loads(json.dumps(self.source_layout, allow_nan=False))
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "frame_id": self.frame_id,
            "source_time_ns": self.source_time_ns,
            "capture_received_ns": self.capture_received_ns,
            "preprocess_ready_ns": self.preprocess_ready_ns,
            "time_quality": self.time_quality,
            "uncertainty_ns": self.uncertainty_ns,
            "size": list(self.size),
            "source_layout": self.source_layout,
            "availability_kind": self.availability_kind,
            "preprocess_version": self.preprocess_version,
        }


@dataclass(frozen=True)
class NumericDecision:
    decision_id: str
    epoch: str
    decision_ns: int
    frames: tuple[NumericFrame, ...]
    actor: dict[str, Any]
    supervision: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            not self.decision_id
            or not self.epoch
            or type(self.decision_ns) is not int
            or self.decision_ns < 0
        ):
            raise ValueError("Invalid numerical decision identity or time")
        object.__setattr__(self, "actor", json.loads(json.dumps(self.actor, allow_nan=False)))
        object.__setattr__(
            self, "supervision", json.loads(json.dumps(self.supervision, allow_nan=False))
        )


class ActorMetadata(Protocol):
    kind: str
    manifest: dict[str, Any]


class NumericActor(ActorMetadata, Protocol):
    def predict(self, actor: dict[str, Any], frames: tuple[NumericFrame, ...]) -> list[float]: ...


@runtime_checkable
class ContextualNumericActor(ActorMetadata, Protocol):
    def predict_decision(
        self, decision: NumericDecision, command_context: dict[str, Any]
    ) -> list[float]: ...


type DecisionActor = NumericActor | ContextualNumericActor


def decision_prediction(
    actor: DecisionActor, decision: NumericDecision, command_context: dict[str, Any] | None = None
) -> list[float]:
    if isinstance(actor, ContextualNumericActor):
        if command_context is None:
            raise ValueError("This actor requires acknowledged command context")
        return actor.predict_decision(decision, command_context)
    return actor.predict(decision.actor, decision.frames)


def validate_decision(decision: NumericDecision, contract: PixelContract) -> str | None:
    reason = validate_frame_history(decision.epoch, decision.decision_ns, decision.frames, contract)
    if reason is not None:
        return reason
    frames = decision.frames
    ages = [(decision.decision_ns - f.source_time_ns) / 1e6 for f in frames]
    if decision.actor.get("image_age_ms") != ages:
        return "image_age_mismatch"
    if not decision.actor.get("ego_mask") or decision.actor.get("image_mask") != [True] * len(
        frames
    ):
        return "incomplete_actor_state"
    return None


def validate_frame_history(
    epoch: str, decision_ns: int, frames: tuple[NumericFrame, ...], contract: PixelContract
) -> str | None:
    """Codec-free structural checks shared by offline evidence and live numerical decisions."""
    if len(frames) != len(contract.history_offsets_ms):
        return "incomplete_history"
    if any(f.epoch != epoch for f in frames):
        return "history_crosses_epoch"
    if len({f.frame_id for f in frames}) != len(frames):
        return "repeated_image"
    if any(f.size != contract.size or f.preprocess_version != contract.resize for f in frames):
        return "pixel_contract_mismatch"
    layout_keys = (
        "size",
        "client_size",
        "format",
        "stride_bytes",
        "color_space",
        "crop",
        "resize_method",
    )
    layouts = [tuple(f.source_layout.get(key) for key in layout_keys) for f in frames]
    if any(layout != layouts[0] for layout in layouts[1:]):
        return "history_layout_changed"
    if any(f.preprocess_ready_ns > decision_ns for f in frames):
        return "image_not_available"
    if any(a.source_time_ns >= b.source_time_ns for a, b in zip(frames, frames[1:])):
        return "nonforward_image_time"
    return None


@dataclass(frozen=True)
class NumericInfer:
    output_dir: Path
    contract: PixelContract = field(default_factory=PixelContract)
    max_decisions: int = 1000
    archive_capacity: int = 8
    archive_bytes: int = 32 * 1024**2

    def __post_init__(self) -> None:
        for value, limit in (
            (self.max_decisions, 100_000),
            (self.archive_capacity, 256),
            (self.archive_bytes, 1024**3),
        ):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError("Invalid numerical inference resource limit")


@dataclass(frozen=True)
class NumericReplay:
    recording_dir: Path
    report_path: Path
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or not 0 <= self.tolerance <= 1e-3:
            raise ValueError("Invalid numerical replay tolerance")
        if self.report_path.suffix.lower() != ".html":
            raise ValueError("Numerical replay report must have an .html suffix")


def rgb_byte_count(size: tuple[int, int] | list[int]) -> int:
    """Validate the RGB shape and derive its exact storage requirement."""
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or any(type(value) is not int or value < 1 for value in size)
    ):
        raise ValueError("Invalid numerical RGB dimensions")
    return size[0] * size[1] * 3


def asset(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError("Numerical asset path escapes its recording")
    return path


def run_numeric(
    request: NumericInfer | NumericReplay,
    actor: NumericActor,
    inputs: Iterable[NumericDecision] | None,
) -> RunResult:
    from fh5.numeric_recording import infer_numeric, replay_numeric

    if isinstance(request, NumericReplay):
        return replay_numeric(request, actor)
    if inputs is None:
        raise ValueError("Numerical inference requires an explicit prepared input source")
    return infer_numeric(request, actor, inputs)
