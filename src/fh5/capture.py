"""Numerical capture contracts, bounded history and deterministic schedule replay."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.capture_metrics import SourceCadence, game_frame_time_unavailable
from fh5.numeric_images import NumericFrame, PixelContract
from fh5.numeric_report import preview_png, write_numeric_report

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class QpcMapping:
    ticks: int
    monotonic_ns: int
    frequency: int
    uncertainty_ns: int

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in asdict(self).values()) or not self.frequency:
            raise ValueError("Invalid QPC mapping")

    def convert(self, ticks: int) -> int:
        # Difference first: a float conversion of a large counter loses precision.
        return self.monotonic_ns + (ticks - self.ticks) * 1_000_000_000 // self.frequency


@dataclass(frozen=True)
class RawCapture:
    present_ticks: int
    mapping: QpcMapping
    size: tuple[int, int]
    bgra: bytes
    layout: dict[str, Any]
    time_quality: str = "synthetic_qpc"
    accumulated_frames: int | None = None

    def __post_init__(self) -> None:
        if type(self.present_ticks) is not int or self.present_ticks < 0:
            raise ValueError("Invalid source ticks")
        if self.accumulated_frames is not None and (
            type(self.accumulated_frames) is not int
            or not 0 <= self.accumulated_frames <= 2**32 - 1
        ):
            raise ValueError("Invalid native accumulated frame count")
        if (
            len(self.size) != 2
            or any(type(v) is not int or v <= 0 for v in self.size)
            or len(self.bgra) != self.size[0] * self.size[1] * 4
        ):
            raise ValueError("BGRA byte count differs from physical crop")
        object.__setattr__(self, "bgra", bytes(self.bgra))
        object.__setattr__(self, "layout", json.loads(json.dumps(self.layout, allow_nan=False)))


@dataclass(frozen=True)
class CaptureEvent:
    received_ns: int
    frame: RawCapture | None = None
    boundary: str | None = None
    processing_ns: int = 0
    reason: str | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in (self.received_ns, self.processing_ns)):
            raise ValueError("Invalid capture event time")
        if len(self.stage_ms) > 12 or any(
            not isinstance(name, str) or not math.isfinite(value) or value < 0
            for name, value in self.stage_ms.items()
        ):
            raise ValueError("Invalid capture stage diagnostic")
        object.__setattr__(self, "stage_ms", dict(self.stage_ms))


@dataclass(frozen=True)
class CaptureConfig:
    pixels: PixelContract = field(default_factory=PixelContract)
    history_capacity: int = 32
    max_age_ms: int = 100
    selection_error_ms: int = 40
    capture_hz: int = 60
    preprocess_hz: int = 60
    observation_hz: int = 20

    def __post_init__(self) -> None:
        for value, low, high in (
            (self.history_capacity, len(self.pixels.history_offsets_ms), 64),
            (self.max_age_ms, 1, 1000),
            (self.selection_error_ms, 0, 250),
            (self.capture_hz, 1, 120),
            (self.preprocess_hz, 1, 120),
            (self.observation_hz, 1, 60),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid bounded capture configuration")
        if self.pixels.resize not in (
            "full-frame-pillow-bilinear-v1",
            "legacy-jpeg-roundtrip-then-bilinear-diagnostic-v1",
        ):
            raise ValueError("Capture implements only the declared Pillow bilinear transform")


@dataclass(frozen=True)
class CaptureReplay:
    output_dir: Path
    config: CaptureConfig
    events: tuple[CaptureEvent, ...]
    decision_times_ns: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.events) > 72_000 or len(self.decision_times_ns) > 36_000:
            raise ValueError("Capture replay exceeds bounded diagnostic budget")
        if sum(len(e.frame.bgra) for e in self.events if e.frame) > 512 * 1024**2:
            raise ValueError("Capture replay raw data exceeds diagnostic byte budget")
        for times in ([e.received_ns for e in self.events], self.decision_times_ns):
            if any(type(t) is not int or t < 0 for t in times) or any(
                a >= b for a, b in zip(times, times[1:])
            ):
                raise ValueError("Capture replay times must increase strictly")


@dataclass(frozen=True)
class CaptureRun:
    output_dir: Path
    config: CaptureConfig
    seconds: float = 30
    archive_limit_bytes: int = 512 * 1024**2
    input_conditions: dict[str, Any] = field(default_factory=dict)
    raw_sample_limit: int = 0
    raw_sample_interval_s: float = 5
    raw_sample_bytes: int = 128 * 1024**2

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds) or not 0.1 <= self.seconds <= 600:
            raise ValueError("Capture probe must last between 0.1 and 600 seconds")
        if not 1024 <= self.archive_limit_bytes <= 2 * 1024**3:
            raise ValueError("Invalid capture archive budget")
        if (
            type(self.raw_sample_limit) is not int
            or not 0 <= self.raw_sample_limit <= 8
            or type(self.raw_sample_bytes) is not int
            or not 1 <= self.raw_sample_bytes <= 256 * 1024**2
            or not math.isfinite(self.raw_sample_interval_s)
            or not 0 <= self.raw_sample_interval_s <= 600
        ):
            raise ValueError("Invalid bounded original-sample budget")
        object.__setattr__(
            self, "input_conditions", json.loads(json.dumps(self.input_conditions, allow_nan=False))
        )


@dataclass(frozen=True)
class Work:
    epoch: int
    frame_id: str
    event: CaptureEvent


def next_preprocess_time(start_ns: int, ready_ns: int, config: CaptureConfig) -> int:
    period = (1_000_000_000 + config.preprocess_hz - 1) // config.preprocess_hz
    return max(start_ns + period, ready_ns)


def preprocess(work: Work, config: CaptureConfig, ready_ns: int) -> NumericFrame:
    raw = work.event.frame
    assert raw is not None
    image = importlib.import_module("PIL.Image")
    rgb = image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    resized = rgb.resize(config.pixels.size, image.Resampling.BILINEAR)
    return NumericFrame(
        epoch=str(work.epoch),
        frame_id=work.frame_id,
        source_time_ns=raw.mapping.convert(raw.present_ticks),
        capture_received_ns=work.event.received_ns,
        preprocess_ready_ns=ready_ns,
        time_quality=raw.time_quality,
        uncertainty_ns=(
            raw.mapping.uncertainty_ns if raw.time_quality != "capture_start_proxy" else None
        ),
        size=config.pixels.size,
        pixels=memoryview(resized.tobytes()),
        source_layout={
            **raw.layout,
            "size": list(raw.size),
            "format": "BGRA",
            "stride_bytes": raw.size[0] * 4,
            "accumulated_frames": raw.accumulated_frames,
            **(
                {"qpc": asdict(raw.mapping), "present_ticks": raw.present_ticks}
                if raw.time_quality != "capture_start_proxy"
                else {"capture_started_ns": raw.present_ticks}
            ),
        },
        preprocess_version=config.pixels.resize,
    )


class CaptureState:
    """One pending raw item and a bounded immutable RGB history; caller owns locking."""

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self.epoch = 0
        self.serial = 0
        self.pending: Work | None = None
        self.history: deque[NumericFrame] = deque(maxlen=config.history_capacity)
        self.counts: Counter[str] = Counter()
        self.last_source_ns: int | None = None
        self.layout: dict[str, Any] | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=512)
        self.cadence = SourceCadence(config.capture_hz)

    def boundary(self, reason: str, now_ns: int) -> None:
        self.epoch += 1
        self.pending = None
        self.history.clear()
        self.layout = None
        self.last_source_ns = None
        self.cadence.previous = None
        self.counts["epoch_changes"] += 1
        self.events.append({"reason": reason, "observed_ns": now_ns, "epoch": str(self.epoch)})

    def offer(self, event: CaptureEvent) -> None:
        if event.boundary:
            self.boundary(event.boundary, event.received_ns)
        raw = event.frame
        if raw is None:
            self.counts[event.reason or "no_new_frame"] += 1
            return
        if raw.present_ticks <= 0:
            self.counts["no_present"] += 1
            return
        source_ns = raw.mapping.convert(raw.present_ticks)
        if source_ns < 0 or source_ns > event.received_ns:
            self.counts["future_present"] += 1
            return
        layout = {**raw.layout, "size": list(raw.size)}
        if self.layout is not None and self.layout != layout:
            self.boundary("layout_changed", event.received_ns)
        self.layout = layout
        if self.last_source_ns is not None and source_ns <= self.last_source_ns:
            self.counts["nonforward_present"] += 1
            return
        self.last_source_ns = source_ns
        self.cadence.observe(raw.time_quality, source_ns)
        self.serial += 1
        self.counts["new_frames"] += 1
        if raw.accumulated_frames is not None:
            self.counts["desktop_presentations_coalesced"] += max(0, raw.accumulated_frames - 1)
        if self.pending is not None:
            self.counts["pending_overwritten"] += 1
        self.pending = Work(self.epoch, f"f{self.serial}", event)

    def take(self) -> Work | None:
        pending, self.pending = self.pending, None
        return pending

    def complete(self, work: Work, frame: NumericFrame) -> None:
        if work.epoch != self.epoch:
            self.counts["old_epoch_completion"] += 1
            return
        self.history.append(frame)
        self.counts["preprocessed"] += 1
        self.counts["history_peak"] = max(self.counts["history_peak"], len(self.history))

    def select(self, now_ns: int) -> tuple[dict[str, Any], tuple[NumericFrame, ...]]:
        row: dict[str, Any] = {
            "decision_id": f"capture-{now_ns}",
            "decision_ns": now_ns,
            "epoch": str(self.epoch),
            "frames": [],
            "status": "skip_not_ready",
            "reason": "incomplete_history",
        }
        history = [f for f in self.history if f.preprocess_ready_ns <= now_ns]
        if not history:
            return row, ()
        anchor = history[-1].source_time_ns
        row["latest_source_ns"] = anchor
        row["latest_age_ms"] = (now_ns - anchor) / 1e6
        if now_ns - anchor > self.config.max_age_ms * 1_000_000:
            row["reason"] = "stale_latest_image"
            return row, ()
        selected = []
        for offset in self.config.pixels.history_offsets_ms:
            target = anchor - offset * 1_000_000
            candidates = [f for f in history if f.source_time_ns <= target]
            if not candidates:
                return row, ()
            frame = candidates[-1]
            if target - frame.source_time_ns > self.config.selection_error_ms * 1_000_000:
                row["reason"] = "history_time_error"
                return row, ()
            selected.append(frame)
        if len({f.frame_id for f in selected}) != len(selected):
            row["reason"] = "repeated_image"
            return row, ()
        row.update(
            status="ready",
            reason=None,
            frames=[f.metadata() for f in selected],
            timing={
                "image_age_s": [(now_ns - f.source_time_ns) / 1e9 for f in selected],
                "adjacent_delta_s": [
                    (b.source_time_ns - a.source_time_ns) / 1e9
                    for a, b in zip(selected, selected[1:])
                ],
            },
        )
        return row, tuple(selected)


def replay_capture(request: CaptureReplay) -> RunResult:
    from fh5.experiment import RunResult, _write_json

    directory = request.output_dir
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "pixels").mkdir()
    (directory / "previews").mkdir()
    state = CaptureState(request.config)
    rows: list[dict[str, Any]] = []
    actions = sorted(
        [(event.received_ns, 1, event) for event in request.events]
        + [(tick, 2, None) for tick in request.decision_times_ns],
        key=lambda x: (x[0], x[1]),
    )
    work: Work | None = None
    ready_at: int | None = None
    next_start = 0

    def advance(now: int) -> None:
        nonlocal work, ready_at, next_start
        while True:
            if work is not None:
                assert ready_at is not None
                if ready_at > now:
                    return
                state.complete(work, preprocess(work, request.config, ready_at))
                work, ready_at = None, None
            if state.pending is None:
                return
            start = max(next_start, state.pending.event.received_ns)
            if start > now:
                return
            work = state.take()
            assert work is not None
            ready_at = start + work.event.processing_ns
            next_start = next_preprocess_time(start, ready_at, request.config)

    for now, _, event in actions:
        advance(now)
        if event is not None:
            state.offer(event)
            advance(now)
        else:
            row, frames = state.select(now)
            stored, previews = [], []
            for frame in frames:
                pixels = bytes(frame.pixels)
                digest = hashlib.sha256(pixels).hexdigest()
                path, preview = f"pixels/{digest}.rgb", f"previews/{digest}.png"
                (directory / path).write_bytes(pixels)
                (directory / preview).write_bytes(preview_png(pixels, frame.size))
                stored.append(dict(frame.metadata(), path=path, sha256=digest))
                previews.append(preview)
            row.update(stored_frames=stored, previews=previews)
            rows.append(row)
    summary = {
        "version": 1,
        "contract": request.config.pixels.metadata(),
        "model": None,
        "timing_kind": "simulated",
        "started_ns": actions[0][0] if actions else 0,
        "ended_ns": actions[-1][0] if actions else 0,
        "commands_sent": False,
        "decisions": rows,
        **state.counts,
        "pending_capacity": 1,
        "history_capacity": request.config.history_capacity,
        "events": list(state.events),
        "cadence": state.cadence.report(),
        "game_frame_time": game_frame_time_unavailable(),
    }
    state.pending = work = None
    state.history.clear()
    summary["resources_released"] = True
    _write_json(directory / "capture.json", summary)
    report = directory / "report.html"
    write_numeric_report(report, summary, directory)
    return RunResult({}, [], [], {"capture": summary}, report)
