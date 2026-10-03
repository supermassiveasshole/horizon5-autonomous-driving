"""Release the sampling lease before replay preparation or learner updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fh5.driving.control import Command
from fh5.driving.events import EventEnvironment, EventInput
from fh5.driving.realtime.model import RealtimeEnvironment, RealtimeRun, TimelineInput
from fh5.evaluation.run import EvaluationEnvironment, EvaluationRun
from fh5.learning.sac.sampler import SACEnvironment, SACSample, SACStart
from fh5.observation.numeric import PixelContract

if TYPE_CHECKING:
    from fh5.learning.sac.realtime_sampler import SACRealtimeEnvironment, SACRealtimeStart


class LearningUnavailable(RuntimeError):
    """An acquisition failed; the adapter must report whether it released resources."""

    def __init__(self, reason: str, *, resources_released: bool):
        super().__init__(reason)
        self.resources_released = resources_released


class RejectedLease(ValueError):
    def __init__(self, released: dict[str, Any]):
        super().__init__("Learning lease source differs from its frozen configuration")
        self.released = released


def _require_source(
    source: SACEnvironment | SACRealtimeEnvironment | EvaluationEnvironment,
    expected: Literal["synthetic", "native"],
) -> None:
    if source.source_kind != expected:
        try:
            released = source.close()
        except Exception as error:
            released = {"resources_released": False, "error": str(error)}
        raise RejectedLease(released)


class _SamplingLease[Sampling: SACEnvironment | SACRealtimeEnvironment]:
    source_kind: Literal["synthetic", "native"]

    def __init__(
        self,
        source: Sampling,
        phase: Callable[[str], None],
        expected: Literal["synthetic", "native"] = "synthetic",
    ):
        _require_source(source, expected)
        self.source_kind = expected
        self.source = source
        self.phase = phase
        self.released: dict[str, Any] | None = None

    def finish(self, recording_dir: Path) -> Path | None:
        try:
            return self.source.finish(recording_dir)
        finally:
            released = self.close()
            if released.get("resources_released") is not True:
                raise ValueError("Sampling lease could not release before learning")
            self.phase("updating")

    def close(self) -> dict[str, Any]:
        if self.released is None:
            self.released = {"resources_released": False}
            try:
                self.released = self.source.close()
            except Exception as error:
                self.released["error"] = str(error)
        return self.released


class SamplingLease(_SamplingLease[SACEnvironment]):
    source_kind: Literal["synthetic"] = "synthetic"

    def start(self, epoch: str, pixels: PixelContract) -> SACStart:
        self.phase("restarting_sampling")
        result = self.source.start(epoch, pixels)
        self.phase("driving")
        return result

    def step(self, command: Command) -> SACSample:
        return self.source.step(command)


class RealtimeSamplingLease(_SamplingLease["SACRealtimeEnvironment"]):
    def start(self, start: SACRealtimeStart) -> RealtimeEnvironment:
        self.phase("restarting_sampling")
        result = self.source.start(start)
        self.phase("driving")
        return result


@dataclass
class _StopMenu:
    source: EventEnvironment
    stopped: Callable[[], bool]
    source_kind: Literal["synthetic", "udp"] = field(init=False)

    def __post_init__(self) -> None:
        self.source_kind = self.source.source_kind

    def now_ns(self) -> int:
        return self.source.now_ns()

    def read(self, period_s: float) -> EventInput:
        value = self.source.read(period_s)
        return replace(value, stop_requested=value.stop_requested or self.stopped())

    def pulse(self, button: str) -> None:
        if self.stopped():
            raise InterruptedError("Learning stop requested before menu action")
        self.source.pulse(button)

    def release(self) -> None:
        self.source.release()

    def close(self) -> None:
        self.source.close()


@dataclass
class StoppingDrive:
    source: RealtimeEnvironment
    stopped: Callable[[], bool]

    def authorize(
        self, request: RealtimeRun, manifest: dict[str, Any], inference_device: str | None
    ) -> None:
        authorize = getattr(self.source, "authorize", None)
        if not callable(authorize):
            raise ValueError("Native sampling requires qualified driving authorization")
        authorize(request, manifest, inference_device)

    @property
    def source_kind(self) -> Literal["synthetic", "shadow", "native"]:
        return self.source.source_kind

    def read(self, period_s: float) -> TimelineInput:
        value = self.source.read(period_s)
        return replace(
            value,
            safety=replace(
                value.safety, stop_requested=value.safety.stop_requested or self.stopped()
            ),
        )

    def signals(self) -> tuple[bool, bool]:
        focused, stop = self.source.signals()
        return focused, stop or self.stopped()

    def send(self, command: Command) -> None:
        if command != Command(0, 0, 0) and self.stopped():
            raise InterruptedError("Learning stop requested before policy command")
        self.source.send(command)

    def close(self) -> dict[str, Any]:
        return self.source.close()


class EvaluationLease:
    source_kind: Literal["synthetic", "native"]

    def __init__(
        self,
        source: EvaluationEnvironment,
        phase: Callable[[str], None],
        stopped: Callable[[], bool],
        expected: Literal["synthetic", "native"] = "synthetic",
        prepared: Callable[[dict[str, Any]], None] | None = None,
    ):
        _require_source(source, expected)
        self.source_kind, self.prepared = expected, prepared
        self.source, self.phase, self.stopped = source, phase, stopped

    def prepare(self, request: EvaluationRun, batch: dict[str, Any]) -> dict[str, Any]:
        interruptible = getattr(self.source, "prepare_stopped", None)
        qualify = getattr(self.source, "prepare", None)
        if not callable(qualify):
            raise ValueError("Native evaluation requires frozen policy qualification")
        if self.stopped():
            raise InterruptedError("Learning stopped before evaluation qualification")
        result: dict[str, Any] = (
            interruptible(request, batch, self.stopped)
            if callable(interruptible)
            else qualify(request, batch)
        )
        if self.prepared is not None:
            self.prepared(result)
        return result

    def event(self, slot_id: str) -> EventEnvironment:
        self.phase("restarting_evaluation")
        return _StopMenu(self.source.event(slot_id), self.stopped)

    def driving(self, slot_id: str, ready_state: dict[str, Any]) -> RealtimeEnvironment:
        self.phase("evaluating")
        return StoppingDrive(self.source.driving(slot_id, ready_state), self.stopped)

    def close(self) -> dict[str, Any]:
        return self.source.close()
