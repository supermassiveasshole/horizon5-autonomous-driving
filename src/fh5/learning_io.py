"""Release the sampling lease before replay preparation or learner updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fh5.control import Command
from fh5.evaluation_run import EvaluationEnvironment
from fh5.events import EventEnvironment, EventInput
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig, RealtimeEnvironment, TimelineInput
from fh5.sac_sampler import SACEnvironment, SACSample, SACStart

if TYPE_CHECKING:
    from fh5.sac_realtime_sampler import SACRealtimeEnvironment


class LearningUnavailable(RuntimeError):
    """An acquisition failed; the adapter must report whether it released resources."""

    def __init__(self, reason: str, *, resources_released: bool):
        super().__init__(reason)
        self.resources_released = resources_released


class RejectedLease(ValueError):
    def __init__(self, released: dict[str, Any]):
        super().__init__("Learning lease must use synthetic external I/O")
        self.released = released


def _require_synthetic(
    source: SACEnvironment | SACRealtimeEnvironment | EvaluationEnvironment,
) -> None:
    if source.source_kind != "synthetic":
        try:
            released = source.close()
        except Exception as error:
            released = {"resources_released": False, "error": str(error)}
        raise RejectedLease(released)


class _SamplingLease[Sampling: SACEnvironment | SACRealtimeEnvironment]:
    source_kind: Literal["synthetic"] = "synthetic"

    def __init__(self, source: Sampling, phase: Callable[[str], None]):
        _require_synthetic(source)
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
    def start(self, epoch: str, pixels: PixelContract) -> SACStart:
        self.phase("restarting_sampling")
        result = self.source.start(epoch, pixels)
        self.phase("driving")
        return result

    def step(self, command: Command) -> SACSample:
        return self.source.step(command)


class RealtimeSamplingLease(_SamplingLease["SACRealtimeEnvironment"]):
    def start(self, identity: str, runtime: RealtimeConfig) -> RealtimeEnvironment:
        self.phase("restarting_sampling")
        result = self.source.start(identity, runtime)
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
    source_kind: Literal["synthetic"] = "synthetic"

    def __init__(
        self,
        source: EvaluationEnvironment,
        phase: Callable[[str], None],
        stopped: Callable[[], bool],
    ):
        _require_synthetic(source)
        self.source, self.phase, self.stopped = source, phase, stopped

    def event(self, slot_id: str) -> EventEnvironment:
        self.phase("restarting_evaluation")
        return _StopMenu(self.source.event(slot_id), self.stopped)

    def driving(self, slot_id: str, ready_state: dict[str, Any]) -> RealtimeEnvironment:
        self.phase("evaluating")
        return StoppingDrive(self.source.driving(slot_id, ready_state), self.stopped)

    def close(self) -> dict[str, Any]:
        return self.source.close()
