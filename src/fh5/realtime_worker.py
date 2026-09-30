"""One persistent inference thread; warmup is separate from the decision schedule."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

from fh5.numeric_images import DecisionActor, NumericDecision, NumericFrame, decision_prediction
from fh5.numeric_recording import numeric_features
from fh5.realtime import RealtimeConfig, RealtimeObservation, SafetyState
from fh5.realtime_state import DecisionState, Work


@dataclass(frozen=True)
class WorkerResult:
    work: Work
    started_ns: int
    returned_ns: int
    features: list[float] | None
    prediction: list[float]
    error: str | None


class InferenceWorker:
    def __init__(self, factory: Callable[[], DecisionActor], config: RealtimeConfig) -> None:
        self.factory, self.config = factory, config
        self.requests: Queue[Work] = Queue(1)
        self.results: Queue[WorkerResult] = Queue(1)
        self.ready, self.done = threading.Event(), threading.Event()
        self.error: str | None = None
        self.manifest: dict[str, Any] = {}
        self.kind = "not_loaded"
        self.warmup_completed = False
        self.worker = threading.Thread(
            target=self._run, name="fh5-persistent-inference", daemon=True
        )
        self.worker.start()

    def _warmup(self, actor: DecisionActor) -> None:
        cfg = self.config
        now = (max(cfg.pixels.history_offsets_ms) + 10) * 1_000_000
        frames = tuple(
            NumericFrame(
                "warmup",
                str(i),
                now - offset * 1_000_000,
                now,
                now,
                "synthetic_warmup",
                0,
                cfg.pixels.size,
                memoryview(bytes(cfg.pixels.size[0] * cfg.pixels.size[1] * 3)),
                {},
            )
            for i, offset in enumerate(cfg.pixels.history_offsets_ms)
        )
        state = DecisionState(cfg, lambda command: None)
        state.update(
            SafetyState(
                "warmup",
                now,
                1,
                True,
                True,
                False,
                cfg.expected_car_ordinal,
                cfg.expected_pi,
                0,
                None,
            )
        )
        work = state.begin(
            now,
            RealtimeObservation(
                "warmup",
                frames,
                {
                    "speed_mps": 0,
                    "velocity_car_mps": [0, 0, 0],
                    "angular_velocity_car_radps": [0, 0, 0],
                },
                now,
            ),
        )
        if work is None:
            raise ValueError("Cannot construct bounded numerical warmup")
        context = None
        if actor.manifest.get("command_context"):
            context = {
                "version": 1,
                "command_index": 0,
                "issued_ns": now - 50_000_000,
                "returned_ns": now - 50_000_000,
                "sent": {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0},
                "owner": "warmup",
            }
        decision_prediction(
            actor, NumericDecision("warmup", "warmup", now, frames, work.actor), context
        )

    def _run(self) -> None:
        try:
            actor = self.factory()
            if (
                actor.manifest.get("numeric_contract", self.config.pixels.metadata())
                != self.config.pixels.metadata()
            ):
                raise ValueError("Frozen actor and real-time numerical contracts differ")
            if actor.manifest.get("command_context") and (
                actor.manifest["command_context"] != "successful-send-return-proxy-v1"
                or self.config.action_offsets_ms != (200, 100, 0)
                or any(
                    actor.manifest.get("bounds", {}).get(key) != getattr(self.config, key)
                    for key in ("max_steer", "max_throttle", "max_brake")
                )
            ):
                raise ValueError("Command-conditioned actor and execution contract differ")
            self.kind, self.manifest = actor.kind, deepcopy(actor.manifest)
            self._warmup(actor)
            self.warmup_completed = True
            self.ready.set()
            while not self.done.is_set():
                try:
                    work = self.requests.get(timeout=0.01)
                except Empty:
                    continue
                prediction: list[float] = []
                features = None
                error = None
                started = time.perf_counter_ns()
                try:
                    features = numeric_features(
                        actor, deepcopy(work.actor), work.observation.frames
                    )
                    prediction = list(
                        decision_prediction(
                            actor,
                            NumericDecision(
                                work.row["decision_id"],
                                work.observation.epoch,
                                work.row["decision_ns"],
                                work.observation.frames,
                                deepcopy(work.actor),
                            ),
                            deepcopy(work.row.get("command_context")),
                        )
                    )
                except Exception as failure:
                    error = f"{type(failure).__name__}: {failure}"
                self.results.put_nowait(
                    WorkerResult(work, started, time.perf_counter_ns(), features, prediction, error)
                )
        except Exception as failure:
            self.error = f"{type(failure).__name__}: {failure}"
        finally:
            self.ready.set()

    def close(self) -> dict[str, Any]:
        self.done.set()
        self.worker.join(timeout=0.5)
        return {
            "resources_released": not self.worker.is_alive(),
            "error": self.error,
            "warmup_completed": self.warmup_completed,
            "worker_limit": 1,
        }
