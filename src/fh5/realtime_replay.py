"""Deterministic external-service fault replay; all actuators and durations are simulated."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Any

from fh5.realtime import RealtimeReplay
from fh5.realtime_report import write_realtime_result
from fh5.realtime_state import DecisionState, Work

if TYPE_CHECKING:
    from fh5.experiment import RunResult


def replay_realtime(request: RealtimeReplay) -> RunResult:
    request.output_dir.mkdir(parents=True, exist_ok=False)
    state = DecisionState(request.config, lambda command: None)
    points = {p.at_ns: p for p in request.inputs}
    replies = iter(request.replies)
    pending: tuple[int, Work, list[float], str | None] | None = None
    start = request.inputs[0].at_ns
    end = request.inputs[-1].at_ns + 1_000_000_000 // request.config.decision_hz
    clock = list(set(range(start, end + 1, 5_000_000)) | set(points) | {end})
    heapq.heapify(clock)
    while clock:
        now = heapq.heappop(clock)
        point = points.pop(now, None)
        if point:
            state.update(point.safety)
            if point.observation:
                state.capture_epoch = point.observation.epoch
        state.supervise(now)
        if pending and pending[0] == now:
            _, work, prediction, error = pending
            pending = None
            state.complete(now, work, prediction, error)
        if point:
            new_work = state.begin(now, point.observation)
            if new_work:
                reply = next(replies, None)
                if reply and reply.delay_ms is not None:
                    due = now + reply.delay_ms * 1_000_000
                    pending = due, new_work, list(reply.prediction), reply.error
                    if due <= end:
                        heapq.heappush(clock, due)
    state.stop(end, "time_limit")
    if state.pending:
        state.pending.row["status"] = "abandoned_inference"
    result: dict[str, Any] = {
        "version": 1,
        "evidence_kind": "synthetic_deadline_replay",
        "commands_sent_to_game": False,
        "decisions": state.decisions,
        "commands": state.commands,
        "stop_reason": state.stop_reason,
        "real_game_validation": False,
        "inference_resources_released": state.pending is None,
        "started_ns": start,
        "ended_ns": end,
    }
    return write_realtime_result(request.output_dir, result)
