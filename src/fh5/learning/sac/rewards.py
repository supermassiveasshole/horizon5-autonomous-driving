"""Index independently settled task state and aggregate rewards for SAC adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.artifacts.io import read_bounded
from fh5.observation.routes import load_route

if TYPE_CHECKING:
    from fh5.result import RunResult


@dataclass(frozen=True)
class RewardIndex:
    task: dict[str, Any]
    context: dict[str, Any]
    states: dict[int, Any]
    steps: dict[int, Any]
    segment_bounds: dict[int, tuple[str, int | None]]


def index_rewards(settled: RunResult, task_file: Path) -> RewardIndex:
    task = settled.summary["attempt_review"]["task"]
    route_path = task_file.parent / task["route_file"]
    if hashlib.sha256(read_bounded(route_path, 128 * 1024**2)).hexdigest() != task["route_sha256"]:
        raise ValueError("Task route changed during SAC replay preparation")
    route = load_route(route_path)
    context = {
        "route_length_m": route["length_m"],
        "checkpoint_ids": [gate["id"] for gate in route["checkpoints"]],
        "max_duration_s": task["max_duration_s"],
        "no_progress_timeout_s": task["no_progress_timeout_s"],
    }
    samples = {sample["packet_index"]: sample for sample in settled.samples}
    states, steps = {}, {}
    segment_bounds: dict[int, tuple[str, int | None]] = {}
    for ordinal, segment in enumerate(settled.summary["rewards"]["segments"]):
        if not segment["steps"]:
            continue
        start = segment["steps"][0]["from_packet_index"]
        end = segment["steps"][-1]["to_packet_index"]
        boundary = samples[start]["received_monotonic_ns"] if ordinal else None
        for packet_index in range(start, end + 1):
            segment_bounds[packet_index] = segment["segment_id"], boundary
        position = samples[start]["route"]
        states[start] = {
            "farthest_confirmed_m": position["confirmed_progress_m"],
            "start_progress_m": position["confirmed_progress_m"],
            "next_checkpoint": position["next_checkpoint"],
            "remaining_s": task["max_duration_s"],
            "no_progress_remaining_s": task["no_progress_timeout_s"],
        }
        for step in segment["steps"]:
            states[step["to_packet_index"]] = step["task_state"]
            steps[step["from_packet_index"]] = (step, segment)
    return RewardIndex(task, context, states, steps, segment_bounds)


def aggregate_reward(
    steps: Iterable[dict[str, Any]], adjustment: dict[str, Any] | None
) -> tuple[float, float]:
    reward, discount = 0.0, 1.0
    for step in steps:
        reward += discount * step["reward"]
        discount *= step["discount"]
    if adjustment is not None:
        reward += discount * adjustment["reward"]
    return reward, discount
