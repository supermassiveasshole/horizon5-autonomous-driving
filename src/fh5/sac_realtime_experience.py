"""Audited SAC transitions from asynchronous numerical execution evidence."""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.collection_store import encode, read_bounded, write_file
from fh5.evaluation_execution import bind_execution_inputs
from fh5.numeric_images import DecisionActor, PixelContract
from fh5.realtime_numeric_replay import (
    read_realtime_decision,
    read_realtime_recording,
    verify_realtime_decision,
)
from fh5.rewards import RewardReplay
from fh5.routes import load_route
from fh5.sac_replay import command_action
from fh5.temporal_features import describe_time

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class SACRealtimePrepare:
    recording_dir: Path
    execution_dir: Path
    task_file: Path
    reward_file: Path
    output_dir: Path
    evidence_file: Path | None = None


def _reward_states(
    settled: RunResult, task_file: Path
) -> tuple[dict[str, Any], dict[int, Any], dict[int, Any]]:
    task = settled.summary["attempt_review"]["task"]
    route_path = task_file.parent / task["route_file"]
    if hashlib.sha256(read_bounded(route_path, 128 * 1024**2)).hexdigest() != task["route_sha256"]:
        raise ValueError("Task route changed during asynchronous experience preparation")
    route = load_route(route_path)
    context = {
        "route_length_m": route["length_m"],
        "checkpoint_ids": [g["id"] for g in route["checkpoints"]],
        "max_duration_s": task["max_duration_s"],
        "no_progress_timeout_s": task["no_progress_timeout_s"],
    }
    samples = {s["packet_index"]: s for s in settled.samples}
    states, steps = {}, {}
    for segment in settled.summary["rewards"]["segments"]:
        if not segment["steps"]:
            continue
        start = segment["steps"][0]["from_packet_index"]
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
    return context, states, steps


def prepare_realtime_experience(request: SACRealtimePrepare, actor: DecisionActor) -> RunResult:
    from fh5.experiment import RunResult, run_experiment

    report = read_realtime_recording(request.execution_dir)
    if (
        report["evidence_kind"] != "synthetic"
        or report["actor_kind"] not in ("frozen-numeric-sac-v1", "frozen-numeric-sac-sampling-v1")
        or report["model"] != actor.manifest
        or report["actor_kind"] != actor.kind
        or report["model"].get("command_context") != "successful-send-return-proxy-v1"
    ):
        raise ValueError("Asynchronous SAC experience requires matching frozen synthetic execution")
    if len(report["decisions"]) > 12_000 or len(report["commands"]) > 20_000:
        raise ValueError("Asynchronous SAC execution exceeds preparation bounds")
    pixels = PixelContract.from_metadata(report["configuration"]["pixels"])
    if pixels.origin != "direct_numeric":
        raise ValueError("Asynchronous SAC experience requires direct numerical pixels")
    request.output_dir.mkdir(parents=True)
    (request.output_dir / "frames").mkdir()
    settled = run_experiment(
        RewardReplay(
            request.recording_dir,
            request.output_dir / "reward",
            request.task_file,
            request.reward_file,
            request.evidence_file,
        )
    )
    bind_execution_inputs(request.execution_dir, report, request.recording_dir, settled)
    task = settled.summary["attempt_review"]["task"]
    if settled.metadata["control_source"] != "policy" or task["control_owner"] != "policy":
        raise ValueError("Asynchronous SAC experience requires actual policy ownership")
    context, states, steps = _reward_states(settled, request.task_file)
    samples = {s["packet_index"]: s for s in settled.samples}
    by_time = {s["received_monotonic_ns"]: s for s in settled.samples}
    observations: dict[str, dict[str, Any]] = {}
    observation_errors = []
    retained_bytes = 0
    decisions = [d for d in report["decisions"] if d["status"] == "accepted"]
    segment_bounds: dict[int, tuple[str, int | None]] = {}
    epoch_segments: dict[str, str] = {}
    for ordinal, segment in enumerate(settled.summary["rewards"]["segments"]):
        if not segment["steps"]:
            continue
        start, end = (
            segment["steps"][0]["from_packet_index"],
            segment["steps"][-1]["to_packet_index"],
        )
        boundary = samples[start]["received_monotonic_ns"] if ordinal else None
        for packet_index in range(start, end + 1):
            segment_bounds[packet_index] = segment["segment_id"], boundary
    for row in decisions:
        try:
            packet_index = by_time[row["telemetry_received_ns"]]["packet_index"]
            if packet_index not in segment_bounds:
                raise ValueError("Observation outside independent reward segment")
            segment_id, boundary = segment_bounds[packet_index]
            if epoch_segments.setdefault(row["epoch"], segment_id) != segment_id:
                raise ValueError("Observation epoch reused after independent recovery")
            verify_realtime_decision(request.execution_dir, row, pixels, actor, 1e-6)
            decision = read_realtime_decision(request.execution_dir, row, pixels)
            if boundary is not None and (
                any(f.source_time_ns < boundary for f in decision.frames)
                or row["command_context"]["returned_ns"] < boundary
                or any(
                    age is not None and row["decision_ns"] - age * 1e6 < boundary
                    for age in decision.actor["action_age_ms"]
                )
            ):
                raise ValueError("Observation history crosses independent recovery boundary")
            frames = []
            for frame in decision.frames:
                digest = hashlib.sha256(frame.pixels).hexdigest()
                name = f"frames/{digest}.rgb"
                if not (request.output_dir / name).exists():
                    retained_bytes += frame.pixels.nbytes
                    if retained_bytes > 512 * 1024**2:
                        raise ValueError("Asynchronous SAC experience exceeds 512 MiB pixels")
                    write_file(request.output_dir / name, bytes(frame.pixels))
                frames.append({**frame.metadata(), "path": name, "sha256": digest})
            observations[row["decision_id"]] = {
                "packet_index": packet_index,
                "decision_id": decision.decision_id,
                "epoch": decision.epoch,
                "decision_ns": decision.decision_ns,
                "actor": decision.actor,
                "frames": frames,
                "timing": describe_time(decision.actor, decision.frames),
            }
        except (OSError, ValueError, TypeError, KeyError) as error:
            observation_errors.append({"decision_id": row["decision_id"], "error": str(error)})

    commands = report["commands"]
    transitions, excluded = [], []
    for number, row in enumerate(decisions):
        index = next(i for i, c in enumerate(commands) if c["decision_id"] == row["decision_id"])
        command, previous, successor = commands[index], commands[index - 1], commands[index + 1]
        following = decisions[number + 1] if number + 1 < len(decisions) else None
        start = by_time[row["telemetry_received_ns"]]["packet_index"]
        first = steps.get(start)
        segment = first[1] if first else None
        terminal_end = segment["final_observation"]["packet_index"] if segment else None
        end = by_time[following["telemetry_received_ns"]]["packet_index"] if following else None
        terminal = bool(
            segment
            and segment["terminated"]
            and terminal_end is not None
            and (end is None or terminal_end <= end)
        )
        if terminal:
            end = terminal_end
        current = observations.get(row["decision_id"])
        after = observations.get(following["decision_id"]) if following and not terminal else None
        reason = None
        if current is None:
            reason = "missing_current_observation"
        elif not terminal and after is None:
            reason = "missing_bootstrap_observation"
        elif following and not terminal and following["epoch"] != row["epoch"]:
            reason = "epoch_boundary"
        elif not terminal and (
            successor["owner"] != "policy"
            or following is None
            or successor["decision_id"] != following["decision_id"]
        ):
            reason = "supervisor_boundary"
        elif end is None or not start < end:
            reason = "nonforward_reward_interval"
        elif (
            not command["returned_ns"]
            < samples[end]["received_monotonic_ns"]
            <= successor["issued_ns"]
        ):
            reason = "response_outside_command_interval"
        interval, cursor = [], start
        while end is not None and cursor < end and cursor in steps:
            step, part = steps[cursor]
            if step["to_packet_index"] > end or not part["reward_usable"] or part is not segment:
                break
            interval.append(step)
            cursor = step["to_packet_index"]
        if not interval or cursor != end:
            reason = reason or "unusable_reward_interval"
        if reason:
            excluded.append({"execution_command_index": index, "reason": reason})
            continue
        assert segment is not None and end is not None and current is not None
        reward, discount = 0.0, 1.0
        for step in interval:
            reward += discount * step["reward"]
            discount *= step["discount"]
        adjustment = segment["terminal_adjustment"] if terminal else None
        if adjustment is not None:
            reward += discount * adjustment["reward"]
        next_ns = following["decision_ns"] if following and not terminal else None
        transitions.append(
            {
                "id": f"async-transition-{index}",
                "execution_command_index": index,
                "control_owner": "policy",
                "epoch": row["epoch"],
                "current": current,
                "next": after,
                "action": command_action(command["sent"]),
                "previous_action": command_action(previous["sent"]),
                "action_elapsed_s": (row["decision_ns"] - previous["returned_ns"]) / 1e9,
                "next_action_elapsed_s": (next_ns - command["returned_ns"]) / 1e9
                if next_ns is not None
                else None,
                "hold_dt_s": (successor["returned_ns"] - command["returned_ns"]) / 1e9,
                "physical_dt_s": sum(step["dt_s"] for step in interval),
                "task_state": states[start],
                "next_task_state": states[end],
                "reward": reward,
                "discount": discount,
                "bootstrap": not terminal,
                "terminated": terminal,
                "truncated": bool(not terminal and end == terminal_end and segment["truncated"]),
                "packet_range": [start, end],
                "reward_steps": interval,
                "terminal_adjustment": adjustment,
                "reward_segment_id": segment["segment_id"],
                "action_time_basis": "asynchronous_send_return_proxy_v1",
                "execution_timing": {
                    "previous_returned_ns": previous["returned_ns"],
                    "decision_ns": row["decision_ns"],
                    "issued_ns": command["issued_ns"],
                    "returned_ns": command["returned_ns"],
                    "next_decision_ns": next_ns,
                    "next_issued_ns": successor["issued_ns"],
                    "next_returned_ns": successor["returned_ns"],
                },
            }
        )
    replay = {
        "version": 3,
        "kind": "sac-numeric-replay-v3",
        "source_kind": "synthetic",
        "source_role": "online",
        "task_contract": task,
        "pixel_contract": pixels.metadata(),
        "task_context": context,
        "task_state_role": "critic_only; frozen BC inputs unchanged",
        "transitions": transitions,
        "excluded": excluded,
        "observation_errors": observation_errors,
        "source_hashes": {
            **settled.summary["rewards"]["source_hashes"],
            "execution": hashlib.sha256(
                read_bounded(request.execution_dir / "realtime-manifest.json", 4096)
            ).hexdigest(),
            "reward": settled.summary["rewards"]["reward_sha256"],
        },
        "sampling_model": report["model"],
        "real_driving_validated": False,
    }
    payload = encode(replay)
    write_file(request.output_dir / "replay.json", payload)
    summary = {
        "eligible_transitions": len(transitions),
        "excluded": excluded,
        "observation_errors": observation_errors,
        "source_kind": "synthetic",
        "real_driving_validated": False,
        "replay_sha256": hashlib.sha256(payload).hexdigest(),
        "commands_sent": False,
    }
    path = request.output_dir / "report.html"
    path.write_text(
        '<!doctype html><meta charset="utf-8"><h1>异步 SAC 经验</h1><pre>'
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>",
        encoding="utf-8",
    )
    return RunResult({}, [], [], {"sac_replay": summary}, path)
