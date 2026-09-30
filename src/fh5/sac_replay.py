"""Audited numerical learning transitions; the first adapter is explicitly synthetic."""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.collection_store import encode, read_bounded, write_file
from fh5.numeric_images import NumericDecision, PixelContract, validate_decision
from fh5.numeric_recording import read_numeric_frame
from fh5.rewards import RewardReplay
from fh5.routes import load_route
from fh5.temporal_features import actor_shape, describe_time

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class SACReplayPrepare:
    recording_dir: Path
    trace_file: Path
    task_file: Path
    reward_file: Path
    output_dir: Path
    evidence_file: Path | None = None


def command_action(command: dict[str, Any]) -> list[float]:
    fields = {"steer_i16": (-32767, 32767), "throttle_u8": (0, 255), "brake_u8": (0, 255)}
    if (
        set(command) != set(fields)
        or any(
            type(command[k]) is not int or not lo <= command[k] <= hi
            for k, (lo, hi) in fields.items()
        )
        or command["throttle_u8"]
        and command["brake_u8"]
    ):
        raise ValueError("SAC requires a valid exclusive two-axis sent command")
    return [command["steer_i16"] / 32767, (command["throttle_u8"] - command["brake_u8"]) / 255]


def _check_history(
    row: dict[str, Any],
    trace: dict[str, Any],
    samples: dict[int, Any],
    boundary_ns: int | None,
) -> None:
    offsets = trace["action_offsets_ms"]
    if offsets != [200, 100, 0]:
        raise ValueError("Synthetic SAC trace currently uses the fixed 200/100/0 ms action history")
    receipts: list[tuple[int, list[float] | None, str]] = [
        (
            trace["initial_issued_ns"],
            command_action(trace["initial_command"]),
            trace["actions"][0]["epoch"],
        )
    ]
    receipts.extend(
        (
            samples[a["from_packet_index"]]["received_monotonic_ns"],
            command_action(a["sent"]) if a["status"] == "sent" else None,
            a["epoch"],
        )
        for a in trace["actions"]
    )
    actions: list[list[float] | None] = []
    ages: list[float | None] = []
    for offset in offsets:
        at = row["decision_ns"] - offset * 1_000_000
        prior = next((r for r in reversed(receipts) if r[0] < at), None)
        if (
            prior
            and prior[1] is not None
            and prior[2] == row["epoch"]
            and (boundary_ns is None or prior[0] >= boundary_ns)
            and at - prior[0] <= 200_000_000
        ):
            actions.append(prior[1])
            ages.append((row["decision_ns"] - prior[0]) / 1e6)
        else:
            actions.append(None)
            ages.append(None)
    actor = row["actor"]
    if (
        actor["actions"] != actions
        or actor["action_mask"] != [a is not None for a in actions]
        or actor["action_age_ms"] != ages
    ):
        raise ValueError("Actor history differs from strictly previous successful commands")


def _observation(
    root: Path,
    row: dict[str, Any],
    sample: dict[str, Any],
    pixels: PixelContract,
    output: Path,
) -> dict[str, Any]:
    frame_bytes = pixels.size[0] * pixels.size[1] * 3
    frames = tuple(read_numeric_frame(root, frame, frame_bytes) for frame in row["frames"])
    decision = NumericDecision(
        row["decision_id"], row["epoch"], row["decision_ns"], frames, row["actor"]
    )
    reason = validate_decision(decision, pixels)
    actor_shape(decision.actor, len(frames))
    if reason:
        raise ValueError("Invalid SAC numerical observation: " + reason)
    if (
        decision.decision_ns != sample["received_monotonic_ns"]
        or decision.actor["ego"]
        != {
            "speed_mps": sample["speed_mps"],
            "velocity_car_mps": sample["motion"]["velocity_car_mps"],
            "angular_velocity_car_radps": sample["motion"]["angular_velocity_car_radps"],
        }
        or decision.actor["ego_age_ms"] != 0
    ):
        raise ValueError("Synchronous synthetic observation must match its telemetry packet")
    saved = []
    for frame in frames:
        digest = hashlib.sha256(frame.pixels).hexdigest()
        relative = f"frames/{digest}.rgb"
        if not (output / relative).exists():
            write_file(output / relative, bytes(frame.pixels))
        saved.append({**frame.metadata(), "path": relative, "sha256": digest})
    return {**row, "frames": saved, "timing": describe_time(decision.actor, frames)}


def prepare_sac_replay(request: SACReplayPrepare) -> RunResult:
    from fh5.experiment import RunResult, run_experiment

    raw = read_bounded(request.trace_file, 32 * 1024**2)
    trace = json.loads(raw)
    if trace.get("version") != 1 or trace.get("kind") != "synthetic-synchronous-action-trace-v1":
        raise ValueError("SAC replay requires an explicit synthetic synchronous trace")
    if not 1 <= len(trace["actions"]) <= 10_000 or not 1 <= len(trace["observations"]) <= 20_000:
        raise ValueError("SAC trace exceeds bounded transition/observation counts")
    pixels = PixelContract.from_metadata(trace["pixel_contract"])
    if pixels.origin != "direct_numeric":
        raise ValueError("SAC replay requires direct numerical pixels")
    if (
        sum(len(row["frames"]) for row in trace["observations"])
        * pixels.size[0]
        * pixels.size[1]
        * 3
        > 512 * 1024**2
    ):
        raise ValueError("SAC trace exceeds 512 MiB decoded frame budget")
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
    if settled.metadata["source_kind"] != "synthetic":
        raise ValueError("Synthetic SAC adapter cannot authenticate live game actions")
    samples = {s["packet_index"]: s for s in settled.samples}
    rewards = settled.summary["rewards"]
    segment_bounds: dict[int, tuple[str, int | None]] = {}
    for ordinal, segment in enumerate(rewards["segments"]):
        if not segment["steps"]:
            continue
        start = segment["steps"][0]["from_packet_index"]
        end = segment["steps"][-1]["to_packet_index"]
        boundary = samples[start]["received_monotonic_ns"] if ordinal else None
        for index in range(start, end + 1):
            segment_bounds[index] = (segment["segment_id"], boundary)
    epoch_segments: dict[str, str] = {}
    observations = {}
    seen, observation_errors = set(), []
    for row in sorted(trace["observations"], key=lambda row: row["packet_index"]):
        index = row["packet_index"]
        if index in seen or index not in samples:
            raise ValueError("Duplicate or unknown SAC observation packet")
        seen.add(index)
        try:
            if index not in segment_bounds:
                raise ValueError("Observation outside an independent reward segment")
            segment_id, boundary = segment_bounds[index]
            bound_segment = epoch_segments.setdefault(row["epoch"], segment_id)
            if bound_segment != segment_id:
                raise ValueError("Observation epoch reused after an independent recovery boundary")
            if boundary is not None and any(
                frame["source_time_ns"] < boundary for frame in row["frames"]
            ):
                raise ValueError("Numerical history crosses an independent recovery boundary")
            _check_history(row, trace, samples, boundary)
            observations[index] = _observation(
                request.trace_file.parent, row, samples[index], pixels, request.output_dir
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            observation_errors.append({"packet_index": index, "error": str(error)})
    task = settled.summary["attempt_review"]["task"]
    route_path = request.task_file.parent / task["route_file"]
    if hashlib.sha256(read_bounded(route_path, 128 * 1024**2)).hexdigest() != task["route_sha256"]:
        raise ValueError("Task route changed during SAC replay preparation")
    route = load_route(route_path)
    task_context = {
        "route_length_m": route["length_m"],
        "checkpoint_ids": [g["id"] for g in route["checkpoints"]],
        "max_duration_s": task["max_duration_s"],
        "no_progress_timeout_s": task["no_progress_timeout_s"],
    }
    task_states = {}
    for segment in rewards["segments"]:
        if not segment["steps"]:
            continue
        index = segment["steps"][0]["from_packet_index"]
        position = samples[index]["route"]
        task_states[index] = {
            "farthest_confirmed_m": position["confirmed_progress_m"],
            "start_progress_m": position["confirmed_progress_m"],
            "next_checkpoint": position["next_checkpoint"],
            "remaining_s": task["max_duration_s"],
            "no_progress_remaining_s": task["no_progress_timeout_s"],
        }
        for step in segment["steps"]:
            task_states[step["to_packet_index"]] = step["task_state"]
    steps = {
        step["from_packet_index"]: (step, segment)
        for segment in rewards["segments"]
        for step in segment["steps"]
    }
    transitions, excluded = [], []
    previous_action: list[float] | None = command_action(trace["initial_command"])
    previous_issued = trace["initial_issued_ns"]
    previous_epoch = trace["actions"][0]["epoch"]
    first_segment = steps.get(trace["actions"][0]["from_packet_index"])
    previous_segment_id = first_segment[1]["segment_id"] if first_segment else None
    last_end = -1
    for number, action in enumerate(trace["actions"]):
        start, end = action["from_packet_index"], action["to_packet_index"]
        if start not in samples or end not in samples or not last_end <= start < end:
            raise ValueError("Unordered, overlapping or unknown SAC action interval")
        last_end = end
        now, until = (samples[i]["received_monotonic_ns"] for i in (start, end))
        first_step = steps.get(start)
        current_segment_id = first_step[1]["segment_id"] if first_step else None
        if action["epoch"] != previous_epoch or current_segment_id != previous_segment_id:
            previous_action = None
        previous_epoch, previous_segment_id = action["epoch"], current_segment_id
        if action["status"] != "sent":
            excluded.append({"action_index": number, "reason": "not_executed_policy_action"})
            previous_action = None
            continue
        sent = command_action(action["sent"])
        reason = "unknown_previous_command" if previous_action is None else None
        if action["owner"] != settled.metadata["control_source"]:
            reason = reason or "control_source_mismatch"
        if action["owner"] != "policy" or action["status"] != "sent":
            reason = reason or "not_executed_policy_action"
        current, following = observations.get(start), observations.get(end)
        if not current:
            reason = reason or "missing_current_observation"
        elif (
            current["epoch"] != action["epoch"]
            or following
            and following["epoch"] != action["epoch"]
        ):
            reason = reason or "epoch_boundary"
        interval, cursor = [], start
        while cursor < end and cursor in steps:
            step, segment = steps[cursor]
            if step["to_packet_index"] > end or not segment["reward_usable"]:
                break
            interval.append((step, segment))
            cursor = step["to_packet_index"]
        if cursor != end or not interval or len({s["segment_id"] for _, s in interval}) != 1:
            reason = reason or "unusable_reward_interval"
        segment = interval[-1][1] if interval else None
        terminal = bool(
            segment
            and segment["terminated"]
            and segment["final_observation"]["packet_index"] == end
        )
        if not terminal and not following:
            reason = reason or "missing_bootstrap_observation"
        if not now < until or not previous_issued < now:
            reason = reason or "invalid_action_time"
        for index in range(start + 1, end + 1):
            feedback = samples.get(index, {}).get("telemetry_controls")
            if (
                not feedback
                or feedback["accel"] != action["sent"]["throttle_u8"]
                or feedback["brake"] != action["sent"]["brake_u8"]
                or abs(feedback["steer"] / 127 - sent[0]) > 1 / 127
            ):
                reason = reason or "synthetic_response_mismatch"
        if reason:
            excluded.append({"action_index": number, "reason": reason})
        else:
            reward, discount = 0.0, 1.0
            for step, _ in interval:
                reward += discount * step["reward"]
                discount *= step["discount"]
            adjustment = segment["terminal_adjustment"] if terminal and segment else None
            if adjustment is not None:
                reward += discount * adjustment["reward"]
            transitions.append(
                {
                    "id": f"transition-{number}",
                    "epoch": action["epoch"],
                    "current": current,
                    "next": following,
                    "action": sent,
                    "previous_action": previous_action,
                    "action_elapsed_s": (now - previous_issued) / 1e9,
                    "hold_dt_s": (until - now) / 1e9,
                    "physical_dt_s": sum(step["dt_s"] for step, _ in interval),
                    "task_state": task_states[start],
                    "next_task_state": task_states[end],
                    "reward": reward,
                    "discount": discount,
                    "bootstrap": not terminal,
                    "terminated": terminal,
                    "truncated": bool(
                        segment
                        and segment["truncated"]
                        and segment["final_observation"]["packet_index"] == end
                    ),
                    "packet_range": [start, end],
                    "reward_steps": [step for step, _ in interval],
                    "terminal_adjustment": adjustment,
                    "reward_segment_id": segment["segment_id"] if segment else None,
                    "action_time_basis": "synthetic_synchronous_application; not a native send-return proxy",
                }
            )
        if action["status"] == "sent":
            previous_action, previous_issued = sent, now
    replay = {
        "version": 1,
        "kind": "sac-numeric-replay-v1",
        "source_kind": "synthetic",
        "pixel_contract": pixels.metadata(),
        "task_context": task_context,
        "task_state_role": "critic_only; frozen BC inputs unchanged",
        "transitions": transitions,
        "excluded": excluded,
        "observation_errors": observation_errors,
        "source_hashes": {
            **rewards["source_hashes"],
            "trace": hashlib.sha256(raw).hexdigest(),
            "reward": rewards["reward_sha256"],
        },
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
    report = request.output_dir / "report.html"
    report.write_text(
        '<!doctype html><meta charset="utf-8"><h1>SAC 转移构造</h1><pre>'
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>",
        encoding="utf-8",
    )
    return RunResult({}, [], [], {"sac_replay": summary}, report)
