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
from fh5.sac_replay import command_action, matches_synthetic_feedback
from fh5.sac_rewards import aggregate_reward, index_rewards
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


def prepare_realtime_experience(request: SACRealtimePrepare, actor: DecisionActor) -> RunResult:
    from fh5.experiment import RunResult, run_experiment

    manifest_path = request.execution_dir / "realtime-manifest.json"
    manifest = read_bounded(manifest_path, 4096)
    report = read_realtime_recording(request.execution_dir)
    if read_bounded(manifest_path, 4096) != manifest:
        raise ValueError("Execution manifest changed during preparation")
    source_kind = report["evidence_kind"]
    if (
        source_kind not in ("synthetic", "native")
        or report["actor_kind"]
        not in (
            "frozen-numeric-sac-v1",
            "frozen-numeric-sac-sampling-v1",
            "frozen-numeric-temporal-bc-v2",
        )
        or report["model"] != actor.manifest
        or report["actor_kind"] != actor.kind
        or report["actor_kind"] != "frozen-numeric-temporal-bc-v2"
        and report["model"].get("command_context") != "successful-send-return-proxy-v1"
    ):
        raise ValueError("Asynchronous SAC experience requires matching frozen execution")
    if source_kind == "native":
        environment = report["environment"]
        provenance = report["model"].get("provenance", {})
        conditions = environment.get("input_conditions", {}).get("conditions", {})
        qualification = environment.get("qualification") or {}
        if (
            report["actor_kind"] != "frozen-numeric-temporal-bc-v2"
            and report["model"].get("source_kind") not in ("native", "mixed")
            or report["model"].get("diagnostic_only") is not False
            or provenance.get("kind") != "continuous_numeric_collection"
            or environment.get("mode") != "numeric_driving"
            or qualification.get("eligible") is not True
            or qualification.get("reasons") != []
            or environment.get("capture", {}).get("source_kind") != "dxgi"
            or conditions.get("status") != "confirmed"
            or conditions != provenance.get("input_conditions")
            or not report["commands_sent_to_game"]
        ):
            raise ValueError("Native SAC experience requires qualified frozen BC or SAC driving")
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
    commands = report["commands"]
    command_indices = {
        command["decision_id"]: index
        for index, command in enumerate(commands)
        if command["decision_id"] is not None
    }
    task = settled.summary["attempt_review"]["task"]
    if settled.metadata["control_source"] != "policy" or task["control_owner"] != "policy":
        raise ValueError("Asynchronous SAC experience requires actual policy ownership")
    if (
        source_kind == "native"
        and report["environment"]["task"]["route_sha256"]
        != (settled.summary["rewards"]["source_hashes"]["route"])
    ):
        raise ValueError("Native execution and independent reward routes differ")
    reward_index = index_rewards(settled, request.task_file)
    context, states, steps = reward_index.context, reward_index.states, reward_index.steps
    samples = {s["packet_index"]: s for s in settled.samples}
    by_time = {s["received_monotonic_ns"]: s for s in settled.samples}
    observations: dict[str, dict[str, Any]] = {}
    observation_errors = []
    decisions = [d for d in report["decisions"] if d["status"] == "accepted"]
    segment_bounds = reward_index.segment_bounds
    epoch_segments: dict[str, str] = {}
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
            command_index = command_indices[row["decision_id"]]
            if boundary is not None and (
                any(f.source_time_ns < boundary for f in decision.frames)
                or command_index > 0
                and commands[command_index - 1]["returned_ns"] < boundary
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

    transitions, excluded = [], []
    for number, row in enumerate(decisions):
        index = command_indices[row["decision_id"]]
        if index == 0:
            excluded.append(
                {"execution_command_index": index, "reason": "missing_previous_command"}
            )
            continue
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
        if previous["returned_ns"] >= row["decision_ns"]:
            reason = "previous_command_not_available"
        elif (
            report["actor_kind"] == "frozen-numeric-temporal-bc-v2"
            and previous["owner"] != "policy"
        ):
            # BC does not wait for SAC's executable action support after release.
            # Resume experience with the next policy-to-policy interval.
            reason = "supervisor_boundary"
        elif current is None:
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
        if source_kind == "synthetic" and reason is None and end is not None:
            for packet_index in range(start + 1, end + 1):
                sample = samples.get(packet_index)
                if sample and sample["received_monotonic_ns"] > command["returned_ns"]:
                    if not matches_synthetic_feedback(
                        command["sent"], sample["telemetry_controls"]
                    ):
                        reason = "synthetic_response_mismatch"
                        break
        if reason:
            excluded.append({"execution_command_index": index, "reason": reason})
            continue
        assert segment is not None and end is not None and current is not None
        adjustment = segment["terminal_adjustment"] if terminal else None
        reward, discount = aggregate_reward(interval, adjustment)
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
    if read_bounded(manifest_path, 4096) != manifest:
        raise ValueError("Execution manifest changed during preparation")
    replay = {
        "version": 3,
        "kind": "sac-numeric-replay-v3",
        "source_kind": source_kind,
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
            "execution": hashlib.sha256(manifest).hexdigest(),
            "reward": settled.summary["rewards"]["reward_sha256"],
        },
        "sampling_model": report["model"],
        "game_application": "unverified",
        "real_driving_validated": False,
    }
    payload = encode(replay)
    write_file(request.output_dir / "replay.json", payload)
    summary = {
        "eligible_transitions": len(transitions),
        "excluded": excluded,
        "observation_errors": observation_errors,
        "source_kind": source_kind,
        "game_application": "unverified",
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
