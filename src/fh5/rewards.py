"""Physical-time reward settlement, separate from independent record validity."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.attempts import AttemptReplay
from fh5.report import write_report

if TYPE_CHECKING:
    from fh5.experiment import RunResult

DRIVING_FAILURES = frozenset({"driving_failure", "wall_riding", "reset_boost", "grass_shortcut"})


@dataclass(frozen=True)
class RewardReplay:
    recording_dir: Path
    output_dir: Path
    task_file: Path
    reward_file: Path
    evidence_file: Path | None = None


def _config(config: Any, horizon: Any) -> tuple[dict[str, Any], dict[str, float]]:
    names = {
        "progress_budget",
        "time_budget",
        "success_bonus",
        "failure_cost",
        "discount_half_lives_per_horizon",
        "max_physics_step_s",
    }
    if (
        not isinstance(config, dict)
        or set(config) != names | {"version"}
        or type(config["version"]) is not int
        or config["version"] != 1
    ):
        raise ValueError("Reward requires a version 1 configuration with known fields")
    for name in names:
        value = config[name]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (value == 0 and name != "discount_half_lives_per_horizon")
        ):
            raise ValueError(f"Invalid reward parameter: {name}")
    if config["max_physics_step_s"] > 0.5:
        raise ValueError("Reward physical steps cannot exceed the verified 0.5 s continuity bound")
    if type(horizon) not in (int, float) or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Reward needs the task's positive finite physical horizon")
    exponent = math.log(2) * config["discount_half_lives_per_horizon"]
    end_discount = math.exp(-exponent)
    success_lower = end_discount * (config["progress_budget"] + config["success_bonus"]) - config[
        "time_budget"
    ] * (-math.expm1(-exponent) / exponent if exponent else 1)
    # Failure can be observed one physical sample after the task deadline.
    failure_upper = (
        config["progress_budget"]
        - math.exp(-exponent * (1 + config["max_physics_step_s"] / horizon))
        * config["failure_cost"]
    )
    if (
        not math.isfinite(success_lower)
        or not math.isfinite(failure_upper)
        or success_lower <= failure_upper
    ):
        raise ValueError("Reward calibration cannot separate legal completion from bounded failure")
    return config, {
        "success_return_lower_bound": success_lower,
        "failure_return_upper_bound": failure_upper,
    }


def _failure(
    sample: dict[str, Any], task: dict[str, Any], events: list[dict[str, Any]]
) -> str | None:
    reviewed = next(
        (
            e["kind"]
            for e in events
            if e["packet_index"] == sample["packet_index"]
            and e["source"] == "independent_review"
            and e["status"] == "confirmed"
            and e["kind"] in DRIVING_FAILURES
        ),
        None,
    )
    if reviewed:
        return str(reviewed)
    if sample["speed_kmh"] > task["max_speed_kmh"]:
        return "speed_limit_exceeded"
    if sample["route"]["status"] in {"missing_checkpoint", "checkpoint_order"}:
        return str(sample["route"]["status"])
    return None


def _settle(
    samples: list[dict[str, Any]],
    fragment: dict[str, Any],
    attempt: dict[str, Any],
    events: list[dict[str, Any]],
    task: dict[str, Any],
    length: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    local = [
        s
        for s in samples
        if s.get("forward_segment_id") == fragment["segment_id"]
        and s["attempt_phase"] == "local_task"
    ]
    quarantine = [
        r
        for r in attempt["reasons"]
        if r
        in {
            "independent_review_missing",
            "geometry_source_reused",
            "control_owner_mismatch",
            "vehicle_mismatch",
            "conditions_changed",
            "human_takeover",
        }
        or r.startswith("suspected_")
    ]
    if any(
        e["kind"] == "destination_changed"
        and e["source"] == "independent_review"
        and e["status"] == "confirmed"
        and e["packet_index"] <= fragment["packet_range"][0]
        for e in events
    ):
        quarantine.append("task_phase_changed")
        local = []
    if not local:
        quarantine.append("local_start_missing")
    quarantine.extend(r for r in attempt["interface_faults"] if r != "interface_fault")
    for diagnostic in attempt["diagnostics"]:
        if diagnostic["kind"] not in {
            "game_time_discontinuity",
            "position_jump",
            "game_clock_wrap",
            "paused",
            "resumed",
        }:
            continue
        index = diagnostic.get("packet_index")
        if (
            index is None
            or not local
            or index <= local[0]["packet_index"]
            and diagnostic["kind"] in {"paused", "resumed"}
        ):
            continue
        explained = any(
            e["source"] == "independent_review"
            and e["status"] == "confirmed"
            and (
                (e["kind"] == "restart" and e["packet_index"] == index)
                or (
                    e["kind"] in {"pause", "rewind", "interface_fault"}
                    and e["packet_index"]
                    <= index
                    <= e.get("resume_packet_index", attempt["packet_range"][1])
                )
            )
            for e in events
        )
        if not explained:
            quarantine.append("unexplained_" + diagnostic["kind"])
    elapsed = 0.0
    total = 0.0
    steps = []
    frontier = local[0]["route"]["confirmed_progress_m"] if local else 0.0
    start = frontier
    rate = config["time_budget"] / task["max_duration_s"]
    decay = math.log(2) * config["discount_half_lives_per_horizon"] / task["max_duration_s"]
    failure = _failure(local[0], task, attempt["events"]) if local else None
    initial_reward = -config["failure_cost"] if failure else 0.0
    total = initial_reward
    last = local[0] if local else None
    last_progress_s = 0.0
    last_progress_m = start
    interruption = None
    unsettled_failure_index = None
    terminal_timing = None
    coalesced = []
    pending_tick = False
    previous = local[0] if local else {}
    previous_packet = previous.get("packet_index", -1)
    for current in local[1:]:
        if failure:
            break
        interruption = next(
            (
                e["kind"]
                for e in attempt["events"]
                if e["source"] == "independent_review"
                and e["status"] == "confirmed"
                and e["kind"] in {"human_takeover", "human_placement", "conditions_changed"}
                and local[0]["packet_index"] < e["packet_index"] <= current["packet_index"]
            ),
            None,
        )
        if interruption:
            break
        dt = (current["game_timestamp_ms"] - previous["game_timestamp_ms"]) / 1000
        if current["packet_index"] != previous_packet + 1:
            quarantine.append("missing_physics_sample")
            break
        previous_packet = current["packet_index"]
        if not 0 <= dt <= config["max_physics_step_s"]:
            quarantine.append("invalid_physics_interval")
            break
        failure = _failure(current, task, attempt["events"])
        if not failure and current["route"]["status"] not in {"matched", "awaiting_checkpoint"}:
            quarantine.append("unconfirmed_path")
            break
        if dt == 0:
            coalesced.append(current["packet_index"])
            pending_tick = current["position_m"] != previous["position_m"]
            if failure:
                quarantine.append("failure_without_physics_interval")
                unsettled_failure_index = current["packet_index"]
                terminal_timing = "same_game_tick_without_transition"
                break
            continue
        pending_tick = False
        if elapsed + dt > task["max_duration_s"] + 1e-9:
            failure = failure or "task_timeout"
        delta = 0.0 if failure else max(0.0, current["route"]["confirmed_progress_m"] - frontier)
        if frontier + delta > last_progress_m + 0.01:
            last_progress_s = elapsed + dt
            last_progress_m = frontier + delta
        elif elapsed + dt - last_progress_s >= task["no_progress_timeout_s"] - 1e-9:
            failure = failure or "no_progress_timeout"
        if (
            not failure
            and elapsed + dt >= task["max_duration_s"] - 1e-9
            and frontier + delta < length - 1e-6
        ):
            failure = "task_timeout"
        if failure:
            delta = 0.0
        frontier += delta
        progress = config["progress_budget"] * delta / (length - start)
        success = frontier >= length - 1e-6 and not quarantine and not failure
        terminal = (
            -config["failure_cost"] if failure else config["success_bonus"] if success else 0.0
        )
        discount = math.exp(-decay * dt)
        time_cost = -rate * (-math.expm1(-decay * dt) / decay if decay else dt)
        reward = time_cost + discount * (progress + terminal)
        total += math.exp(-decay * elapsed) * reward
        elapsed += dt
        last = current
        steps.append(
            {
                "from_packet_index": previous["packet_index"],
                "to_packet_index": current["packet_index"],
                "dt_s": dt,
                "progress_delta_m": delta,
                "progress_reward": progress,
                "time_cost": -rate * dt,
                "terminal_reward": terminal,
                "reward": reward,
                "discount": discount,
                "task_state": {
                    "farthest_confirmed_m": frontier,
                    "start_progress_m": start,
                    "next_checkpoint": current["route"]["next_checkpoint"],
                    "remaining_s": max(0.0, task["max_duration_s"] - elapsed),
                    "no_progress_remaining_s": max(
                        0.0, task["no_progress_timeout_s"] - (elapsed - last_progress_s)
                    ),
                },
            }
        )
        previous = current
        if failure or success:
            break
    if pending_tick:
        quarantine.append("unsettled_same_tick_tail")
    success = bool(steps and steps[-1]["terminal_reward"] > 0)
    terminal_adjustment = None
    if not success and not failure and local:
        boundary_failure = next(
            (
                e
                for e in events
                if e["kind"] in DRIVING_FAILURES | {"stop"}
                and e["source"] == "independent_review"
                and e["status"] == "confirmed"
                and e["packet_index"] == fragment["packet_range"][1] + 1
            ),
            None,
        )
        if boundary_failure:
            failure = (
                "reviewed_abandonment"
                if boundary_failure["kind"] == "stop"
                else boundary_failure["kind"]
            )
            unsettled_failure_index = boundary_failure["packet_index"]
            terminal_timing = "last_valid_state_before_excluded_boundary"
    if unsettled_failure_index is not None:
        total -= math.exp(-decay * elapsed) * config["failure_cost"]
        terminal_adjustment = {
            "event_packet_index": unsettled_failure_index,
            "reason": failure,
            "reward": -config["failure_cost"],
            "credited_at_physics_s": elapsed,
            "timing": terminal_timing,
        }
    final = {"packet_index": last["packet_index"]} if last else None
    terminated = bool(success or failure)
    boundary = next(
        (
            e["kind"]
            for e in events
            if e["source"] == "independent_review"
            and e["status"] == "confirmed"
            and e["packet_index"] == fragment["packet_range"][1] + 1
            and e["kind"]
            in {"pause", "rewind", "interface_fault", "restart", "destination_changed"}
        ),
        "recording_end",
    )
    return {
        "segment_id": fragment["segment_id"],
        "outcome": "quarantined"
        if quarantine
        else "failure"
        if failure
        else "success"
        if success
        else "truncated",
        "termination_reason": failure or ("local_task_complete" if success else None),
        "terminated": terminated,
        "truncated": not terminated,
        "truncation_reason": None if terminated else interruption or boundary,
        "reward_usable": bool(local) and not quarantine,
        "quarantine_reasons": quarantine,
        "physical_duration_s": elapsed,
        "coalesced_same_tick_packets": coalesced,
        "initial_reward": initial_reward,
        "terminal_adjustment": terminal_adjustment,
        "discounted_return": total,
        "steps": steps,
        "final_observation": final,
        "bootstrap_observation": None if terminated or quarantine else final,
    }


def settle_rewards(request: RewardReplay) -> "RunResult":
    from fh5.experiment import RunResult, run_experiment

    task = json.loads(request.task_file.read_text(encoding="utf-8-sig"))
    if not isinstance(task, dict):
        raise ValueError("Reward requires a local task object")
    reward_bytes = request.reward_file.read_bytes()
    config, calibration = _config(
        json.loads(reward_bytes.decode("utf-8-sig")), task.get("max_duration_s")
    )
    request.output_dir.mkdir(parents=True, exist_ok=False)
    base = run_experiment(
        AttemptReplay(
            request.recording_dir,
            request.output_dir / "validity",
            request.task_file,
            request.evidence_file,
        )
    )
    review = base.summary["attempt_review"]
    rewards = {
        "version": 1,
        "rules_version": "local-physical-reward-v1",
        "reward_sha256": hashlib.sha256(reward_bytes).hexdigest(),
        "config": config,
        "calibration": calibration,
        "source_hashes": review["source_hashes"],
        "commands_sent": False,
        "actor_fields_added": [],
        "task_state_owner": "independent_task_manager",
        "transition_export_allowed": False,
        "bootstrap_requires_causal_actor_observation": True,
        "segments": [
            _settle(
                base.samples,
                fragment,
                attempt,
                review["evidence"]["events"],
                review["task"],
                base.summary["route"]["length_m"],
                config,
            )
            for attempt in review["attempts"]
            for fragment in attempt["forward_segments"]
        ],
    }
    (request.output_dir / "reward-config.json").write_bytes(reward_bytes)
    (request.output_dir / "rewards.json").write_text(
        json.dumps(rewards, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = {**base.summary, "rewards": rewards}
    report_path = request.output_dir / "report.html"
    write_report(
        report_path,
        {
            "metadata": base.metadata,
            "samples": base.samples,
            "events": base.events,
            "summary": summary,
        },
    )
    return RunResult(base.metadata, base.samples, base.events, summary, report_path)
