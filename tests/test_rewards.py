"""Reward settlement through the agreed complete experiment-run interface."""

import json
import struct

import pytest
from test_attempts import evidence, protocol
from test_route_check import record, route

from fh5.experiment import run_experiment
from fh5.reward_audit import RewardAudit
from fh5.rewards import RewardReplay


def reward_config(tmp_path, **changes):
    path = tmp_path / "reward.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "progress_budget": 1.0,
                "time_budget": 1.0,
                "success_bonus": 2.0,
                "failure_cost": 3.0,
                "discount_half_lives_per_horizon": 0.0,
                "max_physics_step_s": 0.5,
                **changes,
            }
        )
    )
    return path


def test_complete_local_drive_has_explainable_progress_time_and_terminal_reward(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            evidence(tmp_path, source),
        )
    )
    summary = result.summary["rewards"]
    fragment = summary["segments"][0]
    assert summary["commands_sent"] is False
    assert summary["actor_fields_added"] == []
    assert fragment["outcome"] == "success"
    assert fragment["terminated"] is True
    assert fragment["truncated"] is False
    assert fragment["reward_usable"] is True
    assert fragment["physical_duration_s"] == pytest.approx(0.3)
    assert fragment["discounted_return"] == pytest.approx(2.99)
    steps = fragment["steps"]
    assert [step["progress_delta_m"] for step in steps] == [1, 1, 1]
    assert [step["dt_s"] for step in steps] == pytest.approx([0.1, 0.1, 0.1])
    assert steps[-1]["task_state"]["farthest_confirmed_m"] == 3
    assert steps[-1]["task_state"]["next_checkpoint"] is None
    assert steps[-1]["task_state"]["remaining_s"] == pytest.approx(29.7)
    assert fragment["final_observation"]["packet_index"] == 3
    assert fragment["bootstrap_observation"] is None
    assert (
        result.summary["attempt_review"]["attempts"][0]["formal_result"]["outcome"]
        == "pending_review"
    )


@pytest.mark.parametrize("condition", ["missing_review", "suspected_wall", "policy_claim"])
def test_uncertain_validity_cannot_issue_usable_success_rewards(tmp_path, condition):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    proof = None
    if condition != "missing_review":
        proof = evidence(
            tmp_path,
            source,
            [
                {
                    "packet_index": 1,
                    "kind": "wall_riding",
                    "status": "suspected",
                }
            ],
        )
        if condition == "policy_claim":
            content = json.loads(proof.read_text())
            content["events"][0].update(status="confirmed", source="policy_prediction")
            proof.write_text(json.dumps(content))
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            proof,
        )
    )
    segment = result.summary["rewards"]["segments"][0]
    assert segment["outcome"] == "quarantined"
    assert segment["reward_usable"] is False
    assert segment["terminated"] is False
    assert segment["bootstrap_observation"] is None
    assert all(s["terminal_reward"] == 0 for s in segment["steps"])
    assert segment["quarantine_reasons"]


def test_rewind_preserves_failed_segment_and_never_bootstraps_from_reset(tmp_path):
    bundle = route(tmp_path)
    source = record(
        tmp_path,
        "drive",
        [(0, 0.2), (1, 0.2), (0, 0.2), (3, 0.2), (0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)],
    )
    proof = evidence(
        tmp_path,
        source,
        [
            {"packet_index": 1, "kind": "driving_failure", "status": "confirmed"},
            {"packet_index": 2, "kind": "rewind", "status": "confirmed", "resume_packet_index": 4},
        ],
    )
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            proof,
        )
    )
    failed, recovered = result.summary["rewards"]["segments"]
    assert failed["outcome"] == "failure"
    assert failed["terminated"] is True and failed["truncated"] is False
    assert failed["reward_usable"] is True
    assert failed["steps"][-1]["terminal_reward"] == -3
    assert failed["discounted_return"] < 0
    assert failed["final_observation"]["packet_index"] == 1
    assert failed["bootstrap_observation"] is None
    assert recovered["outcome"] == "success"
    assert recovered["steps"][0]["from_packet_index"] == 4
    assert recovered["discounted_return"] == pytest.approx(2.99)
    assert result.summary["attempt_review"]["attempts"][0]["formal_result"]["outcome"] == "invalid"


@pytest.mark.parametrize("case", ["offroad", "speed", "jump"])
def test_invalid_progress_never_pays_for_reaching_the_geometric_end(tmp_path, case):
    bundle = route(tmp_path)
    points = [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)]
    if case == "offroad":
        points[1] = (1, 2)
    if case == "jump":
        points = [(0, 0.2), (3, 0.2)]
    source = record(tmp_path, "drive", points, speed=10 if case == "speed" else 0)
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            evidence(tmp_path, source),
        )
    )
    segment = result.summary["rewards"]["segments"][0]
    assert segment["outcome"] != "success"
    assert sum(s["progress_delta_m"] for s in segment["steps"]) == 0
    assert all(s["terminal_reward"] <= 0 for s in segment["steps"])
    if case == "speed":
        assert segment["terminated"] is True
        assert segment["termination_reason"] == "speed_limit_exceeded"
        assert segment["discounted_return"] == -3
    else:
        assert segment["reward_usable"] is False
        assert segment["bootstrap_observation"] is None


def test_complete_discounted_returns_prefer_finishing_over_crash_abandon_or_wait(tmp_path):
    bundle = route(tmp_path)
    task = protocol(tmp_path, bundle)
    content = json.loads(task.read_text())
    content.update(max_duration_s=1, no_progress_timeout_s=0.5)
    task.write_text(json.dumps(content))
    config = reward_config(tmp_path, discount_half_lives_per_horizon=1.0)
    cases = {
        "finish": ([(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)], []),
        "crash": (
            [(0, 0.2)],
            [{"packet_index": 0, "kind": "driving_failure", "status": "confirmed"}],
        ),
        "abandon": (
            [(0, 0.2), (1, 0.2), (2.8, 0.2), (2.8, 0.2)],
            [{"packet_index": 3, "kind": "stop", "status": "confirmed"}],
        ),
        "wait": ([(0, 0.2)] * 8, []),
    }
    returns = {}
    for name, (positions, events) in cases.items():
        source = record(tmp_path, name, positions)
        result = run_experiment(
            RewardReplay(
                source,
                tmp_path / (name + "-reward"),
                task,
                config,
                evidence(tmp_path, source, events),
            )
        )
        segment = result.summary["rewards"]["segments"][0]
        assert segment["terminated"] is True and segment["truncated"] is False
        assert segment["outcome"] == ("success" if name == "finish" else "failure")
        returns[name] = segment["discounted_return"]
        if name == "wait":
            assert segment["physical_duration_s"] == pytest.approx(0.5)
            assert segment["termination_reason"] == "no_progress_timeout"
    assert returns["crash"] == -3
    assert returns["finish"] > 0 > max(returns[n] for n in ("crash", "abandon", "wait"))


def test_zero_game_time_cannot_pay_for_moving_and_receipt_latency_is_not_physics(tmp_path):
    bundle = route(tmp_path)
    task = protocol(tmp_path, bundle)
    config = reward_config(tmp_path)
    for name in ("slow_receipt", "moving_at_zero_dt"):
        source = record(tmp_path, name, [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
        path = source / "packets.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if name == "slow_receipt":
            for row in rows:
                row["received_monotonic_ns"] *= 2
        else:
            rows = rows[:2]
            raw = bytearray.fromhex(rows[1]["payload_hex"])
            struct.pack_into("<I", raw, 4, 1000)
            rows[1]["payload_hex"] = raw.hex()
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        result = run_experiment(
            RewardReplay(
                source,
                tmp_path / (name + "-rewards"),
                task,
                config,
                evidence(tmp_path, source),
            )
        )
        segment = result.summary["rewards"]["segments"][0]
        if name == "slow_receipt":
            assert segment["physical_duration_s"] == pytest.approx(0.3)
            assert segment["discounted_return"] == pytest.approx(2.99)
        else:
            assert segment["reward_usable"] is False
            assert segment["outcome"] == "quarantined"
            assert segment["steps"] == []
            assert segment["bootstrap_observation"] is None


def test_repeated_game_ticks_are_settled_only_after_time_advances(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (1.02, 0.2), (2, 0.2), (3, 0.2)])
    path = source / "packets.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    raw = bytearray.fromhex(rows[2]["payload_hex"])
    struct.pack_into("<I", raw, 4, 1100)
    rows[2]["payload_hex"] = raw.hex()
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            evidence(tmp_path, source),
        )
    )
    segment = result.summary["rewards"]["segments"][0]
    assert segment["outcome"] == "success"
    assert segment["physical_duration_s"] == pytest.approx(0.4)
    assert segment["discounted_return"] == pytest.approx(2.986666666666667)
    assert [(s["from_packet_index"], s["to_packet_index"]) for s in segment["steps"]] == [
        (0, 1),
        (1, 3),
        (3, 4),
    ]
    assert segment["coalesced_same_tick_packets"] == [2]


@pytest.mark.parametrize("kind", ["pause", "rewind", "interface_fault", "restart"])
def test_external_recovery_truncates_at_last_real_state_without_inventing_failure(tmp_path, kind):
    bundle = route(tmp_path)
    source = record(
        tmp_path, "drive", [(0, 0.2), (1, 0.2), (0, 0.2), (0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)]
    )
    event = {"packet_index": 2, "kind": kind, "status": "confirmed"}
    if kind != "restart":
        event["resume_packet_index"] = 3
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            evidence(tmp_path, source, [event]),
        )
    )
    first = result.summary["rewards"]["segments"][0]
    assert first["terminated"] is False and first["truncated"] is True
    assert first["truncation_reason"] == kind
    assert first["reward_usable"] is True
    assert first["discounted_return"] == pytest.approx(0.33)
    assert first["final_observation"]["packet_index"] == 1
    assert first["bootstrap_observation"]["packet_index"] == 1
    assert all(s["terminal_reward"] == 0 for s in first["steps"])
    assert all(s["to_packet_index"] <= 1 for s in first["steps"])


def test_corrupt_record_is_not_a_normal_reward_transition(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    path = source / "packets.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["payload_hex"] = "00"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            evidence(tmp_path, source),
        )
    )
    assert result.summary["attempt_review"]["attempts"][0]["outcome"] == "interface_error"
    assert all(not s["reward_usable"] for s in result.summary["rewards"]["segments"])
    assert all(s["bootstrap_observation"] is None for s in result.summary["rewards"]["segments"])


@pytest.mark.parametrize(
    "kind", ["navigation_recomputed", "navigation_hidden", "destination_changed"]
)
def test_navigation_distance_or_line_changes_do_not_create_progress_or_arrival(tmp_path, kind):
    bundle = route(tmp_path)
    # It moves forward, reverses, then returns to the same furthest point.
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (0, 0.2), (1, 0.2), (1, 0.2)])
    proof = evidence(
        tmp_path,
        source,
        [
            {
                "packet_index": 2,
                "kind": kind,
                "status": "confirmed",
                "destination_distance_before_m": 1000,
                "destination_distance_after_m": 0,
            }
        ],
    )
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            proof,
        )
    )
    segments = result.summary["rewards"]["segments"]
    assert all(s["outcome"] != "success" for s in segments)
    assert sum(step["progress_delta_m"] for s in segments for step in s["steps"]) == 1
    if kind == "destination_changed":
        assert segments[0]["truncation_reason"] == "destination_changed"
        assert segments[0]["final_observation"]["packet_index"] == 1
        assert segments[1]["reward_usable"] is False
        assert "task_phase_changed" in segments[1]["quarantine_reasons"]
    else:
        assert len(segments) == 1
        assert segments[0]["discounted_return"] == pytest.approx(0.32)


def test_reward_replay_preserves_existing_output_artifacts(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2)])
    output = tmp_path / "rewarded"
    output.mkdir()
    sentinel = output / "rewards.json"
    sentinel.write_text("existing reward evidence")
    with pytest.raises(FileExistsError):
        run_experiment(
            RewardReplay(
                source,
                output,
                protocol(tmp_path, bundle),
                reward_config(tmp_path),
                evidence(tmp_path, source),
            )
        )
    assert sentinel.read_text() == "existing reward evidence"


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"time_budget": -1},
        {"progress_budget": float("nan")},
        {"max_physics_step_s": True},
        {"discount_half_lives_per_horizon": -1},
        {"failure_cost": 0.1, "discount_half_lives_per_horizon": 1},
    ],
)
def test_reward_configuration_must_be_finite_versioned_and_terminally_calibrated(tmp_path, change):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2)])
    with pytest.raises(ValueError, match="reward|Reward"):
        run_experiment(
            RewardReplay(
                source,
                tmp_path / "rewarded",
                protocol(tmp_path, bundle),
                reward_config(tmp_path, **change),
                evidence(tmp_path, source),
            )
        )
    assert not (tmp_path / "rewarded").exists()


def test_human_takeover_cannot_be_bridged_as_driving_experience(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            evidence(
                tmp_path,
                source,
                [{"packet_index": 2, "kind": "human_takeover", "status": "confirmed"}],
            ),
        )
    )
    segment = result.summary["rewards"]["segments"][0]
    assert segment["reward_usable"] is False
    assert segment["final_observation"]["packet_index"] == 1
    assert segment["truncation_reason"] == "human_takeover"
    assert all(s["to_packet_index"] < 2 for s in segment["steps"])


def test_experiment_exports_complete_counterexample_audit_with_independent_evidence(tmp_path):
    result = run_experiment(
        RewardAudit(
            reward_config(tmp_path, discount_half_lives_per_horizon=1),
            tmp_path / "audit",
        )
    )
    audit = result.summary["reward_audit"]
    assert audit["source_kind"] == "synthetic"
    assert audit["passed"] is True
    cases = {row["case"]: row for row in audit["cases"]}
    assert {
        "complete",
        "early_crash",
        "late_abandon",
        "park",
        "backtrack",
        "wrong_exit",
        "terrain_shortcut",
        "route_jump",
        "missed_checkpoint",
        "navigation_recomputed",
        "navigation_hidden",
        "destination_changed",
    } <= set(cases)
    assert cases["complete"]["return"] > max(
        cases[k]["return"] for k in ("early_crash", "late_abandon", "park")
    )
    assert cases["backtrack"]["progress_m"] == 3
    assert cases["wrong_exit"]["progress_m"] == 0
    assert cases["route_jump"]["progress_m"] == 0
    assert cases["missed_checkpoint"]["outcome"] != "success"
    assert all(row["passed"] for row in cases.values())
    assert all((result.report_path.parent / row["report"]).is_file() for row in cases.values())
    assert "逐段奖励与终止" in result.report_path.read_text(encoding="utf-8")


def test_small_cumulative_progress_prevents_false_stagnation(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(i * 0.006, 0.2) for i in range(9)])
    task = protocol(tmp_path, bundle)
    content = json.loads(task.read_text())
    content["no_progress_timeout_s"] = 0.3
    task.write_text(json.dumps(content))
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            task,
            reward_config(tmp_path),
            evidence(tmp_path, source),
        )
    )
    segment = result.summary["rewards"]["segments"][0]
    assert segment["outcome"] == "truncated"
    assert segment["physical_duration_s"] == pytest.approx(0.8)
    assert segment["termination_reason"] is None


@pytest.mark.parametrize("reviewed_restart", [False, True])
def test_unknown_clock_reset_cannot_create_a_new_reward_eligible_attempt(
    tmp_path, reviewed_restart
):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    path = source / "packets.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for i, row in enumerate(rows[2:], 2):
        raw = bytearray.fromhex(row["payload_hex"])
        struct.pack_into("<I", raw, 4, 100 * i)
        row["payload_hex"] = raw.hex()
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    events = (
        [{"packet_index": 2, "kind": "restart", "status": "confirmed"}] if reviewed_restart else []
    )
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            evidence(tmp_path, source, events),
        )
    )
    segments = result.summary["rewards"]["segments"]
    if reviewed_restart:
        assert segments[-1]["outcome"] == "success"
    else:
        assert all(not s["reward_usable"] for s in segments)
        assert all(s["bootstrap_observation"] is None for s in segments)


@pytest.mark.parametrize("recovery", ["rewind", "pause"])
def test_failure_on_excluded_recovery_entry_still_has_terminal_feedback(tmp_path, recovery):
    bundle = route(tmp_path)
    source = record(
        tmp_path, "drive", [(0, 0.2), (1, 0.2), (0, 0.2), (0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)]
    )
    events = [
        {"packet_index": 2, "kind": "driving_failure", "status": "confirmed"},
        {"packet_index": 2, "kind": recovery, "status": "confirmed", "resume_packet_index": 3},
    ]
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            evidence(tmp_path, source, events),
        )
    )
    segment = result.summary["rewards"]["segments"][0]
    assert segment["outcome"] == "failure" and segment["terminated"] is True
    assert segment["bootstrap_observation"] is None
    assert segment["discounted_return"] == pytest.approx(-2.67)
    assert segment["terminal_adjustment"]["reward"] == -3
    assert segment["terminal_adjustment"]["event_packet_index"] == 2
    assert segment["final_observation"]["packet_index"] == 1
    assert all(step["to_packet_index"] < 2 for step in segment["steps"])


def test_same_tick_failure_keeps_negative_feedback_without_inventing_a_transition(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (1.02, 0.2)])
    path = source / "packets.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    raw = bytearray.fromhex(rows[2]["payload_hex"])
    struct.pack_into("<I", raw, 4, 1100)
    rows[2]["payload_hex"] = raw.hex()
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            evidence(
                tmp_path,
                source,
                [{"packet_index": 2, "kind": "driving_failure", "status": "confirmed"}],
            ),
        )
    )
    segment = result.summary["rewards"]["segments"][0]
    assert segment["terminated"] is True
    assert segment["discounted_return"] == pytest.approx(-2.67)
    assert segment["terminal_adjustment"]["reward"] == -3
    assert segment["terminal_adjustment"]["event_packet_index"] == 2
    assert segment["physical_duration_s"] == 0.1
    assert segment["reward_usable"] is False
    assert segment["bootstrap_observation"] is None
    assert len(segment["steps"]) == 1


def test_restart_does_not_reauthorize_old_task_after_destination_change(tmp_path):
    bundle = route(tmp_path)
    source = record(
        tmp_path, "drive", [(0, 0.2), (1, 0.2), (0, 0.2), (0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)]
    )
    events = [
        {"packet_index": 2, "kind": "destination_changed", "status": "confirmed"},
        {"packet_index": 3, "kind": "restart", "status": "confirmed"},
    ]
    result = run_experiment(
        RewardReplay(
            source,
            tmp_path / "rewarded",
            protocol(tmp_path, bundle),
            reward_config(tmp_path),
            evidence(tmp_path, source, events),
        )
    )
    segments = result.summary["rewards"]["segments"]
    assert segments[0]["truncation_reason"] == "destination_changed"
    assert all(not s["reward_usable"] for s in segments[1:])
    assert all("task_phase_changed" in s["quarantine_reasons"] for s in segments[1:])
    assert all(s["outcome"] != "success" for s in segments)
