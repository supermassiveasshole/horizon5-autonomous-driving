"""Asynchronous action evidence becomes actual SAC updates at the experiment seam."""

import json
import struct
import time
from dataclasses import replace
from pathlib import Path

import pytest
from test_attempts import evidence
from test_evaluation import sha
from test_sac_evaluation import ResponsiveGame
from test_sac_evaluation import sac_policy as sac_policy

from fh5.experiment import Record, run_experiment
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig, RealtimeRun
from fh5.sac_learning import SACResume
from fh5.sac_sampling_actor import SACSamplingActor


def recorded_attempt(
    tmp_path, policy, *, observation_gap=False, conflicting_feedback=False, actor_factory=None
):
    pixels = PixelContract(size=(64, 36))

    def factory():
        if actor_factory is not None:
            return actor_factory()
        return SACSamplingActor(policy, pixels, sha(policy / "policy.json"), exploration_seed=9)

    class DelayedSender(ResponsiveGame):
        gap_command_count = None

        def send(self, command):
            time.sleep(0.008)  # External transport delay, not a delayed/mocked model.
            super().send(command)

        def read(self, period_s):
            point = super().read(period_s)
            if self.sent and not conflicting_feedback:
                command = self.sent[-1][1]
                raw = bytearray(point.raw_packets[0].payload)
                struct.pack_into("<BB", raw, 315, command.throttle_u8, command.brake_u8)
                struct.pack_into("<b", raw, 320, round(command.steer_i16 / 32767 * 127))
                packet = replace(point.raw_packets[0], payload=bytes(raw))
                self.packets[-1] = packet
                point = replace(point, raw_packets=(packet,))
            if observation_gap and len(self.sent) >= 4:
                if self.gap_command_count is None:
                    self.gap_command_count = len(self.sent)
                # Withhold external images until the supervisor's release has
                # reached the adapter, then resume. No wall-clock race with
                # the separate no-decision watchdog is needed for this case.
                if len(self.sent) == self.gap_command_count:
                    return replace(point, observation=None)
            return point

    game = DelayedSender()
    root = tmp_path / "execution"
    report = run_experiment(
        RealtimeRun(
            root, RealtimeConfig(pixels=pixels, reference_count=1, action_lease_ms=100), seconds=0.8
        ),
        realtime_environment=game,
        numeric_actor_factory=factory,
    ).summary["realtime"]
    assert report["stop_reason"] == "time_limit", report
    recording = tmp_path / "recording"
    run_experiment(Record(policy.parent / "record.json", recording), packets=game.packets)
    return root, recording, report, factory


def test_async_experience_preserves_three_clocks_and_continues_actual_sac(tmp_path, sac_policy):
    from fh5.sac_realtime_experience import SACRealtimePrepare

    root, recording, report, actor = recorded_attempt(tmp_path, sac_policy)
    original = (sac_policy / "policy.pt").read_bytes()
    result = run_experiment(
        SACRealtimePrepare(
            recording,
            root,
            sac_policy.parent / "task.json",
            sac_policy.parent / "reward.json",
            tmp_path / "experience",
            evidence(tmp_path, recording),
        ),
        numeric_actor=actor(),
    ).summary["sac_replay"]
    assert result["eligible_transitions"] >= 3, result
    data = json.loads((tmp_path / "experience/replay.json").read_bytes())
    decisions = {d["decision_id"]: d for d in report["decisions"]}
    commands = report["commands"]
    for row in data["transitions"]:
        current = decisions[row["current"]["decision_id"]]
        following = decisions[row["next"]["decision_id"]]
        index = row["execution_command_index"]
        command = commands[index]
        assert row["action_elapsed_s"] == pytest.approx(
            (current["decision_ns"] - commands[index - 1]["returned_ns"]) / 1e9
        )
        assert row["next_action_elapsed_s"] == pytest.approx(
            (following["decision_ns"] - command["returned_ns"]) / 1e9
        )
        assert row["hold_dt_s"] == pytest.approx(
            (commands[index + 1]["returned_ns"] - command["returned_ns"]) / 1e9
        )
        assert row["hold_dt_s"] > row["next_action_elapsed_s"]
        assert row["physical_dt_s"] == pytest.approx(sum(s["dt_s"] for s in row["reward_steps"]))
        # The independently settled fixture has no discount, length 3m, horizon 30s.
        progress = sum(s["progress_delta_m"] for s in row["reward_steps"])
        assert row["reward"] == pytest.approx(progress / 3 - row["physical_dt_s"] / 30)
        assert row["bootstrap"] and not row["terminated"]
        assert row["action"] == [
            command["sent"]["steer_i16"] / 32767,
            (command["sent"]["throttle_u8"] - command["sent"]["brake_u8"]) / 255,
        ]
    assert any(e["reason"] == "missing_bootstrap_observation" for e in data["excluded"])
    trained = run_experiment(
        SACResume(
            sac_policy,
            tmp_path / "continued",
            steps=2,
            additions=((tmp_path / "experience/replay.json", result["replay_sha256"]),),
            expected_checkpoint_sha256=sha(sac_policy / "policy.json"),
        )
    ).summary["sac_learning"]
    assert trained["steps_completed"] == 2
    assert trained["total_steps"] == 5
    assert trained["critic_change_max"] > 0
    assert (sac_policy / "policy.pt").read_bytes() == original

    # A slower successor send happens after the next decision. It must not
    # change that state's executable support or leak future latency into Q.
    for row in data["transitions"]:
        row["execution_timing"]["next_returned_ns"] += 100_000_000
        row["hold_dt_s"] = (
            row["execution_timing"]["next_returned_ns"] - row["execution_timing"]["returned_ns"]
        ) / 1e9
    alternate = tmp_path / "experience/slower-successor.json"
    alternate.write_text(json.dumps(data))
    alternate_result = run_experiment(
        SACResume(
            sac_policy,
            tmp_path / "same-learning",
            steps=2,
            additions=((alternate, sha(alternate)),),
        )
    ).summary["sac_learning"]
    assert alternate_result["learner_state_sha256"] == trained["learner_state_sha256"]


def prepare_attempt(
    tmp_path,
    policy,
    *,
    drop_input=False,
    failure=False,
    recovery=False,
    observation_gap=False,
    coverage=True,
    conflicting_feedback=False,
):
    from fh5.sac_realtime_experience import SACRealtimePrepare

    root, recording, report, actor = recorded_attempt(
        tmp_path, policy, observation_gap=observation_gap, conflicting_feedback=conflicting_feedback
    )
    accepted = [d for d in report["decisions"] if d["status"] == "accepted"]
    events = []
    if failure or recovery:
        packets = [
            json.loads(line) for line in (recording / "packets.jsonl").read_text().splitlines()
        ]
        index = next(
            i
            for i, p in enumerate(packets)
            if p["received_monotonic_ns"] >= accepted[len(accepted) // 2]["telemetry_received_ns"]
        )
        events = (
            [
                {
                    "packet_index": index,
                    "kind": "pause",
                    "status": "confirmed",
                    "resume_packet_index": index + 1,
                }
            ]
            if recovery
            else [{"packet_index": index, "kind": "driving_failure", "status": "confirmed"}]
        )
    if drop_input:
        (root / accepted[len(accepted) // 2]["archive"]["path"]).unlink()
    result = run_experiment(
        SACRealtimePrepare(
            recording,
            root,
            policy.parent / "task.json",
            policy.parent / "reward.json",
            tmp_path / "experience",
            evidence(tmp_path, recording, events, coverage=coverage),
        ),
        numeric_actor=actor(),
    )
    return json.loads((tmp_path / "experience/replay.json").read_bytes()), result


def test_one_missing_numeric_archive_does_not_discard_other_complete_transitions(
    tmp_path, sac_policy
):
    replay, result = prepare_attempt(tmp_path, sac_policy, drop_input=True)
    assert result.summary["sac_replay"]["eligible_transitions"] >= 3
    assert len(replay["observation_errors"]) == 1
    assert {e["reason"] for e in replay["excluded"]} >= {
        "missing_current_observation",
        "missing_bootstrap_observation",
    }
    missing = replay["observation_errors"][0]["decision_id"]
    assert all(
        row["current"]["decision_id"] != missing and row["next"]["decision_id"] != missing
        for row in replay["transitions"]
    )


def test_confirmed_failure_is_terminal_while_missing_tail_is_not_a_fake_failure(
    tmp_path, sac_policy
):
    replay, _ = prepare_attempt(tmp_path, sac_policy, failure=True)
    terminals = [r for r in replay["transitions"] if r["terminated"]]
    assert len(terminals) == 1
    terminal = terminals[0]
    assert terminal["reward"] < -2.9
    assert not terminal["bootstrap"]
    assert terminal["next"] is None and terminal["next_action_elapsed_s"] is None
    assert terminal["reward_steps"][-1]["terminal_reward"] == -3
    assert replay["excluded"]


def test_async_initialized_learner_can_retain_legacy_experience_without_relabeling_its_timing(
    tmp_path, sac_policy
):
    from fh5.sac import SACCriticWarmup
    from fh5.sac_actions import ActionBounds
    from fh5.sac_learning import SACTrain

    _, prepared = prepare_attempt(tmp_path, sac_policy)
    replay = tmp_path / "experience/replay.json"
    run_experiment(
        SACCriticWarmup(
            sac_policy / "bc",
            replay,
            prepared.summary["sac_replay"]["replay_sha256"],
            tmp_path / "warm",
            steps=1,
            bounds=ActionBounds(max_steer=0.4, max_throttle=0.25, max_brake=0.5),
        )
    )
    run_experiment(SACTrain(tmp_path / "warm", replay, tmp_path / "candidate", steps=1))
    old = sac_policy / "experience/replay.json"
    result = run_experiment(
        SACResume(tmp_path / "candidate", tmp_path / "mixed", steps=1, additions=((old, sha(old)),))
    ).summary["sac_learning"]
    assert result["total_steps"] == 2
    stored = json.loads((tmp_path / "mixed/experience/replay.json").read_bytes())
    assert len(stored["source_inventory"]) == 2
    assert {r["action_time_basis"] for r in stored["transitions"]} == {
        "asynchronous_send_return_proxy_v1",
        "synthetic_synchronous_application; not a native send-return proxy",
    }


def test_independent_recovery_cannot_reuse_pre_recovery_epoch_and_history(tmp_path, sac_policy):
    replay, result = prepare_attempt(tmp_path, sac_policy, recovery=True)
    boundary = json.loads((tmp_path / "evidence.json").read_bytes())["events"][0]["packet_index"]
    assert result.summary["sac_replay"]["eligible_transitions"] > 0
    assert all(row["packet_range"][1] < boundary for row in replay["transitions"])
    assert any("epoch" in item["error"] for item in replay["observation_errors"])


def test_supervisor_release_splits_experience_but_later_policy_transitions_remain(
    tmp_path, sac_policy
):
    replay, result = prepare_attempt(tmp_path, sac_policy, observation_gap=True)
    report = json.loads((tmp_path / "execution/report.json").read_bytes())
    releases = [i for i, c in enumerate(report["commands"]) if c["owner"] == "lease_expiry"]
    assert len(releases) == 1
    assert result.summary["sac_replay"]["eligible_transitions"] >= 3
    assert any(e["reason"] == "supervisor_boundary" for e in replay["excluded"])
    assert any(r["execution_command_index"] > releases[0] for r in replay["transitions"])
    assert all(r["execution_command_index"] != releases[0] for r in replay["transitions"])


def test_actor_execution_alone_cannot_supply_independent_reward_validity(tmp_path, sac_policy):
    replay, result = prepare_attempt(tmp_path, sac_policy, coverage=False)
    assert result.summary["sac_replay"]["eligible_transitions"] == 0
    assert replay["excluded"]
    assert all(not r["reason"].startswith("terminal") for r in replay["excluded"])


def test_successful_sends_with_conflicting_feedback_cannot_supply_learning_experience(
    tmp_path, sac_policy
):
    replay, result = prepare_attempt(tmp_path, sac_policy, conflicting_feedback=True)
    assert result.summary["sac_replay"]["eligible_transitions"] == 0
    assert any(e["reason"] == "synthetic_response_mismatch" for e in replay["excluded"])


def test_preparation_does_not_seal_experience_under_a_replaced_execution_identity(
    tmp_path, sac_policy, monkeypatch
):
    from fh5.sac_realtime_experience import SACRealtimePrepare

    root, recording, _, actor = recorded_attempt(tmp_path, sac_policy)
    request = SACRealtimePrepare(
        recording,
        root,
        sac_policy.parent / "task.json",
        sac_policy.parent / "reward.json",
        tmp_path / "experience",
        evidence(tmp_path, recording),
    )
    frozen = actor()
    manifest = root / "realtime-manifest.json"
    original_open = Path.open
    replaced = False

    def external_replacement(path, *args, **kwargs):
        nonlocal replaced
        if not replaced and path.suffix == ".rgb" and path.is_relative_to(root):
            replaced = True
            with original_open(manifest, "wb") as output:
                output.write(b'{"version":1,"report_sha256":"replaced-during-read"}')
        return original_open(path, *args, **kwargs)

    # File-system race only: the recorded model and all internal validators run.
    monkeypatch.setattr(Path, "open", external_replacement)
    with pytest.raises(ValueError, match="Execution manifest changed"):
        run_experiment(request, numeric_actor=frozen)
    assert replaced
    assert not (request.output_dir / "replay.json").exists()
