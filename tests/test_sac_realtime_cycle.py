"""Repeated asynchronous sampling and actual learning through the experiment seam."""

import json
import struct
import time
from dataclasses import replace

import pytest
from test_attempts import evidence
from test_evaluation import sha
from test_sac_evaluation import ResponsiveGame
from test_sac_evaluation import sac_policy as sac_policy

from fh5.experiment import run_experiment
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig


class CycleGame(ResponsiveGame):
    def send(self, command):
        time.sleep(0.008)
        super().send(command)

    def read(self, period_s):
        point = super().read(period_s)
        if self.sent:
            command = self.sent[-1][1]
            raw = bytearray(point.raw_packets[0].payload)
            struct.pack_into("<BB", raw, 315, command.throttle_u8, command.brake_u8)
            struct.pack_into("<b", raw, 320, round(command.steer_i16 / 32767 * 127))
            packet = replace(point.raw_packets[0], payload=bytes(raw))
            self.packets[-1] = packet
            point = replace(point, raw_packets=(packet,))
        return point


class AsyncEnvironment:
    source_kind = "synthetic"

    def __init__(self, root, game_type=CycleGame):
        self.root, self.games, self.closed = root, [], False
        self.game_type = game_type

    def start(self, identity, runtime):
        assert all(game.closed for game in self.games)
        if self.games:
            # External acquisition observes a sealed new candidate before reuse.
            assert (self.root / f"candidate-{len(self.games) - 1:03d}/policy.json").is_file()
        game = self.game_type()
        self.games.append(game)
        return game

    def finish(self, recording):
        assert self.games[-1].closed  # No controller lease may remain during updates.
        return evidence(recording.parent, recording)

    def close(self):
        self.closed = True
        return {"resources_released": all(game.closed for game in self.games)}


def request(tmp_path, policy):
    from fh5.sac_cycle import SACRealtimeCycle

    return SACRealtimeCycle(
        checkpoint_dir=policy,
        recording_config_file=policy.parent / "record.json",
        task_file=policy.parent / "task.json",
        reward_file=policy.parent / "reward.json",
        output_dir=tmp_path / "cycle",
        runtime=RealtimeConfig(pixels=PixelContract(size=(64, 36)), reference_count=1),
        seconds_per_attempt=0.6,
        cycles=2,
        max_updates_per_attempt=2,
        expected_checkpoint_sha256=sha(policy / "policy.json"),
    )


def test_two_async_attempts_release_inputs_then_learn_and_switch_complete_snapshots(
    tmp_path, sac_policy
):
    original = (sac_policy / "policy.pt").read_bytes()
    settings = request(tmp_path, sac_policy)
    environment = AsyncEnvironment(settings.output_dir)
    result = run_experiment(settings, sac_realtime_environment=environment).summary["sac_cycle"]
    assert result["stop_reason"] == "budget_completed", result
    assert environment.closed and len(environment.games) == 2
    assert result["resources_released"] and not result["commands_sent_to_game"]
    assert not result["default_changed"]
    first, second = result["attempts"]
    assert first["total_steps"] == 5 and second["total_steps"] == 7
    assert first["candidate_sha256"] == second["sampling_checkpoint_sha256"]
    assert first["sampling_checkpoint_sha256"] != second["sampling_checkpoint_sha256"]
    for attempt in result["attempts"]:
        replay = json.loads((settings.output_dir / attempt["replay"]).read_bytes())
        assert replay["version"] == 3
        assert (
            replay["sampling_model"]["sac_manifest_sha256"] == attempt["sampling_checkpoint_sha256"]
        )
        assert attempt["learner_updates"] == 2 <= attempt["eligible_transitions"]
        assert attempt["inference_reload_max_error"] == 0
        assert attempt["resources_released"]
        assert all(
            row["action_time_basis"] == "asynchronous_send_return_proxy_v1"
            for row in replay["transitions"]
        )
    assert (sac_policy / "policy.pt").read_bytes() == original


def test_external_stop_retains_attempt_without_learning_or_restarting(tmp_path, sac_policy):
    class StoppedGame(CycleGame):
        def signals(self):
            focused, stop = super().signals()
            return focused, stop or any(command.throttle_u8 for _, command in self.sent)

    settings = request(tmp_path, sac_policy)
    environment = AsyncEnvironment(settings.output_dir, StoppedGame)
    result = run_experiment(settings, sac_realtime_environment=environment).summary["sac_cycle"]
    assert result["stop_reason"] == "stop_requested", result
    assert result["resources_released"] and environment.closed
    assert len(environment.games) == len(result["attempts"]) == 1
    attempt = result["attempts"][0]
    assert attempt["stop_reason"] == "user_stop" and attempt["received_packets"] > 0
    assert attempt["source_assets"] and "learner_updates" not in attempt
    assert not (settings.output_dir / "candidate-000").exists()


@pytest.mark.parametrize(
    "change",
    [{"max_steer": 0.3}, {"reference_count": 2}, {"action_offsets_ms": (180, 80, 0)}],
)
def test_changed_execution_contract_is_rejected_before_acquiring_inputs(
    tmp_path, sac_policy, change
):
    settings = request(tmp_path, sac_policy)
    settings = replace(settings, runtime=replace(settings.runtime, **change))
    environment = AsyncEnvironment(settings.output_dir)
    result = run_experiment(settings, sac_realtime_environment=environment).summary["sac_cycle"]
    assert environment.games == [], result
    assert result["stop_reason"] == "interface_error" and "contract" in result["error"]
    assert result["resources_released"] and environment.closed
    assert result["attempts"] == []


@pytest.mark.parametrize("fault", ["send", "release"])
def test_sampling_fault_preserves_evidence_and_does_not_learn_or_start_again(
    tmp_path, sac_policy, fault
):
    class FaultGame(CycleGame):
        def send(self, command):
            if fault == "send" and command.throttle_u8:
                raise OSError("external command sink unavailable")
            super().send(command)

        def close(self):
            result = super().close()
            return {**result, "resources_released": fault != "release"}

    settings = request(tmp_path, sac_policy)
    environment = AsyncEnvironment(settings.output_dir, FaultGame)
    result = run_experiment(settings, sac_realtime_environment=environment).summary["sac_cycle"]
    assert result["stop_reason"] == "sampling_fault", result
    assert result["resources_released"] is (fault != "release")
    assert environment.closed and len(environment.games) == len(result["attempts"]) == 1
    attempt = result["attempts"][0]
    assert attempt["error"] and attempt["source_assets"]
    assert attempt["received_packets"] > 0 and "learner_updates" not in attempt
    assert not (settings.output_dir / "candidate-000").exists()


def test_without_independent_task_evidence_cycle_preserves_sampling_but_cannot_learn(
    tmp_path, sac_policy
):
    class NoEvidence(AsyncEnvironment):
        def finish(self, recording):
            assert self.games[-1].closed
            return None

    settings = request(tmp_path, sac_policy)
    environment = NoEvidence(settings.output_dir)
    result = run_experiment(settings, sac_realtime_environment=environment).summary["sac_cycle"]
    assert result["stop_reason"] == "no_eligible_experience", result
    assert result["resources_released"] and environment.closed
    assert len(environment.games) == len(result["attempts"]) == 1
    attempt = result["attempts"][0]
    assert attempt["eligible_transitions"] == 0 and "learner_updates" not in attempt
    assert attempt["received_packets"] > 0 and attempt["source_assets"]
