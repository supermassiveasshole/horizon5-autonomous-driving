"""Action-responsive sampling and continued learning at the experiment seam."""

import hashlib
import json
import struct
from dataclasses import replace

import pytest
from test_sac import experience
from test_sac_learning import warm_start

from fh5.experiment import Packet, run_experiment
from fh5.numeric_images import NumericDecision, NumericFrame
from fh5.sac_learning import SACResume, SACTrain


def initial(root):
    replay = warm_start(root)
    checkpoint = root / "initial"
    run_experiment(SACTrain(root / "warm", replay, checkpoint, steps=3))
    return checkpoint


def addition(root):
    folder = root / "new"
    folder.mkdir()
    request = experience(folder, terminal=False)
    request = replace(request, task_file=root / "task.json", reward_file=root / "reward.json")
    result = run_experiment(request).summary["sac_replay"]
    return request.output_dir / "replay.json", result["replay_sha256"]


def test_new_compatible_experience_continues_the_learner_without_rewriting_its_parent(tmp_path):
    checkpoint = initial(tmp_path)
    old_bytes = (checkpoint / "experience/replay.json").read_bytes()
    extra = addition(tmp_path)
    output = tmp_path / "continued"
    result = run_experiment(SACResume(checkpoint, output, steps=2, additions=(extra,))).summary[
        "sac_learning"
    ]
    merged = json.loads((output / "experience/replay.json").read_bytes())
    assert result["total_steps"] == 5
    assert result["actor_updates"] == 1
    assert result["experience_added_transitions"] == 2
    assert len(merged["transitions"]) == 4
    assert len({row["id"] for row in merged["transitions"]}) == 4
    assert len(merged["source_inventory"]) == 2
    assert {row["provenance"]["replay_sha256"] for row in merged["transitions"]} == {
        hashlib.sha256(old_bytes).hexdigest(),
        extra[1],
    }
    assert (checkpoint / "experience/replay.json").read_bytes() == old_bytes
    assert result["bc_transfer_command_error"] is None


class ResponsiveEnvironment:
    source_kind = "synthetic"

    def __init__(self, root):
        self.root = root
        self.template = json.loads((root / "trace.json").read_bytes())["observations"][0]
        self.commands = []
        self.end_positions = []
        self.closed = False
        self.fail_at = None
        self.stop_after = None

    def start(self, epoch, pixels):
        from fh5.control import Command
        from fh5.sac_cycle import SACStart

        self.epoch, self.index, self.x, self.z = epoch, 0, 0.0, 0.2
        self.pixels = pixels
        self.command = Command(0, 0, 0)
        return SACStart(self.sample(), self.command, 900_000_000)

    def step(self, command):
        if self.fail_at is not None and len(self.commands) == self.fail_at:
            raise OSError("synthetic send failed")
        self.commands.append(command)
        self.command = command
        self.index += 1
        if self.index == self.stop_after:
            (self.root / "cycle/stop.request").write_text("requested by external observer")
        self.x += command.throttle_u8 / 255 * 2
        self.z += command.steer_i16 / 32767 * 0.01
        return self.sample()

    def sample(self):
        from fh5.sac_cycle import SACSample

        now = 1_000_000_000 + self.index * 100_000_000
        raw = bytearray(324)
        struct.pack_into("<iI", raw, 0, 1, 1000 + self.index * 100)
        struct.pack_into("<iii", raw, 212, 2941, 6, 999)
        struct.pack_into("<ffff", raw, 244, self.x, 2, self.z, 4)
        raw[315:317] = bytes([self.command.throttle_u8, self.command.brake_u8])
        struct.pack_into("<b", raw, 320, round(self.command.steer_i16 / 32767 * 127))
        state = json.loads(json.dumps(self.template["actor"]))
        rgb = bytes([51, 17, 34] * 64 * 36)
        frames = tuple(
            NumericFrame(
                self.epoch,
                f"{self.index}:{i}",
                now - age,
                now - age + 1,
                now - age + 2,
                "synthetic",
                0,
                self.pixels.size,
                memoryview(rgb),
                {"size": [64, 36], "format": "RGB"},
            )
            for i, age in enumerate((210_000_000, 130_000_000, 10_000_000))
        )
        return SACSample(
            Packet(now, "2026-10-01T00:00:00+00:00", bytes(raw)),
            NumericDecision(f"d{self.index}", self.epoch, now, frames, state),
            done=self.x >= 3,
        )

    def finish(self, recording):
        from test_attempts import evidence

        self.end_positions.append(self.x)
        return evidence(recording.parent, recording)

    def close(self):
        self.closed = True
        return {"resources_released": True}


def cycle_request(root, checkpoint, **changes):
    from fh5.sac_cycle import SACCycle

    return SACCycle(
        checkpoint,
        root / "record.json",
        root / "task.json",
        root / "reward.json",
        root / "cycle",
        cycles=2,
        steps_per_attempt=3,
        **changes,
    )


def test_two_attempts_sample_the_frozen_policy_then_continue_learning_on_their_responses(tmp_path):
    checkpoint = initial(tmp_path)
    original = (checkpoint / "policy.pt").read_bytes()
    environment = ResponsiveEnvironment(tmp_path)
    result = run_experiment(
        cycle_request(tmp_path, checkpoint), sac_environment=environment
    ).summary["sac_cycle"]
    assert result["stop_reason"] == "budget_completed"
    assert environment.closed
    assert result["commands_sent_to_game"] is False
    assert result["default_changed"] is False
    first, second = result["attempts"]
    assert first["eligible_transitions"] == second["eligible_transitions"] == 3
    assert first["total_steps"] == 6
    assert second["total_steps"] == 9
    assert first["candidate_sha256"] == second["sampling_checkpoint_sha256"]
    assert first["sampling_checkpoint_sha256"] != second["sampling_checkpoint_sha256"]
    assert len({r["snapshot_sha256"] for r in first["decisions"]}) == 1
    assert len(environment.commands) == 6
    assert environment.end_positions[0] > 0
    assert (checkpoint / "policy.pt").read_bytes() == original
    for attempt in result["attempts"]:
        rows = json.loads((tmp_path / "cycle" / attempt["replay"]).read_bytes())["transitions"]
        assert rows[0]["previous_action"] == [0, 0]
        assert rows[-1]["bootstrap"] is True  # A sampler bound is not task failure.
        assert rows[-1]["truncated"] is True
        assert [r["action"] for r in rows] == [r["command"] for r in attempt["decisions"]]
        assert attempt["learner_updates"] == len(rows)
        assert attempt["inference_reload_max_error"] == 0


def test_expanded_snapshot_keeps_each_original_replay_and_refuses_double_credit(tmp_path):
    checkpoint = initial(tmp_path)
    extra = addition(tmp_path)
    output = tmp_path / "expanded"
    run_experiment(SACResume(checkpoint, output, steps=2, additions=(extra,)))
    merged = json.loads((output / "experience/replay.json").read_bytes())
    for source in merged["source_inventory"]:
        raw = (output / "experience" / source["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source["replay_sha256"]
        assert json.loads(raw)["source_hashes"] == source["source_hashes"]
    with pytest.raises(ValueError, match="Duplicate"):
        run_experiment(SACResume(output, tmp_path / "duplicate", steps=1, additions=(extra,)))
    assert not (tmp_path / "duplicate").exists()
    source_file = output / "experience" / merged["source_inventory"][0]["path"]
    source_file.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source"):
        run_experiment(SACResume(output, tmp_path / "broken-source", steps=1))
    assert not (tmp_path / "broken-source/policy.json").exists()


def test_stop_during_sampling_archives_the_completed_response_without_more_actions(tmp_path):
    checkpoint = initial(tmp_path)
    environment = ResponsiveEnvironment(tmp_path)
    environment.stop_after = 1
    result = run_experiment(
        cycle_request(tmp_path, checkpoint), sac_environment=environment
    ).summary["sac_cycle"]
    assert result["stop_reason"] == "stop_requested"
    assert len(environment.commands) == 1
    assert len(result["attempts"]) == 1
    assert environment.closed
    assert "latest_candidate" not in result
    assert (tmp_path / "cycle/attempt-000/recording/packets.jsonl").is_file()


def test_failed_send_is_retained_without_training_or_starting_another_attempt(tmp_path):
    checkpoint = initial(tmp_path)
    environment = ResponsiveEnvironment(tmp_path)
    environment.fail_at = 1
    result = run_experiment(
        cycle_request(tmp_path, checkpoint), sac_environment=environment
    ).summary["sac_cycle"]
    assert result["stop_reason"] == "sampling_fault"
    assert len(environment.commands) == 1
    assert environment.closed
    (attempt,) = result["attempts"]
    assert [d["status"] for d in attempt["decisions"]] == ["sent", "failed_or_unknown"]
    assert "latest_candidate" not in result
    trace = json.loads((tmp_path / "cycle/attempt-000/trace.json").read_bytes())
    assert len(trace["actions"]) == 1
    assert len(trace["observations"]) == 2


@pytest.mark.parametrize("fault", ["reward", "task_context", "too_many_updates", "wrong_hash"])
def test_experience_expansion_refuses_incompatible_inputs_or_excess_updates(tmp_path, fault):
    checkpoint = initial(tmp_path)
    path, sha = addition(tmp_path)
    if fault in ("reward", "task_context"):
        value = json.loads(path.read_bytes())
        if fault == "reward":
            value["source_hashes"]["reward"] = "0" * 64
        else:
            value["task_context"]["max_duration_s"] = 60
        path.write_text(json.dumps(value))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "wrong_hash":
        sha = "0" * 64
    with pytest.raises(ValueError):
        run_experiment(
            SACResume(
                checkpoint,
                tmp_path / "bad",
                steps=3 if fault == "too_many_updates" else 1,
                additions=((path, sha),),
            )
        )
    assert not (tmp_path / "bad").exists()


def test_missing_independent_evidence_cannot_feed_self_reported_reward(tmp_path):
    checkpoint = initial(tmp_path)
    environment = ResponsiveEnvironment(tmp_path)
    environment.finish = lambda recording: None
    result = run_experiment(
        cycle_request(tmp_path, checkpoint), sac_environment=environment
    ).summary["sac_cycle"]
    assert result["stop_reason"] == "no_eligible_experience"
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["eligible_transitions"] == 0
    assert "latest_candidate" not in result
    assert environment.closed


def test_cli_can_resume_an_expanded_candidate_after_sample_directories_move(tmp_path):
    from fh5.cli import main

    checkpoint = initial(tmp_path)
    result = run_experiment(
        cycle_request(tmp_path, checkpoint), sac_environment=ResponsiveEnvironment(tmp_path)
    ).summary["sac_cycle"]
    candidate = tmp_path / "cycle" / result["latest_candidate"]
    moved = tmp_path / "portable"
    candidate.rename(moved)
    (tmp_path / "cycle").rename(tmp_path / "old-run")
    assert (
        main(
            [
                "sac-resume",
                "--checkpoint",
                str(moved),
                "--output",
                str(tmp_path / "resumed"),
                "--steps",
                "1",
            ]
        )
        == 0
    )
    summary = json.loads((tmp_path / "resumed/training-report.json").read_bytes())
    assert summary["total_steps"] == 10


def test_cycle_sampling_matches_public_frozen_replay_with_the_recorded_exploration_noise(tmp_path):
    from fh5.sac_learning import SACPolicyReplay

    checkpoint = initial(tmp_path)
    result = run_experiment(
        cycle_request(tmp_path, checkpoint), sac_environment=ResponsiveEnvironment(tmp_path)
    ).summary["sac_cycle"]
    first = result["attempts"][0]["decisions"][0]
    checked = run_experiment(
        SACPolicyReplay(
            checkpoint,
            checkpoint / "experience/replay.json",
            tmp_path / "noise-check.html",
            noise=tuple(first["noise"]),
        )
    ).summary["sac_policy"]["predictions"][0]
    assert first["command"] == pytest.approx(checked["command"], abs=1e-7)
    assert first["log_probability"] == checked["log_probability"]


def test_failure_feedback_remains_a_terminal_training_transition(tmp_path):
    from test_attempts import evidence

    checkpoint = initial(tmp_path)
    environment = ResponsiveEnvironment(tmp_path)
    environment.finish = lambda recording: evidence(
        recording.parent,
        recording,
        [{"packet_index": 3, "kind": "driving_failure", "status": "confirmed"}],
    )
    result = run_experiment(
        cycle_request(tmp_path, checkpoint), sac_environment=environment
    ).summary["sac_cycle"]
    assert result["stop_reason"] == "budget_completed"
    for attempt in result["attempts"]:
        rows = json.loads((tmp_path / "cycle" / attempt["replay"]).read_bytes())["transitions"]
        assert rows[-1]["reward"] < -2
        assert rows[-1]["terminated"] is True
        assert rows[-1]["bootstrap"] is False
        assert attempt["learner_updates"] == 3


def test_finish_failure_keeps_the_attempt_in_the_cycle_report(tmp_path):
    checkpoint = initial(tmp_path)
    environment = ResponsiveEnvironment(tmp_path)

    def fail_finish(recording):
        raise OSError("independent review unavailable")

    environment.finish = fail_finish
    result = run_experiment(
        cycle_request(tmp_path, checkpoint), sac_environment=environment
    ).summary["sac_cycle"]
    assert len(result["attempts"]) == 1
    assert result["stop_reason"] == "sampling_fault"
    assert "independent review unavailable" in result["attempts"][0]["error"]
    assert "latest_candidate" not in result
    assert environment.closed


def test_cli_explicitly_binds_each_new_replay_to_its_digest(tmp_path, capsys):
    from fh5.cli import main

    checkpoint = initial(tmp_path)
    path, digest = addition(tmp_path)
    assert (
        main(
            [
                "sac-resume",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(tmp_path / "cli-expanded"),
                "--steps",
                "2",
                "--add-replay",
                str(path),
                digest,
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["experience_added_transitions"] == 2
    assert summary["total_steps"] == 5


def test_failed_resource_release_does_not_report_a_successful_cycle(tmp_path):
    checkpoint = initial(tmp_path)
    environment = ResponsiveEnvironment(tmp_path)
    environment.close = lambda: {"resources_released": False}
    result = run_experiment(
        cycle_request(tmp_path, checkpoint), sac_environment=environment
    ).summary["sac_cycle"]
    assert result["stop_reason"] == "release_fault"
    assert result["resources_released"] is False


def test_reformatting_independent_evidence_cannot_credit_the_same_recording_again(tmp_path):
    from fh5.sac_replay import SACReplayPrepare

    checkpoint = initial(tmp_path)
    review = tmp_path / "evidence.json"
    review.write_text(json.dumps(json.loads(review.read_bytes()), indent=2))
    prepared_dir = tmp_path / "reprepared"
    prepared = run_experiment(
        SACReplayPrepare(
            tmp_path / "recording",
            tmp_path / "trace.json",
            tmp_path / "task.json",
            tmp_path / "reward.json",
            prepared_dir,
            review,
        )
    ).summary["sac_replay"]
    assert prepared["eligible_transitions"] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        run_experiment(
            SACResume(
                checkpoint,
                tmp_path / "double-credit",
                steps=2,
                additions=((prepared_dir / "replay.json", prepared["replay_sha256"]),),
            )
        )
    assert not (tmp_path / "double-credit").exists()


def test_frame_archive_failure_keeps_sent_commands_and_raw_telemetry(tmp_path):
    checkpoint = initial(tmp_path)

    class BrokenArchive(ResponsiveEnvironment):
        def step(self, command):
            response = super().step(command)
            path = self.root / "cycle/attempt-000/frames"
            if not path.exists():
                path.write_text("synthetic filesystem failure")
            return response

    environment = BrokenArchive(tmp_path)
    result = run_experiment(
        cycle_request(tmp_path, checkpoint), sac_environment=environment
    ).summary["sac_cycle"]
    assert result["stop_reason"] == "sampling_fault"
    (attempt,) = result["attempts"]
    assert len(attempt["decisions"]) == len(environment.commands) == 3
    assert attempt["archive_error"]
    recorded = (tmp_path / "cycle/attempt-000/recording/packets.jsonl").read_bytes().splitlines()
    assert len(recorded) == 4
    assert (tmp_path / "cycle/attempt-000/sampling.json").exists()
    assert "latest_candidate" not in result


@pytest.mark.parametrize("fault", ["epoch", "clock"])
def test_invalid_response_is_archived_but_never_becomes_a_learning_transition(tmp_path, fault):
    checkpoint = initial(tmp_path)

    class BadResponse(ResponsiveEnvironment):
        def step(self, command):
            sample = super().step(command)
            return replace(
                sample,
                decision=replace(
                    sample.decision,
                    **(
                        {"epoch": "bad-epoch"}
                        if fault == "epoch"
                        else {"decision_ns": 1_000_000_000}
                    ),
                ),
            )

    environment = BadResponse(tmp_path)
    result = run_experiment(
        cycle_request(tmp_path, checkpoint), sac_environment=environment
    ).summary["sac_cycle"]
    assert result["stop_reason"] == "sampling_fault"
    (attempt,) = result["attempts"]
    assert len(attempt["decisions"]) == 1
    assert attempt["decisions"][0]["status"] == "sent"
    recorded = (tmp_path / "cycle/attempt-000/recording/packets.jsonl").read_bytes().splitlines()
    assert len(recorded) == 2
    assert json.loads(recorded[-1])["received_monotonic_ns"] == 1_100_000_000
    assert "latest_candidate" not in result


def test_cycle_binds_learner_parent_to_the_snapshot_used_for_sampling(tmp_path):
    checkpoint = initial(tmp_path)
    replacement = tmp_path / "replacement"
    run_experiment(SACResume(checkpoint, replacement, steps=7))

    class SwappedParent(ResponsiveEnvironment):
        def finish(self, recording):
            saved = tmp_path / "preserved-parent"
            for path in (checkpoint, saved, replacement):
                assert path.resolve().is_relative_to(tmp_path.resolve())
            checkpoint.rename(saved)
            replacement.rename(checkpoint)
            return super().finish(recording)

    result = run_experiment(
        replace(cycle_request(tmp_path, checkpoint), cycles=1),
        sac_environment=SwappedParent(tmp_path),
    ).summary["sac_cycle"]
    assert result["stop_reason"] == "interface_error"
    assert "expected parent" in result["error"]
    assert len(result["attempts"]) == 1
    assert "latest_candidate" not in result
    assert not (tmp_path / "cycle/candidate-000").exists()
