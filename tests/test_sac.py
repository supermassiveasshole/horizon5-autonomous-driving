"""Learning transitions and critic initialization at the experiment-run boundary."""

import hashlib
import json
import struct

import pytest
from test_attempts import evidence, protocol
from test_numeric_images import actor_state
from test_rewards import reward_config
from test_route_check import route

from fh5.experiment import Packet, Record, run_experiment
from fh5.numeric_images import PixelContract


def experience(tmp_path, *, terminal=True, host_factor=1):
    bundle = route(tmp_path)
    config = tmp_path / "record.json"
    config.write_text(
        json.dumps(
            {**json.loads((tmp_path / "reference.json").read_bytes()), "control_source": "policy"}
        )
    )
    images = bytes([51, 17, 34] * 64 * 36)
    (tmp_path / "frame.rgb").write_bytes(images)
    packets, observations = [], []
    for index, (x, elapsed) in enumerate([(0, 0), (1, 100), (3 if terminal else 2, 300)]):
        now = 1_000_000_000 + elapsed * host_factor * 1_000_000
        raw = bytearray(324)
        struct.pack_into("<iI", raw, 0, 1, 1000 + elapsed)
        struct.pack_into("<iii", raw, 212, 2941, 6, 999)
        struct.pack_into("<ffff", raw, 244, x, 2, 0.2, 4)
        raw[315] = 51
        packets.append(Packet(now, "2026-10-01T00:00:00+00:00", bytes(raw)))
        state = actor_state()
        state["ego"] = {
            "speed_mps": 4.0,
            "velocity_car_mps": [0.0] * 3,
            "angular_velocity_car_radps": [0.0] * 3,
        }
        state.update(ego_age_ms=0, image_age_ms=[210, 130, 10])
        issued = [(900_000_000, [0.0, 0.0])]
        issued.extend(
            (1_000_000_000 + t * host_factor * 1_000_000, [0.0, 0.2])
            for t in (0, 100)
            if t < elapsed
        )
        history, ages = [], []
        for offset in (200, 100, 0):
            matches = [(at, value) for at, value in issued if at < now - offset * 1_000_000]
            previous = matches[-1] if matches else None
            valid = previous is not None and now - offset * 1_000_000 - previous[0] <= 200_000_000
            history.append(previous[1] if valid else None)
            ages.append((now - previous[0]) / 1e6 if valid else None)
        state.update(
            actions=history, action_mask=[a is not None for a in history], action_age_ms=ages
        )
        frames = [
            {
                "epoch": "attempt-0",
                "frame_id": f"{index}:{slot}",
                "source_time_ns": now - age,
                "capture_received_ns": now - age + 1,
                "preprocess_ready_ns": now - age + 2,
                "time_quality": "synthetic",
                "uncertainty_ns": 0,
                "size": [64, 36],
                "source_layout": {"size": [64, 36], "format": "RGB"},
                "availability_kind": "numeric_ready",
                "preprocess_version": "full-frame-pillow-bilinear-v1",
                "path": "frame.rgb",
                "sha256": hashlib.sha256(images).hexdigest(),
            }
            for slot, age in enumerate((210_000_000, 130_000_000, 10_000_000))
        ]
        observations.append(
            {
                "packet_index": index,
                "decision_id": f"d{index}",
                "epoch": "attempt-0",
                "decision_ns": now,
                "actor": state,
                "frames": frames,
            }
        )
    recording = tmp_path / "recording"
    run_experiment(Record(config, recording), packets=packets)
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "synthetic-synchronous-action-trace-v1",
                "pixel_contract": PixelContract(size=(64, 36)).metadata(),
                "initial_command": {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0},
                "initial_issued_ns": 900_000_000,
                "action_offsets_ms": [200, 100, 0],
                "observations": observations,
                "actions": [
                    {
                        "from_packet_index": i,
                        "to_packet_index": i + 1,
                        "epoch": "attempt-0",
                        "owner": "policy",
                        "status": "sent",
                        "sent": {"steer_i16": 0, "throttle_u8": 51, "brake_u8": 0},
                    }
                    for i in range(2)
                ],
            }
        )
    )
    from fh5.sac_replay import SACReplayPrepare

    task = protocol(tmp_path, bundle)
    task.write_text(json.dumps({**json.loads(task.read_bytes()), "control_owner": "policy"}))

    return SACReplayPrepare(
        recording,
        trace,
        task,
        reward_config(tmp_path),
        tmp_path / "prepared",
        evidence(tmp_path, recording),
    )


def test_replay_joins_numeric_inputs_actual_commands_and_independent_physical_rewards(tmp_path):
    request = experience(tmp_path)
    result = run_experiment(request)
    summary = result.summary["sac_replay"]
    assert summary["eligible_transitions"] == 2
    assert summary["source_kind"] == "synthetic"
    assert summary["real_driving_validated"] is False
    rows = json.loads((request.output_dir / "replay.json").read_bytes())["transitions"]
    assert [r["physical_dt_s"] for r in rows] == pytest.approx([0.1, 0.2])
    assert [r["reward"] for r in rows] == pytest.approx([1 / 3 - 0.1 / 30, 2 / 3 + 2 - 0.2 / 30])
    assert [r["bootstrap"] for r in rows] == [True, False]
    assert [r["action"] for r in rows] == [[0, 0.2], [0, 0.2]]
    assert rows[0]["current"]["timing"]["adjacent_delta_s"] == pytest.approx([0.08, 0.12])
    assert rows[0]["current"]["frames"][0]["path"].endswith(".rgb")
    assert result.report_path.is_file()


def test_task_state_at_current_observation_never_uses_next_step_progress_or_time(tmp_path):
    request = experience(tmp_path)
    run_experiment(request)
    replay = json.loads((request.output_dir / "replay.json").read_bytes())
    first, second = replay["transitions"]
    assert first["task_state"]["farthest_confirmed_m"] == 0
    assert first["task_state"]["remaining_s"] == 30
    assert second["task_state"]["farthest_confirmed_m"] == 1
    assert second["task_state"]["remaining_s"] == pytest.approx(29.9)
    assert second["next_task_state"]["farthest_confirmed_m"] == 3
    assert replay["task_state_role"] == "critic_only; frozen BC inputs unchanged"


def test_critic_warmup_updates_values_without_changing_bc_and_reloads_exactly(tmp_path):
    pytest.importorskip("torch")
    from test_temporal_bc import temporal_fixture

    from fh5.sac import SACCriticReplay, SACCriticWarmup
    from fh5.temporal_bc import TemporalBCTrain

    request = experience(tmp_path)
    prepared = run_experiment(request)
    bc_root = tmp_path / "bc"
    bc_root.mkdir()
    config, _ = temporal_fixture(bc_root)
    run_experiment(TemporalBCTrain(config, bc_root / "model"))
    original = (bc_root / "model/actor.pt").read_bytes()
    warm = run_experiment(
        SACCriticWarmup(
            bc_root / "model",
            request.output_dir / "replay.json",
            prepared.summary["sac_replay"]["replay_sha256"],
            tmp_path / "warm",
            steps=3,
        )
    ).summary["sac"]
    assert warm["steps_completed"] == 3
    assert warm["q_change_max"] > 0
    assert warm["actor_change_max"] == 0
    assert warm["stage"] == "critic_warmup"
    assert warm["actor_optimizer_steps"] == 0
    assert warm["real_driving_validated"] is False
    assert warm["predictions"][0]["critic_task_features"] == [0, 0, 1, 1, 1]
    for command in warm["target_actions"]:
        assert command[0] * 32767 == pytest.approx(round(command[0] * 32767))
        assert command[1] * 255 == pytest.approx(round(command[1] * 255))
    assert (tmp_path / "warm/actor/actor.pt").read_bytes() == original
    replayed = run_experiment(
        SACCriticReplay(
            tmp_path / "warm",
            request.output_dir / "replay.json",
            tmp_path / "reloaded.html",
        )
    ).summary["sac"]
    assert replayed["predictions"] == warm["predictions"]
    manifest_path = tmp_path / "warm/critic.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["stage"] = "unrecognized-future-learner"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported critic checkpoint"):
        run_experiment(
            SACCriticReplay(
                tmp_path / "warm",
                request.output_dir / "replay.json",
                tmp_path / "wrong-stage.html",
            )
        )


@pytest.mark.parametrize("missing_final", [False, True])
def test_recording_end_bootstraps_only_from_its_real_final_observation(tmp_path, missing_final):
    request = experience(tmp_path, terminal=False)
    trace = json.loads(request.trace_file.read_bytes())
    if missing_final:
        trace["observations"].pop()
    request.trace_file.write_text(json.dumps(trace))
    result = run_experiment(request)
    rows = json.loads((request.output_dir / "replay.json").read_bytes())["transitions"]
    if missing_final:
        assert len(rows) == 1
        assert result.summary["sac_replay"]["excluded"] == [
            {"action_index": 1, "reason": "missing_bootstrap_observation"}
        ]
    else:
        assert len(rows) == 2
        assert rows[-1]["truncated"] and rows[-1]["bootstrap"]
        assert not rows[-1]["terminated"]


def test_unsent_action_with_no_receipt_is_excluded_without_inventing_a_command(tmp_path):
    request = experience(tmp_path)
    trace = json.loads(request.trace_file.read_bytes())
    trace["actions"][0].update(status="failed", sent=None)
    request.trace_file.write_text(json.dumps(trace))
    result = run_experiment(request)
    assert result.summary["sac_replay"]["excluded"][0] == {
        "action_index": 0,
        "reason": "not_executed_policy_action",
    }
    rows = json.loads((request.output_dir / "replay.json").read_bytes())["transitions"]
    assert all(row["packet_range"] != [0, 1] for row in rows)


def test_missing_numeric_frame_quarantines_dependent_transitions_without_fill(tmp_path):
    request = experience(tmp_path, terminal=False)
    trace = json.loads(request.trace_file.read_bytes())
    trace["observations"][1]["frames"][0]["path"] = "missing.rgb"
    request.trace_file.write_text(json.dumps(trace))
    result = run_experiment(request)
    assert result.summary["sac_replay"]["eligible_transitions"] == 0
    assert len(result.summary["sac_replay"]["excluded"]) == 2
    assert result.summary["sac_replay"]["observation_errors"][0]["packet_index"] == 1


def test_current_command_cannot_leak_into_its_own_actor_history(tmp_path):
    request = experience(tmp_path)
    trace = json.loads(request.trace_file.read_bytes())
    trace["observations"][0]["actor"]["actions"][-1] = [0.0, 0.2]
    request.trace_file.write_text(json.dumps(trace))
    result = run_experiment(request)
    assert result.summary["sac_replay"]["eligible_transitions"] == 1
    assert result.summary["sac_replay"]["observation_errors"][0]["packet_index"] == 0


def test_policy_trace_cannot_relabel_human_telemetry_as_policy_experience(tmp_path):
    request = experience(tmp_path)
    session = request.recording_dir / "session.json"
    session.write_text(json.dumps({**json.loads(session.read_bytes()), "control_source": "human"}))
    request.task_file.write_text(
        json.dumps({**json.loads(request.task_file.read_bytes()), "control_owner": "human"})
    )
    result = run_experiment(request)
    assert result.summary["sac_replay"]["eligible_transitions"] == 0
    assert {r["reason"] for r in result.summary["sac_replay"]["excluded"]} == {
        "control_source_mismatch"
    }


def test_cli_prepares_warms_and_reloads_without_game_adapters(tmp_path, capsys):
    pytest.importorskip("torch")
    from test_temporal_bc import temporal_fixture

    from fh5.cli import main
    from fh5.temporal_bc import TemporalBCTrain

    request = experience(tmp_path)
    args = ["sac-prepare"]
    for flag, value in (
        ("recording", request.recording_dir),
        ("trace", request.trace_file),
        ("task", request.task_file),
        ("reward", request.reward_file),
        ("evidence", request.evidence_file),
        ("output", request.output_dir),
    ):
        args += ["--" + flag, str(value)]
    assert main(args) == 0
    digest = json.loads(capsys.readouterr().out)["replay_sha256"]
    bc = tmp_path / "bc"
    bc.mkdir()
    config, _ = temporal_fixture(bc)
    run_experiment(TemporalBCTrain(config, bc / "model"))
    replay = request.output_dir / "replay.json"
    assert (
        main(
            [
                "sac-warmup",
                "--model",
                str(bc / "model"),
                "--replay",
                str(replay),
                "--replay-sha256",
                digest,
                "--output",
                str(tmp_path / "warm"),
                "--steps",
                "2",
            ]
        )
        == 0
    )
    warm = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "sac-critic-replay",
                "--checkpoint",
                str(tmp_path / "warm"),
                "--replay",
                str(replay),
                "--report",
                str(tmp_path / "again.html"),
            ]
        )
        == 0
    )
    again = json.loads(capsys.readouterr().out)
    assert again["predictions"] == warm["predictions"]
    assert again["commands_sent"] is False


def test_physics_duration_is_separate_from_host_hold_and_frame_deltas(tmp_path):
    request = experience(tmp_path, host_factor=2)
    run_experiment(request)
    rows = json.loads((request.output_dir / "replay.json").read_bytes())["transitions"]
    assert [r["physical_dt_s"] for r in rows] == pytest.approx([0.1, 0.2])
    assert [r["hold_dt_s"] for r in rows] == pytest.approx([0.2, 0.4])
    assert rows[0]["current"]["timing"]["adjacent_delta_s"] == pytest.approx([0.08, 0.12])


def test_one_held_command_aggregates_discounted_physics_intervals(tmp_path):
    import math

    request = experience(tmp_path)
    trace = json.loads(request.trace_file.read_bytes())
    trace["actions"] = [{**trace["actions"][0], "to_packet_index": 2}]
    # No second issue at 1.1 s: final history still refers to the command at 1.0 s.
    trace["observations"][-1]["actor"].update(
        actions=[[0.0, 0.2], [0.0, 0.2], None],
        action_mask=[True, True, False],
        action_age_ms=[300.0, 300.0, None],
    )
    request.trace_file.write_text(json.dumps(trace))
    reward_config(tmp_path, discount_half_lives_per_horizon=1.0)
    run_experiment(request)
    replay = json.loads((request.output_dir / "replay.json").read_bytes())
    assert not replay["observation_errors"]
    assert len(replay["transitions"]) == 1
    row = replay["transitions"][0]
    # Independently integrate time cost over [0,.3] and place progress/finish
    # impulses at physical times .1 and .3, with a 30-second half-life.
    expected = -(1 - 2**-0.01) / math.log(2) + 2 ** (-1 / 300) / 3 + 8 * 2**-0.01 / 3
    assert row["reward"] == pytest.approx(expected)
    assert row["discount"] == pytest.approx(2**-0.01)
    assert row["physical_dt_s"] == pytest.approx(0.3)


def test_warmup_rejects_out_of_support_actions_instead_of_clipping_them(tmp_path):
    pytest.importorskip("torch")
    from test_temporal_bc import temporal_fixture

    from fh5.sac import SACCriticWarmup
    from fh5.sac_actions import ActionBounds
    from fh5.temporal_bc import TemporalBCTrain

    request = experience(tmp_path)
    prepared = run_experiment(request)
    bc_root = tmp_path / "bc"
    bc_root.mkdir()
    config, _ = temporal_fixture(bc_root)
    run_experiment(TemporalBCTrain(config, bc_root / "model"))
    with pytest.raises(ValueError, match="never clip replay labels"):
        run_experiment(
            SACCriticWarmup(
                bc_root / "model",
                request.output_dir / "replay.json",
                prepared.summary["sac_replay"]["replay_sha256"],
                tmp_path / "warm",
                steps=1,
                bounds=ActionBounds(max_throttle=0.1),
            )
        )
    assert not (tmp_path / "warm").exists()


@pytest.mark.parametrize("fault", ["response", "epoch", "confirmed_failure"])
def test_faults_preserve_reasons_and_do_not_erase_a_real_failure(tmp_path, fault):
    request = experience(tmp_path)
    trace = json.loads(request.trace_file.read_bytes())
    if fault == "response":
        trace["actions"][1]["sent"]["throttle_u8"] = 52
    elif fault == "epoch":
        trace["observations"][1]["epoch"] = "after-reset"
        for frame in trace["observations"][1]["frames"]:
            frame["epoch"] = "after-reset"
    else:
        evidence(
            tmp_path,
            request.recording_dir,
            [{"packet_index": 1, "kind": "driving_failure", "status": "confirmed"}],
        )
    request.trace_file.write_text(json.dumps(trace))
    result = run_experiment(request)
    rows = json.loads((request.output_dir / "replay.json").read_bytes())["transitions"]
    assert result.summary["sac_replay"]["excluded"]
    if fault == "confirmed_failure":
        assert len(rows) == 1
        assert rows[0]["terminated"] and not rows[0]["bootstrap"]
        assert rows[0]["reward"] == pytest.approx(-3 - 0.1 / 30)
    elif fault == "response":
        assert len(rows) == 1
        assert (
            result.summary["sac_replay"]["excluded"][0]["reason"] == "synthetic_response_mismatch"
        )
    else:
        assert rows == []
