"""BC commands bootstrap value learning through the public experiment interface."""

import json
import socket
import struct
from dataclasses import replace

import pytest
from test_attempts import evidence, protocol
from test_evaluation import sha
from test_numeric_drive_cli import drive_config, synthetic_shadow
from test_numeric_drive_cli import eligible_model as eligible_model
from test_numeric_sac_assembly import Desktop, ExternalWorld
from test_rewards import reward_config
from test_sac_evaluation import sac_policy as sac_policy
from test_sac_realtime_experience import recorded_attempt

from fh5.experiment import Packet, Record, run_experiment
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_drive_config import NumericDriveConfiguration
from fh5.realtime_driving import NumericDrivingEnvironment
from fh5.realtime_shadow import ShadowEnvironment
from fh5.realtime_udp import UDPTelemetry
from fh5.sac import SACCriticWarmup
from fh5.sac_actions import ActionBounds
from fh5.sac_learning import SACResume, SACTrain
from fh5.sac_realtime_experience import SACRealtimePrepare


def test_bc_execution_bootstraps_value_learning_without_a_previous_sac_policy(tmp_path, sac_policy):
    bc = sac_policy / "bc"
    original = (bc / "actor.pt").read_bytes()
    execution, recording, report, factory = recorded_attempt(
        tmp_path, sac_policy, actor_factory=lambda: FrozenNumericActor(bc)
    )
    assert report["actor_kind"] == "frozen-numeric-temporal-bc-v2"
    assert not report["model"].get("command_context")
    prepared = run_experiment(
        SACRealtimePrepare(
            recording,
            execution,
            sac_policy.parent / "task.json",
            sac_policy.parent / "reward.json",
            tmp_path / "experience",
            evidence(tmp_path, recording),
        ),
        numeric_actor=factory(),
    ).summary["sac_replay"]
    assert prepared["eligible_transitions"] >= 2, prepared
    replay = tmp_path / "experience/replay.json"
    data = json.loads(replay.read_bytes())
    assert data["source_kind"] == "synthetic"
    assert not data["real_driving_validated"]
    assert any(
        e["execution_command_index"] == 0 and e["reason"] == "missing_previous_command"
        for e in data["excluded"]
    )
    for row in data["transitions"]:
        index = row["execution_command_index"]
        assert index > 0
        previous = report["commands"][index - 1]
        command = report["commands"][index]
        assert row["previous_action"] == [
            previous["sent"]["steer_i16"] / 32767,
            (previous["sent"]["throttle_u8"] - previous["sent"]["brake_u8"]) / 255,
        ]
        assert row["action"] == [
            command["sent"]["steer_i16"] / 32767,
            (command["sent"]["throttle_u8"] - command["sent"]["brake_u8"]) / 255,
        ]
        assert row["action_elapsed_s"] == pytest.approx(
            (row["current"]["decision_ns"] - previous["returned_ns"]) / 1e9
        )
        progress = sum(s["progress_delta_m"] for s in row["reward_steps"])
        assert row["reward"] == pytest.approx(progress / 3 - row["physical_dt_s"] / 30)
    warm = tmp_path / "warm"
    warmed = run_experiment(
        SACCriticWarmup(
            bc,
            replay,
            sha(replay),
            warm,
            steps=2,
            bounds=ActionBounds(max_steer=0.4, max_throttle=0.25, max_brake=0.5),
        )
    ).summary["sac"]
    assert warmed["steps_completed"] == 2
    assert (bc / "actor.pt").read_bytes() == original
    trained = run_experiment(SACTrain(warm, replay, tmp_path / "candidate", steps=2)).summary[
        "sac_learning"
    ]
    assert trained["steps_completed"] == 2 and trained["critic_change_max"] > 0
    assert not trained["real_driving_validated"]
    assert (bc / "actor.pt").read_bytes() == original


def native_bc_recording(tmp_path, model):
    """Simulate only external devices; never open FH5, DXGI or a game controller."""
    config = drive_config(tmp_path, model)
    synthetic_shadow(tmp_path, model, config, native_file_fixture=True)
    plan = NumericDriveConfiguration(config, tmp_path / "execution", 1.5, True)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    world = ExternalWorld(receiver.getsockname())
    world.steer_feedback_scale = 0.4  # Observed steering is not raw XInput.
    telemetry = UDPTelemetry(receiver.getsockname()[1], receiver=receiver)

    def capture():
        camera = world.start()
        camera.source_kind = "dxgi"  # Deliberately simulated device identity.
        return camera

    observations = ShadowEnvironment(
        plan.request,
        plan.capture,
        capture,
        telemetry,
        Desktop(),
        plan.task,
        input_conditions=plan.bindings,
    )
    env = NumericDrivingEnvironment(observations, lambda: world, configuration=plan)
    report = run_experiment(
        plan.request,
        realtime_environment=env,
        numeric_actor_factory=plan.actor,
    ).summary["realtime"]
    assert report["environment"]["qualification"]["eligible"]
    assert report["stop_reason"] == "time_limit", report["stop_reason"]
    assert report["resources_released"] and world.actuator_closed and world.capture_closed
    events = [
        json.loads(line)
        for line in (plan.request.output_dir / report["journal"]["path"]).read_bytes().splitlines()
    ]
    packets = [
        Packet(
            e["data"]["received_monotonic_ns"],
            e["data"]["received_utc"],
            bytes.fromhex(e["data"]["payload_hex"]),
        )
        for e in events
        if e["kind"] == "packet"
    ]
    record_config = tmp_path / "record.json"
    document = json.loads((tmp_path / "reference.json").read_bytes())
    document["control_source"] = "policy"
    record_config.write_text(json.dumps(document))
    recording = tmp_path / "recording"
    run_experiment(Record(record_config, recording, source_kind="udp"), packets=packets)
    task = protocol(tmp_path, plan.task.route_file)
    document = json.loads(task.read_bytes())
    document["control_owner"] = "policy"
    task.write_text(json.dumps(document))
    request = SACRealtimePrepare(
        recording,
        plan.request.output_dir,
        task,
        reward_config(tmp_path),
        tmp_path / "experience",
        evidence(tmp_path, recording),
    )
    return request, plan, report


def test_native_bc_recording_bootstraps_sac_with_filtered_steering_feedback(
    tmp_path, eligible_model
):
    # "native" here exercises the qualified native adapter with boundary fixtures;
    # this test establishes software behavior, never actual FH5 qualification.
    request, plan, report = native_bc_recording(tmp_path, eligible_model)
    original = (eligible_model / "actor.pt").read_bytes()
    prepared = run_experiment(request, numeric_actor=plan.actor()).summary["sac_replay"]
    assert prepared["eligible_transitions"] >= 2, prepared
    replay = request.output_dir / "replay.json"
    data = json.loads(replay.read_bytes())
    assert data["source_kind"] == "native" and not data["real_driving_validated"]
    assert data["game_application"] == "unverified"
    assert data["source_hashes"]["execution"] == sha(
        request.execution_dir / "realtime-manifest.json"
    )
    assert not any(e["reason"] == "synthetic_response_mismatch" for e in data["excluded"])
    for row in data["transitions"]:
        command = report["commands"][row["execution_command_index"]]
        assert row["action"][0] == command["sent"]["steer_i16"] / 32767
        assert row["action_time_basis"] == "asynchronous_send_return_proxy_v1"
    packets = [
        json.loads(line)
        for line in (request.recording_dir / "packets.jsonl").read_bytes().splitlines()
    ]
    assert any(
        struct.unpack_from("<b", bytes.fromhex(packets[r["packet_range"][1]]["payload_hex"]), 320)[
            0
        ]
        != round(r["action"][0] * 127)
        for r in data["transitions"]
    ), "The fixture must actually distinguish filtered feedback from raw steering commands"
    warm = tmp_path / "warm"
    warmed = run_experiment(
        SACCriticWarmup(
            eligible_model,
            replay,
            sha(replay),
            warm,
            steps=2,
            bounds=ActionBounds(max_steer=0.4, max_throttle=0.25, max_brake=0.5),
        )
    ).summary["sac"]
    assert warmed["steps_completed"] == 2 and warmed["actor_change_max"] == 0
    assert warmed["source_kind"] == "native"
    trained = run_experiment(SACTrain(warm, replay, tmp_path / "candidate", steps=2)).summary[
        "sac_learning"
    ]
    assert trained["source_kind"] == "native" and not trained["real_driving_validated"]
    assert trained["critic_change_max"] > 0 and trained["steps_completed"] == 2
    assert json.loads((tmp_path / "candidate/policy.json").read_bytes())["source_kind"] == "native"
    assert (eligible_model / "actor.pt").read_bytes() == original

    # The same complete recording without independent review cannot bootstrap rewards.
    unreviewed = run_experiment(
        replace(request, output_dir=tmp_path / "unreviewed", evidence_file=None),
        numeric_actor=plan.actor(),
    ).summary["sac_replay"]
    assert unreviewed["eligible_transitions"] == 0


def test_mixed_training_retains_each_source_and_rejects_a_false_native_label(
    tmp_path, eligible_model, sac_policy
):
    native, plan, _ = native_bc_recording(tmp_path, eligible_model)
    run_experiment(native, numeric_actor=plan.actor())
    replay = native.output_dir / "replay.json"
    warm = tmp_path / "warm"
    run_experiment(
        SACCriticWarmup(
            eligible_model,
            replay,
            sha(replay),
            warm,
            steps=1,
            bounds=ActionBounds(max_steer=0.4, max_throttle=0.25, max_brake=0.5),
        )
    )
    candidate = tmp_path / "candidate"
    run_experiment(SACTrain(warm, replay, candidate, steps=1))
    original = (candidate / "policy.pt").read_bytes()
    execution, recording, _, factory = recorded_attempt(
        tmp_path / "synthetic", sac_policy, actor_factory=plan.actor
    )
    prepared = run_experiment(
        SACRealtimePrepare(
            recording,
            execution,
            native.task_file,
            native.reward_file,
            tmp_path / "synthetic-experience",
            evidence(tmp_path / "synthetic", recording),
        ),
        numeric_actor=factory(),
    ).summary["sac_replay"]
    assert prepared["source_kind"] == "synthetic" and prepared["eligible_transitions"] >= 2
    mixed = tmp_path / "mixed"
    learned = run_experiment(
        SACResume(
            candidate,
            mixed,
            steps=1,
            additions=((tmp_path / "synthetic-experience/replay.json", prepared["replay_sha256"]),),
        )
    ).summary["sac_learning"]
    assert learned["source_kind"] == "mixed" and not learned["real_driving_validated"]
    assert json.loads((mixed / "policy.json").read_bytes())["source_kind"] == "mixed"
    assert (candidate / "policy.pt").read_bytes() == original
    combined = json.loads((mixed / "experience/replay.json").read_bytes())
    leaves = [
        json.loads((mixed / "experience" / item["path"]).read_bytes())
        for item in combined["source_inventory"]
    ]
    assert {leaf["source_kind"] for leaf in leaves} == {"native", "synthetic"}
    combined["source_kind"] = "native"
    forged = mixed / "experience/false-native.json"
    forged.write_text(json.dumps(combined))
    with pytest.raises(ValueError, match="source kind"):
        run_experiment(
            SACCriticWarmup(eligible_model, forged, sha(forged), tmp_path / "rejected", steps=1)
        )
    assert not (tmp_path / "rejected/critic.json").exists()
