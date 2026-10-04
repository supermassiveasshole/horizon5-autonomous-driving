"""Historical v1 policy recordings remain readable without the retired online driver."""

import base64
import hashlib
import json
import struct

from fh5.experiment import Packet, Record, Replay, run_experiment


def test_replay_preserves_v1_policy_evidence_and_reports_a_damaged_original(tmp_path):
    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    config = tmp_path / "record.json"
    write_json(
        config,
        {
            "schema_version": 1,
            "control_source": "policy",
            "snapshot": {
                key: {"value": None, "status": "unverified"}
                for key in ("vehicle", "variant", "tune", "assists", "event", "environment")
            },
        },
    )
    packets = []
    for index in range(2):
        raw = bytearray(324)
        struct.pack_into("<iI", raw, 0, 1, 1000 + index * 100)
        struct.pack_into("<iii", raw, 212, 2941, 6, 999)
        struct.pack_into("<ffff", raw, 244, index * 0.1, 2, 0.2, 1)
        packets.append(
            Packet(1_000_000_000 + index * 100_000_000, "2026-10-03T00:00:00+00:00", bytes(raw))
        )
    directory = tmp_path / "historical"
    recorded = run_experiment(Record(config, directory), packets=packets)

    neutral = {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0}
    action = {"steer_i16": 3277, "throttle_u8": 51, "brake_u8": 0}
    commands = [
        {
            "issued_ns": at,
            "returned_ns": at + 1_000_000,
            "owner": owner,
            "target": command,
            "sent": command,
            "status": "sent",
        }
        for at, owner, command in (
            (1_020_000_000, "policy", action),
            (1_120_000_000, "stop_guard", neutral),
        )
    ]
    (directory / "commands.jsonl").write_text(
        "".join(json.dumps(command) + "\n" for command in commands), encoding="utf-8"
    )
    write_json(
        directory / "control.json",
        {"version": 1, "stop_reason": "time_limit", "release_sent": True, "commands": commands},
    )
    (directory / "frames").mkdir()
    image = directory / "frames/000000.png"
    image.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        )
    )
    frame = {
        "kind": "frame",
        "frame_index": 0,
        "path": "frames/000000.png",
        "sha256": digest(image),
        "capture_start_ns": 1_005_000_000,
        "capture_end_ns": 1_006_000_000,
        "available_ns": 1_010_000_000,
        "delivered_ns": 1_011_000_000,
        "stored_ns": 1_012_000_000,
        "encoding": "png",
        "image_size": [1, 1],
        "source_size": [1, 1],
    }
    (directory / "vision.jsonl").write_text(json.dumps(frame) + "\n", encoding="utf-8")
    write_json(directory / "vision-config.json", {"max_age_ms": 200})
    write_json(
        directory / "vision-session.json",
        {
            "version": 1,
            "camera_mode": "chase_far",
            "camera_status": "user_reported",
            "camera_pose": "dynamic_unknown",
            "settings": {"max_age_ms": 200},
            "resources_released": True,
            "started_ns": 1_000_000_000,
            "ended_ns": 1_121_000_000,
            "stop_reason": "time_limit",
            "hashes": {
                name: digest(directory / name)
                for name in ("vision-config.json", "vision.jsonl", "packets.jsonl")
            },
        },
    )
    decisions = [
        {
            "decision_ns": 1_020_000_000,
            "observation": {"telemetry_packet_index": 0, "images": [frame]},
            "prediction": [0.1, 0.2],
            "sent": action,
        }
    ]
    journal = directory / "policy-decisions.jsonl"
    journal.write_text(json.dumps(decisions[0]) + "\n", encoding="utf-8")
    route = {
        "version": 1,
        "length_m": 3,
        "low_speed_ready": False,
        "points": [
            {"s_m": 0, "position_m": [0, 2, 0]},
            {"s_m": 3, "position_m": [3, 2, 0]},
        ],
        "checkpoints": [],
        "corridors": [],
    }
    hashes = {
        name: digest(directory / name)
        for name in (
            "packets.jsonl",
            "commands.jsonl",
            "control.json",
            "vision.jsonl",
            "vision-session.json",
            "policy-decisions.jsonl",
        )
    }
    write_json(
        directory / "policy.json",
        {
            "version": 1,
            "stop_reason": "time_limit",
            "release_sent": True,
            "actor_kind": "historical_synthetic_fixture",
            "decisions": decisions,
            "commands": commands,
            "evaluation_route": route,
            "hashes": hashes,
        },
    )
    original_policy = (directory / "policy.json").read_bytes()
    replay = run_experiment(Replay(directory, tmp_path / "replayed.html"))
    assert replay.samples == recorded.samples
    assert replay.metadata["control_source"] == "policy"
    assert replay.summary["policy"]["decisions"] == decisions
    assert replay.summary["policy"]["hashes"] == hashes
    assert replay.summary["policy"]["artifact_errors"] == []
    assert replay.summary["policy"]["formal_validity"] == "pending_independent_review"
    assert replay.summary["route"] == route
    assert replay.summary["control"]["commands"] == commands
    assert replay.summary["control"]["artifact_errors"] == []
    assert replay.summary["vision"]["integrity_errors"] == []
    assert replay.summary["vision"]["frames"][0]["online"]["usable"]
    assert replay.summary["vision"]["frames"][0]["sha256"] == frame["sha256"]
    saved_report = json.loads(replay.report_path.with_suffix(".json").read_bytes())
    assert saved_report["summary"]["policy"]["decisions"] == decisions

    with journal.open("ab") as stream:
        stream.write(b'{"decision_ns":')
    damaged = run_experiment(Replay(directory, tmp_path / "damaged.html"))
    assert damaged.report_path.is_file()
    assert damaged.summary["policy"]["decisions"] == decisions
    assert damaged.summary["route"] == route
    assert (
        "Policy evidence changed: policy-decisions.jsonl"
        in damaged.summary["policy"]["artifact_errors"]
    )
    assert "Invalid policy decision journal" in damaged.summary["policy"]["artifact_errors"]
    assert damaged.summary["policy"]["formal_validity"] == "pending_independent_review"
    assert (directory / "policy.json").read_bytes() == original_policy
