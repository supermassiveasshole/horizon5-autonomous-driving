"""Local attempt verdicts through the agreed experiment-run interface."""

import hashlib
import json
import struct

import pytest

from fh5.evaluation.attempts import AttemptReplay
from fh5.experiment import run_experiment
from tests.observation.test_route_check import record, route


def protocol(tmp_path, bundle):
    path = tmp_path / "task.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "task_id": "reviewed-straight",
                "scope": "local",
                "route_file": str(bundle),
                "route_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "geometry_source_sha256": [],
                "start_mode": "manual_placement",
                "control_owner": "human",
                "expected_car_ordinal": 2941,
                "expected_pi": 999,
                "max_speed_kmh": 20,
                "max_duration_s": 30,
                "no_progress_timeout_s": 10,
            }
        )
    )
    return path


def evidence(tmp_path, source, events=(), *, coverage=True):
    proof = tmp_path / "independent-review.md"
    proof.write_text("Independent synthetic fixture evidence; not real-game recognition.")
    path = tmp_path / "evidence.json"
    count = len((source / "packets.jsonl").read_bytes().splitlines())
    provenance = {
        "source": "independent_review",
        "reviewer": "fixture observer",
        "evidence": ["review"],
    }
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "recording_sha256": hashlib.sha256(
                    (source / "packets.jsonl").read_bytes()
                ).hexdigest(),
                "items": [
                    {
                        "id": "review",
                        "path": proof.name,
                        "sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
                    }
                ],
                "coverage": [
                    {
                        "packet_range": [0, count - 1],
                        "checks": [
                            "wall_riding",
                            "reset_boost",
                            "grass_shortcut",
                            "interventions",
                            "conditions",
                        ],
                        **provenance,
                    }
                ]
                if coverage
                else [],
                "events": [{**event, **provenance} for event in events],
            }
        )
    )
    return path


def test_geometric_completion_without_independent_review_is_quarantined(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    result = run_experiment(AttemptReplay(source, tmp_path / "review", protocol(tmp_path, bundle)))
    review = result.summary["attempt_review"]
    assert review["recording_packet_count"] == 4
    assert len(review["attempts"]) == 1
    attempt = review["attempts"][0]
    assert attempt["packet_range"] == [0, 3]
    assert attempt["outcome"] == "pending_review"
    assert attempt["task_completed"] is True
    assert attempt["confirmed_progress_m"] == 3
    assert attempt["control_owner"] == "human"
    assert attempt["start_mode"] == "manual_placement"
    assert attempt["formal_result"]["outcome"] == "pending_review"
    assert not attempt["unattended"]
    assert not attempt["record_eligible"]


@pytest.mark.parametrize("change", ["bad_task", "partial", "policy_clear", "reused", "tampered"])
def test_evidence_cannot_self_authorize_or_change_after_review(tmp_path, change):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    task = protocol(tmp_path, bundle)
    proof = evidence(tmp_path, source)
    config = json.loads(task.read_text())
    review = json.loads(proof.read_text())
    if change == "bad_task":
        config["max_duration_s"] = float("nan")
    if change == "partial":
        review["coverage"][0]["packet_range"] = [1, 3]
    if change == "policy_clear":
        review["coverage"][0]["source"] = "policy_prediction"
    if change == "reused":
        config["geometry_source_sha256"] = [review["recording_sha256"]]
    if change == "tampered":
        (tmp_path / "independent-review.md").write_text("changed after review")
    task.write_text(json.dumps(config))
    proof.write_text(json.dumps(review))
    request = AttemptReplay(source, tmp_path / "review", task, proof)
    if change in {"bad_task", "tampered"}:
        with pytest.raises(ValueError):
            run_experiment(request)
        assert not (tmp_path / "review").exists()
    else:
        attempt = run_experiment(request).summary["attempt_review"]["attempts"][0]
        assert attempt["outcome"] == "pending_review"
        assert not attempt["record_eligible"]


def test_independent_review_allows_local_completion_with_incidental_contact(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.4), (2, -0.2), (3, 0.2)])
    reviewed = evidence(
        tmp_path,
        source,
        [
            {"packet_index": 1, "kind": "incidental_contact", "status": "confirmed"},
            {"packet_index": 2, "kind": "reasonable_cut", "status": "confirmed"},
        ],
    )
    result = run_experiment(
        AttemptReplay(source, tmp_path / "review", protocol(tmp_path, bundle), reviewed)
    )
    attempt = result.summary["attempt_review"]["attempts"][0]
    assert attempt["outcome"] == "valid_complete"
    assert attempt["record_eligible"] is True
    assert attempt["formal_result"]["outcome"] == "pending_review"
    assert attempt["contact_count"] == 1
    assert attempt["evidence_mode"] == "manual_review"
    assert not attempt["automatic_promotion_allowed"]
    assert len(attempt["forward_segments"]) == 1
    assert attempt["forward_segments"][0]["packet_range"] == [0, 3]
    assert (tmp_path / "review/evidence").is_dir()


@pytest.mark.parametrize(
    "kind", ["wall_riding", "reset_boost", "grass_shortcut", "human_takeover", "rewind"]
)
def test_confirmed_violation_survives_geometric_completion(tmp_path, kind):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    reviewed = evidence(
        tmp_path, source, [{"packet_index": 1, "kind": kind, "status": "confirmed"}]
    )
    result = run_experiment(
        AttemptReplay(source, tmp_path / "review", protocol(tmp_path, bundle), reviewed)
    )
    attempt = result.summary["attempt_review"]["attempts"][0]
    assert attempt["outcome"] == "invalid"
    assert kind in attempt["reasons"]
    assert not attempt["record_eligible"]
    assert attempt["formal_result"]["outcome"] == "invalid"


@pytest.mark.parametrize(
    "source_kind,status", [("independent_review", "suspected"), ("policy_prediction", "confirmed")]
)
def test_suspected_or_policy_only_violation_is_quarantined_not_convicted(
    tmp_path, source_kind, status
):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    reviewed = evidence(
        tmp_path, source, [{"packet_index": 1, "kind": "wall_riding", "status": status}]
    )
    data = json.loads(reviewed.read_text())
    data["events"][0]["source"] = source_kind
    reviewed.write_text(json.dumps(data))
    result = run_experiment(
        AttemptReplay(source, tmp_path / "review", protocol(tmp_path, bundle), reviewed)
    )
    attempt = result.summary["attempt_review"]["attempts"][0]
    assert attempt["outcome"] == "pending_review"
    assert "suspected_wall_riding" in attempt["reasons"]
    assert not attempt["record_eligible"]


def test_restart_preserves_failed_attempt_and_separates_next_completion(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    reviewed = evidence(
        tmp_path, source, [{"packet_index": 2, "kind": "restart", "status": "confirmed"}]
    )
    result = run_experiment(
        AttemptReplay(source, tmp_path / "review", protocol(tmp_path, bundle), reviewed)
    )
    attempts = result.summary["attempt_review"]["attempts"]
    assert len(attempts) == 2
    assert [a["packet_range"] for a in attempts] == [[0, 1], [2, 5]]
    assert [a["outcome"] for a in attempts] == ["driving_failed", "valid_complete"]
    assert attempts[0]["attempt_id"] != attempts[1]["attempt_id"]
    assert not attempts[1]["unattended"]


@pytest.mark.parametrize("kind", ["pause", "rewind"])
def test_recovery_splits_forward_segments_and_never_stitches_progress(tmp_path, kind):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    reviewed = evidence(
        tmp_path, source, [{"packet_index": 2, "kind": kind, "status": "confirmed"}]
    )
    result = run_experiment(
        AttemptReplay(source, tmp_path / "review", protocol(tmp_path, bundle), reviewed)
    )
    attempts = result.summary["attempt_review"]["attempts"]
    assert len(attempts) == 1
    attempt = attempts[0]
    assert [s["packet_range"] for s in attempt["forward_segments"]] == [[0, 1]]
    assert [s["packet_range"] for s in attempt["excluded_spans"]] == [[2, 3]]
    assert attempt["confirmed_progress_m"] == 1
    assert not attempt["task_completed"]
    assert attempt["outcome"] == ("invalid" if kind == "rewind" else "pending_review")


def test_rewind_animation_excluded_and_new_forward_completion_retains_failure(tmp_path):
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
            {"packet_index": 2, "resume_packet_index": 4, "kind": "rewind", "status": "confirmed"},
        ],
    )
    result = run_experiment(
        AttemptReplay(source, tmp_path / "review", protocol(tmp_path, bundle), proof)
    )
    attempt = result.summary["attempt_review"]["attempts"][0]
    assert attempt["outcome"] == "invalid"
    assert attempt["task_completed"]
    assert [s["packet_range"] for s in attempt["forward_segments"]] == [[0, 1], [4, 7]]
    assert attempt["excluded_spans"][0]["packet_range"] == [2, 3]
    assert attempt["confirmed_progress_m"] == 3
    assert all(result.samples[i]["attempt_phase"] == "recovery_excluded" for i in (2, 3))
    assert "reviewed_stop_or_failure" in attempt["reasons"]


def test_empty_recording_retains_interface_failure_and_report_contract(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [])
    result = run_experiment(AttemptReplay(source, tmp_path / "review", protocol(tmp_path, bundle)))
    attempt = result.summary["attempt_review"]["attempts"][0]
    assert attempt["outcome"] == "interface_error"
    assert "no_telemetry" in attempt["reasons"]
    assert attempt["packet_range"] == [0, -1]
    assert attempt["task_packet_range"] is None
    assert not attempt["record_eligible"]
    page = result.report_path.read_text(encoding="utf-8")
    assert 'id="attempt-panel"' in page
    assert 'id="attempt-rows"' in page
    assert attempt["formal_result"]["required_evidence"] == [
        "full_race_start",
        "ordered_official_checkpoints",
        "game_finish_signal",
        "whole_attempt_validity",
    ]


def test_suspected_restart_isolated_and_stop_after_local_finish_is_not_failure(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2), (3, 0.2)])
    task = protocol(tmp_path, bundle)
    proof = evidence(tmp_path, source, [{"packet_index": 4, "kind": "stop", "status": "confirmed"}])
    attempt = run_experiment(AttemptReplay(source, tmp_path / "stopped", task, proof)).summary[
        "attempt_review"
    ]["attempts"][0]
    assert attempt["outcome"] == "valid_complete"
    proof = evidence(
        tmp_path, source, [{"packet_index": 2, "kind": "restart", "status": "suspected"}]
    )
    attempt = run_experiment(AttemptReplay(source, tmp_path / "suspect", task, proof)).summary[
        "attempt_review"
    ]["attempts"][0]
    assert attempt["outcome"] == "pending_review"
    assert "suspected_restart" in attempt["reasons"]


def test_missing_control_journal_cannot_pass_as_successful_autonomous_input(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    session = source / "session.json"
    metadata = json.loads(session.read_text())
    metadata["control_source"] = "calibration"
    session.write_text(json.dumps(metadata))
    task = protocol(tmp_path, bundle)
    config = json.loads(task.read_text())
    config["control_owner"] = "calibration"
    task.write_text(json.dumps(config))
    a = run_experiment(
        AttemptReplay(source, tmp_path / "review", task, evidence(tmp_path, source))
    ).summary["attempt_review"]["attempts"][0]
    assert a["outcome"] == "interface_error"
    assert "control_evidence_incomplete" in a["reasons"]


@pytest.mark.parametrize(
    "artifact,reason",
    [
        ("control.json", "control_evidence_incomplete"),
        ("event-run.json", "event_evidence_incomplete"),
    ],
)
def test_corrupt_sidecar_keeps_complete_telemetry_and_interface_result(tmp_path, artifact, reason):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    (source / artifact).write_text("[]")
    result = run_experiment(
        AttemptReplay(
            source, tmp_path / "review", protocol(tmp_path, bundle), evidence(tmp_path, source)
        )
    )
    assert len(result.samples) == 4
    a = result.summary["attempt_review"]["attempts"][0]
    assert a["outcome"] == "interface_error"
    assert reason in a["reasons"]


def test_string_release_flag_is_not_confirmation_of_control_release(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    neutral = {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0}
    command = {
        "issued_ns": 1,
        "returned_ns": 2,
        "owner": "stop_guard",
        "requested": {},
        "target": neutral,
        "sent": neutral,
        "status": "sent",
    }
    (source / "commands.jsonl").write_text(json.dumps(command) + "\n")
    (source / "control.json").write_text(
        json.dumps(
            {
                "version": 1,
                "stop_reason": "completed",
                "release_sent": "false",
                "commands": [command],
            }
        )
    )
    a = run_experiment(
        AttemptReplay(
            source, tmp_path / "review", protocol(tmp_path, bundle), evidence(tmp_path, source)
        )
    ).summary["attempt_review"]["attempts"][0]
    assert a["outcome"] == "interface_error"
    assert "control_evidence_incomplete" in a["reasons"]


def test_local_entry_and_end_do_not_crop_complete_attempt(tmp_path):
    bundle = route(tmp_path)
    source = record(tmp_path, "drive", [(-5, 0), (0, 0.2), (1, 0.2), (3, 0.2), (6, 3)])
    result = run_experiment(
        AttemptReplay(
            source, tmp_path / "review", protocol(tmp_path, bundle), evidence(tmp_path, source)
        )
    )
    attempt = result.summary["attempt_review"]["attempts"][0]
    assert attempt["packet_range"] == [0, 4]
    assert attempt["task_packet_range"] == [1, 3]
    assert attempt["outcome"] == "valid_complete"
    assert [s["attempt_phase"] for s in result.samples] == [
        "approach",
        "local_task",
        "local_task",
        "local_task",
        "after_local_task",
    ]


def test_unreviewed_activity_loss_cannot_reset_a_failure_into_success(tmp_path):
    bundle = route(tmp_path)
    source = record(
        tmp_path, "drive", [(0, 0.2), (1, 0.2), (0, 0.2), (0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)]
    )
    path = source / "packets.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    raw = bytearray.fromhex(rows[2]["payload_hex"])
    struct.pack_into("<i", raw, 0, 0)
    rows[2]["payload_hex"] = raw.hex()
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = run_experiment(
        AttemptReplay(
            source, tmp_path / "review", protocol(tmp_path, bundle), evidence(tmp_path, source)
        )
    )
    attempt = result.summary["attempt_review"]["attempts"][0]
    assert attempt["outcome"] == "pending_review"
    assert "activity_interrupted" in attempt["reasons"]
    assert not attempt["record_eligible"]


@pytest.mark.parametrize(
    "fault,expected,reason",
    [
        ("speed", "driving_failed", "speed_limit_exceeded"),
        ("timeout", "driving_failed", "task_timeout"),
        ("stall", "driving_failed", "no_progress_timeout"),
        ("gap", "interface_error", "receive_gap"),
        ("corrupt", "interface_error", "unsupported_packet"),
        ("frozen", "interface_error", "game_clock_frozen"),
        ("outside", "pending_review", "unconfirmed_path"),
        ("car", "invalid", "vehicle_mismatch"),
        ("late_placement", "invalid", "human_takeover"),
    ],
)
def test_failures_cannot_be_hidden_by_later_completion(tmp_path, fault, expected, reason):
    bundle = route(tmp_path)
    points = [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)]
    if fault == "outside":
        points[1] = (1, 2)
    if fault == "stall":
        points[1:3] = [(0, 0.2), (0, 0.2)]
    source = record(tmp_path, "drive", points, speed=10 if fault == "speed" else 4)
    path = source / "packets.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == "gap":
        for row in rows[2:]:
            row["received_monotonic_ns"] += 1_000_000_000
    if fault == "corrupt":
        rows[1]["payload_hex"] = "00"
    if fault == "car":
        payload = bytearray.fromhex(rows[2]["payload_hex"])
        struct.pack_into("<I", payload, 212, 123)
        rows[2]["payload_hex"] = payload.hex()
    if fault == "frozen":
        for i, row in enumerate(rows):
            payload = bytearray.fromhex(row["payload_hex"])
            struct.pack_into("<I", payload, 4, 1000)
            row["payload_hex"] = payload.hex()
            row["received_monotonic_ns"] = 1_000_000_000 + i * 200_000_000
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    task = protocol(tmp_path, bundle)
    config = json.loads(task.read_text())
    if fault in {"timeout", "stall"}:
        config["max_duration_s" if fault == "timeout" else "no_progress_timeout_s"] = 0.15
    task.write_text(json.dumps(config))
    events = (
        [{"packet_index": 1, "kind": "human_placement", "status": "confirmed"}]
        if fault == "late_placement"
        else []
    )
    result = run_experiment(
        AttemptReplay(source, tmp_path / "review", task, evidence(tmp_path, source, events))
    )
    attempt = result.summary["attempt_review"]["attempts"][0]
    assert attempt["outcome"] == expected
    assert reason in attempt["reasons"]
    assert not attempt["record_eligible"]
