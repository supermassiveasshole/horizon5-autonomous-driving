"""Evaluation evidence binding through real recording and experiment entry points."""

import hashlib
import json
import struct
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from fh5.driving.realtime.model import RealtimeConfig, RealtimeRun
from fh5.evaluation.prepare import EvaluationReview
from fh5.experiment import Packet, Record, run_experiment
from fh5.learning.bc.actor import FrozenNumericActor
from fh5.observation.numeric import PixelContract
from fh5.observation.recording import read_numeric_frame
from tests.driving.test_realtime import ThreadedGame
from tests.evaluation.test_attempts import evidence
from tests.evaluation.test_evaluation import entry, ledger, prepare, sha
from tests.evaluation.test_evaluation import policy as policy
from tests.observation.test_route_check import record


class PacketGame(ThreadedGame):
    """Synthetic stationary car; no claim of physical vehicle simulation."""

    def __init__(self):
        super().__init__()
        self.packets = []

    def read(self, period_s):
        point = super().read(period_s)
        game_ms = point.safety.game_timestamp_ms % 2**32
        raw = bytearray(324)
        struct.pack_into("<iI", raw, 0, 1, game_ms)
        struct.pack_into("<iii", raw, 212, 2941, 6, 999)
        struct.pack_into("<ffff", raw, 244, 0, 2, 0.2, 0)
        packet = Packet(point.at_ns, datetime.now(UTC).isoformat(), bytes(raw))
        self.packets.append(packet)
        return replace(
            point,
            safety=replace(
                point.safety,
                game_timestamp_ms=game_ms,
                telemetry_packet_index=len(self.packets) - 1,
            ),
            raw_packets=(packet,),
            observation=replace(
                point.observation,
                frames=tuple(
                    replace(f, size=(64, 36), pixels=memoryview(bytes([51, 17, 34] * 64 * 36)))
                    for f in point.observation.frames
                ),
            ),
        )


def execution(
    tmp_path,
    policy,
    *,
    shadow=False,
    source_kind="synthetic",
    reference_mode="no_reference",
    **options,
):
    prepare(tmp_path, policy, [reference_mode])
    pixels = PixelContract(size=(64, 36))
    game = PacketGame()
    game.source_kind = source_kind

    def factory():
        if shadow:
            from fh5.driving.realtime.observation import ShadowNumericActor

            model = tmp_path / "frozen/model"
            return ShadowNumericActor(
                model, pixels, json.loads((model / "model.json").read_bytes())["weights_sha256"]
            )
        return FrozenNumericActor(tmp_path / "frozen/model", pixels)

    report = run_experiment(
        RealtimeRun(
            tmp_path / "execution",
            RealtimeConfig(pixels=pixels, reference_count=1),
            seconds=0.35,
            **options,
        ),
        realtime_environment=game,
        numeric_actor_factory=factory,
    ).summary["realtime"]
    config = tmp_path / "record.json"
    options = json.loads((tmp_path / "evaluation.json").read_bytes())
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "control_source": "human",
                "snapshot": options["conditions"]["snapshot"],
            }
        )
    )
    run_experiment(
        Record(config, tmp_path / "recording", "udp" if source_kind == "shadow" else "synthetic"),
        packets=game.packets,
    )
    row = entry("run-0", tmp_path / "recording")
    row["execution"] = {
        "directory": "execution",
        "manifest_sha256": sha(tmp_path / "execution/realtime-manifest.json"),
    }
    return ledger(tmp_path, [row]), report


def test_batch_binds_real_frozen_predictions_to_the_same_recorded_telemetry(tmp_path, policy):
    path, original = execution(tmp_path, policy)
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", path, tmp_path / "review")
    ).summary["evaluation"]
    proof = summary["executions"][0]
    assert proof["status"] == "bound_diagnostic"
    assert proof["slot_id"] == "run-0"
    assert proof["verified_predictions"] >= 3
    assert proof["observed_reference_modes"] == ["no_reference"]
    assert proof["metrics"]["accepted_decisions"] >= 3
    assert 0.5 <= proof["metrics"]["visual_available_fraction"] <= 1
    assert proof["metrics"]["evidence_kind"] == "synthetic"
    assert proof["metrics"]["decision_count"] == len(original["decisions"])
    assert summary["attempts"][0]["execution_id"] == "run-0"
    assert summary["automatic_promotion_allowed"] is False
    assert summary["closed_loop_validated"] is False


def test_unrelated_packet_stream_cannot_borrow_a_valid_execution_log(tmp_path, policy):
    path, _ = execution(tmp_path, policy)
    packets = tmp_path / "recording/packets.jsonl"
    values = [json.loads(line) for line in packets.read_text().splitlines()]
    for value in values:
        value["received_monotonic_ns"] += 10_000_000_000
    packets.write_text("".join(json.dumps(v) + "\n" for v in values))
    binding = json.loads(path.read_bytes())
    binding["entries"][0]["files"]["packets.jsonl"] = sha(packets)
    path.write_text(json.dumps(binding))
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", path, tmp_path / "review")
    ).summary["evaluation"]
    assert summary["metrics"]["all_attempts"] == 1
    assert summary["executions"][0]["status"] == "quarantined"
    assert summary["executions"][0]["metrics"] is None
    assert "packet stream" in str(summary["executions"][0]["reasons"])


def test_supplied_wrong_execution_quarantines_otherwise_valid_local_result(tmp_path, policy):
    path, _ = execution(tmp_path, policy)
    binding = json.loads(path.read_bytes())["entries"][0]["execution"]
    source = record(tmp_path, "successful-record", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    row = entry("run-0", source, evidence(tmp_path, source))
    row["execution"] = binding
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", ledger(tmp_path, [row]), tmp_path / "review")
    ).summary["evaluation"]
    attempt = summary["attempts"][0]
    assert attempt["local_outcome"] == "valid_complete"
    assert attempt["outcome"] == "pending_review"
    assert attempt["record_eligible"] is False
    assert summary["metrics"]["all_attempts"] == 1
    assert summary["metrics"]["valid_duration_s"]["count"] == 0
    assert "execution_quarantined" in attempt["evidence_gaps"]


def replace_report(tmp_path, ledger_path, report):
    raw = json.dumps(report).encode()
    (tmp_path / "execution/report.json").write_bytes(raw)
    manifest = tmp_path / "execution/realtime-manifest.json"
    manifest.write_text(
        json.dumps({"version": 1, "report_sha256": hashlib.sha256(raw).hexdigest()})
    )
    binding = json.loads(ledger_path.read_bytes())
    binding["entries"][0]["execution"]["manifest_sha256"] = sha(manifest)
    ledger_path.write_text(json.dumps(binding))


def test_execution_metrics_recompute_the_timeline_instead_of_trusting_summary_values(
    tmp_path, policy
):
    path, original = execution(tmp_path, policy)
    original["metrics"] = {"effective_hz": 999999, "maximum_consecutive_skip_ms": -100}
    replace_report(tmp_path, path, original)
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", path, tmp_path / "review")
    ).summary["evaluation"]
    metrics = summary["executions"][0]["metrics"]
    assert 0 < metrics["effective_hz"] < 30
    assert metrics["maximum_consecutive_skip_ms"] >= 0
    assert metrics["newest_image_age_ms"]["p50"] >= 0
    assert 0 < metrics["owner_fraction"]["policy"] <= 1
    assert sum(metrics["owner_fraction"].values()) == pytest.approx(1)
    assert 0 < metrics["longest_recorded_hold_ms"] < 200
    assert metrics["hold_time_basis"] == "successful_send_return_proxy"
    assert summary["execution_metrics"]["bound_runs"] == 1
    assert summary["execution_metrics"]["effective_hz"] == metrics["effective_hz"]


@pytest.mark.parametrize("source_kind", ["synthetic", "shadow"])
def test_shadow_actor_adapter_is_replayed_without_claiming_game_control(
    tmp_path, policy, source_kind
):
    path, original = execution(tmp_path, policy, shadow=True, source_kind=source_kind)
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", path, tmp_path / "review")
    ).summary["evaluation"]
    proof = summary["executions"][0]
    assert proof["status"] == "bound_diagnostic"
    assert proof["game_control_verified"] is False
    assert proof["verified_predictions"] >= 3
    if source_kind == "shadow":
        assert all(
            d["actor"]["actions"] == [None, None, None]
            for d in original["decisions"]
            if "actor" in d
        )


def test_archive_gaps_keep_all_attempts_and_explain_why_execution_is_quarantined(tmp_path, policy):
    path, _ = execution(tmp_path, policy, archive_limit_bytes=1)
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", path, tmp_path / "review")
    ).summary["evaluation"]
    proof = summary["executions"][0]
    assert proof["status"] == "quarantined"
    assert proof["recorded_gaps"]["missing_input_archives"] >= 3
    assert summary["metrics"]["all_attempts"] == 1
    assert summary["execution_metrics"]["unbound_runs"] == 1
    assert summary["execution_metrics"]["effective_hz"] is None


def test_recorded_sent_command_must_match_the_frozen_prediction_and_envelope(tmp_path, policy):
    path, report = execution(tmp_path, policy)
    command = next(c for c in report["commands"] if c["owner"] == "policy")
    command["sent"]["steer_i16"] = -32768
    journal = tmp_path / "execution/realtime-events.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    for event in events:
        if event["kind"] == "command" and event["data"]["decision_id"] == command["decision_id"]:
            event["data"] = command
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    report["journal"]["sha256"] = sha(journal)
    replace_report(tmp_path, path, report)
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", path, tmp_path / "review")
    ).summary["evaluation"]
    proof = summary["executions"][0]
    assert proof["status"] == "quarantined"
    assert proof["metrics"] is None
    assert "command" in str(proof["reasons"])


@pytest.mark.parametrize("change", ["model", "runtime", "pixel", "journal", "reference"])
def test_execution_mismatch_keeps_attempt_but_excludes_execution_metrics(tmp_path, policy, change):
    path, report = execution(
        tmp_path,
        policy,
        reference_mode="reference_assisted" if change == "reference" else "no_reference",
    )
    if change == "model":
        report["model"]["weights_sha256"] = "0" * 64
        replace_report(tmp_path, path, report)
    elif change == "runtime":
        report["configuration"]["action_lease_ms"] = 200
        replace_report(tmp_path, path, report)
    elif change == "pixel":
        row = next(d for d in report["decisions"] if d.get("archive"))
        saved = json.loads((tmp_path / "execution" / row["archive"]["path"]).read_bytes())
        pixels = tmp_path / "execution" / saved["frames"][0]["path"]
        raw = pixels.read_bytes()
        pixels.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    elif change == "journal":
        (tmp_path / "execution/realtime-events.jsonl").write_text("missing")
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", path, tmp_path / "review")
    ).summary["evaluation"]
    proof = summary["executions"][0]
    assert proof["status"] == "quarantined"
    assert proof["reasons"]
    assert proof["metrics"] is None
    assert summary["metrics"]["all_attempts"] == 1
    assert summary["execution_metrics"]["bound_runs"] == 0
    assert "execution_quarantined" in summary["attempts"][0]["evidence_gaps"]
    if change == "reference":
        assert proof["observed_reference_modes"] == ["no_reference"]


def test_cli_reports_failed_execution_verification_without_dropping_the_attempt(
    tmp_path, policy, capsys
):
    from fh5.cli import main

    path, report = execution(tmp_path, policy)
    report["configuration"]["action_lease_ms"] = 200
    replace_report(tmp_path, path, report)
    code = main(
        [
            "evaluation-review",
            "--batch",
            str(tmp_path / "frozen"),
            "--ledger",
            str(path),
            "--output",
            str(tmp_path / "review"),
        ]
    )
    assert code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["quarantined_executions"] == 1
    assert summary["metrics"]["all_attempts"] == 1


def test_a_complete_hash_manifest_cannot_hide_the_missing_final_release(tmp_path, policy):
    path, report = execution(tmp_path, policy)
    release = report["commands"].pop()
    assert release["owner"] == "hard_stop"
    journal = tmp_path / "execution/realtime-events.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    events = [e for e in events if not (e["kind"] == "command" and e["data"] == release)]
    for i, event in enumerate(events):
        event["sequence"] = i
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    report["journal"].update(sha256=sha(journal), offered=len(events), written=len(events))
    replace_report(tmp_path, path, report)
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", path, tmp_path / "review")
    ).summary["evaluation"]
    assert summary["executions"][0]["status"] == "quarantined"
    assert "release" in str(summary["executions"][0]["reasons"])


def test_safety_state_must_be_from_the_linked_telemetry_not_an_unrelated_clock(tmp_path, policy):
    path, report = execution(tmp_path, policy)
    row = next(d for d in report["decisions"] if "actor" in d)
    row["safety_at_decision"]["game_timestamp_ms"] += 7
    saved_path = tmp_path / "execution" / row["archive"]["path"]
    saved = json.loads(saved_path.read_bytes())
    saved["safety_at_decision"] = row["safety_at_decision"]
    saved_path.write_text(json.dumps(saved))
    row["archive"]["sha256"] = sha(saved_path)
    journal = tmp_path / "execution/realtime-events.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    for event in events:
        if (
            event["kind"].startswith("decision_")
            and event["data"]["decision_id"] == row["decision_id"]
        ):
            event["data"]["safety_at_decision"] = row["safety_at_decision"]
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    report["journal"]["sha256"] = sha(journal)
    replace_report(tmp_path, path, report)
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", path, tmp_path / "review")
    ).summary["evaluation"]
    assert summary["executions"][0]["status"] == "quarantined"
    assert "safety" in str(summary["executions"][0]["reasons"])


@pytest.mark.parametrize("timing", ["after_deadline", "after_send"])
def test_accepted_prediction_must_arrive_before_its_command_and_deadline(tmp_path, policy, timing):
    path, report = execution(tmp_path, policy)
    row = next(d for d in report["decisions"] if d["status"] == "accepted")
    command = next(c for c in report["commands"] if c["decision_id"] == row["decision_id"])
    row["inference_returned_ns"] = (
        row["deadline_ns"] + 1_000_000_000
        if timing == "after_deadline"
        else command["issued_ns"] + 1
    )
    journal = tmp_path / "execution/realtime-events.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    for event in events:
        if (
            event["kind"] == "decision_result"
            and event["data"]["decision_id"] == row["decision_id"]
        ):
            event["data"]["inference_returned_ns"] = row["inference_returned_ns"]
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    report["journal"]["sha256"] = sha(journal)
    replace_report(tmp_path, path, report)
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", path, tmp_path / "review")
    ).summary["evaluation"]
    assert summary["executions"][0]["status"] == "quarantined"
    assert summary["execution_metrics"]["effective_hz"] is None
    assert summary["metrics"]["all_attempts"] == 1


def test_malformed_journal_kind_is_isolated_without_losing_the_attempt_report(tmp_path, policy):
    path, report = execution(tmp_path, policy)
    journal = tmp_path / "execution/realtime-events.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    events[0]["kind"] = None
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    report["journal"]["sha256"] = sha(journal)
    replace_report(tmp_path, path, report)
    result = run_experiment(EvaluationReview(tmp_path / "frozen", path, tmp_path / "review"))
    summary = result.summary["evaluation"]
    assert result.report_path.is_file()
    assert summary["executions"][0]["status"] == "quarantined"
    assert summary["executions"][0]["metrics"] is None
    assert summary["metrics"]["all_attempts"] == 1


def test_reproducible_prediction_cannot_validate_an_action_history_that_was_never_sent(
    tmp_path, policy
):
    path, report = execution(tmp_path, policy)
    row = next(d for d in report["decisions"] if d["status"] == "accepted")
    assert row["actor"]["actions"] == [None, None, None]
    row["actor"].update(
        actions=[[0.9, 0.8]] * 3,
        action_mask=[True] * 3,
        action_age_ms=[210.0, 110.0, 10.0],
    )
    archive = tmp_path / "execution" / row["archive"]["path"]
    saved = json.loads(archive.read_bytes())
    saved["actor"] = deepcopy(row["actor"])
    archive.write_text(json.dumps(saved))
    row["archive"]["sha256"] = sha(archive)
    frames = tuple(read_numeric_frame(tmp_path / "execution", f) for f in saved["frames"])
    actor = FrozenNumericActor(tmp_path / "frozen/model", PixelContract(size=(64, 36)))
    row["features"] = actor.input_features(row["actor"], frames)
    row["prediction"] = actor.predict(row["actor"], frames)
    steer, longitudinal = row["prediction"]
    cfg = report["configuration"]
    command = next(c for c in report["commands"] if c["decision_id"] == row["decision_id"])
    command["sent"] = command["target"] = {
        "steer_i16": round(max(-cfg["max_steer"], min(cfg["max_steer"], steer)) * 32767),
        "throttle_u8": round(max(0, min(cfg["max_throttle"], longitudinal)) * 255),
        "brake_u8": round(max(0, min(cfg["max_brake"], -longitudinal)) * 255),
    }
    journal = tmp_path / "execution/realtime-events.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    for event in events:
        if event["data"].get("decision_id") != row["decision_id"]:
            continue
        if event["kind"] == "decision_started":
            event["data"]["actor"] = deepcopy(row["actor"])
        elif event["kind"] == "decision_result":
            event["data"] = deepcopy(row)
        elif event["kind"] == "command":
            event["data"] = deepcopy(command)
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    report["journal"]["sha256"] = sha(journal)
    replace_report(tmp_path, path, report)
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", path, tmp_path / "review")
    ).summary["evaluation"]
    assert summary["executions"][0]["status"] == "quarantined"
    assert summary["metrics"]["all_attempts"] == 1
    assert "action history" in str(summary["executions"][0]["reasons"])
