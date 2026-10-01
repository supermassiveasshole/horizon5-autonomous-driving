"""Automatic start evidence through the public experiment-run seam."""

import json
import shutil
import struct
from dataclasses import replace

import pytest
from test_attempts import evidence
from test_evaluation import policy as policy
from test_evaluation import sha
from test_evaluation_execution import PacketGame
from test_evaluation_run import Batch, request

from fh5.evaluation import EvaluationPrepare, EvaluationReview
from fh5.experiment import run_experiment


def automatic_request(tmp_path, policy, *, handoff_timeout_s=5):
    operation = request(tmp_path, policy)
    task_file = tmp_path / "task.json"
    task = json.loads(task_file.read_bytes())
    task.update(
        version=2,
        start_mode="automatic_event_ready",
        automatic_start={
            "event_file": str(operation.event_config_file),
            "event_sha256": sha(operation.event_config_file),
            "handoff_timeout_s": handoff_timeout_s,
        },
    )
    task_file.write_text(json.dumps(task))
    config_file = tmp_path / "evaluation.json"
    config = json.loads(config_file.read_bytes())
    config["task"]["sha256"] = sha(task_file)
    config_file.write_text(json.dumps(config))
    batch = tmp_path / "automatic-batch"
    run_experiment(EvaluationPrepare(config_file, batch))
    return replace(operation, batch_dir=batch, batch_sha256=sha(batch / "batch.json"))


def test_automatic_starts_are_bound_to_each_attempt_and_recomputed_without_devices(
    tmp_path, policy
):
    operation = automatic_request(tmp_path, policy)
    result = run_experiment(operation, evaluation_environment=Batch())
    batch = result.summary["evaluation"]
    assert batch["verified_starts"] == 2
    assert [s["status"] for s in batch["starts"]] == ["verified", "verified"]
    assert [s["operation"] for s in batch["starts"]] == ["start_ready", "restart_ready"]
    assert all(s["source_kind"] == "synthetic" for s in batch["starts"])
    assert all("automatic_start_unverified" not in a["pending_checks"] for a in batch["attempts"])
    assert all(
        "automatic_restart_not_verified" not in a["evidence_gaps"] for a in batch["attempts"]
    )
    assert batch["automatic_promotion_allowed"] is False
    ledger = json.loads((operation.output_dir / "ledger.json").read_bytes())
    assert len({e["preparation"]["manifest_sha256"] for e in ledger["entries"]}) == 2
    reviewed = run_experiment(
        EvaluationReview(
            operation.output_dir / "frozen",
            operation.output_dir / "ledger.json",
            tmp_path / "replayed",
        )
    ).summary["evaluation"]
    assert reviewed["starts"] == batch["starts"]
    assert reviewed["metrics"]["all_attempts"] == 2


def test_expired_start_handoff_never_sends_a_nonzero_command(tmp_path, policy):
    operation = automatic_request(tmp_path, policy, handoff_timeout_s=0.000001)
    game = Batch()
    result = run_experiment(operation, evaluation_environment=game)
    assert not any(
        c.throttle_u8 or c.brake_u8 or c.steer_i16 for g in game.drives for _, c in g.sent
    )
    assert result.summary["evaluation_run"]["stop_reason"] == "execution_stopped"
    assert result.summary["evaluation"]["verified_starts"] == 0
    assert result.summary["evaluation"]["metrics"]["all_attempts"] == 1
    assert result.summary["evaluation"]["unstarted_slots"] == ["run-1"]


def test_state_change_while_waiting_for_first_prediction_prevents_driving(tmp_path, policy):
    class ChangedBeforeCommand(PacketGame):
        def read(self, period_s):
            point = super().read(period_s)
            if len(self.packets) == 1:
                return replace(point, observation=None)
            raw = bytearray(point.raw_packets[0].payload)
            struct.pack_into("<f", raw, 244, 10)
            packet = replace(point.raw_packets[0], payload=bytes(raw))
            self.packets[-1] = packet
            return replace(point, raw_packets=(packet,))

    class ChangedBatch(Batch):
        def driving(self, slot_id, ready_state):
            game = ChangedBeforeCommand()
            self.drives.append(game)
            return game

    operation = automatic_request(tmp_path, policy)
    game = ChangedBatch()
    result = run_experiment(operation, evaluation_environment=game)
    assert not any(
        c.throttle_u8 or c.brake_u8 or c.steer_i16 for g in game.drives for _, c in g.sent
    )
    assert result.summary["evaluation"]["verified_starts"] == 0
    assert result.summary["evaluation"]["metrics"]["all_attempts"] == 1
    assert result.summary["evaluation"]["unstarted_slots"] == ["run-1"]


def test_neutral_policy_commands_do_not_end_the_start_wait(tmp_path, policy):
    operation = automatic_request(tmp_path, policy)
    config_file = tmp_path / "evaluation.json"
    config = json.loads(config_file.read_bytes())
    config["runtime"].update(max_steer=0, max_throttle=0, max_brake=0)
    config_file.write_text(json.dumps(config))
    frozen = tmp_path / "neutral-batch"
    run_experiment(EvaluationPrepare(config_file, frozen))
    operation = replace(operation, batch_dir=frozen, batch_sha256=sha(frozen / "batch.json"))
    result = run_experiment(operation, evaluation_environment=Batch())
    recorded = json.loads(
        (operation.output_dir / "attempt-0000/execution/report.json").read_bytes()
    )
    assert any(c["owner"] == "policy" and c["status"] == "sent" for c in recorded["commands"])
    assert all(not any(c["sent"].values()) for c in recorded["commands"])
    review = result.summary["evaluation"]
    assert all(e["status"] == "bound_diagnostic" for e in review["executions"])
    assert review["verified_starts"] == 0
    assert review["metrics"]["all_attempts"] == 2
    assert all("nonzero policy command" in str(s["reasons"]) for s in review["starts"])


def test_failed_driver_open_retains_the_started_slot_and_its_unknown_start(tmp_path, policy):
    class FailedSecondDriver(Batch):
        def driving(self, slot_id, ready_state):
            if slot_id == "run-1":
                raise OSError("synthetic driver open failure")
            return super().driving(slot_id, ready_state)

    operation = automatic_request(tmp_path, policy)
    result = run_experiment(operation, evaluation_environment=FailedSecondDriver())
    review = result.summary["evaluation"]
    assert review["metrics"]["all_attempts"] == 2
    assert [s["status"] for s in review["starts"]] == ["verified", "quarantined"]
    assert review["verified_starts"] == 1
    assert review["unstarted_slots"] == []


@pytest.fixture(scope="module")
def recorded_starts(tmp_path_factory, policy):
    root = tmp_path_factory.mktemp("recorded-starts")
    operation = automatic_request(root, policy)
    run_experiment(operation, evaluation_environment=Batch())
    return operation.output_dir


def test_reviewer_checks_state_between_initial_packet_and_first_command(tmp_path, recorded_starts):
    root = tmp_path / "recorded"
    shutil.copytree(recorded_starts, root)
    execution = root / "attempt-0001/execution"
    journal = execution / "realtime-events.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    packets = [e["data"] for e in events if e["kind"] == "packet"]
    first_command = next(e["data"] for e in events if e["kind"] == "command")
    assert packets[1]["received_monotonic_ns"] < first_command["issued_ns"]
    raw = bytearray.fromhex(packets[1]["payload_hex"])
    struct.pack_into("<f", raw, 244, 10)
    packets[1]["payload_hex"] = raw.hex()
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    report_path = execution / "report.json"
    report = json.loads(report_path.read_bytes())
    report["journal"]["sha256"] = sha(journal)
    report_path.write_text(json.dumps(report))
    execution_manifest = execution / "realtime-manifest.json"
    execution_manifest.write_text(json.dumps({"version": 1, "report_sha256": sha(report_path)}))
    packets_path = root / "attempt-0001/recording/packets.jsonl"
    packets_path.write_text("".join(json.dumps(p) + "\n" for p in packets))
    ledger_path = root / "ledger.json"
    ledger = json.loads(ledger_path.read_bytes())
    entry = ledger["entries"][1]
    entry["files"]["packets.jsonl"] = sha(packets_path)
    entry["execution"]["manifest_sha256"] = sha(execution_manifest)
    prep_manifest = root / "attempt-0001/ready/start-manifest.json"
    manifest = json.loads(prep_manifest.read_bytes())
    manifest.update(recording_sha256=sha(packets_path), execution_sha256=sha(execution_manifest))
    prep_manifest.write_text(json.dumps(manifest))
    entry["preparation"]["manifest_sha256"] = sha(prep_manifest)
    ledger_path.write_text(json.dumps(ledger))
    review = run_experiment(
        EvaluationReview(root / "frozen", ledger_path, tmp_path / "reviewed")
    ).summary["evaluation"]
    assert review["executions"][1]["status"] == "bound_diagnostic"
    assert [s["status"] for s in review["starts"]] == ["verified", "quarantined"]
    assert "before first policy command" in str(review["starts"][1]["reasons"])
    assert review["metrics"]["all_attempts"] == 2


@pytest.mark.parametrize("fault", ["missing", "other_slot", "changed_frame"])
def test_start_binding_gaps_do_not_erase_attempts_or_qualify_the_wrong_start(
    tmp_path, recorded_starts, fault
):
    root = tmp_path / "recorded"
    shutil.copytree(recorded_starts, root)
    path = root / "ledger.json"
    ledger = json.loads(path.read_bytes())
    if fault == "missing":
        del ledger["entries"][1]["preparation"]
    elif fault == "other_slot":
        ledger["entries"][1]["preparation"] = ledger["entries"][0]["preparation"]
    else:
        image = sorted((root / "attempt-0001/ready/frames").glob("*.pgm"))[-1]
        image.write_bytes(b"changed frame")
    path.write_text(json.dumps(ledger))
    result = run_experiment(EvaluationReview(root / "frozen", path, tmp_path / "reviewed"))
    review = result.summary["evaluation"]
    assert review["metrics"]["all_attempts"] == 2
    assert [s["status"] for s in review["starts"]] == ["verified", "quarantined"]
    assert "automatic_start_unverified" in review["attempts"][1]["pending_checks"]
    assert review["automatic_promotion_allowed"] is False


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("pixels", "driving frames"),
        ("capture_time", "frame timing"),
        ("telemetry", "stationary telemetry"),
        ("release", "release evidence"),
    ],
)
def test_claimed_readiness_is_recomputed_from_sources_after_manifest_rebinding(
    tmp_path, recorded_starts, fault, reason
):
    root = tmp_path / "recorded"
    shutil.copytree(recorded_starts, root)
    ready = root / "attempt-0001/ready"
    journal = ready / "event-journal.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    screens = [e for e in events if e["kind"] == "screen"]
    if fault == "pixels":
        for event in screens[-2:]:
            frame = ready / event["path"]
            frame.write_bytes(b"P5\n4 2\n255\n" + bytes(8))
            event["sha256"] = sha(frame)
    elif fault == "capture_time":
        screens[-1]["captured_ns"] = 1
    elif fault == "telemetry":
        packets_path = ready / "packets.jsonl"
        packets = [json.loads(line) for line in packets_path.read_text().splitlines()]
        raw = bytearray.fromhex(packets[-1]["payload_hex"])
        struct.pack_into("<i", raw, 212, 123)
        packets[-1]["payload_hex"] = raw.hex()
        packets_path.write_text("".join(json.dumps(p) + "\n" for p in packets))
        for event in events:
            if event["kind"] == "asset" and event["path"] == "packets.jsonl":
                event["sha256"] = sha(packets_path)
    else:
        released = next(e for e in events if e["kind"] == "released")
        released.update(kind="release_failed", error="synthetic release failure")
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    saved_path = ready / "event-run.json"
    saved = json.loads(saved_path.read_bytes())
    saved.update(events=events, journal_sha256=sha(journal))
    saved_path.write_text(json.dumps(saved))
    manifest_path = ready / "start-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"] = {name: sha(ready / name) for name in manifest["files"]}
    manifest_path.write_text(json.dumps(manifest))
    ledger_path = root / "ledger.json"
    ledger = json.loads(ledger_path.read_bytes())
    ledger["entries"][1]["preparation"]["manifest_sha256"] = sha(manifest_path)
    ledger_path.write_text(json.dumps(ledger))
    assert saved["summary"]["ready_verified"] is True
    review = run_experiment(
        EvaluationReview(root / "frozen", ledger_path, tmp_path / "reviewed")
    ).summary["evaluation"]
    assert [s["status"] for s in review["starts"]] == ["verified", "quarantined"]
    assert reason in review["starts"][1]["reasons"][0]
    assert review["metrics"]["all_attempts"] == 2


def test_changed_event_protocol_is_rejected_before_menu_or_driving_inputs(tmp_path, policy):
    operation = automatic_request(tmp_path, policy)
    event = json.loads(operation.event_config_file.read_bytes())
    event["event_run"]["start_radius_m"] = 10
    operation.event_config_file.write_text(json.dumps(event))
    game = Batch()
    with pytest.raises(ValueError, match="frozen automatic start protocol"):
        run_experiment(operation, evaluation_environment=game)
    assert game.menus == []
    assert game.drives == []
    assert game.closed


class CompletingGame(PacketGame):
    """On/off action response ending at a 3 m toy finish, not vehicle dynamics."""

    def __init__(self):
        super().__init__()
        self.position = 0.0
        self.previous_ns = None

    def read(self, period_s):
        point = super().read(period_s)
        command = next((c for sent_ns, c in reversed(self.sent) if sent_ns <= point.at_ns), None)
        speed = 3.5 if command and command.throttle_u8 and self.position < 3 else 0.0
        if self.previous_ns is not None:
            self.position = min(3.0, self.position + speed * (point.at_ns - self.previous_ns) / 1e9)
        self.previous_ns = point.at_ns
        raw = bytearray(point.raw_packets[0].payload)
        struct.pack_into("<ffff", raw, 244, self.position, 2, 0.2, speed)
        struct.pack_into("<f", raw, 40, speed)
        packet = replace(point.raw_packets[0], payload=bytes(raw))
        self.packets[-1] = packet
        return replace(
            point,
            raw_packets=(packet,),
            safety=replace(point.safety, speed_kmh=speed * 3.6),
            observation=replace(
                point.observation,
                ego={
                    "speed_mps": speed,
                    "velocity_car_mps": [0.0, 0.0, speed],
                    "angular_velocity_car_radps": [0.0, 0.0, 0.0],
                },
            ),
        )


class CompletingBatch(Batch):
    def driving(self, slot_id, ready_state):
        game = CompletingGame()
        self.drives.append(game)
        return game


def test_automatic_start_can_complete_local_validity_without_claiming_game_promotion(
    tmp_path, policy
):
    operation = replace(automatic_request(tmp_path, policy), seconds=2)
    game = CompletingBatch()
    run_experiment(operation, evaluation_environment=game)
    assert all(g.position == 3.0 for g in game.drives)
    path = operation.output_dir / "ledger.json"
    ledger = json.loads(path.read_bytes())
    for i, entry in enumerate(ledger["entries"]):
        folder = tmp_path / f"independent-{i}"
        folder.mkdir()
        proof = evidence(folder, operation.output_dir / entry["recording"])
        entry["evidence"] = {"file": str(proof), "sha256": sha(proof)}
    path.write_text(json.dumps(ledger))
    review = run_experiment(
        EvaluationReview(operation.output_dir / "frozen", path, tmp_path / "reviewed")
    ).summary["evaluation"]
    assert review["verified_starts"] == 2
    assert review["metrics"]["outcomes"]["valid_complete"] == 2
    assert all(a["record_eligible"] for a in review["attempts"])
    assert all(a["local_outcome"] == "pending_review" for a in review["attempts"])
    assert review["automatic_promotion_allowed"] is False
    assert review["closed_loop_validated"] is False
