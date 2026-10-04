"""Frozen SAC batches and repeated execution at the experiment boundary."""

import json
import struct
import time
from dataclasses import replace

import pytest

from fh5.evaluation.prepare import EvaluationPrepare, EvaluationReview
from fh5.experiment import run_experiment
from fh5.learning.sac.actions import ActionBounds
from fh5.learning.sac.training import SACPolicyReplay, SACTrain
from tests.evaluation.test_evaluation import prepare, sha
from tests.evaluation.test_evaluation_execution import PacketGame
from tests.evaluation.test_evaluation_run import Batch, request
from tests.learning.sac.test_sac_learning import warm_start
from tests.support.checkpoint_files import prediction_records


@pytest.mark.parametrize("forged", ["context", "executable_support"])
def test_sac_wait_is_replayable_but_an_invented_wait_context_is_quarantined(tmp_path, forged):
    from fh5.driving.realtime.model import RealtimeConfig, RealtimeNumericReplay, RealtimeRun
    from fh5.learning.sac.evaluation_actor import SACEvaluationActor
    from fh5.observation.numeric import PixelContract

    # At this rate the steering support needs >49 ms. Warmup's 50 ms is valid;
    # a send's 15 ms latency leaves too little time at the following 20 Hz tick.
    bounds = ActionBounds(max_steer=0.4, max_throttle=0.25, max_brake=0.5, steer_rate=0.00031)
    replay = warm_start(tmp_path, bounds=bounds)
    model = tmp_path / "candidate"
    run_experiment(SACTrain(tmp_path / "warm", replay, model, steps=1))
    pixels = PixelContract(size=(64, 36))

    def factory():
        return SACEvaluationActor(model, pixels, sha(model / "policy.json"))

    class SlowSend(PacketGame):
        def send(self, command):
            time.sleep(0.015)
            super().send(command)

    root = tmp_path / "execution"
    original = run_experiment(
        RealtimeRun(root, RealtimeConfig(pixels=pixels, reference_count=1), seconds=0.7),
        realtime_environment=SlowSend(),
        numeric_actor_factory=factory,
    ).summary["realtime"]
    waits = [d for d in original["decisions"] if d["status"] == "skip_action_support"]
    assert waits and original["stop_reason"] == "time_limit"
    assert any(
        d["status"] == "accepted" and d["index"] > waits[0]["index"] for d in original["decisions"]
    )
    assert not any(d.get("error") for d in original["decisions"])
    verified = run_experiment(
        RealtimeNumericReplay(root, tmp_path / "verified.html"), numeric_actor=factory()
    ).summary["realtime_numeric_replay"]
    assert verified["verified"] and verified["verified_predictions"] > 0
    assert verified["verified_action_waits"] == len(waits)

    # Rewrite both copies and hashes: independent replay must check the wait,
    # not merely trust a self-consistent journal/report pair.
    report = json.loads((root / "report.json").read_bytes())
    row = next(d for d in report["decisions"] if d["status"] == "skip_action_support")
    if forged == "context":
        row["command_context"]["returned_ns"] -= 100_000_000
    else:
        executable = next(d for d in report["decisions"] if d["status"] == "accepted")
        row["decision_ns"] = executable["decision_ns"]
        row["command_context"] = executable["command_context"]
    journal = root / report["journal"]["path"]
    events = [json.loads(line) for line in journal.read_bytes().splitlines()]
    event = next(
        e
        for e in events
        if e["kind"] == "decision_skipped" and e["data"]["decision_id"] == row["decision_id"]
    )
    event["data"] = row
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    report["journal"]["sha256"] = sha(journal)
    (root / "report.json").write_text(json.dumps(report))
    (root / "realtime-manifest.json").write_text(
        json.dumps({"version": 1, "report_sha256": sha(root / "report.json")})
    )
    invalid = run_experiment(
        RealtimeNumericReplay(root, tmp_path / "invalid.html"), numeric_actor=factory()
    ).summary["realtime_numeric_replay"]
    assert invalid["verified"] is False
    assert "support" in str(invalid["errors"]) or "context" in str(invalid["errors"])


class ResponsiveGame(PacketGame):
    """Small action-responsive fixture, explicitly not FH5 vehicle physics."""

    def __init__(self):
        super().__init__()
        self.position = 0.0
        self.previous_ns = None

    def read(self, period_s):
        point = super().read(period_s)
        command = self.sent[-1][1] if self.sent else None
        speed = (command.throttle_u8 / 255 * 8) if command else 0.0
        if self.previous_ns is not None:
            self.position += speed * (point.at_ns - self.previous_ns) / 1e9
        self.previous_ns = point.at_ns
        raw = bytearray(point.raw_packets[0].payload)
        struct.pack_into("<ffff", raw, 244, self.position, 2, 0.2, speed)
        struct.pack_into("<f", raw, 40, speed)
        actual_speed = struct.unpack_from("<f", raw, 256)[0]
        actual_velocity = struct.unpack_from("<f", raw, 40)[0]
        packet = replace(point.raw_packets[0], payload=bytes(raw))
        self.packets[-1] = packet
        return replace(
            point,
            raw_packets=(packet,),
            safety=replace(point.safety, speed_kmh=actual_speed * 3.6),
            observation=replace(
                point.observation,
                ego={
                    "speed_mps": actual_speed,
                    "velocity_car_mps": [0.0, 0.0, actual_velocity],
                    "angular_velocity_car_radps": [0.0, 0.0, 0.0],
                },
            ),
        )


class ResponsiveBatch(Batch):
    def driving(self, slot_id, ready_state):
        assert self.menus[-1].closed
        game = ResponsiveGame()
        self.drives.append(game)
        return game


@pytest.fixture(scope="module")
def sac_policy(tmp_path_factory):
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("evaluation-sac")
    replay = warm_start(root, bounds=ActionBounds(max_steer=0.4, max_throttle=0.25, max_brake=0.5))
    run_experiment(SACTrain(root / "warm", replay, root / "candidate", steps=3))
    return root / "candidate"


def sac_config(tmp_path, checkpoint):
    _, path = prepare(tmp_path, checkpoint / "bc")
    options = json.loads(path.read_bytes())
    options["version"] = 2
    options["model"] = {
        "kind": "sac",
        "directory": str(checkpoint),
        "manifest_sha256": sha(checkpoint / "policy.json"),
    }
    path.write_text(json.dumps(options))
    return path


def test_frozen_sac_batch_keeps_its_complete_policy_when_the_batch_moves(tmp_path, sac_policy):
    path = sac_config(tmp_path, sac_policy)
    result = run_experiment(EvaluationPrepare(path, tmp_path / "sac-batch"))
    frozen = tmp_path / "sac-batch/model"
    manifest = json.loads(result.report_path.read_bytes())
    assert manifest["version"] == 2
    assert manifest["config"]["model"]["kind"] == "sac"
    assert sha(frozen / "policy.pt") == sha(sac_policy / "policy.pt")
    moved = tmp_path / "moved-batch"
    (tmp_path / "sac-batch").rename(moved)
    replayed = run_experiment(
        SACPolicyReplay(
            moved / "model", sac_policy / "experience/replay.json", tmp_path / "frozen.html"
        )
    ).summary["sac_policy"]
    original = json.loads((sac_policy / "training-report.json").read_bytes())
    assert prediction_records(tmp_path, replayed) == prediction_records(sac_policy, original)
    assert manifest["model_diagnostic_only"] is True


def sac_request(tmp_path, checkpoint):
    operation = request(tmp_path, checkpoint / "bc")
    path = tmp_path / "evaluation.json"
    options = json.loads(path.read_bytes())
    options["version"] = 2
    options["model"] = {
        "kind": "sac",
        "directory": str(checkpoint),
        "manifest_sha256": sha(checkpoint / "policy.json"),
    }
    path.write_text(json.dumps(options))
    frozen = tmp_path / "sac-batch"
    run_experiment(EvaluationPrepare(path, frozen))
    return replace(operation, batch_dir=frozen, batch_sha256=sha(frozen / "batch.json"))


def test_frozen_sac_runs_twice_without_exploration_and_replays_executed_commands(
    tmp_path, sac_policy
):
    operation = sac_request(tmp_path, sac_policy)
    game = ResponsiveBatch()
    before = (sac_policy / "policy.pt").read_bytes()
    result = run_experiment(operation, evaluation_environment=game)
    summary = result.summary["evaluation_run"]
    assert summary["stop_reason"] == "plan_complete", summary
    assert summary["started_slots"] == ["run-0", "run-1"]
    assert result.summary["evaluation"]["execution_metrics"]["bound_runs"] == 2
    for index, drive in enumerate(game.drives):
        recorded = json.loads(
            (operation.output_dir / f"attempt-{index:04d}/execution/report.json").read_bytes()
        )
        assert recorded["actor_kind"] == "frozen-numeric-sac-v1"
        assert recorded["model"]["exploration"] is False
        first = next(d for d in recorded["decisions"] if d["status"] == "accepted")
        assert first["command_context"]["sent"] == {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0}
        assert any(c.throttle_u8 for _, c in drive.sent)
        assert drive.position > 0
    assert summary["resources_released"] is True
    assert result.summary["evaluation"]["automatic_promotion_allowed"] is False
    assert (sac_policy / "policy.pt").read_bytes() == before


def test_sac_runtime_refuses_an_envelope_that_would_reinterpret_its_commands(tmp_path, sac_policy):
    from fh5.driving.realtime.model import RealtimeConfig, RealtimeRun
    from fh5.learning.sac.evaluation_actor import SACEvaluationActor
    from fh5.observation.numeric import PixelContract
    from tests.evaluation.test_evaluation_execution import PacketGame

    game = PacketGame()
    pixels = PixelContract(size=(64, 36))
    result = run_experiment(
        RealtimeRun(
            tmp_path / "wrong-bounds",
            RealtimeConfig(pixels=pixels, reference_count=1, max_throttle=0.1),
            seconds=0.25,
        ),
        realtime_environment=game,
        numeric_actor_factory=lambda: SACEvaluationActor(
            sac_policy, pixels, sha(sac_policy / "policy.json")
        ),
    ).summary["realtime"]
    assert result["stop_reason"] == "model_startup_failed"
    assert not game.sent


def test_context_changed_by_lease_release_discards_the_old_inflight_result(tmp_path):
    from fh5.driving.realtime.model import InferenceReply, RealtimeConfig, RealtimeReplay
    from fh5.observation.numeric import PixelContract
    from tests.driving.test_realtime import BASE, MS, sample

    result = run_experiment(
        RealtimeReplay(
            tmp_path / "changed-context",
            RealtimeConfig(pixels=PixelContract(size=(2, 1)), action_lease_ms=75),
            tuple(sample(ms) for ms in range(250, 551, 50)),
            (InferenceReply(5), InferenceReply(70), *(InferenceReply(5) for _ in range(5))),
            require_command_context=True,
        )
    ).summary["realtime"]
    assert result["decisions"][0]["status"] == "skip_initial_command_context"
    assert result["decisions"][2]["status"] == "discard_command_context_changed"
    assert not any(c["decision_id"] == "d2" for c in result["commands"])
    assert result["decisions"][4]["command_context"]["owner"] == "lease_expiry"
    assert result["decisions"][4]["command_context"]["returned_ns"] == BASE + 375 * MS
    assert result["decisions"][4]["status"] == "accepted"


def test_unacknowledged_initial_neutral_prevents_sac_commands_and_retains_attempt(
    tmp_path, sac_policy
):
    class FailedSend(PacketGame):
        def send(self, command):
            super().send(command)
            raise OSError("synthetic send failed")

    class FailedBatch(Batch):
        def driving(self, slot_id, ready_state):
            game = FailedSend()
            self.drives.append(game)
            return game

    operation = sac_request(tmp_path, sac_policy)
    game = FailedBatch()
    result = run_experiment(operation, evaluation_environment=game)
    assert result.summary["evaluation_run"]["started_slots"] == ["run-0"]
    assert result.summary["evaluation_run"]["unstarted_slots"] == ["run-1"]
    assert all(not (c.throttle_u8 or c.steer_i16 or c.brake_u8) for _, c in game.drives[0].sent)
    assert result.summary["evaluation"]["metrics"]["all_attempts"] == 1


def test_reproducible_sac_prediction_cannot_borrow_an_unsent_command_context(tmp_path, sac_policy):
    operation = sac_request(tmp_path, sac_policy)
    run_experiment(operation, evaluation_environment=Batch())
    root = operation.output_dir / "attempt-0000/execution"
    report = json.loads((root / "report.json").read_bytes())
    row = next(d for d in report["decisions"] if d["status"] == "accepted")
    row["command_context"]["command_index"] = 999
    archived = root / row["archive"]["path"]
    saved = json.loads(archived.read_bytes())
    saved["command_context"] = row["command_context"]
    archived.write_text(json.dumps(saved))
    row["archive"]["sha256"] = sha(archived)
    journal = root / report["journal"]["path"]
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    for event in events:
        if (
            event["kind"].startswith("decision_")
            and event["data"]["decision_id"] == row["decision_id"]
        ):
            event["data"]["command_context"] = row["command_context"]
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    report["journal"]["sha256"] = sha(journal)
    (root / "report.json").write_text(json.dumps(report))
    manifest = root / "realtime-manifest.json"
    manifest.write_text(json.dumps({"version": 1, "report_sha256": sha(root / "report.json")}))
    ledger = operation.output_dir / "ledger.json"
    recorded = json.loads(ledger.read_bytes())
    recorded["entries"][0]["execution"]["manifest_sha256"] = sha(manifest)
    ledger.write_text(json.dumps(recorded))
    result = run_experiment(
        EvaluationReview(operation.output_dir / "frozen", ledger, tmp_path / "review-again")
    ).summary["evaluation"]
    assert result["metrics"]["all_attempts"] == 2
    assert result["executions"][0]["status"] == "quarantined"
    assert "context" in str(result["executions"][0]["reasons"])
    assert result["executions"][1]["status"] == "bound_diagnostic"


def test_registered_sac_batch_uses_the_sac_identity_and_never_promotes_synthetic_results(
    tmp_path, sac_policy
):
    operation = sac_request(tmp_path, sac_policy)
    registry = tmp_path / "usage.sqlite"
    config = tmp_path / "evaluation.json"
    options = json.loads(config.read_bytes())
    options["purpose"] = "final"
    config.write_text(json.dumps(options))
    reserved = tmp_path / "reserved"
    run_experiment(EvaluationPrepare(config, reserved, registry))
    operation = replace(
        operation,
        batch_dir=reserved,
        batch_sha256=sha(reserved / "batch.json"),
        registry_file=registry,
    )
    result = run_experiment(operation, evaluation_environment=Batch()).summary["evaluation"]
    assert result["independence"]["reservation_verified"] is True
    assert result["independence"]["status"] == "no_known_overlap"
    assert result["automatic_promotion_allowed"] is False
    snapshot = json.loads((operation.output_dir / "review/usage-snapshot.json").read_bytes())
    assert snapshot["reservations"][0]["model"] == sha(sac_policy / "policy.json")


def test_sac_review_quarantines_a_policy_send_after_its_context_was_released(tmp_path, sac_policy):
    from copy import deepcopy

    operation = sac_request(tmp_path, sac_policy)
    run_experiment(operation, evaluation_environment=Batch())
    root = operation.output_dir / "attempt-0000/execution"
    report = json.loads((root / "report.json").read_bytes())
    row = [d for d in report["decisions"] if "actor" in d][-1]
    assert row["status"] == "accepted"
    index = next(
        i for i, c in enumerate(report["commands"]) if c["decision_id"] == row["decision_id"]
    )
    policy_send = report["commands"][index]
    midpoint = (row["decision_ns"] + policy_send["issued_ns"]) // 2
    release = deepcopy(report["commands"][0])
    release.update(
        issued_ns=midpoint, returned_ns=midpoint, owner="lease_expiry", valid_until_ns=midpoint
    )
    report["commands"].insert(index, release)
    journal = root / report["journal"]["path"]
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    insertion = next(
        i
        for i, e in enumerate(events)
        if e["kind"] == "command" and e["data"]["decision_id"] == row["decision_id"]
    )
    events.insert(insertion, {"sequence": 0, "kind": "command", "data": release})
    for i, event in enumerate(events):
        event["sequence"] = i
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    report["journal"].update(offered=len(events), written=len(events), sha256=sha(journal))
    (root / "report.json").write_text(json.dumps(report))
    manifest = root / "realtime-manifest.json"
    manifest.write_text(json.dumps({"version": 1, "report_sha256": sha(root / "report.json")}))
    ledger = operation.output_dir / "ledger.json"
    entries = json.loads(ledger.read_bytes())
    entries["entries"][0]["execution"]["manifest_sha256"] = sha(manifest)
    ledger.write_text(json.dumps(entries))
    reviewed = run_experiment(
        EvaluationReview(operation.output_dir / "frozen", ledger, tmp_path / "review-again")
    ).summary["evaluation"]
    assert reviewed["metrics"]["all_attempts"] == 2
    execution = reviewed["executions"][0]
    assert execution["verified_predictions"] > 0  # Numerical replay alone still passes.
    assert execution["status"] == "quarantined"
    assert "context changed before send" in str(execution["reasons"])
    assert reviewed["executions"][1]["status"] == "bound_diagnostic"
    assert reviewed["execution_metrics"]["bound_runs"] == 1


def test_valid_sac_model_replacement_during_driver_open_is_rejected_before_commands(
    tmp_path, sac_policy
):
    import shutil

    from fh5.learning.sac.training import SACResume

    replacement = tmp_path / "replacement"
    run_experiment(SACResume(sac_policy, replacement, steps=1))
    operation = sac_request(tmp_path, sac_policy)

    class SwappingBatch(Batch):
        def driving(self, slot_id, ready_state):
            model = operation.output_dir / "frozen/model"
            for name in ("policy.json", "policy.pt", "training-report.json"):
                shutil.copyfile(replacement / name, model / name)
            return super().driving(slot_id, ready_state)

    game = SwappingBatch()
    result = run_experiment(operation, evaluation_environment=game)
    summary = result.summary["evaluation_run"]
    assert summary["started_slots"] == ["run-0"]
    assert summary["unstarted_slots"] == ["run-1"]
    assert not game.drives[0].sent
    assert summary["attempts"][0]["stop_reason"] == "model_startup_failed"
    assert summary["review_error"]


def test_cli_replays_sac_execution_without_opening_an_environment(tmp_path, sac_policy, capsys):
    from fh5.cli import main

    operation = sac_request(tmp_path, sac_policy)
    run_experiment(operation, evaluation_environment=Batch())
    assert (
        main(
            [
                "realtime-replay",
                str(operation.output_dir / "attempt-0000/execution"),
                "--model",
                str(operation.output_dir / "frozen/model"),
                "--report",
                str(tmp_path / "cli-replay.html"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["verified"] is True
    assert result["commands_sent_to_game"] is False
