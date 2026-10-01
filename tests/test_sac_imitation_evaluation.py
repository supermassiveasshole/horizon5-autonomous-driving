"""Development-driven guidance changes at the experiment boundary, without devices."""

import json
import struct
from dataclasses import replace

import pytest
from test_attempts import evidence
from test_evaluation import sha
from test_evaluation_execution import PacketGame
from test_evaluation_start import CompletingBatch, automatic_request
from test_sac_learning import warm_start

from fh5.evaluation import EvaluationPrepare
from fh5.experiment import run_experiment
from fh5.sac_actions import ActionBounds
from fh5.sac_learning import SACResume, SACTrain


class SteeringDragGame(PacketGame):
    """Toy straight course: steering drag reduces progress; no FH5 claim."""

    def __init__(self):
        super().__init__()
        self.position = 0.0
        self.previous_ns = None

    def read(self, period_s):
        point = super().read(period_s)
        sent = [(at, command) for at, command in self.sent if at <= point.at_ns]
        command = sent[-1][1] if sent else None
        speed = (
            min(3.5, max(0.3, 8 - 100 * abs(command.steer_i16 / 32767)))
            if command and command.throttle_u8 and self.position < 3
            else 0.0
        )
        speed = struct.unpack("<f", struct.pack("<f", speed))[0]
        if self.previous_ns is not None:
            start = max(self.previous_ns, sent[-1][0]) if sent else self.previous_ns
            self.position = min(3.0, self.position + speed * (point.at_ns - start) / 1e9)
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


class SteeringDragBatch(CompletingBatch):
    def driving(self, slot_id, ready_state):
        game = SteeringDragGame()
        self.drives.append(game)
        return game


@pytest.fixture(scope="module")
def evaluated_guidance(tmp_path_factory):
    root = tmp_path_factory.mktemp("imitation-evaluation")
    replay = warm_start(root, bounds=ActionBounds(max_steer=0.4, max_throttle=0.25, max_brake=0.5))
    setup = root / "setup"
    setup.mkdir()
    operation = automatic_request(setup, root / "warm/actor")
    run_experiment(
        SACTrain(
            root / "warm",
            replay,
            root / "candidate",
            steps=30,
            actor_lr=0.001,
            imitation_weights=(1.0, 0.5, 0.0),
            imitation_protocol_batch=operation.batch_dir,
        )
    )
    registry = root / "usage.sqlite"
    path = evaluate_pair(root, setup / "evaluation.json", operation, root / "candidate", registry)
    return root, path, registry, operation


def evaluate_pair(root, template, operation, checkpoint, registry):
    root.mkdir(parents=True, exist_ok=True)
    comparison = {"version": 1}
    for side, kind, model in (
        ("incumbent", "bc", checkpoint / "bc"),
        ("candidate", "sac", checkpoint),
    ):
        config = json.loads(template.read_bytes())
        config["version"] = 2
        config["model"] = {
            "kind": kind,
            "directory": str(model),
            "manifest_sha256": sha(model / ("policy.json" if kind == "sac" else "model.json")),
        }
        if kind == "bc":
            config["version"] = 1
            del config["model"]["kind"]
        path = root / (side + ".json")
        path.write_text(json.dumps(config))
        batch = root / (side + "-batch")
        run_experiment(EvaluationPrepare(path, batch, registry))
        request = replace(
            operation,
            batch_dir=batch,
            batch_sha256=sha(batch / "batch.json"),
            output_dir=root / (side + "-run"),
            seconds=5,
            registry_file=registry,
        )
        run_experiment(request, evaluation_environment=SteeringDragBatch())
        ledger_path = request.output_dir / "ledger.json"
        ledger = json.loads(ledger_path.read_bytes())
        for index, entry in enumerate(ledger["entries"]):
            proof_dir = root / f"{side}-proof-{index}"
            proof_dir.mkdir()
            proof = evidence(proof_dir, request.output_dir / entry["recording"])
            entry["evidence"] = {"file": str(proof), "sha256": sha(proof)}
        ledger_path.write_text(json.dumps(ledger))
        comparison[side] = {
            "batch": str(batch),
            "batch_sha256": sha(batch / "batch.json"),
            "ledger": str(ledger_path),
            "ledger_sha256": sha(ledger_path),
        }
    path = root / "comparison.json"
    path.write_text(json.dumps(comparison))
    return path


def test_development_review_keeps_all_attempts_and_preserves_resume_state(
    tmp_path, evaluated_guidance
):
    root, comparison, registry, _ = evaluated_guidance
    result = run_experiment(
        SACResume(
            root / "candidate",
            tmp_path / "reviewed",
            steps=0,
            imitation_comparison=comparison,
            imitation_registry=registry,
        )
    )
    summary = result.summary["sac_learning"]
    review = summary["imitation_review"]
    assert review["scope"] == "synthetic_development_only"
    assert review["actual_policy_attempts"] == {"incumbent": 2, "candidate": 2}
    assert review["default_changed"] is False
    assert summary["real_driving_validated"] is False
    original = json.loads((root / "candidate/training-report.json").read_bytes())
    assert summary["learner_state_sha256"] == original["learner_state_sha256"]
    resumed = run_experiment(SACResume(tmp_path / "reviewed", tmp_path / "resumed", steps=0))
    assert resumed.summary["sac_learning"]["imitation"] == summary["imitation"]


@pytest.mark.parametrize("fault", ["missing_execution", "wall_riding", "no_registry"])
def test_incomplete_or_invalid_development_evidence_cannot_weaken_guidance(
    tmp_path, evaluated_guidance, fault
):
    root, comparison, registry, _ = evaluated_guidance
    bindings = json.loads(comparison.read_bytes())
    if fault != "no_registry":
        from pathlib import Path

        old_ledger = Path(bindings["candidate"]["ledger"])
        ledger = json.loads(old_ledger.read_bytes())
        for row in ledger["entries"]:
            row["recording"] = str((old_ledger.parent / row["recording"]).resolve())
            for key in ("execution", "preparation"):
                row[key]["directory"] = str((old_ledger.parent / row[key]["directory"]).resolve())
        if fault == "missing_execution":
            ledger["entries"][0].pop("execution")
        else:
            source = Path(ledger["entries"][0]["recording"])
            proof = evidence(
                tmp_path,
                source,
                [{"packet_index": 1, "kind": "wall_riding", "status": "confirmed"}],
            )
            ledger["entries"][0]["evidence"] = {"file": str(proof), "sha256": sha(proof)}
        path = tmp_path / "ledger.json"
        path.write_text(json.dumps(ledger))
        bindings["candidate"].update(ledger=str(path), ledger_sha256=sha(path))
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(bindings))
    summary = run_experiment(
        SACResume(
            root / "candidate",
            tmp_path / "retained",
            steps=0,
            imitation_comparison=path,
            imitation_registry=None if fault == "no_registry" else registry,
        )
    ).summary["sac_learning"]
    assert summary["imitation_review"]["advanced"] is False
    assert summary["imitation"]["weight"] == 1.0
    assert summary["imitation_review"]["reasons"]
    assert (
        summary["imitation_review"]["comparison"]["reviews"]["candidate"]["metrics"]["all_attempts"]
        == 2
    )
    assert (
        summary["learner_state_sha256"]
        == json.loads((root / "candidate/training-report.json").read_bytes())[
            "learner_state_sha256"
        ]
    )


def test_improved_candidate_weakens_guidance_and_can_be_frozen_for_next_evaluation(
    tmp_path, evaluated_guidance
):
    root, comparison, registry, operation = evaluated_guidance
    trained = run_experiment(
        SACResume(
            root / "candidate",
            tmp_path / "weakened",
            steps=0,
            imitation_comparison=comparison,
            imitation_registry=registry,
        )
    ).summary["sac_learning"]
    assert trained["imitation_review"]["advanced"], trained["imitation_review"]["reasons"]
    assert trained["imitation"]["weight"] == 0.5
    config = json.loads((root / "candidate.json").read_bytes())
    config["model"] = {
        "kind": "sac",
        "directory": str(tmp_path / "weakened"),
        "manifest_sha256": sha(tmp_path / "weakened/policy.json"),
    }
    path = tmp_path / "next.json"
    path.write_text(json.dumps(config))
    run_experiment(EvaluationPrepare(path, tmp_path / "next-batch"))
    assert (tmp_path / "next-batch/model/policy.json").exists()
    with pytest.raises(ValueError, match="exact current candidate"):
        run_experiment(
            SACResume(
                tmp_path / "weakened",
                tmp_path / "stale",
                steps=0,
                imitation_comparison=comparison,
                imitation_registry=registry,
            )
        )
    second = evaluate_pair(
        tmp_path / "fresh",
        root / "setup/evaluation.json",
        operation,
        tmp_path / "weakened",
        registry,
    )
    exited = run_experiment(
        SACResume(
            tmp_path / "weakened",
            tmp_path / "exited",
            steps=0,
            imitation_comparison=second,
            imitation_registry=registry,
        )
    ).summary["sac_learning"]
    assert exited["imitation_review"]["advanced"], exited["imitation_review"]["reasons"]
    assert exited["imitation"]["weight"] == 0
    assert exited["imitation"]["phase"] == "exited"
    resumed = run_experiment(
        SACResume(tmp_path / "exited", tmp_path / "continued", steps=2)
    ).summary["sac_learning"]
    assert resumed["imitation"]["phase"] == "exited"
    assert resumed["imitation"]["teacher_evaluations"] == 0
    assert all(step.get("imitation_weight", 0) == 0 for step in resumed["updates"])
    proof = tmp_path / "exited" / exited["imitation"]["transitions"][0]["review"]
    proof.write_bytes(b"{}")
    with pytest.raises(ValueError, match="evidence changed"):
        run_experiment(SACResume(tmp_path / "exited", tmp_path / "corrupted", steps=0))
    independent = run_experiment(SACResume(tmp_path / "continued", tmp_path / "copied", steps=0))
    assert independent.summary["sac_learning"]["imitation"]["phase"] == "exited"
