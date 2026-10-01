"""Persistent candidate decisions at the experiment-run seam."""

import io
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_attempts import evidence
from test_evaluation import sha
from test_evaluation_start import automatic_request
from test_sac_imitation_evaluation import SteeringDragBatch
from test_sac_learning import warm_start

from fh5.evaluation import EvaluationPrepare
from fh5.experiment import run_experiment
from fh5.sac_actions import ActionBounds
from fh5.sac_learning import SACTrain


@pytest.fixture(scope="module")
def candidates(tmp_path_factory):
    root = tmp_path_factory.mktemp("persistent-candidates")
    replay = warm_start(root, bounds=ActionBounds(max_steer=0.4, max_throttle=0.25, max_brake=0.5))
    setup = root / "setup"
    setup.mkdir()
    operation = automatic_request(setup, root / "warm/actor")
    registry = root / "usage.sqlite"
    bindings, checkpoints = {}, {}
    for side, steps in (("incumbent", 0), ("candidate", 30)):
        checkpoint = root / (side + "-model")
        run_experiment(
            SACTrain(
                root / "warm",
                replay,
                checkpoint,
                steps=steps,
                actor_lr=0.001,
                imitation_weights=(1.0, 0.5, 0.0),
                imitation_protocol_batch=operation.batch_dir,
            )
        )
        checkpoints[side] = str(checkpoint)
        config = json.loads((setup / "evaluation.json").read_bytes())
        config["version"] = 2
        config["model"] = {
            "kind": "sac",
            "directory": str(checkpoint),
            "manifest_sha256": sha(checkpoint / "policy.json"),
        }
        path = root / (side + "-evaluation.json")
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
        ledger_file = request.output_dir / "ledger.json"
        ledger = json.loads(ledger_file.read_bytes())
        for index, row in enumerate(ledger["entries"]):
            proof_dir = root / f"{side}-proof-{index}"
            proof_dir.mkdir()
            proof = evidence(proof_dir, request.output_dir / row["recording"])
            row["evidence"] = {"file": str(proof), "sha256": sha(proof)}
        ledger_file.write_text(json.dumps(ledger))
        bindings[side] = {
            "batch": str(batch),
            "batch_sha256": sha(batch / "batch.json"),
            "ledger": str(ledger_file),
            "ledger_sha256": sha(ledger_file),
        }
    return root, bindings, checkpoints, registry


def record_config(root, candidates, *, missing_execution=False, candidate_outcome=None):
    _, originals, checkpoints, _ = candidates
    bindings = json.loads(json.dumps(originals))
    if missing_execution or candidate_outcome:
        original_file = Path(bindings["candidate"]["ledger"])
        ledger = json.loads(original_file.read_bytes())
        for row in ledger["entries"]:
            row["recording"] = str((original_file.parent / row["recording"]).resolve())
            for key in ("preparation", "execution"):
                row[key]["directory"] = str(
                    (original_file.parent / row[key]["directory"]).resolve()
                )
        if missing_execution:
            ledger["entries"][0].pop("execution")
        if candidate_outcome:
            row = ledger["entries"][0]
            if candidate_outcome == "unknown":
                row["evidence"] = None
            else:
                source = Path(row["recording"])
                index = len((source / "packets.jsonl").read_bytes().splitlines()) - 1
                proof = evidence(
                    root,
                    source,
                    [{"packet_index": index, "kind": candidate_outcome, "status": "confirmed"}],
                )
                row["evidence"] = {"file": str(proof), "sha256": sha(proof)}
        path = root / "incomplete-ledger.json"
        path.write_text(json.dumps(ledger))
        bindings["candidate"].update(ledger=str(path), ledger_sha256=sha(path))
    comparison = root / "comparison.json"
    comparison.write_text(json.dumps({"version": 1, **bindings}))
    config = root / "record.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "comparison": {"file": str(comparison), "sha256": sha(comparison)},
                "checkpoints": checkpoints,
            }
        )
    )
    return config


def test_record_retains_both_learners_but_missing_evidence_cannot_replace_default(
    tmp_path, candidates
):
    from fh5.candidate_store import CandidateHistory, CandidateRecord

    config = record_config(tmp_path, candidates, missing_execution=True)
    store = tmp_path / "versions"
    result = run_experiment(CandidateRecord(config, store, None, candidates[3])).summary[
        "candidate_store"
    ]
    assert result["scope"] == "synthetic_development_only"
    assert result["default"]["model_sha256"] == sha(
        Path(candidates[2]["incumbent"]) / "policy.json"
    )
    assert result["explorer"]["model_sha256"] == sha(
        Path(candidates[2]["candidate"]) / "policy.json"
    )
    assert result["selection"] == "retain_incumbent"
    assert "candidate:actual_synthetic_policy_execution_incomplete" in result["reasons"]
    assert result["default_changed"] is False
    assert result["real_driving_validated"] is False
    history = run_experiment(CandidateHistory(store)).summary["candidate_store"]
    assert history["revision"] == result["revision"]
    assert len(history["history"]) == 1
    for role, side in (("default", "incumbent"), ("explorer", "candidate")):
        checkpoint = store / history[role]["archive"] / "checkpoint"
        assert (checkpoint / "policy.json").read_bytes() == (
            Path(candidates[2][side]) / "policy.json"
        ).read_bytes()


def test_faster_qualified_candidate_can_be_rolled_back_without_erasing_exploration(
    tmp_path, candidates
):
    from fh5.candidate_store import CandidateHistory, CandidateRecord, CandidateRollback

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    store = tmp_path / "versions"
    original = run_experiment(
        CandidateRecord(
            record_config(first, candidates, missing_execution=True), store, None, candidates[3]
        )
    ).summary["candidate_store"]
    promoted = run_experiment(
        CandidateRecord(
            record_config(second, candidates), store, original["revision"], candidates[3]
        )
    ).summary["candidate_store"]
    assert promoted["selection"] == "prefer_candidate_locally"
    assert promoted["default"]["model_sha256"] == original["explorer"]["model_sha256"]
    assert promoted["synthetic_default_changed"] is True
    rolled = run_experiment(
        CandidateRollback(
            store,
            promoted["revision"],
            original["revision"],
            "Restore reliable baseline",
            candidates[3],
        )
    ).summary["candidate_store"]
    assert rolled["default"] == original["default"]
    assert rolled["explorer"] == promoted["explorer"]
    assert rolled["default_changed"] is False
    history = run_experiment(CandidateHistory(store)).summary["candidate_store"]["history"]
    assert len(history) == 3
    assert [row["operation"] for row in history] == ["selection", "selection", "rollback"]
    assert history[0] == original
    assert history[1] == promoted
    assert history[2]["target_revision"] == original["revision"]


@pytest.mark.parametrize("outcome", ["driving_failure", "wall_riding", "unknown"])
def test_faster_candidate_is_retained_as_aggressive_only_with_complete_legal_evidence(
    tmp_path, candidates, outcome
):
    from fh5.candidate_store import CandidateRecord

    result = run_experiment(
        CandidateRecord(
            record_config(tmp_path, candidates, candidate_outcome=outcome),
            tmp_path / "store",
            None,
            candidates[3],
        )
    ).summary["candidate_store"]
    assert result["default"]["model_sha256"] == sha(
        Path(candidates[2]["incumbent"]) / "policy.json"
    )
    assert result["selection"] == "retain_incumbent"
    expected = {"no_reference": result["explorer"]} if outcome == "driving_failure" else {}
    assert result["aggressive_by_reference"] == expected
    if outcome == "driving_failure":
        assert "candidate:reliability_regressed:no_reference" in result["reasons"]


def test_cli_persists_across_processes_and_rejects_competing_stale_writes(tmp_path, candidates):
    config = record_config(tmp_path, candidates)
    store = tmp_path / "versions"
    command = [sys.executable, "-m", "fh5"]
    saved = subprocess.run(
        command
        + [
            "candidate-record",
            "--config",
            str(config),
            "--store",
            str(store),
            "--registry",
            str(candidates[3]),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert saved.returncode == 0, saved.stderr
    original = json.loads(saved.stdout)
    args = command + [
        "candidate-rollback",
        "--store",
        str(store),
        "--expected-revision",
        original["revision"],
        "--target-revision",
        original["revision"],
        "--reason",
        "Independent process rollback",
        "--registry",
        str(candidates[3]),
    ]
    processes = [
        subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(2)
    ]
    try:
        outputs = [p.communicate(timeout=120) for p in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    assert sorted(p.returncode for p in processes) == [0, 2], outputs
    failed = next(stderr for p, (_, stderr) in zip(processes, outputs) if p.returncode == 2)
    assert "revision changed" in failed
    assert "Traceback" not in failed
    read = subprocess.run(
        command + ["candidate-history", "--store", str(store)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert read.returncode == 0, read.stderr
    history = json.loads(read.stdout)
    assert len(history["history"]) == 2
    assert history["history"][0] == original
    assert history["explorer"] == original["explorer"]


@pytest.fixture(scope="module")
def retained_store(tmp_path_factory, candidates):
    from fh5.candidate_store import CandidateRecord

    root = tmp_path_factory.mktemp("retained-version-store")
    store = root / "versions"
    run_experiment(
        CandidateRecord(
            record_config(root, candidates, missing_execution=True), store, None, candidates[3]
        )
    )
    return store


def test_history_detects_changed_retained_decision(tmp_path, retained_store):
    from fh5.candidate_store import CandidateHistory

    store = tmp_path / "versions"
    shutil.copytree(retained_store, store)
    prior = run_experiment(CandidateHistory(store)).summary["candidate_store"]
    report = store / prior["comparison"]
    report.write_bytes(report.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="evidence changed"):
        run_experiment(CandidateHistory(store))


@pytest.mark.parametrize("fault", ["stale", "archive", "source", "disk"])
def test_failed_update_or_rollback_leaves_committed_roles_and_history_intact(
    tmp_path, retained_store, candidates, monkeypatch, fault
):
    from fh5.candidate_store import CandidateHistory, CandidateRecord, CandidateRollback

    store = tmp_path / "versions"
    shutil.copytree(retained_store, store)
    prior = run_experiment(CandidateHistory(store)).summary["candidate_store"]
    revision = prior["revision"]
    operation = CandidateRollback(store, revision, revision, "Check fault handling", candidates[3])
    if fault == "stale":
        operation = replace(operation, expected_revision="0" * 64)
    if fault == "archive":
        (store / prior["default"]["archive"] / "checkpoint/policy.pt").write_bytes(
            b"broken learner"
        )
    original_open = Path.open

    def injected(path, mode="r", *args, **kwargs):
        if fault == "source" and path.resolve() == Path(prior["qualification"]["comparison_file"]):
            return io.BytesIO(b"{}\n")
        if (
            fault == "disk"
            and mode == "xb"
            and path.name == "policy.pt"
            and path.is_relative_to(store)
        ):
            raise OSError("simulated archive disk failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", injected)
    if fault == "disk":
        operation = CandidateRecord(
            record_config(tmp_path, candidates), store, revision, candidates[3]
        )
    with pytest.raises((ValueError, OSError)):
        run_experiment(operation)
    assert run_experiment(CandidateHistory(store)).summary["candidate_store"] == prior
