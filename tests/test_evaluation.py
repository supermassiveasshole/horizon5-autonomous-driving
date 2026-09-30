"""Frozen batch evaluation through the agreed experiment-run boundary."""

import hashlib
import json
from dataclasses import asdict

import pytest
from test_attempts import evidence, protocol
from test_route_check import record, route

from fh5.experiment import run_experiment
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig


@pytest.fixture(scope="module")
def policy(tmp_path_factory):
    pytest.importorskip("torch")
    from test_temporal_bc import temporal_fixture

    from fh5.temporal_bc import TemporalBCTrain

    root = tmp_path_factory.mktemp("evaluation-policy")
    config, _ = temporal_fixture(root)
    run_experiment(TemporalBCTrain(config, root / "model"))
    return root / "model"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(tmp_path, policy, modes=None, purpose="development"):
    from fh5.evaluation import EvaluationPrepare

    bundle = route(tmp_path)
    task = protocol(tmp_path, bundle)
    modes = modes or ["no_reference"]
    config = tmp_path / "evaluation.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "purpose": purpose,
                "model": {"directory": str(policy), "manifest_sha256": sha(policy / "model.json")},
                "task": {"file": str(task), "sha256": sha(task)},
                "conditions": {
                    "snapshot": json.loads((tmp_path / "reference/session.json").read_bytes())[
                        "snapshot"
                    ],
                    "camera": "chase_far",
                    "navigation": "full_racing_line",
                    "task_basis": "independent_local_task",
                },
                "runtime": {
                    **asdict(
                        RealtimeConfig(pixels=PixelContract(size=(64, 36)), reference_count=1)
                    ),
                    "pixels": PixelContract(size=(64, 36)).metadata(),
                },
                "plan": [
                    {"id": f"run-{i}", "reference_mode": mode} for i, mode in enumerate(modes)
                ],
                "criteria": {
                    "min_valid_attempts": 1,
                    "reliability_tolerance": 0.05,
                    "min_time_improvement_fraction": 0.02,
                    "anomalies": "quarantine_keep_in_denominator",
                    "exploration": False,
                    "rewind": False,
                },
            }
        )
    )
    result = run_experiment(EvaluationPrepare(config, tmp_path / "frozen"))
    return result, config


def ledger(tmp_path, entries):
    manifest = tmp_path / "frozen/batch.json"
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "batch_sha256": sha(manifest),
                "entries": entries,
            }
        )
    )
    return path


def entry(slot, source, proof=None):
    return {
        "slot_id": slot,
        "recording": str(source),
        "files": {p.name: sha(p) for p in source.iterdir() if p.suffix in (".json", ".jsonl")},
        "evidence": {"file": str(proof), "sha256": sha(proof)} if proof else None,
    }


def separate_clock(source, offset_ns):
    path = source / "packets.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["received_monotonic_ns"] += offset_ns
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_frozen_batch_retains_mixed_outcomes_and_unstarted_slots(tmp_path, policy):
    from fh5.evaluation import EvaluationReview

    prepared, _ = prepare(tmp_path, policy, ["no_reference"] * 4 + ["reference_assisted"] * 2)
    rows = []
    for i, name in enumerate(("success", "fast", "wall", "unreviewed", "empty")):
        folder = tmp_path / name
        folder.mkdir()
        source = record(
            folder,
            "drive",
            [] if name == "empty" else [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)],
            speed=30 if name == "fast" else 4,
        )
        separate_clock(source, i * 1_000_000_000)
        events = (
            [{"packet_index": 1, "kind": "wall_riding", "status": "confirmed"}]
            if name == "wall"
            else []
        )
        proof = evidence(folder, source, events) if name not in ("unreviewed", "empty") else None
        rows.append(entry(f"run-{i}", source, proof))
    result = run_experiment(
        EvaluationReview(tmp_path / "frozen", ledger(tmp_path, rows), tmp_path / "review")
    )
    batch = result.summary["evaluation"]
    assert prepared.summary["evaluation"]["state"] == "prepared"
    assert batch["planned_runs"] == 6
    assert batch["recorded_runs"] == 5
    assert batch["unstarted_slots"] == ["run-5"]
    assert batch["metrics"]["all_attempts"] == 5
    assert batch["metrics"]["outcomes"] == {
        "valid_complete": 1,
        "driving_failed": 1,
        "invalid": 1,
        "pending_review": 1,
        "interface_error": 1,
    }
    assert batch["metrics"]["valid_fraction_all_attempts"] == 0.2
    assert batch["metrics"]["classified_driving_attempts"] == 3
    assert batch["metrics"]["valid_fraction_classified_driving"] == pytest.approx(1 / 3)
    assert batch["metrics"]["valid_duration_s"] == {
        "count": 1,
        "min": 0.3,
        "median": 0.3,
        "max": 0.3,
    }
    assert batch["by_reference"]["no_reference"]["all_attempts"] == 4
    assert batch["by_reference"]["reference_assisted"]["all_attempts"] == 1
    assert batch["commands_sent"] is False
    assert batch["automatic_promotion_allowed"] is False
    assert batch["diagnostic_only"] is True
    assert all(a["evidence_gaps"] for a in batch["attempts"])
    assert (tmp_path / "review/batch-report.json").is_file()
    assert result.report_path.is_file()


def test_damaged_recording_does_not_erase_other_attempts_or_its_own_failure(tmp_path, policy):
    from fh5.evaluation import EvaluationReview

    prepare(tmp_path, policy, ["no_reference"] * 2)
    entries = []
    for i in range(2):
        folder = tmp_path / f"source-{i}"
        folder.mkdir()
        source = record(folder, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
        separate_clock(source, i * 1_000_000_000)
        entries.append(entry(f"run-{i}", source, evidence(folder, source)))
    frozen_ledger = ledger(tmp_path, entries)
    (tmp_path / "source-1/drive/packets.jsonl").write_text("damaged after binding")
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", frozen_ledger, tmp_path / "review")
    ).summary["evaluation"]
    assert summary["metrics"]["all_attempts"] == 2
    assert [r["outcome"] for r in summary["attempts"]] == ["valid_complete", "interface_error"]
    assert summary["unresolved_recordings"] == 1
    assert summary["attempts"][1]["valid_duration_s"] is None
    assert "source changed" in summary["attempts"][1]["read_error"]


@pytest.mark.parametrize("change", ["reference", "actions", "vehicle"])
def test_runtime_must_agree_with_model_and_task_before_freezing(tmp_path, policy, change):
    from fh5.evaluation import EvaluationPrepare

    _, config = prepare(tmp_path, policy)
    options = json.loads(config.read_bytes())
    if change == "reference":
        options["runtime"]["reference_count"] = 2
    elif change == "actions":
        options["runtime"]["action_offsets_ms"] = [100, 0]
    else:
        options["runtime"]["expected_pi"] = 998
    config.write_text(json.dumps(options))
    with pytest.raises(ValueError, match="runtime"):
        run_experiment(EvaluationPrepare(config, tmp_path / "incompatible"))
    assert not (tmp_path / "incompatible/batch.json").exists()


def test_cli_freezes_protocol_and_reviews_every_restart_without_using_developer_files(
    tmp_path, policy, capsys
):
    from fh5.cli import main

    _, config = prepare(tmp_path, policy, purpose="final")
    output = tmp_path / "cli-batch"
    assert main(["evaluation-prepare", "--config", str(config), "--output", str(output)]) == 0
    frozen = json.loads(capsys.readouterr().out)
    assert frozen["purpose"] == "final" and frozen["planned_runs"] == 1
    config.write_text("changed after freeze")
    (tmp_path / "task.json").write_text("changed after freeze")
    (tmp_path / "route/route.json").write_text("changed after freeze")
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    proof = evidence(
        tmp_path, source, [{"packet_index": 2, "kind": "restart", "status": "confirmed"}]
    )
    path = ledger(tmp_path, [entry("run-0", source, proof)])
    binding = json.loads(path.read_bytes())
    binding["batch_sha256"] = sha(output / "batch.json")
    path.write_text(json.dumps(binding))
    assert (
        main(
            [
                "evaluation-review",
                "--batch",
                str(output),
                "--ledger",
                str(path),
                "--output",
                str(tmp_path / "review"),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["purpose"] == "final" and summary["extra_attempts"] == 1
    assert summary["metrics"]["all_attempts"] == 2
    assert summary["metrics"]["outcomes"]["driving_failed"] == 1
    assert summary["metrics"]["outcomes"]["valid_complete"] == 1
    assert summary["automatic_promotion_allowed"] is False


def test_preparing_evaluation_does_not_perturb_the_callers_learning_randomness(tmp_path, policy):
    import torch

    before = torch.get_rng_state().clone()
    prepare(tmp_path, policy)
    assert torch.equal(torch.get_rng_state(), before)


def test_copy_with_relabelled_session_cannot_count_as_an_independent_repeat(tmp_path, policy):
    import shutil

    from fh5.evaluation import EvaluationReview

    prepare(tmp_path, policy, ["no_reference"] * 2)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    proof = evidence(tmp_path, source)
    duplicate = tmp_path / "copy"
    shutil.copytree(source, duplicate)
    metadata = json.loads((duplicate / "session.json").read_bytes())
    metadata["annotation"] = "another name does not make a new driving attempt"
    (duplicate / "session.json").write_text(json.dumps(metadata))
    path = ledger(tmp_path, [entry("run-0", source, proof), entry("run-1", duplicate, proof)])
    with pytest.raises(ValueError, match="independent"):
        run_experiment(EvaluationReview(tmp_path / "frozen", path, tmp_path / "review"))


def test_distinct_empty_recordings_still_count_as_interface_attempts(tmp_path, policy):
    from fh5.evaluation import EvaluationReview

    prepare(tmp_path, policy, ["no_reference"] * 2)
    entries = []
    for i in range(2):
        source = record(tmp_path, f"empty-{i}", [])
        entries.append(entry(f"run-{i}", source))
    summary = run_experiment(
        EvaluationReview(tmp_path / "frozen", ledger(tmp_path, entries), tmp_path / "review")
    ).summary["evaluation"]
    assert summary["metrics"]["all_attempts"] == 2
    assert summary["metrics"]["outcomes"]["interface_error"] == 2
    assert summary["unresolved_recordings"] == 0


@pytest.mark.parametrize("asset", ["model/actor.pt", "route/reference.json", "batch.json"])
def test_changed_frozen_assets_cannot_be_used_to_reinterpret_a_batch(tmp_path, policy, asset):
    from fh5.evaluation import EvaluationReview

    prepare(tmp_path, policy)
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    frozen_ledger = ledger(tmp_path, [entry("run-0", source, evidence(tmp_path, source))])
    path = tmp_path / "frozen" / asset
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed|different frozen batch"):
        run_experiment(EvaluationReview(tmp_path / "frozen", frozen_ledger, tmp_path / "review"))
    assert not (tmp_path / "review").exists()


@pytest.mark.parametrize("change", ["missing_runtime_bound", "missing_model", "bad_conditions"])
def test_incomplete_protocol_cannot_publish_a_partly_implicit_freeze(tmp_path, policy, change):
    from fh5.evaluation import EvaluationPrepare

    _, config = prepare(tmp_path, policy)
    options = json.loads(config.read_bytes())
    if change == "missing_runtime_bound":
        del options["runtime"]["action_lease_ms"]
    elif change == "missing_model":
        options["model"] = {}
    else:
        options["conditions"] = None
    config.write_text(json.dumps(options))
    with pytest.raises(ValueError, match="[Ee]valuation"):
        run_experiment(EvaluationPrepare(config, tmp_path / "incomplete"))
    assert not (tmp_path / "incomplete/batch.json").exists()
