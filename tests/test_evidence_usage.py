"""Evidence use is persistent and final evaluation never erases known reuse."""

import json
from dataclasses import replace

import pytest
from test_attempts import evidence
from test_evaluation import entry, ledger, prepare, sha
from test_evaluation import policy as policy
from test_route_check import record

from fh5.evaluation import EvaluationPrepare, EvaluationReview
from fh5.experiment import run_experiment


def test_training_recording_cannot_be_relabelled_as_independent_final_evidence(tmp_path, policy):
    from fh5.evidence_usage import RecordUsage

    prepare(tmp_path, policy, purpose="final")
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    registry = tmp_path / "usage.sqlite"
    registered = run_experiment(
        RecordUsage(registry, (source,), "training", tmp_path / "registered")
    )
    assert registered.summary["evidence_usage"]["recordings"] == 1
    result = run_experiment(
        EvaluationReview(
            tmp_path / "frozen",
            ledger(tmp_path, [entry("run-0", source, evidence(tmp_path, source))]),
            tmp_path / "review",
            registry_file=registry,
        )
    )
    report = result.summary["evaluation"]
    assert report["independence"]["status"] == "known_overlap"
    assert report["independence"]["conflicts"][0]["prior_role"] == "training"
    assert report["independence"]["independence_proven"] is False
    assert report["metrics"]["all_attempts"] == 1
    assert report["automatic_promotion_allowed"] is False
    assert json.loads((tmp_path / "review/usage-snapshot.json").read_bytes())["uses"]


def test_final_batch_is_reserved_before_new_recordings_and_review_is_idempotent(tmp_path, policy):
    _, config = prepare(tmp_path, policy, purpose="final")
    registry = tmp_path / "usage.sqlite"
    run_experiment(EvaluationPrepare(config, tmp_path / "reserved", registry_file=registry))
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    binding = ledger(tmp_path, [entry("run-0", source, evidence(tmp_path, source))])
    document = json.loads(binding.read_bytes())
    document["batch_sha256"] = sha(tmp_path / "reserved/batch.json")
    binding.write_text(json.dumps(document))
    snapshots = []
    for index in range(2):
        result = run_experiment(
            EvaluationReview(tmp_path / "reserved", binding, tmp_path / f"review-{index}", registry)
        )
        usage = result.summary["evaluation"]["independence"]
        assert usage["status"] == "no_known_overlap"
        assert usage["reservation_verified"] is True
        assert usage["independence_proven"] is False
        snapshots.append((tmp_path / f"review-{index}/usage-snapshot.json").read_bytes())
    assert snapshots[0] == snapshots[1]


def test_missing_registry_preserves_attempts_but_reports_unknown_independence(tmp_path, policy):
    prepare(tmp_path, policy, purpose="final")
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    missing = tmp_path / "missing.sqlite"
    result = run_experiment(
        EvaluationReview(
            tmp_path / "frozen",
            ledger(tmp_path, [entry("run-0", source)]),
            tmp_path / "review",
            missing,
        )
    )
    assert result.summary["evaluation"]["metrics"]["all_attempts"] == 1
    assert result.summary["evaluation"]["independence"]["status"] == "unknown"
    assert result.summary["evaluation"]["independence"]["error"]
    assert not missing.exists()
    assert result.report_path.is_file()


def test_cli_registers_selection_and_reports_reused_final_data_without_devices(
    tmp_path, policy, capsys
):
    from fh5.cli import main

    _, config = prepare(tmp_path, policy, purpose="final")
    registry = tmp_path / "usage.sqlite"
    reserved = tmp_path / "reserved"
    assert (
        main(
            [
                "evaluation-prepare",
                "--config",
                str(config),
                "--output",
                str(reserved),
                "--registry",
                str(registry),
            ]
        )
        == 0
    )
    capsys.readouterr()
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    assert (
        main(
            [
                "evidence-use",
                "--registry",
                str(registry),
                "--role",
                "selection",
                "--recording",
                str(source),
                "--output",
                str(tmp_path / "selection"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["role"] == "selection"
    path = ledger(tmp_path, [entry("run-0", source)])
    contents = json.loads(path.read_bytes())
    contents["batch_sha256"] = sha(reserved / "batch.json")
    path.write_text(json.dumps(contents))
    assert (
        main(
            [
                "evaluation-review",
                "--batch",
                str(reserved),
                "--ledger",
                str(path),
                "--output",
                str(tmp_path / "review"),
                "--registry",
                str(registry),
            ]
        )
        == 4
    )
    report = json.loads(capsys.readouterr().out)
    assert report["independence"]["status"] == "known_overlap"
    assert report["metrics"]["all_attempts"] == 1


def test_repeated_execution_registers_every_started_recording_in_the_reserved_batch(
    tmp_path, policy
):
    from test_evaluation_run import Batch, request

    operation = request(tmp_path, policy)
    registry = tmp_path / "usage.sqlite"
    reserved = tmp_path / "reserved"
    run_experiment(EvaluationPrepare(tmp_path / "evaluation.json", reserved, registry))
    result = run_experiment(
        replace(
            operation,
            batch_dir=reserved,
            batch_sha256=sha(reserved / "batch.json"),
            registry_file=registry,
        ),
        evaluation_environment=Batch(),
    )
    evaluation = result.summary["evaluation"]
    assert evaluation["metrics"]["all_attempts"] == 2
    assert evaluation["independence"]["reservation_verified"] is True
    assert evaluation["independence"]["status"] == "no_known_overlap"
    snapshot = json.loads((operation.output_dir / "review/usage-snapshot.json").read_bytes())
    assert len(snapshot["uses"]) == 4
    assert {r["role"] for r in snapshot["uses"]} == {"selection"}


def test_future_dated_source_is_not_treated_as_recorded_after_a_reservation(tmp_path, policy):
    _, config = prepare(tmp_path, policy, purpose="final")
    registry = tmp_path / "usage.sqlite"
    run_experiment(EvaluationPrepare(config, tmp_path / "reserved", registry))
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    metadata = json.loads((source / "session.json").read_bytes())
    metadata["created_utc"] = "2099-01-01T00:00:00+00:00"
    (source / "session.json").write_text(json.dumps(metadata))
    binding = ledger(tmp_path, [entry("run-0", source)])
    content = json.loads(binding.read_bytes())
    content["batch_sha256"] = sha(tmp_path / "reserved/batch.json")
    binding.write_text(json.dumps(content))
    result = run_experiment(
        EvaluationReview(tmp_path / "reserved", binding, tmp_path / "review", registry)
    )
    assert result.summary["evaluation"]["independence"]["status"] == "unknown"


@pytest.mark.parametrize("later_use", ["training", "selection", "another_final"])
def test_later_use_or_another_final_batch_cannot_reuse_the_same_recording(
    tmp_path, policy, later_use
):
    import shutil

    from fh5.evidence_usage import RecordUsage

    _, config = prepare(tmp_path, policy, purpose="final")
    registry = tmp_path / "usage.sqlite"
    reserved = tmp_path / "reserved"
    run_experiment(EvaluationPrepare(config, reserved, registry))
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    binding = ledger(tmp_path, [entry("run-0", source)])
    data = json.loads(binding.read_bytes())
    data["batch_sha256"] = sha(reserved / "batch.json")
    binding.write_text(json.dumps(data))
    first = run_experiment(EvaluationReview(reserved, binding, tmp_path / "first", registry))
    assert first.summary["evaluation"]["independence"]["status"] == "no_known_overlap"
    if later_use == "another_final":
        reserved = tmp_path / "second-batch"
        run_experiment(EvaluationPrepare(config, reserved, registry))
        data["batch_sha256"] = sha(reserved / "batch.json")
        binding.write_text(json.dumps(data))
    else:
        alias = tmp_path / "renamed-recording"
        shutil.copytree(source, alias)
        metadata = json.loads((alias / "session.json").read_bytes())
        metadata["annotation"] = "renaming and annotating does not make an independent recording"
        (alias / "session.json").write_text(json.dumps(metadata))
        run_experiment(RecordUsage(registry, (alias,), later_use, tmp_path / "later"))
    second = run_experiment(EvaluationReview(reserved, binding, tmp_path / "second", registry))
    usage = second.summary["evaluation"]["independence"]
    assert usage["status"] == "known_overlap"
    expected = "final" if later_use == "another_final" else later_use
    assert expected in {c["prior_role"] for c in usage["conflicts"]}
    assert second.summary["evaluation"]["metrics"]["all_attempts"] == 1


def test_failed_registration_does_not_partially_change_existing_use_history(tmp_path):
    from fh5.evidence_usage import RecordUsage

    registry = tmp_path / "usage.sqlite"
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2)])
    run_experiment(RecordUsage(registry, (source,), "training", tmp_path / "first"))
    before = (tmp_path / "first/usage-snapshot.json").read_bytes()
    with pytest.raises(OSError):
        run_experiment(
            RecordUsage(registry, (source, tmp_path / "missing"), "selection", tmp_path / "bad")
        )
    run_experiment(RecordUsage(registry, (source,), "training", tmp_path / "second"))
    assert (tmp_path / "second/usage-snapshot.json").read_bytes() == before
    assert not (tmp_path / "bad").exists()


def test_cli_damaged_registry_returns_input_error_without_replacing_history(tmp_path, capsys):
    from fh5.cli import main

    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2)])
    registry = tmp_path / "usage.sqlite"
    registry.write_bytes(b"damaged existing registry")
    assert (
        main(
            [
                "evidence-use",
                "--registry",
                str(registry),
                "--role",
                "training",
                "--recording",
                str(source),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert "registry" in capsys.readouterr().err.lower()
    assert registry.read_bytes() == b"damaged existing registry"


@pytest.mark.parametrize("change", ["replace_failed", "omit_failed"])
def test_registered_slot_cannot_hide_a_previous_failure(tmp_path, policy, change):
    _, config = prepare(tmp_path, policy, purpose="final")
    registry = tmp_path / "usage.sqlite"
    batch = tmp_path / "reserved"
    run_experiment(EvaluationPrepare(config, batch, registry))
    failed = record(tmp_path, "failed", [(0, 0.2), (1, 0.2)], speed=30)
    original = ledger(tmp_path, [entry("run-0", failed, evidence(tmp_path, failed))])
    data = json.loads(original.read_bytes())
    data["batch_sha256"] = sha(batch / "batch.json")
    original.write_text(json.dumps(data))
    first = run_experiment(EvaluationReview(batch, original, tmp_path / "first", registry))
    assert first.summary["evaluation"]["metrics"]["outcomes"]["driving_failed"] == 1
    first_bytes = (tmp_path / "first/batch-report.json").read_bytes()
    success = record(tmp_path, "success", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    data["entries"] = [entry("run-0", success)] if change == "replace_failed" else []
    original.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="registered.*slot|registered.*attempt"):
        run_experiment(EvaluationReview(batch, original, tmp_path / "second", registry))
    assert (tmp_path / "first/batch-report.json").read_bytes() == first_bytes
    assert not (tmp_path / "second/batch-report.json").exists()


def test_unstarted_slot_can_be_added_without_changing_previously_reviewed_attempts(
    tmp_path, policy
):
    from test_evaluation import separate_clock

    _, config = prepare(tmp_path, policy, ["no_reference"] * 2, purpose="final")
    registry = tmp_path / "usage.sqlite"
    batch = tmp_path / "reserved"
    run_experiment(EvaluationPrepare(config, batch, registry))
    first = record(tmp_path, "first-drive", [(0, 0.2), (1, 0.2)])
    path = ledger(tmp_path, [entry("run-0", first)])
    data = json.loads(path.read_bytes())
    data["batch_sha256"] = sha(batch / "batch.json")
    path.write_text(json.dumps(data))
    partial = run_experiment(EvaluationReview(batch, path, tmp_path / "partial", registry))
    assert partial.summary["evaluation"]["unstarted_slots"] == ["run-1"]
    second = record(tmp_path, "second-drive", [(0, 0.2), (1, 0.2)])
    separate_clock(second, 10_000_000_000)
    data["entries"].append(entry("run-1", second))
    path.write_text(json.dumps(data))
    complete = run_experiment(EvaluationReview(batch, path, tmp_path / "complete", registry))
    assert complete.summary["evaluation"]["metrics"]["all_attempts"] == 2
    assert complete.summary["evaluation"]["unstarted_slots"] == []
    assert complete.summary["evaluation"]["independence"]["status"] == "no_known_overlap"


def test_legacy_registry_preserves_history_and_marks_missing_old_slot_bindings_unknown(
    tmp_path, policy
):
    import sqlite3

    from fh5.evidence_usage import RecordUsage

    _, config = prepare(tmp_path, policy, purpose="final")
    registry = tmp_path / "usage.sqlite"
    batch = tmp_path / "reserved"
    run_experiment(EvaluationPrepare(config, batch, registry))
    source = record(tmp_path, "drive", [(0, 0.2), (1, 0.2)])
    path = ledger(tmp_path, [entry("run-0", source)])
    data = json.loads(path.read_bytes())
    data["batch_sha256"] = sha(batch / "batch.json")
    path.write_text(json.dumps(data))
    run_experiment(EvaluationReview(batch, path, tmp_path / "original", registry))
    before = json.loads((tmp_path / "original/usage-snapshot.json").read_bytes())
    # Fixture: the first registry format had no per-slot bindings or migration marker.
    with sqlite3.connect(registry) as db:
        db.execute("DROP TABLE slots")
        db.execute("DROP TABLE IF EXISTS legacy_reviews")
        db.execute("PRAGMA user_version=1")

    migrated = run_experiment(EvaluationReview(batch, path, tmp_path / "migrated", registry))
    usage = migrated.summary["evaluation"]["independence"]
    assert usage["status"] == "unknown"
    assert "legacy_slot_history_unavailable" in usage["unknown_reasons"]
    assert usage["reservation_verified"] is True
    after = json.loads((tmp_path / "migrated/usage-snapshot.json").read_bytes())
    for field in ("registry_id", "uses", "reservations"):
        assert after[field] == before[field]

    # The original database remains usable, and later uses still expose overlap.
    run_experiment(RecordUsage(registry, (source,), "training", tmp_path / "training"))
    reviewed = run_experiment(EvaluationReview(batch, path, tmp_path / "reviewed", registry))
    assert reviewed.summary["evaluation"]["independence"]["status"] == "known_overlap"
    assert "training" in {
        c["prior_role"] for c in reviewed.summary["evaluation"]["independence"]["conflicts"]
    }
    run_experiment(EvaluationPrepare(config, tmp_path / "next-batch", registry))
