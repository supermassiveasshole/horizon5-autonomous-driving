"""Candidate retention through the public experiment boundary."""

import hashlib
import json
from pathlib import Path

import pytest
from test_sac_learning import warm_start

from fh5.candidate_archive import CandidateArchive
from fh5.experiment import run_experiment
from fh5.sac_learning import SACResume, SACTrain


def test_archived_candidate_keeps_its_identity_and_training_after_source_is_moved(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    replay = warm_start(source)
    run_experiment(SACTrain(source / "warm", replay, source / "candidate", steps=3))
    checkpoint = source / "candidate"
    raw = (checkpoint / "policy.json").read_bytes()
    identity = hashlib.sha256(raw).hexdigest()
    expected = run_experiment(SACResume(checkpoint, tmp_path / "expected", steps=5)).summary[
        "sac_learning"
    ]

    result = run_experiment(
        CandidateArchive(
            checkpoint,
            tmp_path / "archive",
            expected_checkpoint_sha256=identity,
            reason="Retain candidate for further exploration after comparison",
        )
    ).summary["candidate_archive"]
    archived = tmp_path / "archive/checkpoint"
    assert (archived / "policy.json").read_bytes() == raw
    assert result["checkpoint_sha256"] == identity
    assert result["default_changed"] is False
    assert result["driving_qualification"] == "not_established_by_archive"
    # Both resolved paths stay in this isolated test directory; no user data is moved.
    moved = tmp_path / "source-unavailable"
    assert source.resolve().is_relative_to(tmp_path.resolve())
    assert moved.resolve().is_relative_to(tmp_path.resolve())
    source.rename(moved)

    continued = run_experiment(SACResume(archived, tmp_path / "continued", steps=5)).summary[
        "sac_learning"
    ]
    assert continued["learner_state_sha256"] == expected["learner_state_sha256"]
    assert continued["predictions"] == expected["predictions"]
    restored_manifest = json.loads((tmp_path / "continued/policy.json").read_bytes())
    assert restored_manifest["continuation"]["parent_checkpoint_sha256"] == identity
    assert (
        json.loads((archived / "policy.json").read_bytes())["learner_state_sha256"]
        != (continued["learner_state_sha256"])
    )


def test_cli_restores_an_exact_candidate_without_changing_the_archive(tmp_path, capsys):
    from fh5.cli import main

    replay = warm_start(tmp_path)
    model = tmp_path / "candidate"
    trained = run_experiment(SACTrain(tmp_path / "warm", replay, model, steps=3))
    raw = (model / "policy.json").read_bytes()
    identity = hashlib.sha256(raw).hexdigest()
    archive = tmp_path / "archive"
    assert (
        main(
            [
                "candidate-archive",
                "--checkpoint",
                str(model),
                "--checkpoint-sha256",
                identity,
                "--output",
                str(archive),
                "--reason",
                "Retain exploratory candidate",
            ]
        )
        == 0
    )
    saved = json.loads(capsys.readouterr().out)
    manifest = (archive / "archive.json").read_bytes()
    output = tmp_path / "restored"
    assert (
        main(
            [
                "candidate-restore",
                "--archive",
                str(archive),
                "--archive-sha256",
                saved["archive_sha256"],
                "--output",
                str(output),
                "--reason",
                "Resume exploration",
            ]
        )
        == 0
    )
    restored = json.loads(capsys.readouterr().out)
    assert restored["checkpoint_sha256"] == identity
    assert restored["default_changed"] is False
    assert (output / "policy.json").read_bytes() == raw
    assert (archive / "archive.json").read_bytes() == manifest
    proof = json.loads((output / "restored-from.json").read_bytes())
    assert proof["archive_sha256"] == saved["archive_sha256"]
    resumed = run_experiment(SACResume(output, tmp_path / "reloaded", steps=0))
    assert (
        resumed.summary["sac_learning"]["learner_state_sha256"]
        == (trained.summary["sac_learning"]["learner_state_sha256"])
    )


@pytest.fixture
def candidate(tmp_path):
    replay = warm_start(tmp_path)
    path = tmp_path / "candidate"
    run_experiment(SACTrain(tmp_path / "warm", replay, path, steps=3))
    return path, hashlib.sha256((path / "policy.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("operation", ["archive", "restore"])
def test_storage_failure_never_publishes_a_loadable_candidate(
    tmp_path, candidate, monkeypatch, operation
):
    from fh5.candidate_archive import CandidateRestore

    checkpoint, identity = candidate
    archive = tmp_path / "archive"
    request = CandidateArchive(checkpoint, archive, identity, "Preserve candidate")
    artifact = archive / "checkpoint"
    if operation == "restore":
        archived = run_experiment(request).summary["candidate_archive"]
        artifact = tmp_path / "restored"
        request = CandidateRestore(archive, artifact, archived["archive_sha256"], "Restore")
    original_open = Path.open

    def fail_disk_write(path, mode="r", *args, **kwargs):
        if path == artifact / "policy.pt" and mode == "xb":
            raise OSError("Simulated full disk")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_disk_write)
    with pytest.raises(OSError, match="full disk"):
        run_experiment(request)
    assert not (artifact / "policy.json").exists()
    assert not (artifact / "restored-from.json").exists()
    assert (checkpoint / "policy.json").exists()
    if operation == "archive":
        assert not (archive / "archive.json").exists()


@pytest.mark.parametrize("version", [2, 3, 4])
def test_archive_keeps_training_phase_history_and_excludes_unrelated_files(tmp_path, version):
    first = tmp_path / "first"
    if version == 3:
        from test_sac_mixture import mixture_inputs

        initial, demo = mixture_inputs(tmp_path)
        run_experiment(
            SACResume(initial, first, steps=3, additions=(demo,), demonstration_fraction=0.5)
        )
    else:
        replay = warm_start(tmp_path)
        options = {"imitation_weights": (1.0, 0.0)} if version == 4 else {}
        run_experiment(SACTrain(tmp_path / "warm", replay, first, steps=3, **options))
    current = tmp_path / "current"
    run_experiment(SACResume(first, current, steps=2))
    raw = (current / "policy.json").read_bytes()
    manifest = json.loads(raw)
    assert manifest["version"] == version
    (current / "unrelated.txt").write_text("Not part of the learning snapshot")
    archive = tmp_path / "archive"
    result = run_experiment(
        CandidateArchive(
            current, archive, hashlib.sha256(raw).hexdigest(), "Keep training progress"
        )
    ).summary["candidate_archive"]
    assert result["total_steps"] == 5
    assert not (archive / "checkpoint/unrelated.txt").exists()
    assert (archive / "checkpoint/policy.json").read_bytes() == raw
    for prior in manifest["history"]:
        for field in ("checkpoint", "report"):
            assert (archive / "checkpoint" / prior[field]).read_bytes() == (
                current / prior[field]
            ).read_bytes()
    continued = run_experiment(SACResume(archive / "checkpoint", tmp_path / "next", steps=1))
    assert continued.summary["sac_learning"]["total_steps"] == 6
    if version == 3:
        assert continued.summary["sac_learning"]["sampling"]["available"] == {
            "demonstration": 4,
            "online": 2,
        }
    if version == 4:
        assert continued.summary["sac_learning"]["imitation"]["weight"] == 1.0


@pytest.mark.parametrize("operation", ["archive", "restore"])
@pytest.mark.parametrize("corrupt", ["pixel", "teacher", "history", "weights"])
def test_changed_dependencies_cannot_be_published(tmp_path, candidate, operation, corrupt):
    from fh5.candidate_archive import CandidateRestore

    initial, _ = candidate
    current = tmp_path / "current"
    run_experiment(SACResume(initial, current, steps=0))
    manifest = json.loads((current / "policy.json").read_bytes())
    identity = hashlib.sha256((current / "policy.json").read_bytes()).hexdigest()
    target = tmp_path / "archive"
    request = CandidateArchive(current, target, identity, "Keep candidate")
    source = current
    if operation == "restore":
        result = run_experiment(request).summary["candidate_archive"]
        source = target / "checkpoint"
        request = CandidateRestore(
            target, tmp_path / "restored", result["archive_sha256"], "Restore"
        )
        target = tmp_path / "restored"
    replay = json.loads((source / "experience/replay.json").read_bytes())
    names = {
        "pixel": "experience/" + replay["transitions"][0]["current"]["frames"][0]["path"],
        "teacher": "bc/actor.pt",
        "history": manifest["history"][0]["report"],
        "weights": "policy.pt",
    }
    (source / names[corrupt]).write_bytes(b"changed dependency")
    with pytest.raises((ValueError, OSError)):
        run_experiment(request)
    assert not target.exists()


def test_existing_destinations_and_wrong_bindings_leave_sources_unchanged(tmp_path, candidate):
    from fh5.candidate_archive import CandidateRestore

    checkpoint, identity = candidate
    original = (checkpoint / "policy.pt").read_bytes()
    archive = tmp_path / "archive"
    request = CandidateArchive(checkpoint, archive, identity, "Retain")
    saved = run_experiment(request).summary["candidate_archive"]
    archive_bytes = (archive / "archive.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(request)
    with pytest.raises(ValueError, match="identity"):
        run_experiment(CandidateArchive(checkpoint, tmp_path / "wrong", "0" * 64, "Retain"))
    with pytest.raises(ValueError, match="outside"):
        run_experiment(CandidateArchive(checkpoint, checkpoint / "child", identity, "Retain"))
    with pytest.raises(ValueError, match="changed"):
        run_experiment(CandidateRestore(archive, tmp_path / "bad-restore", "0" * 64, "Restore"))
    with pytest.raises(FileExistsError):
        run_experiment(CandidateRestore(archive, checkpoint, saved["archive_sha256"], "Restore"))
    assert (checkpoint / "policy.pt").read_bytes() == original
    assert (archive / "archive.json").read_bytes() == archive_bytes
    assert not (tmp_path / "wrong").exists()
    assert not (tmp_path / "bad-restore").exists()


def test_restore_rejects_changed_archive_inventory_without_exposing_output(tmp_path, candidate):
    from fh5.candidate_archive import CandidateRestore

    checkpoint, identity = candidate
    archive = tmp_path / "archive"
    result = run_experiment(CandidateArchive(checkpoint, archive, identity, "Retain")).summary[
        "candidate_archive"
    ]
    manifest = json.loads((archive / "archive.json").read_bytes())
    manifest["files"].pop("bc/actor.pt")
    raw = json.dumps(manifest).encode()
    (archive / "archive.json").write_bytes(raw)
    with pytest.raises(ValueError, match="changed"):
        run_experiment(
            CandidateRestore(archive, tmp_path / "old-id", result["archive_sha256"], "Restore")
        )
    with pytest.raises(ValueError, match="inventory"):
        run_experiment(
            CandidateRestore(
                archive, tmp_path / "missing", hashlib.sha256(raw).hexdigest(), "Restore"
            )
        )
    assert not (tmp_path / "missing").exists()
