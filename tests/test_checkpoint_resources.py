"""Growing checkpoint evidence does not decide whether learning may continue."""

import hashlib
import json
import tracemalloc
from pathlib import Path

import pytest
from checkpoint_files import history_entries
from test_sac_learning import warm_start

from fh5.candidate_archive import CandidateArchive, CandidateRestore
from fh5.experiment import run_experiment
from fh5.sac_learning import SACResume, SACTrain


@pytest.mark.parametrize("flow", ["continue", "archive_restore"])
def test_resume_streams_large_training_evidence_and_preserves_learning_state(tmp_path, flow):
    torch = pytest.importorskip("torch")
    replay = warm_start(tmp_path)
    first = tmp_path / "first"
    run_experiment(SACTrain(tmp_path / "warm", replay, first, steps=3))
    whole = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "whole", steps=5)
    ).summary["sac_learning"]

    # Whitespace leaves the JSON's meaning intact while crossing the legacy 128 MiB
    # acceptance ceiling. Seal the representation as a valid external checkpoint.
    report = first / "training-report.json"
    with report.open("ab") as stream:
        padding = b" " * 1024**2
        for _ in range(129 if flow == "continue" else 257):
            stream.write(padding)
    with report.open("rb") as stream:
        report_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = json.loads((first / "policy.json").read_text())
    saved = torch.load(first / "policy.pt", map_location="cpu", weights_only=True)
    manifest["training_report_sha256"] = report_sha
    saved["metadata"]["training_report_sha256"] = report_sha
    torch.save(saved, first / "policy.pt")
    with (first / "policy.pt").open("rb") as stream:
        manifest["weights_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (first / "policy.json").write_text(json.dumps(manifest), encoding="utf-8")

    continued = tmp_path / "continued"
    tracemalloc.start()
    try:
        source = first
        if flow == "archive_restore":
            identity = hashlib.sha256((first / "policy.json").read_bytes()).hexdigest()
            archive = tmp_path / "archive"
            retained = run_experiment(
                CandidateArchive(first, archive, identity, "Retain complete training evidence")
            ).summary["candidate_archive"]
            source = tmp_path / "restored"
            restored = run_experiment(
                CandidateRestore(archive, source, retained["archive_sha256"], "Continue training")
            ).summary["candidate_restore"]
            assert restored["checkpoint_sha256"] == identity
            assert restored["total_steps"] == 3
        result = run_experiment(SACResume(source, continued, steps=2)).summary["sac_learning"]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Merely removing the size rejection while still loading the whole report fails
    # this check. This measures Python allocations; it is not a claim about Torch RSS.
    assert peak < report.stat().st_size
    assert result["total_steps"] == 5
    assert result["learner_state_sha256"] == whole["learner_state_sha256"]
    assert result["predictions"] == whole["predictions"]
    checkpoint = json.loads((continued / "policy.json").read_text())
    retained = continued / history_entries(continued, checkpoint)[-1]["report"]
    assert retained.stat().st_size > 128 * 1024**2
    with retained.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == report_sha

    # The larger retained evidence remains mandatory and is checked again next time.
    with retained.open("r+b") as stream:
        stream.write(b"!")
    with pytest.raises(ValueError, match="history changed"):
        run_experiment(SACResume(continued, tmp_path / "corrupt", steps=1))
    assert not (tmp_path / "corrupt/policy.json").exists()


def test_changed_history_during_copy_is_not_published_as_a_resumable_checkpoint(
    tmp_path, monkeypatch
):
    replay = warm_start(tmp_path)
    first = tmp_path / "first"
    run_experiment(SACTrain(tmp_path / "warm", replay, first, steps=1))
    manifest = json.loads((first / "policy.json").read_text())
    name = history_entries(first, manifest)[0]["report"]
    output = tmp_path / "continued"
    original_open = Path.open
    changed = False

    def concurrent_change(path, mode="r", *args, **kwargs):
        nonlocal changed
        if path == output / name and mode == "xb":
            with original_open(first / name, "r+b") as stream:
                stream.write(b"!")
            changed = True
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", concurrent_change)
    with pytest.raises(ValueError, match="history changed during copy"):
        run_experiment(SACResume(first, output, steps=1))
    assert changed
    assert not (output / "policy.json").exists()


@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_history_is_rechecked_before_publishing_completed_training(tmp_path, monkeypatch, fault):
    replay = warm_start(tmp_path)
    first = tmp_path / "first"
    run_experiment(SACTrain(tmp_path / "warm", replay, first, steps=1))
    manifest = json.loads((first / "policy.json").read_text())
    output = tmp_path / "continued"
    name = history_entries(first, manifest)[0]["report"]
    target = output / name
    original_open = Path.open
    fault_injected = False

    def history_disappears_after_updates(path, mode="r", *args, **kwargs):
        nonlocal fault_injected
        if path == output / "policy.pt" and mode == "xb":
            if fault == "missing":
                target.unlink()
            else:
                with original_open(target, "wb") as stream:
                    stream.write(b"{}")
            fault_injected = True
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", history_disappears_after_updates)
    with pytest.raises((ValueError, OSError)):
        run_experiment(SACResume(first, output, steps=1))
    assert fault_injected
    assert not (output / "policy.json").exists()
    assert (first / name).exists()
