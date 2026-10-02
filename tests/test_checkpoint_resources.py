"""Growing checkpoint evidence does not decide whether learning may continue."""

import hashlib
import json
import tracemalloc
from pathlib import Path

import pytest
from test_sac_learning import warm_start

from fh5.experiment import run_experiment
from fh5.sac_learning import SACResume, SACTrain


def test_resume_streams_large_training_evidence_and_preserves_learning_state(tmp_path):
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
        for _ in range(129):
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
        result = run_experiment(SACResume(first, continued, steps=2)).summary["sac_learning"]
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
    retained = continued / checkpoint["history"][-1]["report"]
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
    name = manifest["history"][0]["report"]
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
