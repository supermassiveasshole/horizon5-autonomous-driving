"""Long-lived checkpoint ancestry uses portable independent records."""

import hashlib
import json

import pytest

from fh5.evaluation.candidate_archive import CandidateArchive, CandidateRestore
from fh5.experiment import run_experiment
from fh5.learning.sac.training import SACResume, SACTrain
from tests.learning.sac.test_sac_learning import warm_start
from tests.support.checkpoint_files import history_entries


def test_checkpoint_history_is_indexed_portable_and_preserves_actual_learning(tmp_path):
    replay = warm_start(tmp_path)
    first = tmp_path / "first"
    run_experiment(SACTrain(tmp_path / "warm", replay, first, steps=3))
    first_manifest = json.loads((first / "policy.json").read_text())
    assert isinstance(first_manifest["history"], dict)
    assert first_manifest["history"]["kind"] == "linked-checkpoint-history-v1"
    assert first_manifest["history"]["count"] == 1
    first_identity = hashlib.sha256((first / "policy.json").read_bytes()).hexdigest()
    expected = run_experiment(SACResume(first, tmp_path / "expected", steps=2)).summary[
        "sac_learning"
    ]
    archive = tmp_path / "archive"
    archived = run_experiment(CandidateArchive(first, archive, first_identity, "Preserve ancestry"))
    restored = tmp_path / "restored"
    run_experiment(
        CandidateRestore(
            archive, restored, archived.summary["candidate_archive"]["archive_sha256"], "Continue"
        )
    )
    child = tmp_path / "child"
    result = run_experiment(SACResume(restored, child, steps=2)).summary["sac_learning"]
    assert result["learner_state_sha256"] == expected["learner_state_sha256"]
    assert result["predictions"] == expected["predictions"]
    child_manifest = json.loads((child / "policy.json").read_text())
    assert child_manifest["history"]["count"] == 2
    assert child_manifest["history"]["head"] != first_manifest["history"]["head"]
    ancestor_index = first_manifest["history"]["head"]["path"]
    assert (child / ancestor_index).read_bytes() == (first / ancestor_index).read_bytes()
    entries = history_entries(child, child_manifest)
    assert len(entries) == 2
    assert entries[1]["checkpoint_sha256"] == first_identity
    assert (child / entries[0]["report"]).read_bytes() == (
        tmp_path / "warm/training-report.json"
    ).read_bytes()
    assert (child / entries[1]["report"]).read_bytes() == (
        first / "training-report.json"
    ).read_bytes()


def rebind_history(root, history, stage="policy"):
    """Construct a compatible external artifact, preserving actual learner state."""
    torch = pytest.importorskip("torch")
    manifest_file = root / (stage + ".json")
    weights_file = root / (stage + ".pt")
    manifest = json.loads(manifest_file.read_bytes())
    manifest["history"] = history
    saved = torch.load(weights_file, map_location="cpu", weights_only=True)
    saved["metadata"] = {k: v for k, v in manifest.items() if k != "weights_sha256"}
    torch.save(saved, weights_file)
    manifest["weights_sha256"] = hashlib.sha256(weights_file.read_bytes()).hexdigest()
    manifest_file.write_text(json.dumps(manifest))


def test_legacy_inline_ancestry_migrates_without_changing_learning_or_source(tmp_path):
    replay = warm_start(tmp_path)
    rebind_history(tmp_path / "warm", [], "critic")
    whole = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "whole", steps=5)
    ).summary["sac_learning"]
    first = tmp_path / "first"
    run_experiment(SACTrain(tmp_path / "warm", replay, first, steps=3))
    legacy_entries = history_entries(first)
    rebind_history(first, legacy_entries)
    source = (first / "policy.json").read_bytes()
    weights = (first / "policy.pt").read_bytes()
    output = tmp_path / "continued"
    continued = run_experiment(SACResume(first, output, steps=2)).summary["sac_learning"]
    assert continued["learner_state_sha256"] == whole["learner_state_sha256"]
    assert continued["predictions"] == whole["predictions"]
    assert (first / "policy.json").read_bytes() == source
    assert (first / "policy.pt").read_bytes() == weights
    entries = history_entries(output)
    assert entries[:-1] == legacy_entries
    assert (output / entries[-1]["checkpoint"]).read_bytes() == source
    manifest = json.loads((output / "policy.json").read_bytes())
    assert manifest["history"]["kind"] == "linked-checkpoint-history-v1"
    assert manifest["history"]["count"] == 2


@pytest.mark.parametrize("fault", ["missing", "changed", "count", "kind", "escape"])
def test_linked_ancestry_must_be_intact_before_resume_publishes_output(tmp_path, fault):
    replay = warm_start(tmp_path)
    first = tmp_path / "first"
    run_experiment(SACTrain(tmp_path / "warm", replay, first, steps=1))
    manifest = json.loads((first / "policy.json").read_bytes())
    history = manifest["history"]
    node = first / history["head"]["path"]
    if fault == "missing":
        node.unlink()
    elif fault == "changed":
        node.write_text("{}")
    else:
        if fault == "count":
            history["count"] += 1
        elif fault == "kind":
            history["kind"] = "unsupported-history"
        else:
            outside = tmp_path / "outside.json"
            outside.write_bytes(node.read_bytes())
            history["head"]["path"] = "../outside.json"
        rebind_history(first, history)
    original = (first / "policy.json").read_bytes()
    output = tmp_path / "continued"
    with pytest.raises((ValueError, OSError)):
        run_experiment(SACResume(first, output, steps=2))
    assert not output.exists()
    assert (first / "policy.json").read_bytes() == original
