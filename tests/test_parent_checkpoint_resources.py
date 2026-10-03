"""Growing sealed learner evidence through parent acknowledgement and continuation."""

import hashlib
import json
import tracemalloc
from pathlib import Path

import pytest
from test_candidate_store import candidates as candidates
from test_evaluation import sha
from test_learning_loop import SharedBackend, loop_request
from test_learning_loop import seeded_loop as seeded_loop
from test_learning_recovery import interrupt_selection
from test_learning_update_recovery import interrupted_update

from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue
from fh5.sac_learning import SACResume


def extend_sealed_report(checkpoint, *, policy_padding_mb=0):
    report = checkpoint / "training-report.json"
    # Keep every learned value intact while crossing the old 128 MiB gate.
    # This tests representation compatibility and verifies that the parent does
    # not keep the entire serialized report in memory.
    with report.open("ab") as stream:
        block = b" " * 1024**2
        for _ in range(129):
            stream.write(block)
    rebind_report(checkpoint)
    with (checkpoint / "policy.json").open("ab") as stream:
        for _ in range(policy_padding_mb):
            stream.write(block)
    return report


def rebind_report(checkpoint):
    torch = pytest.importorskip("torch")
    report = checkpoint / "training-report.json"
    with report.open("rb") as stream:
        report_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest_file = checkpoint / "policy.json"
    manifest = json.loads(manifest_file.read_bytes())
    saved = torch.load(checkpoint / "policy.pt", map_location="cpu", weights_only=True)
    manifest["training_report_sha256"] = report_sha
    saved["metadata"]["training_report_sha256"] = report_sha
    torch.save(saved, checkpoint / "policy.pt")
    with (checkpoint / "policy.pt").open("rb") as stream:
        manifest["weights_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")


def test_parent_adopts_large_pending_report_then_finishes_the_original_updates(
    tmp_path, seeded_loop
):
    root, pending = interrupted_update(tmp_path, seeded_loop)
    checkpoint = root / "round-000/updates-000"
    report = extend_sealed_report(checkpoint, policy_padding_mb=5)
    identity = sha(checkpoint / "policy.json")
    backend = SharedBackend(seeded_loop[0])
    tracemalloc.start()
    try:
        recovered = run_experiment(
            LearningContinue(root, sha(root / "state.json")), learning_environment=backend
        ).summary["learning_loop"]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert recovered["stop_reason"] == "stop_requested", recovered.get("error")
    assert recovered["latest_learner"]["sha256"] == identity
    assert recovered["learner_updates"] == pending["learner_updates"] == 0
    assert recovered["eligible_transitions"] == 3
    assert not backend.leases and backend.closed
    assert peak < report.stat().st_size

    expected = run_experiment(
        SACResume(root / "round-000/learning/candidate-000", tmp_path / "reference", steps=3)
    ).summary["sac_learning"]
    (root / "stop.request").unlink()
    continued_backend = SharedBackend(seeded_loop[0])
    completed = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=continued_backend
    ).summary["learning_loop"]
    assert completed["stop_reason"] == "budget_completed", completed.get("error")
    assert completed["rounds_completed"] == 1
    assert completed["learner_updates"] == completed["eligible_transitions"] == 3
    assert completed["latest_learner"]["total_steps"] == 33
    assert completed["latest_learner"]["learner_state_sha256"] == expected["learner_state_sha256"]
    assert len(continued_backend.leases) == 1  # Only the frozen evaluation lease.
    assert sha(checkpoint / "policy.json") == identity
    learned = json.loads(
        (Path(completed["latest_learner"]["directory"]) / "training-report.json").read_bytes()
    )
    assert learned["predictions"] == expected["predictions"]


def test_parent_adopts_large_sampling_report_then_uses_its_original_credit(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    root = request.output_dir
    interrupt_selection(request, seeded_loop[0], "before_learned", scenario="stopped_updates")
    learning = root / "round-000/learning"
    checkpoint = learning / "candidate-000"
    report = extend_sealed_report(checkpoint)
    identity = sha(checkpoint / "policy.json")
    # The child result is not acknowledged by the parent yet. Rebind its actual
    # learner to the equivalent enlarged report, leaving all progress unchanged.
    result_file = learning / "attempt-000/cycle-result.json"
    result = json.loads(result_file.read_bytes())
    result["candidate_sha256"] = identity
    result_file.write_text(json.dumps(result), encoding="utf-8")
    summary_file = learning / "summary.json"
    summary = json.loads(summary_file.read_bytes())
    summary["attempts"][0] = result
    summary_file.write_text(json.dumps(summary), encoding="utf-8")
    backend = SharedBackend(seeded_loop[0])
    tracemalloc.start()
    try:
        recovered = run_experiment(
            LearningContinue(root, sha(root / "state.json")), learning_environment=backend
        ).summary["learning_loop"]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert recovered["stop_reason"] == "stop_requested", recovered.get("error")
    assert recovered["latest_learner"]["sha256"] == identity
    assert recovered["learner_updates"] == 0 and recovered["eligible_transitions"] == 3
    assert recovered["recoveries"][-1]["kind"] == "sealed_sampling"
    assert not backend.leases and backend.closed
    assert peak < report.stat().st_size

    expected = run_experiment(SACResume(checkpoint, tmp_path / "reference", steps=3)).summary[
        "sac_learning"
    ]
    (root / "stop.request").unlink()
    completed_backend = SharedBackend(seeded_loop[0])
    completed = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=completed_backend
    ).summary["learning_loop"]
    assert completed["stop_reason"] == "budget_completed", completed.get("error")
    assert completed["learner_updates"] == completed["eligible_transitions"] == 3
    assert completed["rounds_completed"] == 1
    assert completed["latest_learner"]["learner_state_sha256"] == expected["learner_state_sha256"]
    assert len(completed_backend.leases) == 1
    assert sha(checkpoint / "policy.json") == identity


@pytest.mark.parametrize("damage", ["bad_json", "changed_report"])
def test_invalid_pending_report_cannot_advance_parent_or_acquire_an_environment(
    tmp_path, seeded_loop, damage
):
    root, _ = interrupted_update(tmp_path, seeded_loop)
    checkpoint = root / "round-000/updates-000"
    state = root / "state.json"
    before = state.read_bytes()
    report = checkpoint / "training-report.json"
    if damage == "bad_json":
        report.write_bytes(report.read_bytes().rstrip()[:-1] + b',"diagnostics":[1,]}\n')
        # Even hash-consistent unused diagnostics must be valid JSON.
        rebind_report(checkpoint)
    else:
        report.write_bytes(report.read_bytes() + b" ")
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises(ValueError):
        run_experiment(LearningContinue(root, sha(state)), learning_environment=backend)
    assert state.read_bytes() == before
    assert not (root / "round-000/update-history/000000.json").exists()
    assert not (root / "round-000/updates-001").exists()
    assert not backend.leases and backend.closed
