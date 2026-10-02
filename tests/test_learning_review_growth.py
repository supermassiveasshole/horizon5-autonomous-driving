"""Rebuild derived evaluation reviews without a publication-count ceiling."""

import json

import pytest
from test_candidate_store import candidates as candidates
from test_evaluation import sha
from test_learning_loop import SharedBackend, loop_request
from test_learning_loop import seeded_loop as seeded_loop
from test_learning_recovery import interrupt_selection

from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue


@pytest.mark.parametrize("legacy", [False, True])
def test_retained_review_directories_cannot_exhaust_continuation(tmp_path, seeded_loop, legacy):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_parent_review_ack")
    root = request.output_dir
    state = root / "state.json"
    prior = json.loads(state.read_bytes())
    if legacy:
        prior["rounds"][0].pop("review_publication", None)
        prior["rounds"][0]["review_directories"] = [str(root / "round-000/reviewed")]
        state.write_text(json.dumps(prior))
    # Retained partial publications, not eleven new driving evaluations.
    for index in range(1, 11):
        directory = root / f"round-000/reviewed-{index:03d}"
        directory.mkdir()
        (directory / "partial.txt").write_text(f"retained diagnostic {index}")
    originals = {
        path: sha(path)
        for directory in (root / "round-000").glob("reviewed*")
        for path in directory.rglob("*")
        if path.is_file()
    }
    backend = SharedBackend(seeded_loop[0])
    result = run_experiment(
        LearningContinue(root, sha(state)), learning_environment=backend
    ).summary["learning_loop"]
    assert result["stop_reason"] == "budget_completed", result.get("error")
    assert result["learner_updates"] == prior["learner_updates"] == 3
    assert result["latest_learner"] == prior["latest_learner"]
    assert not backend.leases and backend.closed and result["resources_released"]
    row = result["rounds"][0]
    assert row["review_publication"] == {
        "directory": str(root / "round-000/reviewed-011"),
        "sequence": 11,
    }
    assert (root / "round-000/reviewed-011/batch-report.json").is_file()
    assert row.get("review_directories", []) == prior["rounds"][0].get("review_directories", [])
    assert all(sha(path) == digest for path, digest in originals.items())
