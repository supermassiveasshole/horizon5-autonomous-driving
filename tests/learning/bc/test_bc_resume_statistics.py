"""Optional aggregate statistics cannot own the learner's ability to resume."""

import json
from pathlib import Path

import pytest

from fh5.experiment import run_experiment
from fh5.learning.bc.schedule import ScheduledBCResume, ScheduledBCTrain
from tests.learning.bc.test_bc_resume_resources import (
    PressureAfterTwoUpdates,
    resume_inputs,
    rewrite_learner_artifact,
)
from tests.learning.bc.test_learning_schedule import Resources


@pytest.mark.parametrize("statistics", [None, {"duration_s": "lost", "time_gradient_l1": []}])
def test_optional_statistics_loss_preserves_exact_bc_continuation(tmp_path, statistics):
    torch = pytest.importorskip("torch")
    schedule, _ = resume_inputs(tmp_path)
    reference = run_experiment(
        ScheduledBCTrain(schedule, tmp_path / "reference"), learning_resources=Resources()
    ).summary["learning_schedule"]
    parent = tmp_path / "parent"
    stopped = run_experiment(
        ScheduledBCTrain(schedule, parent), learning_resources=PressureAfterTwoUpdates()
    ).summary["learning_schedule"]
    root = Path(stopped["learner_checkpoint"]["directory"])
    saved = torch.load(root / "learner.pt", weights_only=True)
    if statistics is None:
        saved["metadata"].pop("statistics")
    else:
        saved["metadata"]["statistics"] = statistics
    expected = rewrite_learner_artifact(torch, parent, stopped, saved)
    continued = run_experiment(
        ScheduledBCResume(parent, tmp_path / "continued", expected), learning_resources=Resources()
    ).summary["learning_schedule"]
    assert continued["steps_this_run"] == 4
    assert (
        continued["learner_checkpoint"]["learner_state_sha256"]
        == reference["learner_checkpoint"]["learner_state_sha256"]
    )
    manifest = json.loads((tmp_path / "continued/learner/learner.json").read_bytes())
    assert manifest["statistics"]["status"] == "partial"
    model = json.loads((tmp_path / "continued/candidate/model.json").read_bytes())
    assert model["training"]["statistics_status"] == "partial"
