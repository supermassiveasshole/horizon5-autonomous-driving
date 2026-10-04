"""Actual numerical batches are not admitted by an invented byte ceiling."""

import hashlib
import json
from copy import deepcopy

from fh5.experiment import run_experiment
from fh5.learning.sac.critic import SACCriticWarmup
from fh5.learning.sac.training import SACResume, SACTrain
from tests.learning.sac.test_critic_pixel_residency import critic_corpus


def test_explicit_batch_can_cross_the_old_float_image_ceiling_and_remain_recoverable(tmp_path):
    model, path, _, _ = critic_corpus(tmp_path, count=1)
    document = json.loads(path.read_bytes())
    template = document["transitions"][0]
    frames = template["current"]["frames"]
    width, height = frames[0]["size"]
    bytes_per_row = len(frames) * width * height * 3 * 4
    count = (256 * 1024**2) // bytes_per_row + 1  # Cross the old gate, not a runtime budget.
    rows = []
    for i in range(count):
        row = deepcopy(template)
        row["id"] = f"batch-row-{i}"
        rows.append(row)
    document["transitions"] = rows
    path.write_text(json.dumps(document))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    warm = tmp_path / "warm"
    run_experiment(SACCriticWarmup(model, path, digest, warm, steps=1))
    output = tmp_path / "learned"
    trained = run_experiment(SACTrain(warm, path, output, steps=1, batch_size=count)).summary[
        "sac_learning"
    ]
    assert trained["steps_completed"] == 1
    assert trained["stop_reason"] == "budget_completed"
    assert sum(trained["sampling"]["sampled"].values()) == count
    restored = run_experiment(SACResume(output, tmp_path / "restored", steps=0)).summary[
        "sac_learning"
    ]
    assert restored["learner_state_sha256"] == trained["learner_state_sha256"]
    assert restored["predictions"]["sha256"] == trained["predictions"]["sha256"]
