"""Growing observation metadata must not become a resident learner corpus."""

import gc
import hashlib
import json
import sqlite3
import tempfile
import tracemalloc
from copy import deepcopy
from pathlib import Path

import pytest

from fh5.experiment import run_experiment
from fh5.learning.sac.critic import SACCriticWarmup
from fh5.learning.sac.training import SACResume, SACTrain
from tests.learning.sac.test_critic_resume import warm_inputs
from tests.support.checkpoint_files import update_records


def test_sac_loads_selected_observations_without_retaining_the_metadata_corpus(tmp_path):
    model, original, _ = warm_inputs(tmp_path)
    replay = json.loads(original.read_bytes())
    terminal = replay["transitions"][-1]
    assert terminal["terminated"]
    replay["transitions"] = []
    for i in range(128):
        row = deepcopy(terminal)
        row["id"] = f"terminal-{i}"
        replay["transitions"].append(row)
    retained, states = [], []
    padding = "observation-diagnostic;" * 4096
    for variant in ("small", "large"):
        if variant == "large":
            for row in replay["transitions"]:
                # Metadata is distinct per observation, but cannot change actor inputs.
                row["current"]["diagnostic"] = row["id"] + padding
        path = original.with_name(variant + ".json")
        path.write_text(json.dumps(replay), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        warm = tmp_path / (variant + "-warm")
        run_experiment(SACCriticWarmup(model, path, digest, warm, steps=1))
        gc.collect()
        tracemalloc.start()

        def stop(step):
            if step == 0:
                retained.append(tracemalloc.get_traced_memory()[0])
            return False

        try:
            result = run_experiment(
                SACTrain(warm, path, tmp_path / variant, steps=2), sac_stop_requested=stop
            ).summary["sac_learning"]
        finally:
            tracemalloc.stop()
        assert result["steps_completed"] == 2
        states.append(result["learner_state_sha256"])
    # Allow incidental Python bookkeeping, but not retention of most of the
    # added corpus. This is an observed-growth assertion, not a runtime cap.
    assert retained[1] - retained[0] < len(padding) * len(replay["transitions"]) // 2
    assert states[0] == states[1]


@pytest.mark.parametrize("failure", ["index", "pixels", "memory", "roles"])
def test_training_input_failure_saves_completed_updates_without_consuming_sampling_rng(
    tmp_path, monkeypatch, failure
):
    model, replay, digest = warm_inputs(tmp_path)
    warm = tmp_path / "warm"
    run_experiment(SACCriticWarmup(model, replay, digest, warm, steps=1))
    connected, opened = sqlite3.connect, Path.open
    armed, faults = False, []
    temporary = tmp_path / "temporary"
    temporary.mkdir()

    class UnavailableIndex(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if armed and sql.lstrip().upper().startswith("SELECT"):
                faults.append(True)
                if failure == "memory":
                    raise MemoryError("learning input allocation unavailable")
                raise sqlite3.OperationalError("learning input read unavailable")
            return super().execute(sql, *args, **kwargs)

    def connect(database, *args, **kwargs):
        prefix = "fh5-provenance-" if failure == "roles" else "fh5-learning-data-"
        if failure != "pixels" and Path(database).parent.name.startswith(prefix):
            kwargs["factory"] = UnavailableIndex
        return connected(database, *args, **kwargs)

    def open_file(path, mode="r", *args, **kwargs):
        if failure == "pixels" and armed and path.suffix == ".rgb" and mode == "rb":
            faults.append(True)
            raise OSError("learning input read unavailable")
        return opened(path, mode, *args, **kwargs)

    def stop(step):
        nonlocal armed
        armed = step == 3
        return False

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "tempdir", str(temporary))
        patch.setattr(sqlite3, "connect", connect)
        patch.setattr(Path, "open", open_file)
        trained = run_experiment(
            SACTrain(warm, replay, tmp_path / "stopped", steps=5), sac_stop_requested=stop
        ).summary["sac_learning"]
    assert faults
    assert not list(temporary.iterdir())
    assert trained["steps_completed"] == 3
    assert trained["stop_reason"] == "training_data_unavailable"
    assert "learning input" in trained["training_error"]
    assert trained["predictions"]["status"] == ("complete" if failure == "roles" else "unavailable")
    resumed = run_experiment(
        SACResume(tmp_path / "stopped", tmp_path / "resumed", steps=2)
    ).summary["sac_learning"]
    whole = run_experiment(SACTrain(warm, replay, tmp_path / "whole", steps=5)).summary[
        "sac_learning"
    ]
    assert resumed["total_steps"] == 5
    assert resumed["learner_state_sha256"] == whole["learner_state_sha256"]
    assert update_records(tmp_path / "stopped") + update_records(tmp_path / "resumed") == (
        update_records(tmp_path / "whole")
    )
    assert sum(trained["sampling"]["sampled"].values()) == 3 * 2
