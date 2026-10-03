"""Growing required experience is streamed through real learner checkpoints."""

import hashlib
import json
import sqlite3
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest

from fh5.experiment import run_experiment
from fh5.learning.sac.critic import SACCriticResume, SACCriticWarmup
from tests.learning.sac.test_critic_pixel_residency import critic_corpus
from tests.learning.sac.test_critic_resume import warm_inputs


def test_sealed_pixels_can_cross_the_old_total_without_losing_updates(tmp_path):
    # The sample count is chosen only to cross the historical rejection gate.
    # All RGB files are real, contract-sized and checked by the CPU actor.
    frame_bytes = 512 * 288 * 3
    count = (512 * 1024**2) // (3 * frame_bytes) + 1
    model, replay, digest, source_bytes = critic_corpus(tmp_path, count=count)
    assert source_bytes > 512 * 1024**2
    first_root, resumed_root = tmp_path / "first", tmp_path / "resumed"
    first = run_experiment(
        SACCriticWarmup(model, replay, digest, first_root, steps=2),
        sac_stop_requested=lambda step: step == 1,
    ).summary["sac"]
    assert first["steps_completed"] == 1
    manifest = json.loads((first_root / "critic.json").read_bytes())
    assert manifest["experience"]["frame_bytes"] == source_bytes
    assert manifest["experience"]["frame_files"] == count * 3
    parent = (first_root / "critic.json").read_bytes()
    resumed = run_experiment(SACCriticResume(first_root, resumed_root)).summary["sac"]
    assert resumed["steps_completed"] == 1 and resumed["total_steps"] == 2
    assert resumed["phase_status"] == "complete"
    assert resumed["actor_change_max"] == resumed["reload_max_abs_error"] == 0
    assert (first_root / "critic.json").read_bytes() == parent
    source_digest = hashlib.sha256(bytes([51, 17, 34]) * (512 * 288)).hexdigest()
    for frame in (resumed_root / "experience").glob("corpus-*.rgb"):
        with frame.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == source_digest


@pytest.mark.parametrize("fault", ["index", "copy"])
def test_failed_experience_storage_cleans_index_and_preserves_parent(tmp_path, monkeypatch, fault):
    model, replay, digest = warm_inputs(tmp_path)
    parent = tmp_path / "parent"
    run_experiment(
        SACCriticWarmup(model, replay, digest, parent, steps=2),
        sac_stop_requested=lambda step: step == 1,
    )
    original = (parent / "critic.json").read_bytes()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    output = tmp_path / "failed"
    opened, connected = Path.open, sqlite3.connect
    observed = []

    def connect(database, *args, **kwargs):
        if fault == "index" and Path(database).parent.parent == temporary:
            observed.append(True)
            raise sqlite3.OperationalError("database or disk is full")
        return connected(database, *args, **kwargs)

    def open_file(path, mode="r", *args, **kwargs):
        if (
            fault == "copy"
            and path.is_relative_to(output)
            and path.suffix == ".rgb"
            and mode == "xb"
        ):
            observed.append(True)
            raise OSError("disk is full while copying experience")
        return opened(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "tempdir", str(temporary))
        patch.setattr(sqlite3, "connect", connect)
        patch.setattr(Path, "open", open_file)
        with pytest.raises(OSError, match="disk is full"):
            run_experiment(SACCriticResume(parent, output))
    assert observed == [True]
    assert not list(temporary.iterdir())
    assert not (output / "critic.json").exists()
    assert (parent / "critic.json").read_bytes() == original
    retry = run_experiment(SACCriticResume(parent, tmp_path / "retry")).summary["sac"]
    assert retry["steps_completed"] == 1 and retry["total_steps"] == 2


def test_repeated_frame_paths_are_deduplicated_but_conflicting_digests_reject(tmp_path):
    model, replay, digest = warm_inputs(tmp_path)
    first = tmp_path / "first"
    run_experiment(SACCriticWarmup(model, replay, digest, first, steps=1))
    manifest = json.loads((first / "critic.json").read_bytes())
    assert manifest["experience"]["frame_files"] == 1
    content = json.loads(replay.read_bytes())
    terminal = content["transitions"][-1]
    terminal["next"] = deepcopy(terminal["current"])
    terminal["next"]["frames"][0]["sha256"] = "0" * 64
    replay.write_text(json.dumps(content))
    digest = hashlib.sha256(replay.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="conflicting contents"):
        run_experiment(SACCriticWarmup(model, replay, digest, tmp_path / "conflict", steps=1))
    assert not (tmp_path / "conflict/critic.json").exists()
