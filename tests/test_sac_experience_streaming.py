"""Bounded cold experience assembly through the public continuation request."""

import json
from pathlib import Path

import pytest
from test_candidate_store import candidates as candidates
from test_sac_frame_cache import FRAME_BYTES, varied_candidate
from test_sac_frame_cache import saved_candidate as saved_candidate

from fh5.experiment import run_experiment
from fh5.sac_learning import SACPolicyReplay


def test_expansion_copies_one_frame_at_a_time_and_replays_after_moving(tmp_path, saved_candidate):
    checkpoint, _ = varied_candidate(tmp_path, saved_candidate)
    report = json.loads((checkpoint / "training-report.json").read_bytes())
    copied = report["experience_expansion"]
    assert copied["frame_files"] == 5
    assert copied["frame_bytes"] == 5 * FRAME_BYTES
    assert copied["copy_peak_frame_bytes"] == FRAME_BYTES
    assert copied["files_deleted"] == 0
    assert copied["mode"] == "verified-frame-stream"
    assert report["steps_completed"] == 0
    (tmp_path / "incoming").rename(tmp_path / "unused-source")
    moved = tmp_path / "moved"
    checkpoint.rename(moved)
    restored = run_experiment(
        SACPolicyReplay(moved, moved / "experience/replay.json", tmp_path / "restored.html")
    ).summary["sac_policy"]
    assert restored["predictions"] == report["predictions"]
    assert restored["commands_sent"] is False


def test_changed_cold_source_before_copy_cannot_publish_a_candidate(tmp_path, saved_candidate):
    mkdir = Path.mkdir
    changed = []

    def create_directory(path, *args, **kwargs):
        result = mkdir(path, *args, **kwargs)
        if path.name == "expanded":
            frame = next((tmp_path / "incoming/prepared/frames").glob("*.rgb"))
            frame.write_bytes(b"x" * frame.stat().st_size)
            changed.append(frame)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "mkdir", create_directory)
        with pytest.raises(ValueError, match="pixel hash mismatch"):
            varied_candidate(tmp_path, saved_candidate)
    assert len(changed) == 1
    assert not (tmp_path / "varied-candidate").exists()


def test_failed_copy_releases_temporary_files_and_preserves_parent(tmp_path, saved_candidate):
    parent, _ = saved_candidate
    originals = {path: path.read_bytes() for path in parent.rglob("*.rgb")}
    parent_manifest = (parent / "policy.json").read_bytes()
    opened = Path.open
    copies = []

    def open_file(path, *args, **kwargs):
        if path.parent.name == "frames" and path.parent.parent.name == "expanded":
            if args and args[0] == "xb":
                copies.append(path)
                if len(copies) == 2:
                    raise OSError("injected cold-copy write failure")
        return opened(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", open_file)
        with pytest.raises(OSError, match="cold-copy write failure"):
            varied_candidate(tmp_path, saved_candidate)
    assert len(copies) == 2
    assert not copies[0].parent.parent.exists()
    assert not (tmp_path / "varied-candidate").exists()
    assert (parent / "policy.json").read_bytes() == parent_manifest
    assert all(path.read_bytes() == raw for path, raw in originals.items())
