"""Bounded cold experience assembly through the public continuation request."""

import json
import shutil
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

import pytest

from fh5.experiment import run_experiment
from fh5.learning.sac.training import SACPolicyReplay
from tests.evaluation.test_candidate_store import candidates as candidates
from tests.learning.sac.test_sac_frame_cache import FRAME_BYTES, varied_candidate
from tests.learning.sac.test_sac_frame_cache import saved_candidate as saved_candidate
from tests.support.checkpoint_files import prediction_records


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
    assert prediction_records(tmp_path, restored) == prediction_records(moved, report)
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


def test_small_experience_copy_does_not_allocate_the_global_frame_read_limit(
    tmp_path, saved_candidate
):
    mkdir, opened = Path.mkdir, Path.open
    measured = []

    def create_directory(path, *args, **kwargs):
        result = mkdir(path, *args, **kwargs)
        if path.name == "expanded":
            tracemalloc.start()
        return result

    def open_file(path, *args, **kwargs):
        if path.name == "replay.json" and path.parent.name == "expanded":
            if args and args[0] in ("x", "xb"):
                measured.append(tracemalloc.get_traced_memory()[1])
                tracemalloc.stop()
        return opened(path, *args, **kwargs)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, "mkdir", create_directory)
            patch.setattr(Path, "open", open_file)
            varied_candidate(tmp_path, saved_candidate)
    finally:
        tracemalloc.stop()
    (tmp_path / "copy-memory-observation.json").write_text(json.dumps({"peak_bytes": measured}))
    # Includes Python path/I/O/metadata overhead, independently of report counters.
    assert len(measured) == 1 and measured[0] < 64 * 1024


def test_frozen_replay_pixel_reads_do_not_allocate_the_cache_capacity(tmp_path, saved_candidate):
    checkpoint, replay = saved_candidate
    opened = Path.open
    peaks = []

    class MeasuredReader:
        def __init__(self, stream):
            self.stream = stream

        def read(self, size=-1):
            tracemalloc.start()
            try:
                return self.stream.read(size)
            finally:
                peaks.append(tracemalloc.get_traced_memory()[1])
                tracemalloc.stop()

    @contextmanager
    def measure(stream):
        with stream:
            yield MeasuredReader(stream)

    def open_file(path, *args, **kwargs):
        stream = opened(path, *args, **kwargs)
        if path.suffix == ".rgb" and args and args[0] == "rb":
            return measure(stream)
        return stream

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", open_file)
        result = run_experiment(
            SACPolicyReplay(checkpoint, replay, tmp_path / "replay.html")
        ).summary["sac_policy"]
    (tmp_path / "read-memory-observation.json").write_text(json.dumps({"peak_bytes": peaks}))
    assert prediction_records(tmp_path, result) and result["commands_sent"] is False
    assert peaks and max(peaks) < 64 * 1024


@pytest.mark.parametrize("change", ["grow", "shrink"])
def test_pixel_file_change_between_stat_and_read_is_rejected(tmp_path, saved_candidate, change):
    checkpoint, _ = saved_candidate
    copied = tmp_path / "source"
    shutil.copytree(checkpoint, copied)
    frame = next((copied / "experience/frames").glob("*.rgb"))
    original = frame.read_bytes()
    opened = Path.open
    changed = []

    def open_file(path, *args, **kwargs):
        if path == frame and args and args[0] == "rb":
            with opened(frame, "wb") as stream:
                stream.write(original + b"x" if change == "grow" else original[:-1])
            changed.append(True)
        return opened(path, *args, **kwargs)

    report = tmp_path / "rejected.html"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", open_file)
        with pytest.raises(ValueError, match="changed size while reading"):
            run_experiment(SACPolicyReplay(copied, copied / "experience/replay.json", report))
    assert changed == [True]
    assert not report.exists() and not report.with_suffix(".json").exists()
