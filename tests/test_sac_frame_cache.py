"""Bounded raw frame retention through frozen and learning experiment requests."""

import hashlib
import io
import json
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from pathlib import Path

import pytest
from test_candidate_store import candidates as candidates
from test_sac import experience

from fh5.experiment import run_experiment
from fh5.sac_learning import SACPolicyReplay, SACResume

FRAME_BYTES = 64 * 36 * 3


@pytest.fixture(scope="module")
def saved_candidate(candidates):
    checkpoint = candidates[0] / "candidate-model"
    return checkpoint, checkpoint / "experience/replay.json"


def test_frozen_replay_shares_raw_frames_within_an_explicit_byte_budget(tmp_path, saved_candidate):
    checkpoint, replay = saved_candidate
    baseline = run_experiment(
        SACPolicyReplay(checkpoint, replay, tmp_path / "baseline.html")
    ).summary["sac_policy"]
    bounded = run_experiment(
        SACPolicyReplay(checkpoint, replay, tmp_path / "bounded.html", raw_cache_bytes=FRAME_BYTES)
    ).summary["sac_policy"]
    assert bounded["predictions"] == baseline["predictions"]
    cache = bounded["raw_frame_cache"]
    assert cache["budget_bytes"] == cache["peak_bytes"] == FRAME_BYTES
    assert cache["retained_bytes"] <= FRAME_BYTES
    assert cache["hits"] > 0 and cache["unique_frames"] == 1
    assert cache["storage_dtype"] == "uint8"
    assert cache["files_deleted"] == 0 and bounded["commands_sent"] is False


def varied_candidate(root, saved_candidate):
    parent, _ = saved_candidate
    incoming = root / "incoming"
    incoming.mkdir()
    request = replace(
        experience(incoming, terminal=False, timeline=[(0, 0), (1.3, 100), (2.1, 300)]),
        task_file=parent.parent / "task.json",
        reward_file=parent.parent / "reward.json",
    )
    trace = json.loads(request.trace_file.read_bytes())
    for index, observation in enumerate(trace["observations"]):
        for slot, frame in enumerate(observation["frames"]):
            colour = (index + slot) % 4
            raw = bytes([30 + colour * 25, 17, 34] * 64 * 36)
            path = incoming / f"colour-{colour}.rgb"
            path.write_bytes(raw)
            frame.update(path=path.name, sha256=hashlib.sha256(raw).hexdigest())
    request.trace_file.write_text(json.dumps(trace))
    prepared = run_experiment(request).summary["sac_replay"]
    checkpoint = root / "varied-candidate"
    result = run_experiment(
        SACResume(
            parent,
            checkpoint,
            steps=0,
            additions=((request.output_dir / "replay.json", prepared["replay_sha256"]),),
        )
    ).summary["sac_learning"]
    assert result["steps_completed"] == 0
    return checkpoint, checkpoint / "experience/replay.json"


def test_evicted_frames_reload_without_changing_frozen_predictions_or_deleting_sources(
    tmp_path, saved_candidate
):
    checkpoint, replay = varied_candidate(tmp_path, saved_candidate)
    originals = {path: path.read_bytes() for path in checkpoint.rglob("*.rgb")}
    baseline = run_experiment(
        SACPolicyReplay(checkpoint, replay, tmp_path / "baseline.html")
    ).summary["sac_policy"]
    bounded = run_experiment(
        SACPolicyReplay(checkpoint, replay, tmp_path / "bounded.html", raw_cache_bytes=FRAME_BYTES)
    ).summary["sac_policy"]
    assert bounded["predictions"] == baseline["predictions"]
    cache = bounded["raw_frame_cache"]
    assert cache["unique_frames"] == 5 and cache["evictions"] > 0
    assert cache["peak_bytes"] == cache["retained_bytes"] == FRAME_BYTES
    assert cache["misses"] > cache["unique_frames"]
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    assert cache["files_deleted"] == 0


def test_resume_keeps_complete_learning_state_when_only_the_frame_budget_changes(
    tmp_path, saved_candidate
):
    checkpoint, _ = varied_candidate(tmp_path, saved_candidate)
    baseline = run_experiment(SACResume(checkpoint, tmp_path / "baseline", steps=0)).summary[
        "sac_learning"
    ]
    continued = tmp_path / "bounded"
    bounded = run_experiment(
        SACResume(checkpoint, continued, steps=0, raw_cache_bytes=FRAME_BYTES)
    ).summary["sac_learning"]
    assert bounded["learner_state_sha256"] == baseline["learner_state_sha256"]
    assert bounded["predictions"] == baseline["predictions"]
    cache = bounded["raw_frame_cache"]
    assert cache["peak_bytes"] <= FRAME_BYTES and cache["evictions"] > 0
    replayed = run_experiment(
        SACPolicyReplay(continued, continued / "experience/replay.json", tmp_path / "restored.html")
    ).summary["sac_policy"]
    assert replayed["predictions"] == bounded["predictions"]
    assert replayed["raw_frame_cache"]["budget_bytes"] == FRAME_BYTES


def test_cli_can_select_the_raw_frame_budget_for_frozen_replay(tmp_path, saved_candidate):
    from fh5.cli import main

    checkpoint, replay = saved_candidate
    output = io.StringIO()
    with redirect_stdout(output):
        code = main(
            [
                "sac-policy-replay",
                "--checkpoint",
                str(checkpoint),
                "--replay",
                str(replay),
                "--report",
                str(tmp_path / "cli.html"),
                "--raw-cache-bytes",
                str(FRAME_BYTES),
            ]
        )
    assert code == 0
    result = json.loads(output.getvalue())
    assert result["raw_frame_cache"]["budget_bytes"] == FRAME_BYTES
    assert result["raw_frame_cache"]["peak_bytes"] <= FRAME_BYTES


def test_cold_frame_changed_after_initial_predictions_is_rejected_on_reload(
    tmp_path, saved_candidate
):
    checkpoint, _ = varied_candidate(tmp_path, saved_candidate)
    digest = hashlib.sha256(bytes([30, 17, 34] * 64 * 36)).hexdigest()
    original = checkpoint / "experience/frames" / (digest + ".rgb")
    original_bytes = original.read_bytes()
    continued = tmp_path / "continued"
    copied_replay = continued / "experience/replay.json"
    opened = Path.open
    changed = []

    @contextmanager
    def writing_copy(stream):
        with stream:
            yield stream
        # After the learner has inspected its inputs and archived the source,
        # change an external cold frame that the later prediction pass must reload.
        with opened(original, "wb") as target:
            target.write(b"x" * len(original_bytes))
        changed.append(True)

    def open_file(path, *args, **kwargs):
        stream = opened(path, *args, **kwargs)
        return writing_copy(stream) if path == copied_replay else stream

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, "open", open_file)
            with pytest.raises(ValueError, match="pixel hash mismatch"):
                run_experiment(
                    SACResume(checkpoint, continued, steps=0, raw_cache_bytes=FRAME_BYTES)
                )
        assert changed == [True]
        assert not (continued / "policy.json").exists()
    finally:
        original.write_bytes(original_bytes)


@pytest.mark.parametrize("budget", [0, True, FRAME_BYTES - 1])
def test_invalid_or_too_small_cache_does_not_publish_a_policy_report(
    tmp_path, saved_candidate, budget
):
    checkpoint, replay = saved_candidate
    report = tmp_path / "invalid.html"
    with pytest.raises(ValueError, match="cache.*budget"):
        run_experiment(SACPolicyReplay(checkpoint, replay, report, raw_cache_bytes=budget))
    assert not report.exists()


def test_actual_updates_and_complete_resume_are_independent_of_frame_eviction(
    tmp_path, saved_candidate
):
    checkpoint, _ = varied_candidate(tmp_path, saved_candidate)
    baseline = run_experiment(SACResume(checkpoint, tmp_path / "baseline", steps=2)).summary[
        "sac_learning"
    ]
    output = tmp_path / "bounded"
    bounded = run_experiment(
        SACResume(checkpoint, output, steps=2, raw_cache_bytes=FRAME_BYTES)
    ).summary["sac_learning"]
    assert bounded["steps_completed"] == 2 and bounded["actor_updates"] == 1
    assert bounded["encoder_change_max"] > 0 and bounded["actor_change_max"] > 0
    assert bounded["learner_state_sha256"] == baseline["learner_state_sha256"]
    assert bounded["updates"] == baseline["updates"]
    assert bounded["predictions"] == baseline["predictions"]
    assert bounded["raw_frame_cache"]["peak_bytes"] <= FRAME_BYTES
    assert bounded["raw_frame_cache"]["evictions"] > 0
    restored = run_experiment(
        SACPolicyReplay(output, output / "experience/replay.json", tmp_path / "restored.html")
    ).summary["sac_policy"]
    assert restored["predictions"] == bounded["predictions"]
