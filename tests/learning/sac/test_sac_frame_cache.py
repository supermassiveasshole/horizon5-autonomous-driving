"""Bounded raw frame retention through frozen and learning experiment requests."""

import hashlib
import io
import json
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from pathlib import Path

import pytest

from fh5.experiment import run_experiment
from fh5.learning.sac.training import SACPolicyReplay, SACResume
from tests.evaluation.test_candidate_store import candidates as candidates
from tests.learning.sac.test_sac import experience
from tests.support.checkpoint_files import prediction_records, update_records

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
    assert prediction_records(tmp_path, bounded) == prediction_records(tmp_path, baseline)
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
    assert prediction_records(tmp_path, bounded) == prediction_records(tmp_path, baseline)
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
    assert prediction_records(tmp_path, replayed) == prediction_records(continued, bounded)
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
        mode = args[0] if args else kwargs.get("mode", "r")
        return writing_copy(stream) if path == copied_replay and mode in ("xb", "wb") else stream

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


@pytest.mark.parametrize("budget", [-1, True, 0.5])
def test_invalid_cache_budget_does_not_publish_a_policy_report(tmp_path, saved_candidate, budget):
    checkpoint, replay = saved_candidate
    report = tmp_path / "invalid.html"
    with pytest.raises(ValueError, match="cache.*budget"):
        run_experiment(SACPolicyReplay(checkpoint, replay, report, raw_cache_bytes=budget))
    assert not report.exists()


@pytest.mark.parametrize("budget", [0, FRAME_BYTES - 1, 512 * 1024**2 + 1])
def test_cache_capacity_controls_retention_instead_of_admitting_frames(
    tmp_path, saved_candidate, budget
):
    checkpoint, replay = saved_candidate
    baseline = run_experiment(
        SACPolicyReplay(checkpoint, replay, tmp_path / "baseline.html", raw_cache_bytes=FRAME_BYTES)
    ).summary["sac_policy"]
    uncached = run_experiment(
        SACPolicyReplay(checkpoint, replay, tmp_path / "uncached.html", raw_cache_bytes=budget)
    ).summary["sac_policy"]
    assert prediction_records(tmp_path, uncached) == prediction_records(tmp_path, baseline)
    cache = uncached["raw_frame_cache"]
    assert cache["budget_bytes"] == budget
    assert cache["peak_bytes"] <= budget
    if budget < FRAME_BYTES:
        assert cache["peak_bytes"] == cache["retained_bytes"] == cache["retained_frames"] == 0
        assert cache["bypassed_frames"] > 0
    else:
        # This is permission to retain bytes, not a preallocation or required size.
        assert cache["peak_bytes"] == FRAME_BYTES
    assert cache["files_deleted"] == 0


def test_new_training_does_not_invent_a_cache_budget_and_keeps_exact_updates(tmp_path):
    from fh5.learning.sac.training import SACTrain
    from tests.learning.sac.test_sac_learning import warm_start

    replay = warm_start(tmp_path)
    output = tmp_path / "uncached"
    first = run_experiment(SACTrain(tmp_path / "warm", replay, output, steps=3)).summary[
        "sac_learning"
    ]
    cache = first["raw_frame_cache"]
    assert cache["budget_bytes"] == cache["retained_bytes"] == cache["peak_bytes"] == 0
    assert cache["bypassed_frames"] > 0
    second = run_experiment(SACResume(output, tmp_path / "continued", steps=2)).summary[
        "sac_learning"
    ]
    assert second["raw_frame_cache"]["budget_bytes"] == 0
    whole = run_experiment(
        SACTrain(
            tmp_path / "warm", replay, tmp_path / "cached", steps=5, raw_cache_bytes=FRAME_BYTES
        )
    ).summary["sac_learning"]
    assert second["learner_state_sha256"] == whole["learner_state_sha256"]
    assert second["predictions"] == whole["predictions"]
    assert update_records(output) + update_records(tmp_path / "continued") == update_records(
        tmp_path / "cached"
    )
    reloaded = run_experiment(
        SACPolicyReplay(tmp_path / "continued", replay, tmp_path / "reloaded.html")
    ).summary["sac_policy"]
    assert reloaded["raw_frame_cache"]["budget_bytes"] == 0
    assert reloaded["predictions"]["sha256"] == second["predictions"]["sha256"]


def test_legacy_cache_default_is_preserved_and_can_be_explicitly_disabled(tmp_path):
    from fh5.learning.sac.training import SACTrain
    from tests.learning.sac.test_sac_learning import warm_start

    torch = pytest.importorskip("torch")
    replay = warm_start(tmp_path)
    checkpoint = tmp_path / "legacy"
    run_experiment(
        SACTrain(tmp_path / "warm", replay, checkpoint, steps=1, raw_cache_bytes=512 * 1024**2)
    )
    # Older public checkpoints omitted the field and inherited 512 MiB.
    saved = torch.load(checkpoint / "policy.pt", map_location="cpu", weights_only=True)
    saved["metadata"]["configuration"].pop("raw_cache_bytes")
    torch.save(saved, checkpoint / "policy.pt")
    manifest = {
        **saved["metadata"],
        "weights_sha256": hashlib.sha256((checkpoint / "policy.pt").read_bytes()).hexdigest(),
    }
    (checkpoint / "policy.json").write_text(json.dumps(manifest))
    inherited = run_experiment(SACResume(checkpoint, tmp_path / "inherited", steps=2)).summary[
        "sac_learning"
    ]
    disabled = run_experiment(
        SACResume(checkpoint, tmp_path / "disabled", steps=2, raw_cache_bytes=0)
    ).summary["sac_learning"]
    assert inherited["raw_frame_cache"]["budget_bytes"] == 512 * 1024**2
    assert disabled["raw_frame_cache"]["retained_bytes"] == 0
    assert disabled["learner_state_sha256"] == inherited["learner_state_sha256"]
    assert disabled["predictions"] == inherited["predictions"]


@pytest.mark.parametrize("budget", [0, FRAME_BYTES - 1])
def test_training_and_resume_bypass_retention_without_changing_real_updates(
    tmp_path, saved_candidate, budget
):
    checkpoint, _ = varied_candidate(tmp_path, saved_candidate)
    first = run_experiment(
        SACResume(checkpoint, tmp_path / "first", steps=1, raw_cache_bytes=budget)
    ).summary["sac_learning"]
    second = run_experiment(SACResume(tmp_path / "first", tmp_path / "second", steps=1)).summary[
        "sac_learning"
    ]
    whole = run_experiment(
        SACResume(checkpoint, tmp_path / "whole", steps=2, raw_cache_bytes=FRAME_BYTES)
    ).summary["sac_learning"]
    assert first["raw_frame_cache"]["bypassed_frames"] > 0
    assert second["raw_frame_cache"]["budget_bytes"] == budget
    assert second["learner_state_sha256"] == whole["learner_state_sha256"]
    assert second["predictions"] == whole["predictions"]
    assert update_records(tmp_path / "first") + update_records(tmp_path / "second") == (
        update_records(tmp_path / "whole")
    )


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
    assert update_records(output) == update_records(tmp_path / "baseline")
    assert bounded["predictions"] == baseline["predictions"]
    assert bounded["raw_frame_cache"]["peak_bytes"] <= FRAME_BYTES
    assert bounded["raw_frame_cache"]["evictions"] > 0
    restored = run_experiment(
        SACPolicyReplay(output, output / "experience/replay.json", tmp_path / "restored.html")
    ).summary["sac_policy"]
    assert prediction_records(tmp_path, restored) == prediction_records(output, bounded)


@pytest.mark.parametrize("failure", [OSError, MemoryError])
def test_post_update_prediction_resource_failure_retains_completed_learning(
    tmp_path, saved_candidate, monkeypatch, failure
):
    checkpoint, _ = varied_candidate(tmp_path, saved_candidate)
    output = tmp_path / "candidate"
    original_open = Path.open
    post_update = []

    def unavailable_cold_input(path, mode="r", *args, **kwargs):
        if path == output / "diagnostics/predictions.jsonl" and mode == "xb":
            post_update.append(True)
        if post_update and path.suffix == ".rgb" and mode == "rb":
            raise failure("cold input unavailable during post-update prediction")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", unavailable_cold_input)
        trained = run_experiment(
            SACResume(checkpoint, output, steps=1, raw_cache_bytes=FRAME_BYTES)
        ).summary["sac_learning"]
    assert trained["steps_completed"] == 1
    assert trained["predictions"]["status"] == "unavailable"
    assert trained["predictions"]["sha256"] is None
    assert "cold input unavailable" in trained["predictions"]["error"]
    continued = run_experiment(SACResume(output, tmp_path / "continued", steps=2)).summary[
        "sac_learning"
    ]
    whole = run_experiment(
        SACResume(checkpoint, tmp_path / "whole", steps=3, raw_cache_bytes=FRAME_BYTES)
    ).summary["sac_learning"]
    assert continued["learner_state_sha256"] == whole["learner_state_sha256"]
