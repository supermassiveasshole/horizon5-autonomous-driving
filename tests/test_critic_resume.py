"""Finite Q preheating continuation through the experiment-run interface."""

import hashlib
import json

import pytest
from test_sac import experience
from test_temporal_bc import temporal_fixture

from fh5.experiment import run_experiment
from fh5.sac import SACCriticWarmup
from fh5.temporal_bc import TemporalBCTrain


def warm_inputs(root):
    request = experience(root)
    prepared = run_experiment(request).summary["sac_replay"]
    bc = root / "bc"
    bc.mkdir()
    config, _ = temporal_fixture(bc)
    run_experiment(TemporalBCTrain(config, bc / "model"))
    return bc / "model", request.output_dir / "replay.json", prepared["replay_sha256"]


def test_stopped_preheating_resumes_remaining_budget_with_same_q_state_and_frozen_bc(tmp_path):
    pytest.importorskip("torch")
    from fh5.sac import SACCriticResume

    model, replay, digest = warm_inputs(tmp_path)
    original = {name: (model / name).read_bytes() for name in ("model.json", "actor.pt")}
    whole = run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "whole", steps=8, batch_size=1)
    ).summary["sac"]
    first = run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "first", steps=8, batch_size=1),
        sac_stop_requested=lambda step: step == 3,
    ).summary["sac"]
    second = run_experiment(SACCriticResume(tmp_path / "first", tmp_path / "second")).summary["sac"]
    assert first["stop_reason"] == "stop_requested"
    assert first["total_steps"] == 3
    assert first["warmup_remaining_steps"] == 5
    assert second["steps_completed"] == 5
    assert second["total_steps"] == 8
    assert second["warmup_remaining_steps"] == 0
    assert second["phase_status"] == "complete"
    assert first["updates"] + second["updates"] == whole["updates"]
    assert second["learner_state_sha256"] == whole["learner_state_sha256"]
    assert second["predictions"] == whole["predictions"]
    assert second["actor_change_max"] == 0
    assert second["actor_optimizer_steps"] == 0
    for name, data in original.items():
        assert (tmp_path / "second/actor" / name).read_bytes() == data
        assert (model / name).read_bytes() == data
    manifest = json.loads((tmp_path / "second/critic.json").read_bytes())
    assert manifest["version"] == 2
    assert manifest["configuration"]["steps"] == 8
    assert len(manifest["history"]) == 1


def test_sac_handoff_requires_the_finite_warmup_to_finish_and_accepts_resumed_q(tmp_path):
    pytest.importorskip("torch")
    from fh5.sac import SACCriticResume
    from fh5.sac_learning import SACTrain

    model, replay, digest = warm_inputs(tmp_path)
    first = tmp_path / "first"
    run_experiment(
        SACCriticWarmup(model, replay, digest, first, steps=4),
        sac_stop_requested=lambda step: step == 1,
    )
    with pytest.raises(ValueError, match="Finish finite critic warm-up"):
        run_experiment(
            SACTrain(first, first / "experience/replay.json", tmp_path / "premature", steps=2)
        )
    complete = tmp_path / "complete"
    run_experiment(SACCriticResume(first, complete))
    result = run_experiment(
        SACTrain(complete, complete / "experience/replay.json", tmp_path / "sac", steps=2)
    ).summary["sac_learning"]
    assert result["bc_transfer_command_error"] == 0
    assert result["steps_completed"] == 2
    assert result["actor_updates"] == 1
    assert (tmp_path / "sac/bc/actor.pt").read_bytes() == (model / "actor.pt").read_bytes()


def test_cli_resumes_a_portable_snapshot_without_original_bc_or_replay(tmp_path, capsys):
    pytest.importorskip("torch")
    from fh5.cli import main

    model, replay, digest = warm_inputs(tmp_path)
    run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "first", steps=6),
        sac_stop_requested=lambda step: step == 2,
    )
    moved = tmp_path / "portable"
    (tmp_path / "first").rename(moved)
    (model / "actor.pt").write_bytes(b"original workspace model no longer available")
    replay.write_text("{}")
    assert (
        main(
            [
                "sac-warmup-resume",
                "--checkpoint",
                str(moved),
                "--output",
                str(tmp_path / "continued"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["steps_completed"] == 4
    assert result["total_steps"] == 6
    assert result["phase_status"] == "complete"
    assert result["actor_change_max"] == result["reload_max_abs_error"] == 0


def test_warmup_history_survives_handoff_and_subsequent_sac_continuation(tmp_path):
    pytest.importorskip("torch")
    from fh5.sac import SACCriticResume
    from fh5.sac_learning import SACResume, SACTrain

    model, replay, digest = warm_inputs(tmp_path)
    run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "first", steps=3),
        sac_stop_requested=lambda step: step == 1,
    )
    run_experiment(SACCriticResume(tmp_path / "first", tmp_path / "second", steps=1))
    run_experiment(SACCriticResume(tmp_path / "second", tmp_path / "third"))
    run_experiment(
        SACTrain(
            tmp_path / "third", tmp_path / "third/experience/replay.json", tmp_path / "sac", steps=0
        )
    )
    manifest = json.loads((tmp_path / "sac/policy.json").read_bytes())
    assert len(manifest["history"]) == 3
    for entry, source in zip(manifest["history"], ("first", "second", "third")):
        assert (tmp_path / "sac" / entry["checkpoint"]).read_bytes() == (
            tmp_path / source / "critic.json"
        ).read_bytes()
        assert (tmp_path / "sac" / entry["report"]).read_bytes() == (
            tmp_path / source / "training-report.json"
        ).read_bytes()
    run_experiment(SACResume(tmp_path / "sac", tmp_path / "continued", steps=1))
    continued = json.loads((tmp_path / "continued/policy.json").read_bytes())
    assert len(continued["history"]) == 4
    (tmp_path / "continued" / continued["history"][0]["report"]).write_text("{}")
    with pytest.raises(ValueError, match="continuation history changed"):
        run_experiment(SACResume(tmp_path / "continued", tmp_path / "invalid", steps=1))


@pytest.mark.parametrize("fault", ["pixels", "replay", "report", "bc", "weights", "history"])
def test_preheating_resume_rejects_corrupted_dependencies_before_publishing(tmp_path, fault):
    pytest.importorskip("torch")
    from fh5.sac import SACCriticResume

    model, replay, digest = warm_inputs(tmp_path)
    run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "first", steps=4),
        sac_stop_requested=lambda step: step == 1,
    )
    source = tmp_path / "second"
    run_experiment(SACCriticResume(tmp_path / "first", source, steps=1))
    manifest = json.loads((source / "critic.json").read_bytes())
    target = {
        "pixels": next((source / "experience/frames").glob("*.rgb")),
        "replay": source / "experience/replay.json",
        "report": source / "training-report.json",
        "bc": source / "actor/model.json",
        "weights": source / "critic.pt",
        "history": source / manifest["history"][0]["report"],
    }[fault]
    target.write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        run_experiment(SACCriticResume(source, tmp_path / "invalid"))
    assert not (tmp_path / "invalid/critic.json").exists()


def test_resume_cannot_extend_budget_and_zero_steps_preserves_phase_without_reinitializing(
    tmp_path,
):
    pytest.importorskip("torch")
    from fh5.sac import SACCriticResume

    model, replay, digest = warm_inputs(tmp_path)
    stopped = run_experiment(
        SACCriticWarmup(model, replay, digest, tmp_path / "first", steps=4),
        sac_stop_requested=lambda step: step == 1,
    ).summary["sac"]
    with pytest.raises(ValueError, match="remaining warm-up budget"):
        run_experiment(SACCriticResume(tmp_path / "first", tmp_path / "too-many", steps=4))
    copied = run_experiment(
        SACCriticResume(tmp_path / "first", tmp_path / "copy", steps=0)
    ).summary["sac"]
    assert copied["learner_state_sha256"] == stopped["learner_state_sha256"]
    assert copied["total_steps"] == 1 and copied["warmup_remaining_steps"] == 3
    assert copied["phase_status"] == "warming"
    finished = run_experiment(SACCriticResume(tmp_path / "copy", tmp_path / "finished")).summary[
        "sac"
    ]
    assert finished["steps_completed"] == 3 and finished["phase_status"] == "complete"
    again = run_experiment(SACCriticResume(tmp_path / "finished", tmp_path / "again")).summary[
        "sac"
    ]
    assert again["steps_completed"] == 0
    assert again["learner_state_sha256"] == finished["learner_state_sha256"]


def test_frozen_critic_report_cannot_overwrite_the_resumable_weights(tmp_path):
    pytest.importorskip("torch")
    from fh5.sac import SACCriticReplay

    model, replay, digest = warm_inputs(tmp_path)
    source = tmp_path / "warm"
    run_experiment(SACCriticWarmup(model, replay, digest, source, steps=1))
    target = source / "critic.pt"
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(SACCriticReplay(source, source / "experience/replay.json", target))
    assert target.read_bytes() == original


def test_legacy_warmup_still_replays_and_initializes_sac_but_cannot_claim_portable_resume(tmp_path):
    torch = pytest.importorskip("torch")
    from fh5.sac import SACCriticReplay, SACCriticResume
    from fh5.sac_learning import SACTrain

    model, replay, digest = warm_inputs(tmp_path)
    source = tmp_path / "legacy"
    expected = run_experiment(SACCriticWarmup(model, replay, digest, source, steps=2)).summary[
        "sac"
    ]
    manifest = json.loads((source / "critic.json").read_bytes())
    saved = torch.load(source / "critic.pt", map_location="cpu", weights_only=True)
    del saved["metadata"], saved["step"]
    torch.save(saved, source / "critic.pt")
    legacy = {
        key: manifest[key]
        for key in (
            "stage",
            "bounds",
            "command_quantization",
            "replay_sha256",
            "actor_manifest_sha256",
            "configuration",
        )
    }
    legacy.update(
        version=1,
        weights_sha256=hashlib.sha256((source / "critic.pt").read_bytes()).hexdigest(),
        report_file="training-report.json",
        report_sha256=manifest["training_report_sha256"],
    )
    (source / "critic.json").write_text(json.dumps(legacy))
    replayed = run_experiment(SACCriticReplay(source, replay, tmp_path / "legacy.html")).summary[
        "sac"
    ]
    assert replayed["predictions"] == expected["predictions"]
    with pytest.raises(ValueError, match="sealed version 2"):
        run_experiment(SACCriticResume(source, tmp_path / "invalid"))
    learned = run_experiment(SACTrain(source, replay, tmp_path / "sac", steps=2)).summary[
        "sac_learning"
    ]
    assert learned["steps_completed"] == 2
    assert learned["bc_transfer_command_error"] == 0


def test_warmup_resume_rejects_a_self_consistent_but_unsupported_target_update_contract(tmp_path):
    torch = pytest.importorskip("torch")
    from fh5.sac import SACCriticResume

    model, replay, digest = warm_inputs(tmp_path)
    source = tmp_path / "warm"
    run_experiment(
        SACCriticWarmup(model, replay, digest, source, steps=3),
        sac_stop_requested=lambda step: step == 1,
    )
    saved = torch.load(source / "critic.pt", map_location="cpu", weights_only=True)
    saved["metadata"]["configuration"]["target_tau"] = 0.2
    torch.save(saved, source / "critic.pt")
    manifest = {
        **saved["metadata"],
        "weights_sha256": hashlib.sha256((source / "critic.pt").read_bytes()).hexdigest(),
    }
    (source / "critic.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported critic configuration"):
        run_experiment(SACCriticResume(source, tmp_path / "invalid"))


def test_stop_file_saves_only_after_a_complete_update_and_can_resume(tmp_path):
    pytest.importorskip("torch")
    from fh5.sac import SACCriticResume

    model, replay, digest = warm_inputs(tmp_path)
    output = tmp_path / "stopped"

    def request_file_after_check(step):
        if step == 2:
            (output / "stop.request").write_text("external file request")
        return False

    stopped = run_experiment(
        SACCriticWarmup(model, replay, digest, output, steps=6),
        sac_stop_requested=request_file_after_check,
    ).summary["sac"]
    assert stopped["stop_reason"] == "stop_requested"
    # The request arrived after this iteration's file check; this update completes first.
    assert stopped["total_steps"] == 3
    resumed = run_experiment(SACCriticResume(output, tmp_path / "resumed")).summary["sac"]
    assert resumed["steps_completed"] == 3 and resumed["total_steps"] == 6
    assert not (tmp_path / "resumed/stop.request").exists()


@pytest.mark.parametrize(
    "limit", ["REPORT_LIMIT_BYTES", "WEIGHTS_LIMIT_BYTES", "MANIFEST_LIMIT_BYTES"]
)
def test_capacity_failure_never_publishes_an_unreadable_warmup_snapshot(
    tmp_path, monkeypatch, limit
):
    pytest.importorskip("torch")
    from fh5 import sac_checkpoint

    model, replay, digest = warm_inputs(tmp_path)
    # Scale the storage capacity, not the learner or filesystem, to exercise overflow cheaply.
    monkeypatch.setattr(sac_checkpoint, limit, 1, raising=False)
    output = tmp_path / "overflow"
    with pytest.raises(ValueError, match="capacity"):
        run_experiment(SACCriticWarmup(model, replay, digest, output, steps=2))
    assert not (output / "critic.json").exists()


def test_combined_history_capacity_is_checked_before_publishing_the_next_snapshot(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    from fh5 import sac_checkpoint
    from fh5.sac import SACCriticResume

    model, replay, digest = warm_inputs(tmp_path)
    first = tmp_path / "first"
    run_experiment(SACCriticWarmup(model, replay, digest, first, steps=2))
    # Parent history fits; adding this segment's report/manifest no longer fits.
    prior_size = sum(
        (first / name).stat().st_size for name in ("critic.json", "training-report.json")
    )
    monkeypatch.setattr(sac_checkpoint, "HISTORY_LIMIT_BYTES", prior_size + 1, raising=False)
    output = tmp_path / "overflow"
    with pytest.raises(ValueError, match="capacity"):
        run_experiment(SACCriticResume(first, output, steps=0))
    assert not (output / "critic.json").exists()


def test_sac_handoff_cannot_publish_a_report_its_resume_reader_will_refuse(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from fh5 import sac_checkpoint
    from fh5.sac_learning import SACTrain

    model, replay, digest = warm_inputs(tmp_path)
    first = tmp_path / "first"
    run_experiment(SACCriticWarmup(model, replay, digest, first, steps=1))
    monkeypatch.setattr(
        sac_checkpoint, "REPORT_LIMIT_BYTES", (first / "training-report.json").stat().st_size
    )
    output = tmp_path / "overflow"
    with pytest.raises(ValueError, match="capacity"):
        run_experiment(SACTrain(first, replay, output, steps=3))
    assert not (output / "policy.json").exists()
