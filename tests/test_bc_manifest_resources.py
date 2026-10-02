"""Growing BC diagnostics must not gate or stay resident in downstream learning."""

import hashlib
import json
import tracemalloc

import pytest
from test_critic_resume import warm_inputs

from fh5.candidate_archive import CandidateArchive, CandidateRestore
from fh5.experiment import run_experiment
from fh5.sac import SACCriticResume, SACCriticWarmup
from fh5.sac_learning import SACPolicyReplay, SACResume, SACTrain


def add_loss_history(model):
    path = model / "model.json"
    manifest = json.loads(path.read_bytes())
    stats = manifest.pop("training")
    stats.pop("losses")
    # Real, finite diagnostic scalars, written incrementally. This is archived
    # history growth, not a request for half a million new optimizer updates.
    block = ",".join(["0.123456789"] * 1024)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest)[:-1] + ',"training":')
        stream.write(json.dumps(stats)[:-1] + ',"losses":[')
        for index in range(512):
            if index:
                stream.write(",")
            stream.write(block)
        stream.write("]}}\n")
    return len(block) * 512


def test_growing_bc_diagnostics_do_not_stay_resident_during_updates(tmp_path):
    model, replay, replay_sha = warm_inputs(tmp_path)
    baseline = run_experiment(
        SACCriticWarmup(model, replay, replay_sha, tmp_path / "baseline", steps=2)
    ).summary["sac"]
    history_size = add_loss_history(model)
    with (model / "model.json").open("rb") as stream:
        model_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    resident = []

    def observe(step):
        if step == 0:
            resident.append(tracemalloc.get_traced_memory()[0])
        return False

    tracemalloc.start()
    try:
        result = run_experiment(
            SACCriticWarmup(model, replay, replay_sha, tmp_path / "learned", steps=2),
            sac_stop_requested=observe,
        ).summary["sac"]
    finally:
        tracemalloc.stop()
    assert resident and resident[0] < history_size, "BC diagnostic history remains in RAM"
    assert result["learner_state_sha256"] == baseline["learner_state_sha256"]
    with (tmp_path / "learned/actor/model.json").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == model_sha


def test_sac_streams_bc_manifest_history_during_training_and_continuation(tmp_path):
    model, replay, replay_sha = warm_inputs(tmp_path)
    warm = tmp_path / "warm"
    run_experiment(SACCriticWarmup(model, replay, replay_sha, warm, steps=2))
    baseline = run_experiment(SACTrain(warm, replay, tmp_path / "baseline", steps=2)).summary[
        "sac_learning"
    ]
    history_size = add_loss_history(model)
    large_warm = tmp_path / "large-warm"
    run_experiment(SACCriticWarmup(model, replay, replay_sha, large_warm, steps=2))
    observed = []

    def stop(step):
        if step == 0:
            observed.append(tracemalloc.get_traced_memory()[0])
        return step == 1

    tracemalloc.start()
    try:
        run_experiment(
            SACTrain(large_warm, replay, tmp_path / "partial", steps=2),
            sac_stop_requested=stop,
        )
    finally:
        tracemalloc.stop()
    assert observed and observed[0] < history_size, "SAC keeps BC diagnostics in RAM"
    result = run_experiment(
        SACResume(tmp_path / "partial", tmp_path / "continued", steps=1)
    ).summary["sac_learning"]
    assert result["total_steps"] == 2
    assert result["learner_state_sha256"] == baseline["learner_state_sha256"]


def test_large_bc_manifest_survives_learning_replay_archive_and_restore(tmp_path):
    model, replay, replay_sha = warm_inputs(tmp_path)
    run_experiment(SACCriticWarmup(model, replay, replay_sha, tmp_path / "small-warm", steps=2))
    baseline = run_experiment(
        SACTrain(tmp_path / "small-warm", replay, tmp_path / "baseline", steps=2)
    ).summary["sac_learning"]
    manifest_file = model / "model.json"
    # Representation compatibility across the old 256 MiB gate. The real
    # growing-history tests above establish the structural memory behavior.
    with manifest_file.open("ab") as stream:
        block = b" " * 1024**2
        for _ in range(257):
            stream.write(block)
    with manifest_file.open("rb") as stream:
        manifest_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    first, warm = tmp_path / "first", tmp_path / "warm"
    candidate, archive, restored = (
        tmp_path / name for name in ("candidate", "archive", "restored")
    )
    tracemalloc.start()
    try:
        run_experiment(
            SACCriticWarmup(model, replay, replay_sha, first, steps=2),
            sac_stop_requested=lambda step: step == 1,
        )
        run_experiment(SACCriticResume(first, warm))
        result = run_experiment(SACTrain(warm, replay, candidate, steps=2)).summary["sac_learning"]
        reloaded = run_experiment(SACPolicyReplay(candidate, replay, tmp_path / "replay.html"))
        assert reloaded.summary["sac_policy"]["predictions"]["records"] == 2
        assert reloaded.summary["sac_policy"]["predictions"]["status"] == "complete"
        checkpoint_sha = hashlib.sha256((candidate / "policy.json").read_bytes()).hexdigest()
        retained = run_experiment(
            CandidateArchive(candidate, archive, checkpoint_sha, "Retain grown BC manifest")
        ).summary["candidate_archive"]
        run_experiment(
            CandidateRestore(archive, restored, retained["archive_sha256"], "Restore grown BC")
        )
        continuation = run_experiment(SACResume(restored, tmp_path / "continued", steps=0)).summary[
            "sac_learning"
        ]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result["learner_state_sha256"] == baseline["learner_state_sha256"]
    assert continuation["learner_state_sha256"] == baseline["learner_state_sha256"]
    assert peak < manifest_file.stat().st_size
    with (restored / "bc/model.json").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest_sha


@pytest.mark.parametrize("bad_history", ["[1,]", '{1:"bad key"}', '{"lost_value":}', "[1e+]"])
def test_unread_bc_diagnostics_still_require_valid_json(tmp_path, bad_history):
    model, replay, replay_sha = warm_inputs(tmp_path)
    path = model / "model.json"
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw.rstrip()[:-1] + ',"diagnostics":' + bad_history + "}", encoding="utf-8")
    with pytest.raises(ValueError):
        run_experiment(SACCriticWarmup(model, replay, replay_sha, tmp_path / "rejected", steps=1))
    assert not (tmp_path / "rejected").exists()
