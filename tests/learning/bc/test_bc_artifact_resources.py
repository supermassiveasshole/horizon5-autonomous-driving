"""BC artifacts grow without duplicate byte payloads or file-size admission gates."""

import hashlib
import json
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

import pytest

from fh5.experiment import run_experiment
from fh5.learning.sac.critic import SACCriticResume, SACCriticWarmup
from fh5.learning.sac.training import SACResume, SACTrain
from tests.learning.sac.test_critic_resume import warm_inputs
from tests.support.checkpoint_files import update_records


def grow_weights(model, mebibytes):
    path = model / "actor.pt"
    # A valid Torch archive may contain additional records. Keep the real
    # model/tensors identical while testing storage growth independently of RAM
    # needed for larger networks. These bytes are never mocked or compressed.
    with ZipFile(path, "a") as archive:
        prefix = archive.namelist()[0].split("/")[0]
        with archive.open(prefix + "/resource-evidence.bin", "w") as member:
            block = bytes(1024**2)
            for _ in range(mebibytes):
                member.write(block)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = json.loads((model / "model.json").read_bytes())
    manifest["weights_sha256"] = digest
    (model / "model.json").write_text(json.dumps(manifest), encoding="utf-8")
    return digest, path.stat().st_size


def test_large_bc_weights_stream_through_critic_sac_and_exact_continuation(tmp_path):
    model, replay, digest = warm_inputs(tmp_path)
    warm = tmp_path / "small-warm"
    small_critic = run_experiment(
        SACCriticWarmup(model, replay, digest, warm, steps=2, batch_size=1)
    ).summary["sac"]
    small_sac = run_experiment(
        SACTrain(warm, replay, tmp_path / "small-sac", steps=2, batch_size=1)
    ).summary["sac_learning"]
    weights_sha, size = grow_weights(model, 257)  # Cross the old 256 MiB admission gate.
    first, complete = tmp_path / "first", tmp_path / "complete"
    learned, resumed = tmp_path / "learned", tmp_path / "resumed"
    tracemalloc.start()
    try:
        stopped = run_experiment(
            SACCriticWarmup(model, replay, digest, first, steps=2, batch_size=1),
            sac_stop_requested=lambda step: step == 1,
        ).summary["sac"]
        critic = run_experiment(SACCriticResume(first, complete)).summary["sac"]
        run_experiment(
            SACTrain(complete, replay, learned, steps=2, batch_size=1),
            sac_stop_requested=lambda step: step == 1,
        )
        result = run_experiment(SACResume(learned, resumed, steps=1)).summary["sac_learning"]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert stopped["total_steps"] == 1 and critic["total_steps"] == result["total_steps"] == 2
    assert critic["learner_state_sha256"] == small_critic["learner_state_sha256"]
    assert result["learner_state_sha256"] == small_sac["learner_state_sha256"]
    assert update_records(first) + update_records(complete) == update_records(warm)
    assert update_records(learned) + update_records(resumed) == update_records(
        tmp_path / "small-sac"
    )
    assert peak < size, "Weight serialization must not be retained as a complete Python byte array"
    for copy in (model, first / "actor", complete / "actor", learned / "bc", resumed / "bc"):
        with (copy / "actor.pt").open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == weights_sha


@pytest.mark.parametrize("learner", ["critic", "sac"])
@pytest.mark.parametrize("failure", ["changed_copy", "storage_unavailable"])
def test_bc_copy_failure_preserves_sources_without_publishing_checkpoint(
    tmp_path, monkeypatch, learner, failure
):
    model, replay, digest = warm_inputs(tmp_path)
    warm = tmp_path / "warm"
    if learner == "sac":
        run_experiment(SACCriticWarmup(model, replay, digest, warm, steps=2))
    source = model if learner == "critic" else warm
    original = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    failed = tmp_path / "failed"
    target = failed / ("actor" if learner == "critic" else "bc") / "actor.pt"
    open_file = Path.open
    triggered = []

    @contextmanager
    def changed_copy(stream):
        with stream:
            yield stream
        # Simulate an external storage change after writing, before publication.
        with open_file(target, "r+b") as stored:
            first = stored.read(1)
            stored.seek(0)
            stored.write(bytes([first[0] ^ 1]))

    def open_with_failure(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == target and mode == "xb":
            triggered.append(path)
            if failure == "storage_unavailable":
                raise OSError("BC copy storage unavailable")
            return changed_copy(open_file(path, *args, **kwargs))
        return open_file(path, *args, **kwargs)

    def request(output):
        if learner == "critic":
            return SACCriticWarmup(model, replay, digest, output, steps=2)
        return SACTrain(warm, replay, output, steps=2)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_with_failure)
        error = OSError if failure == "storage_unavailable" else ValueError
        with pytest.raises(error, match="storage unavailable|changed during copy"):
            run_experiment(request(failed))
    assert triggered
    assert not (failed / ("critic.json" if learner == "critic" else "policy.json")).exists()
    assert all(path.read_bytes() == value for path, value in original.items())
    result = run_experiment(request(tmp_path / "retry"))
    summary = result.summary["sac" if learner == "critic" else "sac_learning"]
    assert summary["total_steps"] == 2
