"""Preserve recoverable training under tensor-to-Python allocation pressure."""

import io

import pytest

from fh5.experiment import run_experiment
from fh5.learning.sac.training import SACResume, SACTrain
from tests.learning.sac.test_sac_learning import warm_start


def test_checkpoint_and_resume_do_not_expand_complete_tensors_into_python_byte_lists(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    replay = warm_start(tmp_path)
    baseline = run_experiment(
        SACTrain(tmp_path / "warm", replay, tmp_path / "baseline", steps=5)
    ).summary["sac_learning"]
    original = torch.Tensor.tolist
    conversions = []

    def constrained_list(tensor):
        if tensor.dtype == torch.uint8:
            # Fault only Python byte materialization, not Torch training/storage.
            # The transfer quantum comes from the standard-library I/O buffer;
            # there is no bound on tensor size or completed learning steps.
            conversions.append(tensor.numel())
            if tensor.numel() > io.DEFAULT_BUFFER_SIZE:
                raise MemoryError("whole tensor Python byte list unavailable")
        return original(tensor)

    with monkeypatch.context() as allocation:
        allocation.setattr(torch.Tensor, "tolist", constrained_list)
        first = run_experiment(
            SACTrain(tmp_path / "warm", replay, tmp_path / "first", steps=3)
        ).summary["sac_learning"]
        resumed = run_experiment(
            SACResume(tmp_path / "first", tmp_path / "resumed", steps=2)
        ).summary["sac_learning"]
    assert first["total_steps"] == 3 and resumed["total_steps"] == 5
    assert resumed["learner_state_sha256"] == baseline["learner_state_sha256"]
    assert resumed["predictions"] == baseline["predictions"]
    assert conversions and max(conversions) <= io.DEFAULT_BUFFER_SIZE
