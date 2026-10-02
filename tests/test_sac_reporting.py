"""Optional presentation failures must not discard actual trained learner state."""

from pathlib import Path

from checkpoint_files import prediction_records
from test_sac_learning import warm_start

from fh5.experiment import run_experiment
from fh5.sac_learning import SACPolicyReplay, SACResume, SACTrain


def test_html_write_failure_preserves_training_and_continuation(tmp_path, monkeypatch):
    replay = warm_start(tmp_path)
    candidate, resumed = tmp_path / "candidate", tmp_path / "resumed"
    original_open = Path.open

    def unavailable_html(path, mode="r", *args, **kwargs):
        if path.suffix == ".html" and any(flag in mode for flag in "wx"):
            raise OSError("presentation filesystem unavailable")
        return original_open(path, mode, *args, **kwargs)

    # Only the external file sink fails; actual Torch updates and snapshot readers run.
    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", unavailable_html)
        trained = run_experiment(SACTrain(tmp_path / "warm", replay, candidate, steps=2))
        assert trained.summary["sac_learning"]["steps_completed"] == 2
        assert trained.summary["sac_learning"]["presentation"]["status"] == "unavailable"
        assert trained.report_path == candidate / "training-report.json"
        continued = run_experiment(SACResume(candidate, resumed, steps=2))
        assert continued.summary["sac_learning"]["steps_completed"] == 2
        assert continued.summary["sac_learning"]["total_steps"] == 4
        assert continued.report_path.is_file()
    restored = run_experiment(
        SACPolicyReplay(resumed, resumed / "experience/replay.json", tmp_path / "rebuilt.html")
    )
    assert restored.report_path.is_file()
    assert prediction_records(tmp_path, restored.summary["sac_policy"]) == prediction_records(
        resumed, continued.summary["sac_learning"]
    )


def test_critic_report_failure_still_leaves_a_usable_warm_start(tmp_path, monkeypatch):
    original_open = Path.open

    def failed_critic_display(path, mode="r", *args, **kwargs):
        if path == tmp_path / "warm/report.html" and any(flag in mode for flag in "wx"):
            raise MemoryError("optional display allocation failed")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failed_critic_display)
    replay = warm_start(tmp_path)
    result = run_experiment(SACTrain(tmp_path / "warm", replay, tmp_path / "candidate", steps=2))
    assert result.summary["sac_learning"]["steps_completed"] == 2
    assert result.report_path.is_file()
