"""Completed numerical BC remains usable when optional presentation is unavailable."""

import hashlib
import json
from pathlib import Path

import pytest
from test_temporal_bc import temporal_fixture

from fh5.experiment import run_experiment
from fh5.temporal_bc import TemporalBCReplay, TemporalBCTrain


@pytest.mark.parametrize("failure", [OSError, MemoryError])
def test_html_failure_preserves_verified_bc_and_allows_prediction_rebuild(
    tmp_path, monkeypatch, failure
):
    config, snapshot = temporal_fixture(tmp_path)
    model = tmp_path / "model"
    original_open = Path.open

    def unavailable_html(path, mode="r", *args, **kwargs):
        if path.suffix == ".html" and any(flag in mode for flag in "wx"):
            raise failure("optional HTML output unavailable")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", unavailable_html)
        result = run_experiment(TemporalBCTrain(config, model))
    summary = result.summary["temporal_bc"]
    assert summary["training"]["steps_completed"] == 2
    assert summary["presentation"]["status"] == "unavailable"
    assert result.report_path == model / "report.json"
    manifest = json.loads((model / "model.json").read_bytes())
    assert (
        manifest["verification"]["sha256"]
        == hashlib.sha256(result.report_path.read_bytes()).hexdigest()
    )
    assert summary["training"]["reload_max_abs_error"] <= 1e-6
    rebuilt = run_experiment(TemporalBCReplay(model, snapshot, tmp_path / "rebuilt.html"))
    assert rebuilt.summary["temporal_bc"]["verification"]["status"] == "verified"
    assert rebuilt.summary["temporal_bc"]["decisions"] == summary["decisions"]
    assert rebuilt.report_path.is_file()


def test_preview_failure_keeps_predictions_and_defers_optional_images(tmp_path, monkeypatch):
    config, snapshot = temporal_fixture(tmp_path)
    model = tmp_path / "model"
    original_open = Path.open
    attempts = []

    def unavailable_preview(path, mode="r", *args, **kwargs):
        if path.suffix == ".png" and any(flag in mode for flag in "wx"):
            attempts.append(path)
            raise OSError("optional preview storage unavailable")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", unavailable_preview)
        result = run_experiment(TemporalBCTrain(config, model))
    summary = result.summary["temporal_bc"]
    assert summary["training"]["steps_completed"] == 2
    assert summary["preview_export"]["status"] == "unavailable"
    assert len(attempts) == 1  # Stop spending resources once this optional sink has failed.
    assert len(summary["decisions"]) == 12
    assert all(r["previews"] == [None, None, None] for r in summary["decisions"])
    rebuilt = run_experiment(TemporalBCReplay(model, snapshot, tmp_path / "rebuilt.html"))
    after = rebuilt.summary["temporal_bc"]
    assert after["verification"]["status"] == "verified"
    assert [r["prediction"] for r in after["decisions"]] == [
        r["prediction"] for r in summary["decisions"]
    ]
    assert all((model / p).is_file() for r in after["decisions"] for p in r["previews"])


def test_partial_preview_is_not_published_and_replay_rebuilds_valid_pixels(tmp_path, monkeypatch):
    from PIL import Image

    config, snapshot = temporal_fixture(tmp_path)
    model = tmp_path / "model"
    original_write = Path.write_bytes

    def partial_preview(path, data):
        if path.suffix == ".png":
            original_write(path, data[:16])
            raise OSError("preview storage exhausted during write")
        return original_write(path, data)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "write_bytes", partial_preview)
        result = run_experiment(TemporalBCTrain(config, model))
    assert result.summary["temporal_bc"]["preview_export"]["status"] == "unavailable"
    rebuilt = run_experiment(TemporalBCReplay(model, snapshot, tmp_path / "rebuilt.html"))
    for row in rebuilt.summary["temporal_bc"]["decisions"]:
        for relative in row["previews"]:
            with Image.open(model / relative) as frame:
                assert frame.size == (64, 36)
                assert frame.getpixel((0, 0)) == (51, 17, 34)


def test_required_prediction_evidence_failure_is_not_reported_as_verified(tmp_path, monkeypatch):
    config, _ = temporal_fixture(tmp_path)
    model = tmp_path / "model"
    original_open = Path.open

    def unavailable_evidence(path, mode="r", *args, **kwargs):
        if path == model / "report.json" and any(flag in mode for flag in "wx"):
            raise OSError("required numerical evidence unavailable")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", unavailable_evidence)
        with pytest.raises(OSError, match="required numerical evidence"):
            run_experiment(TemporalBCTrain(config, model))
    assert (model / "actor.pt").is_file()
    assert "verification" not in json.loads((model / "model.json").read_bytes())


def test_partial_optional_image_releases_space_for_required_evidence(tmp_path, monkeypatch):
    config, snapshot = temporal_fixture(tmp_path)
    model = tmp_path / "model"
    original_write, original_open = Path.write_bytes, Path.open
    partial_files = []

    def fills_remaining_space(path, data):
        if path.suffix == ".png":
            original_write(path, data[:16])
            partial_files.append(path)
            raise OSError("ENOSPC while writing optional preview")
        return original_write(path, data)

    def requires_reclaimed_space(path, mode="r", *args, **kwargs):
        if (
            path == model / "report.json"
            and any(flag in mode for flag in "wx")
            and any(p.exists() for p in partial_files)
        ):
            raise OSError("ENOSPC: uncommitted optional output still occupies space")
        return original_open(path, mode, *args, **kwargs)

    # The external sink has room for required evidence only after reclaiming
    # the unpublished preview. No internal learner or publisher is replaced.
    with monkeypatch.context() as fault:
        fault.setattr(Path, "write_bytes", fills_remaining_space)
        fault.setattr(Path, "open", requires_reclaimed_space)
        result = run_experiment(TemporalBCTrain(config, model))
    assert result.summary["temporal_bc"]["preview_export"]["status"] == "unavailable"
    assert partial_files and all(not p.exists() for p in partial_files)
    replay = run_experiment(TemporalBCReplay(model, snapshot, tmp_path / "rebuilt.html"))
    assert replay.summary["temporal_bc"]["verification"]["status"] == "verified"


@pytest.mark.parametrize(
    "extension,section", [(".html", "presentation"), (".png", "preview_export")]
)
def test_optional_cleanup_failure_preserves_the_original_error(
    tmp_path, monkeypatch, extension, section
):
    config, snapshot = temporal_fixture(tmp_path)
    model = tmp_path / "model"
    original_open, original_unlink = Path.open, Path.unlink

    def failed_output(path, mode="r", *args, **kwargs):
        if path.suffix == extension and any(flag in mode for flag in "wx"):
            raise OSError("original optional output failure")
        return original_open(path, mode, *args, **kwargs)

    def failed_cleanup(path, *args, **kwargs):
        if path.suffix == extension:
            raise PermissionError("cleanup denied")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", failed_output)
        fault.setattr(Path, "unlink", failed_cleanup)
        result = run_experiment(TemporalBCTrain(config, model))
    diagnostic = result.summary["temporal_bc"][section]
    assert diagnostic["error"] == "OSError: original optional output failure"
    assert diagnostic["cleanup_error"] == "PermissionError: cleanup denied"
    replay = run_experiment(TemporalBCReplay(model, snapshot, tmp_path / "rebuilt.html"))
    assert replay.summary["temporal_bc"]["verification"]["status"] == "verified"


def test_partial_html_releases_space_for_model_verification(tmp_path, monkeypatch):
    config, snapshot = temporal_fixture(tmp_path)
    model = tmp_path / "model"
    original_text, original_open = Path.write_text, Path.open

    def partial_html(path, data, *args, **kwargs):
        if path.suffix == ".html":
            original_text(path, data[:16], *args, **kwargs)
            raise OSError("ENOSPC while writing optional HTML")
        return original_text(path, data, *args, **kwargs)

    def requires_reclaimed_space(path, mode="r", *args, **kwargs):
        if (
            path == model / "model.json"
            and any(flag in mode for flag in "wx")
            and (model / "report.html").exists()
        ):
            raise OSError("ENOSPC: partial HTML still occupies space")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "write_text", partial_html)
        fault.setattr(Path, "open", requires_reclaimed_space)
        result = run_experiment(TemporalBCTrain(config, model))
    assert result.summary["temporal_bc"]["presentation"]["status"] == "unavailable"
    assert result.report_path == model / "report.json"
    assert not (model / "report.html").exists()
    replay = run_experiment(TemporalBCReplay(model, snapshot, tmp_path / "rebuilt.html"))
    assert replay.summary["temporal_bc"]["verification"]["status"] == "verified"
