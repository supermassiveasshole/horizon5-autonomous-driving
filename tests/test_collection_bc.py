"""Continuous recordings become causal numerical training inputs at the experiment seam."""

import hashlib
import json
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import pytest
from test_collection_dataset import dataset_inputs

from fh5.experiment import run_experiment


def prepare_inputs(tmp_path):
    return dataset_inputs(tmp_path, vary_action=True, size=(64, 36))


def test_collection_export_preserves_pixels_past_inputs_and_separate_holdout(tmp_path):
    from fh5.collection_bc import CollectionBCPrepare

    config = prepare_inputs(tmp_path)
    output = tmp_path / "numeric"
    result = run_experiment(CollectionBCPrepare(config, output))
    selection = output / "selection.json"
    train = json.loads((output / "dataset.json").read_bytes())
    final = json.loads((output / "evaluation.json").read_bytes())
    assert {g["split"] for g in train["groups"]} == {"train", "development"}
    assert {g["split"] for g in final["groups"]} == {"evaluation"}
    assert {r["decision_id"] for r in train["decisions"]}.isdisjoint(
        r["decision_id"] for r in final["decisions"]
    )
    row = next(r for r in train["decisions"] if r["source_sequence"] == 5)
    actor = row["views"]["no_reference"]
    assert actor["actions"][-1][0] == -1  # strictly previous input, not next poll's label
    assert all(a > 0 for a in actor["action_age_ms"] if a is not None)
    assert row["supervision"]["action"][0] > 0.5
    assert row["supervision"]["label_poll_ns"] > row["decision_ns"]
    assert actor["image_age_ms"] == [200, 100, 0]
    assert row["views"]["reference_assisted"] == actor  # missing optional reference
    assert not any(actor["reference"]["mask"])
    for frame in row["frames"]:
        assert (output / frame["path"]).read_bytes() == bytes([51, 17, 34] * (64 * 36))
    assert (
        train["provenance"]["selection_sha256"]
        == hashlib.sha256(selection.read_bytes()).hexdigest()
    )
    assert result.summary["collection_bc"]["diagnostic_only"] is True
    assert result.summary["collection_bc"]["evaluation"]["metrics"] == "withheld"
    assert not result.summary["collection_bc"]["commands_sent"]


def test_exported_collection_trains_and_reloads_without_final_holdout_feedback(tmp_path):
    pytest.importorskip("torch")
    from fh5.collection_bc import CollectionBCPrepare
    from fh5.numeric_drive_config import NumericDriveConfiguration
    from fh5.temporal_bc import TemporalBCReplay, TemporalBCTrain

    config = prepare_inputs(tmp_path)
    # Exercise the documented preparation settings with real sealed test sources.
    # The source stays diagnostic; matching runtime history is not game qualification.
    example = json.loads(
        (Path(__file__).parents[1] / "configs/collection-bc.example.json").read_text()
    )
    example["sources"] = json.loads(config.read_text())["sources"]
    config.write_text(json.dumps(example))
    output = tmp_path / "numeric"
    open_file = Path.open

    def unavailable_html(path, *args, **kwargs):
        if path == output / "report.html":
            raise OSError("optional HTML unavailable")
        return open_file(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", unavailable_html)
        prepared = run_experiment(CollectionBCPrepare(config, output))
    assert prepared.report_path == output / "report.json"
    assert prepared.report_path.is_file()
    assert prepared.summary["collection_bc"]["presentation"]["status"] == "unavailable"
    dataset = output / "dataset.json"
    train_config = tmp_path / "train.json"
    train_config.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": str(dataset),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "seed": 9,
                "steps": 2,
                "batch_size": 4,
                "learning_rate": 0.001,
                "device": "cpu",
                "time_mode": "actual",
            }
        )
    )
    trained = run_experiment(TemporalBCTrain(train_config, tmp_path / "model"))
    summary = trained.summary["temporal_bc"]
    assert summary["training"]["steps_completed"] == 2
    assert {r["split"] for r in summary["decisions"]} == {"train", "development"}
    assert all(r["timing"]["adjacent_delta_s"] == [0.1, 0.1] for r in summary["decisions"])
    assert summary["model"]["provenance"]["diagnostic_only"]
    assert summary["model"]["diagnostic_only"]
    reloaded = run_experiment(
        TemporalBCReplay(tmp_path / "model", dataset, tmp_path / "replay.html")
    )
    assert reloaded.summary["temporal_bc"]["decisions"] == summary["decisions"]
    assert reloaded.summary["temporal_bc"]["verification"]["max_abs_error"] <= 1e-6

    from test_route_check import route

    examples = Path(__file__).parents[1] / "configs"
    capture = json.loads((examples / "capture-dxgi.example.json").read_text())
    capture["pixels"] = json.loads((tmp_path / "model/model.json").read_text())["numeric_contract"]
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps(capture))
    settings = json.loads((examples / "realtime-drive.example.json").read_text())
    settings["capture_config"] = str(capture_path)
    settings["model"] = {"directory": str(tmp_path / "model"), "device": "cpu"}
    settings["task"]["route_file"] = str(route(tmp_path))
    settings["task"]["end_margin_m"] = 0.5  # fixture route is only three metres
    driving = tmp_path / "drive.json"
    driving.write_text(json.dumps(settings))
    plan = NumericDriveConfiguration(driving, tmp_path / "not-opened", 1, False)
    assert "action_history_mismatch" not in plan.qualification["reasons"]
    assert "diagnostic_model" in plan.qualification["reasons"]
    assert not plan.qualification["eligible"]


def test_optional_independent_reference_only_changes_paired_reference_branch(tmp_path):
    from test_routes import recording

    from fh5.collection_bc import CollectionBCPrepare
    from fh5.routes import BuildRoute

    config = prepare_inputs(tmp_path)
    reference_source = recording(tmp_path, [(x, 0) for x in range(0, 101, 5)])
    run_experiment(BuildRoute(reference_source, tmp_path / "route", 0, 20))
    options = json.loads(config.read_bytes())
    options["reference"] = {
        "route_file": str(tmp_path / "route/route.json"),
        "independence_evidence": ["Separate synthetic recording; different session and timestamps"],
    }
    config.write_text(json.dumps(options))
    output = tmp_path / "numeric"
    run_experiment(CollectionBCPrepare(config, output))
    data = json.loads((output / "dataset.json").read_bytes())
    assert (output / "reference/route.json").read_bytes() == (
        tmp_path / "route/route.json"
    ).read_bytes()
    assert data["provenance"]["reference"]["status"] == "loaded"
    assert any(
        any(r["views"]["reference_assisted"]["reference"]["mask"]) for r in data["decisions"]
    )
    for row in data["decisions"]:
        plain, assisted = row["views"]["no_reference"], row["views"]["reference_assisted"]
        assert not any(plain["reference"]["mask"])
        assert {k: v for k, v in plain.items() if k != "reference"} == {
            k: v for k, v in assisted.items() if k != "reference"
        }


def test_collection_export_rejects_changed_pixels_before_publishing(tmp_path):
    from fh5.collection_bc import CollectionBCPrepare

    config = prepare_inputs(tmp_path)
    source = Path(json.loads(config.read_bytes())["sources"][0]["recording"])
    next((source / "blocks").rglob("*.rgb")).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        run_experiment(CollectionBCPrepare(config, tmp_path / "numeric"))
    assert not (tmp_path / "numeric").exists()


def test_preparation_accepts_the_configured_reservoir_budget(tmp_path):
    from fh5.collection_bc import CollectionBCPrepare

    config = prepare_inputs(tmp_path)
    run_experiment(CollectionBCPrepare(config, tmp_path / "original"))
    settings = json.loads(config.read_bytes())
    settings["rules"]["max_samples_per_attempt"] = 5001
    config.write_text(json.dumps(settings))
    run_experiment(CollectionBCPrepare(config, tmp_path / "larger-budget"))
    for name in ("dataset.json", "evaluation.json"):
        original = json.loads((tmp_path / "original" / name).read_bytes())
        current = json.loads((tmp_path / "larger-budget" / name).read_bytes())
        assert current["decisions"] == original["decisions"]
        assert current["provenance"]["envelope"]["max_samples_per_attempt"] == 5001


def test_unavailable_preparation_index_preserves_sources_for_retry(tmp_path, monkeypatch):
    from fh5.collection_bc import CollectionBCPrepare

    config = prepare_inputs(tmp_path)
    output = tmp_path / "prepared"
    attempted = []

    def unavailable_storage(path, *args, **kwargs):
        attempted.append(Path(path))
        raise sqlite3.OperationalError("disk I/O error")

    with monkeypatch.context() as fault:
        fault.setattr(sqlite3, "connect", unavailable_storage)
        with pytest.raises(OSError, match="Cannot index collection dataset"):
            run_experiment(CollectionBCPrepare(config, output))
    assert attempted and all(not path.parent.exists() for path in attempted)
    assert not output.exists()
    result = run_experiment(CollectionBCPrepare(config, output))
    assert result.summary["collection_dataset"]["ready_for_software_training"]


def test_collection_export_accepts_histories_beyond_old_frame_quota(tmp_path):
    from test_collection import Stream, input_at, request

    from fh5.collection_bc import CollectionBCPrepare
    from fh5.collection_dataset import CollectionDatasetReview
    from fh5.numeric_images import PixelContract

    config = dataset_inputs(tmp_path)
    options = json.loads(config.read_bytes())
    options["rules"]["max_samples_per_attempt"] = 500
    size = (480, 270)
    pixels = bytes([51, 17, 34]) * (480 * 270)

    def points(index):
        for sequence in range(500):
            point = input_at(250 + index * 100_000 + sequence * 50)
            # Pace the external source so this export test does not test writer overload.
            time.sleep(0.005)
            yield replace(
                point,
                frames=tuple(
                    replace(frame, size=size, pixels=memoryview(pixels)) for frame in point.frames
                ),
            )

    for index, source in enumerate(options["sources"]):
        folder = tmp_path / f"long-source-{index}"
        folder.mkdir()
        req = request(folder, pixels=PixelContract(size=size), block_rows=250)
        recorded = run_experiment(req, collection_environment=Stream(points(index)))
        assert recorded.summary["collection"]["written_rows"] == 500
        review_path = Path(source["review"])
        review = json.loads(review_path.read_bytes())
        review["session_sha256"] = hashlib.sha256(
            (req.output_dir / "session.json").read_bytes()
        ).hexdigest()
        review["attempts"][0]["end_sequence"] = 500
        review["attempts"][0]["intervals"][0]["end_sequence"] = 500
        review_path.write_text(json.dumps(review))
        source["recording"] = str(req.output_dir)
    config.write_text(json.dumps(options))

    output = tmp_path / "numeric"
    prepared = run_experiment(CollectionBCPrepare(config, output)).summary["collection_bc"]
    train = json.loads((output / "dataset.json").read_bytes())
    final = json.loads((output / "evaluation.json").read_bytes())
    assert len(train["decisions"]) == 990
    assert len(final["decisions"]) == 495
    assert prepared["decoded_frame_budget_bytes"] > 512 * 1024**2
    # Distinct timestamped frames may share pixels; this is not resident-memory usage.
    assert len(list((output / "pixels").iterdir())) == 1
    assert next((output / "pixels").iterdir()).read_bytes() == pixels
    assert {r["group"] for r in train["decisions"]} == {"group-0", "group-1"}
    assert {r["group"] for r in final["decisions"]} == {"group-2"}
    assert train["decisions"][-1]["source_sequence"] == 498
    assert train["decisions"][-1]["views"]["no_reference"]["image_age_ms"] == [200, 100, 0]
    verified = run_experiment(
        CollectionDatasetReview(output / "selection.json", tmp_path / "verified.html")
    )
    assert verified.summary["collection_dataset"]["verified"]


def test_old_prepare_config_requests_migration_before_reading_selection(tmp_path):
    from fh5.collection_bc import CollectionBCPrepare

    config = tmp_path / "old-prepare.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": "legacy-selection.json",
                "dataset_sha256": "0" * 64,
                "action_history_offsets_ms": [100, 50, 0],
                "max_action_age_ms": 100,
                "waypoint_distances_m": [5, 10, 20],
            }
        )
    )
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="requires v2.*source configuration"):
        run_experiment(CollectionBCPrepare(config, output))
    assert not output.exists()
