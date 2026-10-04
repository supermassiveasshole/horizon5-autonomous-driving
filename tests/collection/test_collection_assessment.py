"""Frozen candidates are compared and finally evaluated through experiment runs."""

import hashlib
import json

import pytest

from fh5.collection.bc import CollectionBCPrepare
from fh5.experiment import run_experiment
from fh5.learning.bc.training import TemporalBCTrain
from tests.collection.test_collection_bc import prepare_inputs

pytest.importorskip("torch")


def candidates(tmp_path):
    prepare = prepare_inputs(tmp_path)
    numeric = tmp_path / "numeric"
    run_experiment(CollectionBCPrepare(prepare, numeric))
    dataset = numeric / "dataset.json"
    result = {}
    for name, seed in (("candidate", 9), ("baseline", 7)):
        cfg = tmp_path / (name + ".json")
        cfg.write_text(
            json.dumps(
                {
                    "version": 1,
                    "dataset": str(dataset),
                    "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    "seed": seed,
                    "steps": 2,
                    "batch_size": 4,
                    "learning_rate": 0.001,
                    "device": "cpu",
                    "time_mode": "actual",
                }
            )
        )
        model = tmp_path / name
        run_experiment(TemporalBCTrain(cfg, model))
        result[name] = {
            "directory": str(model),
            "manifest_sha256": hashlib.sha256((model / "model.json").read_bytes()).hexdigest(),
        }
    return numeric, result


def assess_config(tmp_path, numeric, models, mode):
    dataset = numeric / ("evaluation.json" if mode == "final" else "dataset.json")
    cfg = tmp_path / (mode + ".json")
    cfg.write_text(
        json.dumps(
            {
                "version": 1,
                "mode": mode,
                "dataset": str(dataset),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                **models,
                "device": "cpu",
            }
        )
    )
    return cfg


def test_final_evaluation_uses_prebound_holdout_and_keeps_frozen_models_unchanged(tmp_path):
    from fh5.collection.assessment import CollectionBCAssess

    numeric, models = candidates(tmp_path)
    before = {
        p: p.read_bytes() for name in models for p in (tmp_path / name).rglob("*") if p.is_file()
    }
    result = run_experiment(
        CollectionBCAssess(
            assess_config(tmp_path, numeric, models, "final"), tmp_path / "assessment"
        )
    )
    summary = result.summary["collection_assessment"]
    assert summary["mode"] == "final"
    assert summary["baseline"]["status"] == "comparable"
    assert summary["selection_allowed"] is False
    assert summary["diagnostic_only"] is True
    assert summary["commands_sent"] is False
    assert len(summary["decisions"]) == 20
    assert {r["split"] for r in summary["decisions"]} == {"evaluation"}
    assert all(r["baseline_prediction"] is not None for r in summary["decisions"])
    assert summary["metrics"]["no_reference"]["candidate"]["count"] == 10
    assert summary["metrics"]["no_reference"]["copy_recent_action"]["mae"] == [0, 0]
    assert summary["verification"]["candidate_reload_max_abs_error"] <= 1e-6
    assert summary["verification"]["baseline_reload_max_abs_error"] <= 1e-6
    assert {
        p: p.read_bytes() for name in models for p in (tmp_path / name).rglob("*") if p.is_file()
    } == before


def test_development_assessment_does_not_predict_final_or_training_samples(tmp_path):
    from fh5.collection.assessment import CollectionBCAssess

    numeric, models = candidates(tmp_path)
    result = run_experiment(
        CollectionBCAssess(
            assess_config(tmp_path, numeric, models, "development"), tmp_path / "assessment"
        )
    )
    summary = result.summary["collection_assessment"]
    assert summary["selection_allowed"]
    assert len(summary["decisions"]) == 20
    assert {r["split"] for r in summary["decisions"]} == {"development"}
    assert summary["baseline"]["status"] == "comparable"


@pytest.mark.parametrize("fault", ["rehash_holdout", "swapped_partition", "changed_model", "pixel"])
def test_assessment_rejects_unbound_or_corrupted_inputs_before_reporting(tmp_path, fault):
    from fh5.collection.assessment import CollectionBCAssess

    numeric, models = candidates(tmp_path)
    config = assess_config(tmp_path, numeric, models, "final")
    options = json.loads(config.read_bytes())
    if fault == "rehash_holdout":
        path = numeric / "evaluation.json"
        data = json.loads(path.read_bytes())
        data["decisions"][0]["supervision"]["action"] = [0, 0]
        path.write_text(json.dumps(data))
        options["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "swapped_partition":
        options["mode"] = "development"
    elif fault == "changed_model":
        path = tmp_path / "candidate/model.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        next((numeric / "pixels").glob("*.rgb")).write_bytes(b"broken")
    config.write_text(json.dumps(options))
    with pytest.raises(ValueError):
        run_experiment(CollectionBCAssess(config, tmp_path / "assessment"))
    assert not (tmp_path / "assessment").exists()


def test_missing_baseline_still_reports_copy_action_and_missing_strata(tmp_path):
    from fh5.collection.assessment import CollectionBCAssess

    numeric, models = candidates(tmp_path)
    models["baseline"] = None
    result = run_experiment(
        CollectionBCAssess(
            assess_config(tmp_path, numeric, models, "final"), tmp_path / "assessment"
        )
    )
    summary = result.summary["collection_assessment"]
    assert summary["baseline"]["status"] == "not_provided"
    metrics = summary["metrics"]["no_reference"]
    assert metrics["baseline"]["count"] == 0 and metrics["copy_recent_action"]["count"] == 10
    assert metrics["strata"]["brake"]["candidate"]["mae"] is None
    assert metrics["strata"]["release_rt"]["candidate"]["count"] == 0


@pytest.mark.parametrize("fault", ["session", "sequence", "clock"])
def test_training_rejects_source_ranges_that_contradict_actual_heldout_samples(tmp_path, fault):
    numeric, _ = candidates(tmp_path)
    data = json.loads((numeric / "dataset.json").read_bytes())
    heldout = json.loads((numeric / "evaluation.json").read_bytes())
    for group in heldout["groups"]:
        original = group["id"]
        group.update(id=original + "-renamed", split="train")
        for source in group["source_ranges"]:
            if fault == "session":
                source["session_sha256"] = "0" * 64
            elif fault == "sequence":
                source["start_sequence"] += 1000
                source["end_sequence"] += 1000
            else:
                source["first_ns"] += 100_000_000_000
                source["last_ns"] += 100_000_000_000
        for row in heldout["decisions"]:
            if row["group"] == original:
                row["group"] = group["id"]
    data["groups"] += heldout["groups"]
    data["decisions"] += heldout["decisions"]
    altered = numeric / "contradictory-training.json"
    altered.write_text(json.dumps(data))
    options = json.loads((tmp_path / "candidate.json").read_bytes())
    options.update(
        dataset=str(altered), dataset_sha256=hashlib.sha256(altered.read_bytes()).hexdigest()
    )
    config = tmp_path / "contradictory-config.json"
    config.write_text(json.dumps(options))
    with pytest.raises(ValueError, match="source range"):
        run_experiment(TemporalBCTrain(config, tmp_path / "contradictory-model"))
    assert not (tmp_path / "contradictory-model").exists()


@pytest.mark.parametrize("kind", ["old_source", "seen_holdout", "candidate_seen_holdout"])
def test_seen_or_incompatible_models_cannot_claim_fair_heldout_scores(tmp_path, kind):
    from fh5.collection.assessment import CollectionBCAssess

    numeric, models = candidates(tmp_path)
    if kind == "old_source":
        from tests.learning.bc.test_temporal_bc import temporal_fixture

        folder = tmp_path / "old"
        folder.mkdir()
        cfg, _ = temporal_fixture(folder)
        reason = "legacy_or_unknown_collection_source"
    else:
        data = json.loads((numeric / "dataset.json").read_bytes())
        heldout = json.loads((numeric / "evaluation.json").read_bytes())
        data["groups"] += [dict(g, split="train") for g in heldout["groups"]]
        data["decisions"] += heldout["decisions"]
        altered = numeric / "leaky-training.json"
        altered.write_text(json.dumps(data))
        options = json.loads((tmp_path / "baseline.json").read_bytes())
        options.update(
            dataset=str(altered), dataset_sha256=hashlib.sha256(altered.read_bytes()).hexdigest()
        )
        cfg = tmp_path / "leaky-config.json"
        cfg.write_text(json.dumps(options))
        reason = "baseline_seen_heldout"
    root = tmp_path / "incompatible"
    run_experiment(TemporalBCTrain(cfg, root))
    models["baseline"] = {
        "directory": str(root),
        "manifest_sha256": hashlib.sha256((root / "model.json").read_bytes()).hexdigest(),
    }
    if kind == "candidate_seen_holdout":
        models["candidate"], models["baseline"] = models["baseline"], None
        with pytest.raises(ValueError, match="overlaps"):
            run_experiment(
                CollectionBCAssess(
                    assess_config(tmp_path, numeric, models, "final"), tmp_path / "assessment"
                )
            )
        assert not (tmp_path / "assessment").exists()
        return
    result = run_experiment(
        CollectionBCAssess(
            assess_config(tmp_path, numeric, models, "final"), tmp_path / "assessment"
        )
    )
    summary = result.summary["collection_assessment"]
    assert summary["baseline"]["status"] == "incompatible"
    assert reason in summary["baseline"]["reasons"]
    assert all(row["baseline_prediction"] is None for row in summary["decisions"])
    assert summary["metrics"]["no_reference"]["candidate"]["count"] == 10
    assert summary["metrics"]["no_reference"]["baseline"]["count"] == 0
