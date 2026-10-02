"""Exact temporal diagnostics through complete public training and replay runs."""

import hashlib
import json
import math
import sqlite3
import tracemalloc
from copy import deepcopy
from pathlib import Path

import pytest
from test_collection_assessment import assess_config
from test_collection_bc import prepare_inputs
from test_temporal_bc import temporal_fixture

from fh5.collection_assessment import CollectionBCAssess
from fh5.collection_bc import CollectionBCPrepare
from fh5.experiment import run_experiment
from fh5.temporal_bc import TemporalBCReplay, TemporalBCTrain


def metric_inputs(tmp_path):
    config, snapshot = temporal_fixture(tmp_path)
    document = json.loads(snapshot.read_bytes())
    template = document["decisions"][0]
    document["decisions"] = []
    document["groups"].append(
        {"id": "second-training-attempt", "split": "train", "evidence_id": "independent-fourth"}
    )

    def sample(index, group, target, *, variant=0, history=True, eligible=True, speed=10):
        row = deepcopy(template)
        row.update(
            decision_id=f"observation-{index}",
            group=group,
            epoch=group,
            variant=variant,
            bc_eligible=eligible,
        )
        row["decision_ns"] += index * 1_000_000_000
        row["supervision"]["action"] = target
        for frame in row["frames"]:
            frame["epoch"] = group
            frame["frame_id"] = f"{index}:{frame['frame_id']}"
            for field in ("source_time_ns", "capture_received_ns", "preprocess_ready_ns"):
                frame[field] += index * 1_000_000_000
        for actor in row["views"].values():
            actor["ego"]["speed_mps"] = speed
            if history:
                actor["actions"][-1] = [0.0, 0.0]
                actor["action_mask"][-1] = True
                actor["action_age_ms"][-1] = 10
        document["decisions"].append(row)

    # Ten nominal errors are 0/16 through 9/16 on both axes. Their sum is
    # 45/16, squared sum is 285/256, and nearest-rank p90 is the ninth: 8/16.
    for index in range(10):
        group = "attempt-0" if index < 5 else "second-training-attempt"
        sample(index, group, [index / 16, -index / 16], speed=0 if index < 2 else 10)
    sample(10, "attempt-0", [-0.5, 0.5], variant=1, speed=2)
    sample(11, "second-training-attempt", [-0.5, 0.5], variant=2, speed=20)
    # No history is available in the independent held-out observation. The
    # history ablation must therefore leave its prediction and errors intact.
    sample(12, "attempt-1", [0.25, 0.0], history=False)
    # This valid observation must appear in predictions but never in metrics.
    sample(13, "attempt-2", [-1.0, 1.0], eligible=False)
    snapshot.write_text(json.dumps(document), encoding="utf-8")
    options = json.loads(config.read_bytes())
    options.update(steps=1, dataset_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
    config.write_text(json.dumps(options), encoding="utf-8")
    return config, snapshot


def test_temporal_metrics_preserve_exact_percentiles_groups_and_history_baselines(tmp_path):
    config, snapshot = metric_inputs(tmp_path)
    model = tmp_path / "model"
    trained = run_experiment(TemporalBCTrain(config, model)).summary["temporal_bc"]
    empty = {"count": 0, "mae": None, "rmse": None, "p90_absolute_error": None}
    for view in ("no_reference", "reference_assisted"):
        nominal = trained["metrics"]["train"][view]["nominal"]
        assert nominal["count"] == 10
        assert nominal["attempt_groups"] == 2
        assert nominal["copy_recent_action"] == {
            "count": 10,
            "mae": [9 / 32, 9 / 32],
            "rmse": [math.sqrt(57 / 512), math.sqrt(57 / 512)],
            "p90_absolute_error": [0.5, 0.5],
        }
        assert {name: group["count"] for name, group in nominal["strata"].items()} == {
            "stationary": 2,
            "coast": 1,
            "brake": 9,
            "right": 6,
        }
        assert nominal["strata"]["stationary"]["attempt_groups"] == 1
        assert nominal["strata"]["right"]["attempt_groups"] == 2
        reselected = trained["metrics"]["train"][view]["reselected"]
        assert reselected["count"] == reselected["attempt_groups"] == 2
        assert reselected["copy_recent_action"] == {
            "count": 2,
            "mae": [0.5, 0.5],
            "rmse": [0.5, 0.5],
            "p90_absolute_error": [0.5, 0.5],
        }
        assert {name: group["count"] for name, group in reselected["strata"].items()} == {
            "startup": 1,
            "left": 2,
            "throttle": 2,
        }
        development = trained["metrics"]["development"][view]["nominal"]
        assert development["copy_recent_action"] == empty
        assert development["without_action_history"] == {key: development[key] for key in empty}
        for split, variant in (
            ("development", "reselected"),
            ("evaluation", "nominal"),
            ("evaluation", "reselected"),
        ):
            assert trained["metrics"][split][view][variant] == {
                **empty,
                "attempt_groups": 0,
                "strata": {},
                "copy_recent_action": empty,
                "without_action_history": empty,
            }
    assert len(trained["decisions"]) == 28
    replayed = run_experiment(
        TemporalBCReplay(model, snapshot, tmp_path / "replayed.html")
    ).summary["temporal_bc"]
    assert replayed["metrics"] == trained["metrics"]
    assert replayed["verification"]["status"] == "verified"


def assessment_inputs(tmp_path, *, audit_items=0):
    prepare, _ = prepare_inputs(tmp_path)
    numeric = tmp_path / "numeric"
    run_experiment(CollectionBCPrepare(prepare, numeric))
    dataset, heldout = numeric / "dataset.json", numeric / "evaluation.json"
    document = json.loads(heldout.read_bytes())
    audit_bytes = 0
    for position, row in enumerate(document["decisions"]):
        row["supervision"]["action"] = [position / 16, -position / 16]
        for actor in row["views"].values():
            actor["actions"][-1] = [0.0, 0.0]
            actor["action_mask"][-1] = True
            actor["action_age_ms"][-1] = 10
            actor["ego"]["speed_mps"] = 0 if position < 2 else 10
        if audit_items:
            # Distinct structured label-review evidence for each held-out
            # observation. It remains visible in the complete assessment.
            audit = [
                {
                    "review_sequence": index,
                    "observation": row["decision_id"],
                    "reviewed_action": row["supervision"]["action"],
                }
                for index in range(audit_items)
            ]
            row["supervision"]["label_audit"] = audit
            audit_bytes += len(json.dumps(audit).encode())
    heldout.write_text(json.dumps(document), encoding="utf-8")
    training = json.loads(dataset.read_bytes())
    training["provenance"]["final_dataset_sha256"] = hashlib.sha256(
        heldout.read_bytes()
    ).hexdigest()
    dataset.write_text(json.dumps(training), encoding="utf-8")
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": str(dataset),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "seed": 9,
                "steps": 1,
                "batch_size": 4,
                "learning_rate": 0.001,
                "device": "cpu",
                "time_mode": "actual",
            }
        )
    )
    model = tmp_path / "model"
    run_experiment(TemporalBCTrain(config, model))
    models = {
        "candidate": {
            "directory": str(model),
            "manifest_sha256": hashlib.sha256((model / "model.json").read_bytes()).hexdigest(),
        },
        "baseline": None,
    }
    return assess_config(tmp_path, numeric, models, "final"), audit_bytes


def test_assessment_keeps_complete_evidence_without_retaining_label_audit_in_memory(tmp_path):
    config, audit_bytes = assessment_inputs(tmp_path, audit_items=1024)
    output = tmp_path / "assessment"
    tracemalloc.start()
    try:
        result = run_experiment(CollectionBCAssess(config, output))
        resident = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    assert resident < audit_bytes, (
        "Returned assessment retains the complete structured label-audit corpus",
        resident,
        audit_bytes,
    )
    summary = result.summary["collection_assessment"]
    assert len(summary["decisions"]) == 20
    assert summary["verification"]["candidate_reload_max_abs_error"] <= 1e-6
    relocated = tmp_path / "relocated assessment"
    output.rename(relocated)
    # Returned evidence is independent of the closed input index and of where
    # the published report directory subsequently moves.
    first, last = summary["decisions"][0], summary["decisions"][-1]
    assert first["supervision"]["label_audit"][-1]["review_sequence"] == 1023
    assert last["supervision"]["label_audit"][0]["observation"] in last["decision_id"]
    assert json.loads((relocated / "report.json").read_bytes())["decisions"][-1] == last
    empty = {"count": 0, "mae": None, "rmse": None, "p90_absolute_error": None}
    for view in ("no_reference", "reference_assisted"):
        metrics = summary["metrics"][view]
        assert metrics["candidate"]["count"] == 10
        assert metrics["independent_groups"] == 1
        assert metrics["baseline"] == empty
        assert metrics["copy_recent_action"] == {
            "count": 10,
            "mae": [9 / 32, 9 / 32],
            "rmse": [math.sqrt(57 / 512), math.sqrt(57 / 512)],
            "p90_absolute_error": [0.5, 0.5],
        }
        assert {key: values["candidate"]["count"] for key, values in metrics["strata"].items()} == {
            "startup": 0,
            "left": 0,
            "right": 6,
            "release_rt": 0,
            "coast": 1,
            "brake": 9,
        }
        assert metrics["strata"]["startup"] == {
            "candidate": empty,
            "baseline": empty,
            "copy_recent_action": empty,
            "independent_groups": 0,
        }


@pytest.mark.parametrize("failure", [sqlite3.OperationalError, MemoryError])
def test_assessment_metric_capacity_failure_retains_verified_predictions(
    tmp_path, monkeypatch, failure
):
    config, _ = assessment_inputs(tmp_path)
    connect = sqlite3.connect

    def no_metric_capacity(database, *args, **kwargs):
        if Path(database).parent.name.startswith("fh5-assessment-metrics-"):
            raise failure("metric storage exhausted")
        return connect(database, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(sqlite3, "connect", no_metric_capacity)
        result = run_experiment(CollectionBCAssess(config, tmp_path / "assessment"))
    summary = result.summary["collection_assessment"]
    assert summary["metrics"]["status"] == "unavailable"
    assert "metric storage exhausted" in summary["metrics"]["error"]
    assert summary["verification"]["candidate_reload_max_abs_error"] <= 1e-6
    assert len(summary["decisions"]) == 20
    evidence = json.loads((tmp_path / "assessment/report.json").read_bytes())
    assert evidence["metrics"] == summary["metrics"]
    assert evidence["decisions"] == summary["decisions"]
    rebuilt = run_experiment(CollectionBCAssess(config, tmp_path / "rebuilt")).summary[
        "collection_assessment"
    ]
    assert rebuilt["decisions"] == summary["decisions"]
    assert rebuilt["metrics"]["no_reference"]["candidate"]["count"] == 10
