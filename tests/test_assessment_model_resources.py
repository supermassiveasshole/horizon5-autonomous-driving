"""Frozen assessment reads model evidence without retaining training diagnostics."""

import gc
import hashlib
import json
import shutil
import tracemalloc
from pathlib import Path

import pytest
from test_bc_manifest_resources import add_loss_history
from test_prediction_metrics import assessment_inputs

from fh5.collection_assessment import CollectionBCAssess
from fh5.experiment import run_experiment


def bind_model(config, model):
    with (model / "model.json").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    options = json.loads(config.read_bytes())
    for name in ("candidate", "baseline"):
        entry = options[name]
        if entry and (config.parent / entry["directory"]).resolve() == model.resolve():
            entry["manifest_sha256"] = digest
    config.write_text(json.dumps(options), encoding="utf-8")
    return digest


def test_assessment_does_not_retain_growing_frozen_model_loss_history(tmp_path, monkeypatch):
    config, _ = assessment_inputs(tmp_path)
    reference = run_experiment(CollectionBCAssess(config, tmp_path / "reference")).summary[
        "collection_assessment"
    ]
    model = tmp_path / "model"
    history_bytes = add_loss_history(model)
    manifest_hash = bind_model(config, model)
    output = tmp_path / "assessment"
    observed = []
    mkdir = Path.mkdir

    def observe_publication(path, *args, **kwargs):
        if path == output:
            observed.append(tracemalloc.get_traced_memory()[0])
        return mkdir(path, *args, **kwargs)

    gc.collect()
    with monkeypatch.context() as observation:
        observation.setattr(Path, "mkdir", observe_publication)
        tracemalloc.start()
        try:
            result = run_experiment(CollectionBCAssess(config, output)).summary[
                "collection_assessment"
            ]
        finally:
            tracemalloc.stop()
    assert observed and observed[0] < history_bytes, (
        "Assessment retains unused frozen training history",
        observed,
        history_bytes,
    )
    assert result["decisions"] == reference["decisions"]
    assert result["metrics"] == reference["metrics"]
    assert result["verification"] == reference["verification"]
    with (model / "model.json").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest_hash


def test_assessment_accepts_large_frozen_manifest_representation(tmp_path):
    config, _ = assessment_inputs(tmp_path)
    options = json.loads(config.read_bytes())
    options["baseline"] = dict(options["candidate"])
    config.write_text(json.dumps(options), encoding="utf-8")
    reference = run_experiment(CollectionBCAssess(config, tmp_path / "reference")).summary[
        "collection_assessment"
    ]
    model = tmp_path / "model"
    # Legal JSON whitespace crosses the former 128 MiB reader gate. This
    # isolates representation compatibility, not growing sample cardinality.
    with (model / "model.json").open("ab") as stream:
        for _ in range(129):
            stream.write(b" " * 1024**2)
    original_hash = bind_model(config, model)
    result = run_experiment(CollectionBCAssess(config, tmp_path / "assessment")).summary[
        "collection_assessment"
    ]
    assert result["baseline"]["status"] == "comparable"
    assert result["decisions"] == reference["decisions"]
    assert result["metrics"] == reference["metrics"]
    assert result["verification"] == reference["verification"]
    with (model / "model.json").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == original_hash


@pytest.mark.parametrize("entry", ["candidate", "baseline"])
def test_assessment_validates_discarded_manifest_fields_before_publication(tmp_path, entry):
    config, _ = assessment_inputs(tmp_path)
    model = tmp_path / "model"
    if entry == "baseline":
        baseline = tmp_path / "baseline"
        shutil.copytree(model, baseline)
        options = json.loads(config.read_bytes())
        options["baseline"] = {
            "directory": str(baseline),
            "manifest_sha256": options["candidate"]["manifest_sha256"],
        }
        config.write_text(json.dumps(options), encoding="utf-8")
        model = baseline
    path = model / "model.json"
    # The checksum matches the new bytes. A malformed field that is not used
    # by the evaluator must still fail complete JSON validation.
    path.write_bytes(path.read_bytes().rstrip()[:-1] + b',"unused_diagnostic":[0,]}')
    bind_model(config, model)
    output = tmp_path / "assessment"
    with pytest.raises(ValueError):
        run_experiment(CollectionBCAssess(config, output))
    assert not output.exists()
