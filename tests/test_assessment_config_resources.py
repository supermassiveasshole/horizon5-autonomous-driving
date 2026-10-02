"""Assessment configuration size never replaces schema and input validation."""

import hashlib
import json
import tracemalloc

import pytest

from fh5.collection_assessment import CollectionBCAssess
from fh5.experiment import run_experiment


def assessment_config(tmp_path, *, padding=0, extra=b""):
    config = {
        "version": 1,
        "mode": "final",
        "device": "cpu",
        "candidate": {"directory": "missing-model", "manifest_sha256": "0" * 64},
        "baseline": None,
        "dataset": "evaluation.json",
        "dataset_sha256": "0" * 64,
    }
    path = tmp_path / "assessment.json"
    with path.open("wb") as stream:
        stream.write(json.dumps(config).encode()[:-1] + extra + b"}")
        for _ in range(padding):
            stream.write(b" " * 1024**2)
    return path


def test_large_assessment_config_reaches_real_input_validation_without_retention(tmp_path):
    config = assessment_config(tmp_path, padding=2)
    output = tmp_path / "assessment"
    tracemalloc.start()
    try:
        with pytest.raises(FileNotFoundError) as missing:
            run_experiment(CollectionBCAssess(config, output))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # This deliberately stops before loading a model. A missing artifact is
    # still rejected after a valid configuration, regardless of JSON whitespace.
    assert missing.value.filename == str(tmp_path / "missing-model/model.json")
    assert peak < config.stat().st_size
    assert not output.exists()


@pytest.mark.parametrize("extra", [b',"unexpected":true', b',"unexpected":[1,]'])
def test_strict_assessment_projection_rejects_unknown_or_malformed_fields(tmp_path, extra):
    config = assessment_config(tmp_path, padding=2, extra=extra)
    output = tmp_path / "assessment"
    with pytest.raises(ValueError) as invalid:
        run_experiment(CollectionBCAssess(config, output))
    assert "bounded limit" not in str(invalid.value)
    assert not output.exists()


def test_assessment_rejects_changed_private_configuration_before_inputs(
    tmp_path, corrupt_private_reads
):
    config = assessment_config(tmp_path)
    original = config.read_bytes()
    output = tmp_path / "assessment"
    with corrupt_private_reads(config, b'"final"', b'"other"') as changed:
        with pytest.raises(ValueError, match="changed"):
            run_experiment(CollectionBCAssess(config, output))
    assert changed
    assert config.read_bytes() == original
    assert not output.exists()


def test_large_assessment_configuration_predicts_exactly_and_binds_all_bytes(tmp_path):
    from test_prediction_metrics import assessment_inputs

    config, _ = assessment_inputs(tmp_path)
    original = json.loads(config.read_bytes())
    model_file = tmp_path / "model/model.json"
    frozen = model_file.read_bytes()
    reference = run_experiment(CollectionBCAssess(config, tmp_path / "reference")).summary[
        "collection_assessment"
    ]
    with config.open("ab") as stream:
        for _ in range(2):
            stream.write(b" " * 1024**2)
    with config.open("rb") as stream:
        expected_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result = run_experiment(CollectionBCAssess(config, tmp_path / "assessment")).summary[
        "collection_assessment"
    ]
    assert result["config"] == original
    assert result["config_sha256"] == expected_hash
    assert result["config_sha256"] != reference["config_sha256"]
    assert result["decisions"] == reference["decisions"]
    assert result["metrics"] == reference["metrics"]
    assert result["verification"] == reference["verification"]
    assert model_file.read_bytes() == frozen
