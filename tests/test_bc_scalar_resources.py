"""Explicit BC scalar settings respect the actual learner interface."""

import json
from dataclasses import replace

import pytest
from test_bc_loss_resources import loss_inputs

from fh5.experiment import run_experiment


@pytest.mark.parametrize(
    "field,value",
    [("seed", 2**32 + 7), ("seed", -(2**63)), ("seed", 2**64 - 1), ("learning_rate", 0.125)],
)
def test_bc_uses_explicit_scalar_settings_and_replays_real_updates(tmp_path, field, value):
    torch = pytest.importorskip("torch")
    request, replay = loss_inputs(tmp_path, steps=1)
    settings = json.loads(request.config_file.read_bytes())
    settings[field] = value
    request.config_file.write_text(json.dumps(settings), encoding="utf-8")
    trained = run_experiment(request).summary["temporal_bc"]
    assert trained["training"]["steps_completed"] == 1
    assert trained["training"]["time_gradient_l1"] > 0
    replayed = run_experiment(replay).summary["temporal_bc"]
    assert replayed["decisions"] == trained["decisions"]
    original = torch.load(request.output_dir / "actor.pt", weights_only=True)["actor"]
    repeated_dir = tmp_path / "repeated"
    repeated = run_experiment(replace(request, output_dir=repeated_dir)).summary["temporal_bc"]
    restored = torch.load(repeated_dir / "actor.pt", weights_only=True)["actor"]
    assert original.keys() == restored.keys()
    assert all(torch.equal(value, restored[key]) for key, value in original.items())
    assert repeated["decisions"] == trained["decisions"]
    manifest = json.loads((request.output_dir / "model.json").read_bytes())
    assert manifest["config"][field] == value
    if field == "learning_rate":
        settings[field] = 0.001
        request.config_file.write_text(json.dumps(settings), encoding="utf-8")
        other = tmp_path / "lower-learning-rate"
        run_experiment(replace(request, output_dir=other))
        lower = torch.load(other / "actor.pt", weights_only=True)["actor"]
        assert any(not torch.equal(value, lower[key]) for key, value in original.items())


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", -(2**63) - 1),
        ("seed", 2**64),
        ("seed", True),
        ("seed", 7.0),
        ("learning_rate", True),
        ("learning_rate", float("nan")),
        ("learning_rate", float("inf")),
        ("learning_rate", 0.0),
        ("learning_rate", -0.1),
    ],
)
def test_bc_rejects_invalid_scalar_interface_values_before_training(tmp_path, field, value):
    request, _ = loss_inputs(tmp_path, steps=1)
    settings = json.loads(request.config_file.read_bytes())
    settings[field] = value
    request.config_file.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError):
        run_experiment(request)
    assert not request.output_dir.exists()
