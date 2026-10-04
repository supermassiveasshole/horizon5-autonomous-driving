"""Explicit legacy-source diagnostics keep the real model's training provenance."""

import hashlib
import json

import pytest

from fh5.cli import main
from fh5.driving.realtime.observation import ShadowNumericActor
from fh5.experiment import run_experiment
from fh5.learning.bc.training import TemporalBCTrain
from fh5.observation.numeric import NumericDecision, NumericInfer, PixelContract
from fh5.observation.recording import read_numeric_frame
from tests.driving.test_numeric_drive_cli import drive_config
from tests.learning.bc.test_temporal_bc import temporal_fixture

pytest.importorskip("torch")


def test_shadow_source_diagnostic_keeps_old_contract_and_uses_direct_numeric_pixels(
    tmp_path, capsys
):
    config, dataset = temporal_fixture(tmp_path)
    data = json.loads(dataset.read_text())
    data["pixel_contract"]["origin"] = "legacy_offline"
    dataset.write_text(json.dumps(data))
    options = json.loads(config.read_text())
    options["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    config.write_text(json.dumps(options))
    trained = run_experiment(TemporalBCTrain(config, tmp_path / "model"))
    metadata = json.loads((tmp_path / "model/model.json").read_text())
    digest = metadata["weights_sha256"]
    pixels = PixelContract(size=(64, 36))
    with pytest.raises(ValueError, match="source diagnostic"):
        ShadowNumericActor(tmp_path / "model", pixels, digest)
    model = ShadowNumericActor(
        tmp_path / "model", pixels, digest, allow_legacy_source_diagnostic=True
    )
    entry = data["decisions"][0]
    decision = NumericDecision(
        "native-diagnostic",
        entry["epoch"],
        entry["decision_ns"],
        tuple(read_numeric_frame(dataset.parent, f) for f in entry["frames"]),
        entry["views"]["no_reference"],
    )
    result = run_experiment(
        NumericInfer(tmp_path / "direct", pixels), numeric_actor=model, numeric_inputs=[decision]
    )
    assert (
        result.summary["numeric"]["decisions"][0]["prediction"]
        == trained.summary["temporal_bc"]["decisions"][0]["prediction"]
    )
    assert model.manifest["numeric_contract"]["origin"] == "direct_numeric"
    assert model.manifest["training_numeric_contract"]["origin"] == "legacy_offline"
    assert model.manifest["source_compatibility"] == "explicit_unvalidated_legacy_to_direct_shadow"
    assert (
        model.manifest["diagnostic_only"]
        and not model.manifest["new_capture_distribution_validated"]
    )
    assert metadata["numeric_contract"]["origin"] == "legacy_offline"
    with pytest.raises(ValueError, match="source diagnostic"):
        ShadowNumericActor(
            tmp_path / "model",
            PixelContract(size=(128, 72)),
            digest,
            allow_legacy_source_diagnostic=True,
        )
    with pytest.raises(ValueError, match="expected model"):
        ShadowNumericActor(
            tmp_path / "model", pixels, "0" * 64, allow_legacy_source_diagnostic=True
        )

    model_bytes = (tmp_path / "model/model.json").read_bytes()
    shared_config = drive_config(tmp_path, tmp_path / "model")
    capture_path = tmp_path / "capture.json"
    capture = json.loads(capture_path.read_text())
    capture["pixels"]["origin"] = "direct_numeric"
    capture_path.write_text(json.dumps(capture))
    output = tmp_path / "not-created"
    args = ["--config", str(shared_config), "--output", str(output)]
    capsys.readouterr()
    assert main(["realtime-shadow", *args]) == 2
    assert "source diagnostic" in json.loads(capsys.readouterr().err)["message"]
    assert main(["realtime-shadow", *args, "--allow-legacy-source-diagnostic"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["legacy_source_diagnostic"]
    assert report["training_pixel_contract"]["origin"] == "legacy_offline"
    assert report["configuration"]["pixels"]["origin"] == "direct_numeric"
    assert not report["devices_opened"] and not report["commands_sent_to_game"]
    assert main(["realtime-drive", *args]) == 2
    assert "source diagnostic" in json.loads(capsys.readouterr().err)["message"]
    assert not output.exists()
    assert (tmp_path / "model/model.json").read_bytes() == model_bytes
