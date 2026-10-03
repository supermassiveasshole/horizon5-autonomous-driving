"""One frozen configuration serves read-only shadow and guarded driving."""

import json

import pytest
from test_numeric_drive_cli import drive_config
from test_realtime_driving import numeric_driving_model as numeric_driving_model
from test_realtime_shadow import Capture, Desktop, Telemetry
from test_temporal_bc import inference_threads as inference_threads

from fh5.cli import main


def test_same_configuration_supports_shadow_then_driving_validation(
    tmp_path, numeric_driving_model, capsys
):
    config = drive_config(tmp_path, numeric_driving_model)
    original = config.read_bytes()
    shadow_output = tmp_path / "shadow-not-created"
    assert (
        main(
            [
                "realtime-shadow",
                "--config",
                str(config),
                "--output",
                str(shadow_output),
                "--hz",
                "10",
                "--seconds",
                "45",
            ]
        )
        == 0
    )
    shadow = json.loads(capsys.readouterr().out)
    assert shadow["status"] == "validated_only"
    assert not shadow["devices_opened"] and not shadow["commands_sent_to_game"]
    assert shadow["configuration"]["decision_hz"] == 10
    assert not shadow_output.exists()
    assert config.read_bytes() == original

    drive_output = tmp_path / "drive-not-created"
    assert main(["realtime-drive", "--config", str(config), "--output", str(drive_output)]) == 0
    driving = json.loads(capsys.readouterr().out)
    assert driving["status"] == "validated_only"
    assert not driving["devices_opened"] and not driving["commands_sent_to_game"]
    assert not driving["qualification"]["eligible"]
    assert "missing_shadow_evidence" in driving["qualification"]["reasons"]
    assert not drive_output.exists()
    assert config.read_bytes() == original


def test_shadow_rejects_changed_bound_model_before_opening_devices(
    tmp_path, numeric_driving_model, capsys, monkeypatch
):
    config = drive_config(tmp_path, numeric_driving_model)
    document = json.loads(config.read_text())
    document["model"]["manifest_sha256"] = "0" * 64
    config.write_text(json.dumps(document))

    def reject_device(*args, **kwargs):
        pytest.fail("Changed model must be rejected before constructing native adapters")

    monkeypatch.setattr("fh5.numeric_drive_config.UDPTelemetry", reject_device)
    monkeypatch.setattr("fh5.dxgi_windows.WindowsDXGIFrames", reject_device)
    monkeypatch.setattr("fh5.live.WindowsDesktop", reject_device)
    monkeypatch.setattr("fh5.live.XboxController", reject_device)
    monkeypatch.setattr("fh5.capture_resources.WindowsResources", reject_device)
    output = tmp_path / "not-created"
    assert (
        main(["realtime-shadow", "--config", str(config), "--output", str(output), "--live"]) == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert "manifest changed" in error["message"].lower()
    assert not output.exists()


def test_old_shadow_configuration_explains_shared_configuration_migration(
    tmp_path, numeric_driving_model, capsys
):
    config = drive_config(tmp_path, numeric_driving_model)
    document = json.loads(config.read_text())
    del document["shadow"]
    del document["model"]["manifest_sha256"]
    config.write_text(json.dumps(document))
    output = tmp_path / "not-created"
    assert main(["realtime-shadow", "--config", str(config), "--output", str(output)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "realtime-drive.example.json" in error["message"]
    assert "model.manifest_sha256" in error["message"]
    assert "shadow" in error["message"] and "null" in error["message"]
    assert not output.exists()


@pytest.mark.parametrize("inference_threads", [1], indirect=True)
def test_shadow_live_ignores_stale_evidence_without_constructing_a_controller(
    tmp_path, numeric_driving_model, inference_threads, capsys, monkeypatch
):
    config = drive_config(tmp_path, numeric_driving_model)
    document = json.loads(config.read_text())
    document["shadow"] = {
        "directory": "previous-shadow-no-longer-present",
        "manifest_sha256": "0" * 64,
    }
    config.write_text(json.dumps(document))
    original = config.read_bytes()
    capture, telemetry = Capture(), Telemetry()
    controller_attempts = []

    def reject_controller():
        controller_attempts.append(True)
        pytest.fail("Read-only shadow must never construct a controller")

    monkeypatch.setattr("fh5.dxgi_windows.WindowsDXGIFrames", lambda target: capture)
    monkeypatch.setattr("fh5.numeric_drive_config.UDPTelemetry", lambda port: telemetry)
    monkeypatch.setattr("fh5.live.WindowsDesktop", Desktop)
    monkeypatch.setattr("fh5.live.XboxController", reject_controller)
    monkeypatch.setattr("fh5.capture_resources.WindowsResources", lambda: None)
    output = tmp_path / "shadow"
    assert (
        main(
            [
                "realtime-shadow",
                "--config",
                str(config),
                "--output",
                str(output),
                "--seconds",
                "1.5",
                "--live",
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    accepted = [row for row in report["decisions"] if row["status"] == "accepted"]
    assert accepted and all(len(row["prediction"]) == 2 for row in accepted)
    assert report["evidence_kind"] == "shadow"
    assert report["stop_reason"] == "time_limit"
    assert report["resources_released"] and capture.closed and telemetry.closed
    assert not report["commands_sent_to_game"] and not controller_attempts
    assert not report["evidence"]["training_eligible"]
    assert config.read_bytes() == original
    assert not capsys.readouterr().err
