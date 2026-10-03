"""Numerical driving configuration through CLI, without any native device access."""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from test_realtime_driving import Controller
from test_realtime_driving import numeric_driving_model as numeric_driving_model
from test_realtime_shadow import Capture, Desktop, Telemetry
from test_route_check import route
from test_temporal_bc import temporal_fixture

from fh5.cli import main
from fh5.experiment import run_experiment
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_drive_cli import native_driving_environment
from fh5.numeric_drive_config import NumericDriveConfiguration
from fh5.realtime_driving import NumericDrivingEnvironment
from fh5.realtime_model import ShadowNumericActor
from fh5.realtime_shadow import ShadowEnvironment
from fh5.temporal_bc import TemporalBCTrain


@pytest.fixture(scope="module")
def eligible_model(tmp_path_factory):
    # External native-source FILE fixture. It trains a real tiny actor, but is
    # deliberately not game evidence or a candidate for user driving.
    folder = tmp_path_factory.mktemp("simulated-native-files")
    config, snapshot = temporal_fixture(folder)
    data = json.loads(snapshot.read_text())
    conditions = {
        "version": 1,
        "id": "simulated-native-file-fixture",
        "status": "confirmed",
        "test_fixture": "simulated external files; NOT real capture qualification",
    }
    data["provenance"] = {
        "kind": "continuous_numeric_collection",
        "diagnostic_only": False,
        "input_conditions": conditions,
        "vehicle": {"expected_car_ordinal": 2941, "expected_pi": 999},
        "config": {"action_history_offsets_ms": [200, 100, 0], "max_action_age_ms": 200},
        "test_fixture": conditions["test_fixture"],
    }
    snapshot.write_text(json.dumps(data))
    training = json.loads(config.read_text())
    training["dataset_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    config.write_text(json.dumps(training))
    import torch

    torch.set_num_threads(1)
    run_experiment(TemporalBCTrain(config, folder / "model"))
    return folder / "model"


def drive_config(tmp_path, model):
    model_bytes = (model / "model.json").read_bytes()
    metadata = json.loads(model_bytes)
    capture = json.loads(
        (Path(__file__).parents[1] / "configs/capture-dxgi.example.json").read_text()
    )
    capture["pixels"] = metadata["numeric_contract"]
    if metadata.get("provenance", {}).get("input_conditions"):
        capture["input_conditions"] = metadata["provenance"]["input_conditions"]
        capture["target"]["condition_id"] = capture["input_conditions"]["id"]
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps(capture))
    task_file = route(tmp_path)
    path = tmp_path / "drive.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "capture_config": str(capture_path),
                "model": {
                    "directory": str(model),
                    "device": "cpu",
                },
                "task": {
                    "route_file": str(task_file),
                    "end_margin_m": 0.5,
                },
                "decision": {"reference_count": 1},
                "port": 5300,
                "shadow": None,
            }
        )
    )
    return path


def test_dry_run_reports_ineligible_candidate_without_opening_devices(
    tmp_path, numeric_driving_model, capsys
):
    config = drive_config(tmp_path, numeric_driving_model)
    output = tmp_path / "not-created"
    assert main(["realtime-drive", "--config", str(config), "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "validated_only"
    assert not report["devices_opened"] and not report["qualification"]["eligible"]
    assert "diagnostic_model" in report["qualification"]["reasons"]
    assert "missing_shadow_evidence" in report["qualification"]["reasons"]
    assert not output.exists()


def synthetic_shadow(tmp_path, model, config, *, native_file_fixture=False):
    plan = NumericDriveConfiguration(config, tmp_path / "shadow", 2, False)
    capture = Capture()
    if native_file_fixture:
        capture.source_kind = "dxgi"  # External source identity is simulated in this test only.
    bindings = dict(plan.bindings)
    bindings["shadow_config_sha256"] = bindings.pop("drive_config_sha256")
    env = ShadowEnvironment(
        plan.request,
        plan.capture,
        lambda: capture,
        Telemetry(),
        Desktop(),
        plan.task,
        input_conditions=bindings,
    )
    metadata = json.loads((model / "model.json").read_text())
    actor = ShadowNumericActor(model, plan.capture.pixels, metadata["weights_sha256"])
    actor.actor.torch.set_num_threads(1)
    run_experiment(plan.request, realtime_environment=env, numeric_actor_factory=lambda: actor)
    root = json.loads(config.read_text())
    root["shadow"] = {
        "directory": str(plan.request.output_dir),
    }
    config.write_text(json.dumps(root))


def test_synthetic_shadow_cannot_qualify_native_driving(tmp_path, numeric_driving_model, capsys):
    config = drive_config(tmp_path, numeric_driving_model)
    synthetic_shadow(tmp_path, numeric_driving_model, config)
    assert (
        main(["realtime-drive", "--config", str(config), "--output", str(tmp_path / "drive")]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert "shadow_not_native" in report["qualification"]["reasons"]
    assert "shadow_evidence_not_verified" not in report["qualification"]["reasons"]
    assert report["qualification"]["shadow"]["accepted_decisions"] > 0
    assert not report["qualification"]["eligible"]


def test_live_option_cannot_bypass_diagnostic_candidate(tmp_path, numeric_driving_model, capsys):
    config = drive_config(tmp_path, numeric_driving_model)
    output = tmp_path / "drive"
    assert main(["realtime-drive", "--config", str(config), "--output", str(output), "--live"]) == 2
    report = json.loads(capsys.readouterr().err)
    assert report["status"] == "error"
    assert "diagnostic_model" in report["message"]
    assert not output.exists()


@pytest.mark.parametrize("changed", ["shadow_manifest", "model_manifest", "weights", "route"])
def test_changed_bound_manifest_rejects_before_devices(
    tmp_path, numeric_driving_model, capsys, changed
):
    config = drive_config(tmp_path, numeric_driving_model)
    if changed == "shadow_manifest":
        synthetic_shadow(tmp_path, numeric_driving_model, config)
    root = json.loads(config.read_text())
    section, key = {
        "shadow_manifest": ("shadow", "manifest_sha256"),
        "model_manifest": ("model", "manifest_sha256"),
        "weights": ("model", "expected_sha256"),
        "route": ("task", "expected_route_sha256"),
    }[changed]
    root[section][key] = "0" * 64
    config.write_text(json.dumps(root))
    output = tmp_path / "drive"
    assert main(["realtime-drive", "--config", str(config), "--output", str(output), "--live"]) == 2
    report = json.loads(capsys.readouterr().err)
    assert any(word in report["message"].lower() for word in ("changed", "differs"))
    assert not output.exists()


@pytest.mark.parametrize(
    "changed,reason",
    [
        ("device", "shadow_inference_device_mismatch"),
        ("cadence", "shadow_decision_configuration_mismatch"),
    ],
)
def test_shadow_evidence_does_not_transfer_to_different_runtime(
    tmp_path, numeric_driving_model, capsys, changed, reason
):
    config = drive_config(tmp_path, numeric_driving_model)
    synthetic_shadow(tmp_path, numeric_driving_model, config)
    root = json.loads(config.read_text())
    if changed == "device":
        root["model"]["device"] = "cuda"
    else:
        root["decision"]["decision_hz"] = 10
    config.write_text(json.dumps(root))
    assert (
        main(["realtime-drive", "--config", str(config), "--output", str(tmp_path / "drive")]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert reason in report["qualification"]["reasons"]
    assert not report["qualification"]["eligible"]


@pytest.mark.parametrize("explicit_hashes", [False, True])
def test_qualified_file_fixture_reaches_guarded_native_adapter(
    tmp_path, eligible_model, explicit_hashes, monkeypatch
):
    config = drive_config(tmp_path, eligible_model)
    synthetic_shadow(tmp_path, eligible_model, config, native_file_fixture=True)
    if explicit_hashes:
        root = json.loads(config.read_text())
        model_bytes = (eligible_model / "model.json").read_bytes()
        root["model"].update(
            expected_sha256=json.loads(model_bytes)["weights_sha256"],
            manifest_sha256=hashlib.sha256(model_bytes).hexdigest(),
        )
        root["task"]["expected_route_sha256"] = hashlib.sha256(
            Path(root["task"]["route_file"]).read_bytes()
        ).hexdigest()
        root["shadow"]["manifest_sha256"] = hashlib.sha256(
            (tmp_path / "shadow/realtime-manifest.json").read_bytes()
        ).hexdigest()
        config.write_text(json.dumps(root))
    original = config.read_bytes()
    capture, telemetry, controller = Capture(), Telemetry(), Controller()
    monkeypatch.setattr("fh5.dxgi_windows.WindowsDXGIFrames", lambda target: capture)
    monkeypatch.setattr("fh5.numeric_drive_config.UDPTelemetry", lambda port: telemetry)
    monkeypatch.setattr("fh5.live.WindowsDesktop", Desktop)
    monkeypatch.setattr("fh5.live.XboxController", lambda: controller)
    monkeypatch.setattr("fh5.capture_resources.WindowsResources", lambda: None)
    plan = NumericDriveConfiguration(config, tmp_path / "drive", 1.5, True)
    shadow = plan.qualification["shadow"]
    assert (
        shadow["manifest_sha256"]
        == hashlib.sha256((tmp_path / "shadow/realtime-manifest.json").read_bytes()).hexdigest()
    )
    assert (
        plan.model_hash == hashlib.sha256((eligible_model / "model.json").read_bytes()).hexdigest()
    )
    env = native_driving_environment(plan)
    r = run_experiment(
        plan.request, realtime_environment=env, numeric_actor_factory=plan.actor
    ).summary["realtime"]
    assert r["environment"]["qualification"]["eligible"], r["stop_reason"]
    assert controller.active.is_set(), r["stop_reason"]
    assert r["commands_sent_to_game"] and r["resources_released"]
    assert controller.closed and telemetry.closed and capture.closed
    assert r["real_game_validation"] is False
    assert config.read_bytes() == original


@pytest.mark.parametrize(
    "changed,reason",
    [("model", "shadow_model_manifest_sha256_mismatch"), ("route", "shadow_task_mismatch")],
)
def test_replacing_selected_artifact_does_not_reuse_old_shadow(
    tmp_path, eligible_model, changed, reason
):
    copied = tmp_path / "model-copy"
    shutil.copytree(eligible_model, copied)
    config = drive_config(tmp_path, copied)
    synthetic_shadow(tmp_path, copied, config, native_file_fixture=True)
    previous = NumericDriveConfiguration(config, tmp_path / "drive", 1, True)
    assert previous.qualification["eligible"]
    path = copied / "model.json" if changed == "model" else previous.task.route_file
    # Equivalent JSON is still a different frozen artifact. No manual hashes are
    # present in this config; the old recording must retain its original binding.
    path.write_bytes(path.read_bytes() + b"\n")
    current = NumericDriveConfiguration(config, tmp_path / "drive", 1, True)
    assert not current.qualification["eligible"]
    assert reason in current.qualification["reasons"]


@pytest.mark.parametrize("changed", ["bindings", "capture", "request"])
def test_qualification_cannot_authorize_a_different_adapter(tmp_path, eligible_model, changed):
    config = drive_config(tmp_path, eligible_model)
    synthetic_shadow(tmp_path, eligible_model, config, native_file_fixture=True)
    plan = NumericDriveConfiguration(config, tmp_path / "drive", 1, True)
    capture, telemetry, controller = Capture(), Telemetry(), Controller()
    bindings = dict(plan.bindings)
    capture_config = plan.capture
    request = plan.request
    if changed == "bindings":
        bindings["inference_device"] = "cuda"
    elif changed == "capture":
        capture_config = replace(capture_config, capture_hz=30)
    else:
        request = replace(request, seconds=0.5)
    observations = ShadowEnvironment(
        request,
        capture_config,
        lambda: capture,
        telemetry,
        Desktop(),
        plan.task,
        input_conditions=bindings,
    )
    env = NumericDrivingEnvironment(observations, lambda: controller, configuration=plan)
    r = run_experiment(request, realtime_environment=env, numeric_actor_factory=plan.actor).summary[
        "realtime"
    ]
    assert "differs" in r["stop_reason"]
    assert not controller.commands and not capture.closed
    assert not r["environment"]["controller_created"]
    assert telemetry.closed and r["resources_released"]


def test_model_replacement_after_qualification_stops_before_capture(tmp_path, eligible_model):
    copied = tmp_path / "model-copy"
    shutil.copytree(eligible_model, copied)
    config = drive_config(tmp_path, copied)
    synthetic_shadow(tmp_path, copied, config, native_file_fixture=True)
    plan = NumericDriveConfiguration(config, tmp_path / "drive", 1, True)
    manifest = copied / "model.json"
    content = json.loads(manifest.read_text())
    content["unexpected_change"] = True
    manifest.write_text(json.dumps(content))
    capture, telemetry, controller = Capture(), Telemetry(), Controller()
    observations = ShadowEnvironment(
        plan.request,
        plan.capture,
        lambda: capture,
        telemetry,
        Desktop(),
        plan.task,
        input_conditions=plan.bindings,
    )
    env = NumericDrivingEnvironment(observations, lambda: controller, configuration=plan)
    r = run_experiment(
        plan.request, realtime_environment=env, numeric_actor_factory=plan.actor
    ).summary["realtime"]
    assert r["stop_reason"] == "model_startup_failed"
    assert "manifest changed" in r["inference"]["error"]
    assert not controller.commands and not capture.closed
    assert telemetry.closed and r["resources_released"]


@pytest.mark.parametrize("asset_kind", ["input", "pixels"])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_damaged_shadow_assets_cannot_qualify_driving(
    tmp_path, eligible_model, capsys, asset_kind, damage
):
    config = drive_config(tmp_path, eligible_model)
    synthetic_shadow(tmp_path, eligible_model, config, native_file_fixture=True)
    output = tmp_path / "drive"
    args = ["realtime-drive", "--config", str(config), "--output", str(output)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["qualification"]["eligible"]
    recording = tmp_path / "shadow"
    report = json.loads((recording / "report.json").read_text())
    row = next(d for d in report["decisions"] if d["status"] == "accepted")
    path = recording / row["archive"]["path"]
    if asset_kind == "pixels":
        path = recording / json.loads(path.read_text())["frames"][0]["path"]
    if damage == "missing":
        path.unlink()
    else:
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
    assert main(args) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "error"
    assert not output.exists()


def test_cpu_worker_cannot_use_cuda_timing_qualification(tmp_path, eligible_model):
    config = drive_config(tmp_path, eligible_model)
    synthetic_shadow(tmp_path, eligible_model, config, native_file_fixture=True)
    # Simulate saved CUDA evidence at the external file boundary. The executing
    # actor below remains a real CPU model; no CUDA or native device is opened.
    root = json.loads(config.read_text())
    root["model"]["device"] = "cuda"
    recording = tmp_path / "shadow"
    report = json.loads((recording / "report.json").read_text())
    report["environment"]["input_conditions"]["inference_device"] = "cuda"
    report["inference"]["inference_device"] = "cuda"
    payload = json.dumps(report).encode()
    (recording / "report.json").write_bytes(payload)
    manifest = recording / "realtime-manifest.json"
    data = json.loads(manifest.read_text())
    data["report_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest.write_text(json.dumps(data))
    root["shadow"]["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config.write_text(json.dumps(root))
    plan = NumericDriveConfiguration(config, tmp_path / "drive", 1, True)
    assert plan.qualification["eligible"]
    capture, telemetry, controller = Capture(), Telemetry(), Controller()
    observations = ShadowEnvironment(
        plan.request,
        plan.capture,
        lambda: capture,
        telemetry,
        Desktop(),
        plan.task,
        input_conditions=plan.bindings,
    )
    env = NumericDrivingEnvironment(observations, lambda: controller, configuration=plan)
    r = run_experiment(
        plan.request,
        realtime_environment=env,
        numeric_actor_factory=lambda: FrozenNumericActor(eligible_model, plan.capture.pixels),
    ).summary["realtime"]
    assert "inference device differs" in r["stop_reason"]
    assert not controller.commands and not capture.closed
    assert not r["environment"]["controller_created"]
    assert telemetry.closed and r["resources_released"]
