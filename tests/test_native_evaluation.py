"""Native orchestration with real models and explicitly simulated external I/O."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_evaluation import sha
from test_evaluation_run import Menu
from test_evaluation_start import automatic_request
from test_numeric_drive_cli import drive_config, synthetic_shadow
from test_numeric_drive_cli import eligible_model as eligible_model
from test_realtime_driving import Controller
from test_realtime_shadow import Capture, Desktop, Telemetry

from fh5.cli import main
from fh5.evaluation import EvaluationPrepare
from fh5.experiment import run_experiment
from fh5.realtime_driving import NumericDrivingEnvironment
from fh5.realtime_shadow import ShadowEnvironment


def native_request(tmp_path, model):
    operation = automatic_request(tmp_path, model)
    path = tmp_path / "evaluation.json"
    config = json.loads(path.read_bytes())
    config["version"] = 3
    config["model"].update(kind="bc", device="cpu")
    metadata = json.loads((model / "model.json").read_bytes())
    config["conditions"]["numeric_input_conditions"] = metadata["provenance"]["input_conditions"]
    path.write_text(json.dumps(config))
    batch = tmp_path / "native-batch"
    run_experiment(EvaluationPrepare(path, batch))
    return replace(operation, batch_dir=batch, batch_sha256=sha(batch / "batch.json"))


def test_native_batch_freezes_numeric_conditions_and_execution_device(tmp_path, eligible_model):
    operation = native_request(tmp_path, eligible_model)
    batch = json.loads((operation.batch_dir / "batch.json").read_bytes())
    assert batch["kind"] == "frozen-local-evaluation-v3"
    assert batch["config"]["model"]["device"] == "cpu"
    assert batch["model_diagnostic_only"] is False
    assert batch["config"]["conditions"]["numeric_input_conditions"]["test_fixture"] == (
        "simulated external files; NOT real capture qualification"
    )
    assert not operation.output_dir.exists()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_comparison_requires_the_same_frozen_device(tmp_path, eligible_model, device):
    from fh5.candidate_selection import CandidateCompare

    operation = native_request(tmp_path, eligible_model)
    config_file = tmp_path / "evaluation.json"
    config = json.loads(config_file.read_bytes())
    config["model"]["device"] = device
    config_file.write_text(json.dumps(config))
    candidate = tmp_path / "candidate-batch"
    # Configuration declaration only: no CUDA model/device is opened.
    run_experiment(EvaluationPrepare(config_file, candidate))
    bindings = {}
    for side, batch in (("incumbent", operation.batch_dir), ("candidate", candidate)):
        ledger = tmp_path / f"{side}-ledger.json"
        digest = sha(batch / "batch.json")
        ledger.write_text(json.dumps({"version": 1, "batch_sha256": digest, "entries": []}))
        bindings[side] = {
            "batch": str(batch),
            "batch_sha256": digest,
            "ledger": str(ledger),
            "ledger_sha256": sha(ledger),
        }
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"version": 1, **bindings}))
    output = tmp_path / "comparison"
    request = CandidateCompare(comparison, output)
    if device == "cuda":
        with pytest.raises(ValueError, match="conditions"):
            run_experiment(request)
        assert not output.exists()
    else:
        result = run_experiment(request).summary["candidate_selection"]
        assert result["conditions"]["inference_device"] == "cpu"
        assert result["promotion_allowed"] is False


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_version_three_cannot_silently_use_the_legacy_synthetic_runner(
    tmp_path, eligible_model, device
):
    from test_evaluation_run import Batch

    operation = native_request(tmp_path, eligible_model)
    path = tmp_path / "evaluation.json"
    config = json.loads(path.read_bytes())
    config["model"]["device"] = device
    path.write_text(json.dumps(config))
    batch = tmp_path / "declared-device"
    run_experiment(EvaluationPrepare(path, batch))
    operation = replace(operation, batch_dir=batch, batch_sha256=sha(batch / "batch.json"))
    environment = Batch()
    with pytest.raises(ValueError, match="native"):
        run_experiment(operation, evaluation_environment=environment)
    assert not environment.menus and not environment.drives and not operation.output_dir.exists()


def prepared_drive(tmp_path, model, operation):
    folder = tmp_path / "drive-qualification"
    folder.mkdir()
    path = drive_config(folder, model)
    config = json.loads(path.read_bytes())
    batch = json.loads((operation.batch_dir / "batch.json").read_bytes())
    config["decision"] = {k: v for k, v in batch["config"]["runtime"].items() if k != "pixels"}
    config["task"]["route_file"] = str(operation.batch_dir / "route/route.json")
    config["task"]["expected_route_sha256"] = sha(operation.batch_dir / "route/route.json")
    path.write_text(json.dumps(config))
    synthetic_shadow(folder, model, path, native_file_fixture=True)
    return path


class ExternalDevices:
    """Explicit simulated native I/O identity, never real Windows resources."""

    def __init__(self, fail_restart=False):
        self.menus, self.controllers, self.captures, self.telemetry = [], [], [], []
        self.fail_restart = fail_restart

    def menu(self, config_file: Path, port: int):
        menu = Menu(bool(self.menus), bool(self.menus) and self.fail_restart)
        menu.source_kind = "udp"
        self.menus.append(menu)
        return menu

    def drive(self, plan):
        capture, telemetry, controller = Capture(), Telemetry(), Controller()
        self.captures.append(capture)
        self.telemetry.append(telemetry)
        self.controllers.append(controller)
        observations = ShadowEnvironment(
            plan.request,
            plan.capture,
            lambda: capture,
            telemetry,
            Desktop(),
            plan.task,
            input_conditions=plan.bindings,
        )
        return NumericDrivingEnvironment(observations, lambda: controller, configuration=plan)


def test_native_batch_runs_two_qualified_attempts_and_preserves_full_evidence(
    tmp_path, eligible_model
):
    from fh5.evaluation_native import NativeEvaluationEnvironment

    operation = native_request(tmp_path, eligible_model)
    config = prepared_drive(tmp_path, eligible_model, operation)
    operation = replace(operation, live=True, seconds=1.2)
    devices = ExternalDevices()
    environment = NativeEvaluationEnvironment(
        config, menu_factory=devices.menu, driving_factory=devices.drive
    )
    result = run_experiment(operation, evaluation_environment=environment)
    summary = result.summary["evaluation_run"]
    assert summary["stop_reason"] == "plan_complete", summary
    assert summary["started_slots"] == ["run-0", "run-1"]
    assert summary["commands_sent_to_game"] and summary["resources_released"]
    assert result.summary["evaluation"]["execution_metrics"]["bound_runs"] == 2
    assert result.summary["evaluation"]["verified_starts"] == 2
    assert result.summary["evaluation"]["automatic_promotion_allowed"] is False
    assert devices.menus[1].pulses == ["START", "X", "A", "A"]
    assert all(c.active.is_set() and c.closed for c in devices.controllers)
    assert all(m.closed for m in devices.menus)
    assert all(c.closed for c in devices.captures + devices.telemetry)


def test_native_evaluation_cli_checks_without_opening_devices(tmp_path, eligible_model, capsys):
    operation = native_request(tmp_path, eligible_model)
    config = prepared_drive(tmp_path, eligible_model, operation)
    assert (
        main(
            [
                "evaluation-run",
                "--batch",
                str(operation.batch_dir),
                "--batch-sha256",
                operation.batch_sha256,
                "--event-config",
                str(operation.event_config_file),
                "--driving-config",
                str(config),
                "--output",
                str(operation.output_dir),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "validated_only" and not report["devices_opened"]
    assert report["planned_runs"] == 2
    assert not operation.output_dir.exists()


@pytest.mark.parametrize("fault", ["no_optin", "too_long", "legacy_batch", "changed_conditions"])
def test_unqualified_native_batch_never_acquires_menu_or_driving_devices(
    tmp_path, eligible_model, fault
):
    from fh5.evaluation_native import NativeEvaluationEnvironment

    operation = native_request(tmp_path, eligible_model)
    config = prepared_drive(tmp_path, eligible_model, operation)
    operation = replace(operation, live=True)
    if fault == "no_optin":
        operation = replace(operation, live=False)
    elif fault == "too_long":
        operation = replace(operation, seconds=31)
    elif fault == "legacy_batch":
        operation = replace(
            operation,
            batch_dir=tmp_path / "automatic-batch",
            batch_sha256=sha(tmp_path / "automatic-batch/batch.json"),
        )
    else:
        root = json.loads(config.read_bytes())
        capture = Path(root["capture_config"])
        document = json.loads(capture.read_bytes())
        document["input_conditions"]["changed_hud"] = True
        capture.write_text(json.dumps(document))
    devices = ExternalDevices()
    environment = NativeEvaluationEnvironment(
        config, menu_factory=devices.menu, driving_factory=devices.drive
    )
    with pytest.raises(ValueError):
        run_experiment(operation, evaluation_environment=environment)
    assert not devices.menus and not devices.controllers and not operation.output_dir.exists()


def test_native_restart_failure_preserves_first_attempt_without_opening_next_driver(
    tmp_path, eligible_model
):
    from fh5.evaluation_native import NativeEvaluationEnvironment

    operation = native_request(tmp_path, eligible_model)
    config = prepared_drive(tmp_path, eligible_model, operation)
    operation = replace(operation, live=True, seconds=1.2)
    devices = ExternalDevices(fail_restart=True)
    environment = NativeEvaluationEnvironment(
        config, menu_factory=devices.menu, driving_factory=devices.drive
    )
    result = run_experiment(operation, evaluation_environment=environment)
    summary = result.summary["evaluation_run"]
    assert summary["stop_reason"] == "ready_unconfirmed"
    assert summary["started_slots"] == ["run-0"] and summary["unstarted_slots"] == ["run-1"]
    assert result.summary["evaluation"]["metrics"]["all_attempts"] == 1
    assert len(devices.controllers) == 1 and summary["resources_released"]
    assert all(m.closed for m in devices.menus)


def test_native_configuration_change_between_attempts_stops_before_next_menu(
    tmp_path, eligible_model
):
    from fh5.evaluation_native import NativeEvaluationEnvironment

    operation = native_request(tmp_path, eligible_model)
    config = prepared_drive(tmp_path, eligible_model, operation)
    operation = replace(operation, live=True, seconds=1.2)

    class ChangingDevices(ExternalDevices):
        def drive(self, plan):
            environment = super().drive(plan)
            controller = self.controllers[-1]
            previous_close = controller.close

            def close():
                previous_close()
                original = json.loads(config.read_bytes())
                original["port"] = 5400
                config.write_text(json.dumps(original))

            controller.close = close  # The external device callback changes an external file.
            return environment

    devices = ChangingDevices()
    environment = NativeEvaluationEnvironment(
        config, menu_factory=devices.menu, driving_factory=devices.drive
    )
    result = run_experiment(operation, evaluation_environment=environment)
    summary = result.summary["evaluation_run"]
    assert summary["stop_reason"] == "interface_error"
    assert "configuration changed" in summary["error"]
    assert summary["started_slots"] == ["run-0"] and summary["unstarted_slots"] == ["run-1"]
    assert len(devices.menus) == len(devices.controllers) == 1
    assert (
        summary["resources_released"]
        and result.summary["evaluation"]["metrics"]["all_attempts"] == 1
    )


@pytest.mark.parametrize("fault", ["closing", "close_error"])
def test_native_menu_capture_must_release_before_a_driving_lease(tmp_path, eligible_model, fault):
    import threading

    from fh5.evaluation_native import NativeEvaluationEnvironment
    from fh5.live_event import BoundedFrames

    unblocked, finished = threading.Event(), threading.Event()

    class Source:
        frame = None

        def capture(self):
            return self.frame

        def close(self):
            try:
                if fault == "closing":
                    unblocked.wait(60)
                else:
                    raise OSError("external capture close failed")
            finally:
                finished.set()

    class ClosingMenu(Menu):
        source_kind = "udp"

        def __init__(self):
            super().__init__(False)
            self.source = Source()
            self.frames = BoundedFrames(lambda: self.source, confirm_release=True)

        def read(self, period_s):
            observed = super().read(period_s)
            self.source.frame = observed.frame
            return replace(observed, frame=self.frames.capture())

        def close(self):
            super().close()
            self.frames.close()

    class Devices(ExternalDevices):
        def menu(self, path, plan):
            result = ClosingMenu()
            self.menus.append(result)
            return result

    operation = replace(native_request(tmp_path, eligible_model), live=True, seconds=1.2)
    config = prepared_drive(tmp_path, eligible_model, operation)
    devices = Devices()
    environment = NativeEvaluationEnvironment(
        config, menu_factory=devices.menu, driving_factory=devices.drive
    )
    try:
        result = run_experiment(operation, evaluation_environment=environment)
    finally:
        unblocked.set()
        assert finished.wait(2)
    summary = result.summary["evaluation_run"]
    assert not summary["resources_released"]
    assert summary["stop_reason"] == "interface_error"
    assert not devices.controllers
    assert summary["started_slots"] == []
    assert summary["unstarted_slots"] == ["run-0", "run-1"]


def test_native_runner_accepts_a_qualified_environment_contract(tmp_path, eligible_model):
    from fh5.evaluation_native import NativeEvaluationEnvironment

    operation = replace(native_request(tmp_path, eligible_model), live=True, seconds=1.2)
    config = prepared_drive(tmp_path, eligible_model, operation)
    devices = ExternalDevices()
    qualified = NativeEvaluationEnvironment(
        config, menu_factory=devices.menu, driving_factory=devices.drive
    )

    class EnvironmentContract:
        source_kind = "native"

        def prepare(self, request, batch):
            return qualified.prepare(request, batch)

        def event(self, slot):
            return qualified.event(slot)

        def driving(self, slot, ready):
            return qualified.driving(slot, ready)

        def close(self):
            return qualified.close()

    result = run_experiment(operation, evaluation_environment=EnvironmentContract())
    assert result.summary["evaluation_run"]["stop_reason"] == "plan_complete"
    assert result.summary["evaluation"]["execution_metrics"]["bound_runs"] == 2


@pytest.mark.parametrize("interrupted", [False, True])
def test_failed_menu_acquisition_does_not_claim_all_resources_released(
    tmp_path, eligible_model, interrupted
):
    from fh5.evaluation_native import NativeEvaluationEnvironment

    class Devices(ExternalDevices):
        def menu(self, path, plan):
            if interrupted:
                raise KeyboardInterrupt("interrupted external capture allocation")
            raise OSError("external capture allocation failed; cleanup status unknown")

    operation = replace(native_request(tmp_path, eligible_model), live=True)
    config = prepared_drive(tmp_path, eligible_model, operation)
    devices = Devices()
    result = run_experiment(
        operation,
        evaluation_environment=NativeEvaluationEnvironment(
            config, menu_factory=devices.menu, driving_factory=devices.drive
        ),
    )
    summary = result.summary["evaluation_run"]
    assert summary["stop_reason"] == ("user_stop" if interrupted else "interface_error")
    assert not summary["resources_released"]
    assert not devices.controllers and not summary["started_slots"]
