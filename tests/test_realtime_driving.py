"""Actual frozen numerical BC at the experiment seam, with external devices simulated."""

import hashlib
import json
import threading
import time
from dataclasses import replace

import pytest
from test_realtime_shadow import Capture, Desktop, Telemetry
from test_route_check import route
from test_temporal_bc import temporal_fixture

from fh5.capture import CaptureConfig
from fh5.cli import main
from fh5.experiment import run_experiment
from fh5.live import NEUTRAL
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig, RealtimeNumericReplay, RealtimeRun
from fh5.realtime_driving import NumericDrivingEnvironment
from fh5.realtime_shadow import LocalTask, ShadowEnvironment
from fh5.temporal_bc import TemporalBCTrain


@pytest.fixture(scope="module")
def numeric_driving_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("driving-model")
    config, _ = temporal_fixture(root)
    run_experiment(TemporalBCTrain(config, root / "trained"))
    return root / "trained"


class Controller:
    def __init__(self):
        self.commands = []
        self.closed = False
        self.active = threading.Event()

    def send(self, command):
        self.commands.append(command)
        if command != NEUTRAL:
            self.active.set()

    def close(self):
        self.closed = True


def setup_drive(tmp_path, *, desktop=None, factory=None, source_kind="synthetic"):
    task_file = route(tmp_path)
    task = LocalTask(
        task_file, hashlib.sha256(task_file.read_bytes()).hexdigest(), end_margin_m=0.5
    )
    request = RealtimeRun(
        tmp_path / "drive",
        RealtimeConfig(pixels=PixelContract(size=(64, 36)), reference_count=1),
        seconds=1.5,
    )
    capture, telemetry, controller = Capture(), Telemetry(), Controller()
    observations = ShadowEnvironment(
        request,
        CaptureConfig(pixels=request.config.pixels),
        lambda: capture,
        telemetry,
        desktop or Desktop(),
        task,
    )
    environment = NumericDrivingEnvironment(
        observations, factory or (lambda: controller), source_kind=source_kind
    )
    return request, environment, controller, capture, telemetry


def test_frozen_bc_drives_bounded_commands_and_replays_without_devices(
    tmp_path, numeric_driving_model, monkeypatch
):
    from PIL import Image

    request, environment, controller, capture, telemetry = setup_drive(tmp_path)
    actor = FrozenNumericActor(numeric_driving_model, request.config.pixels)
    actor.torch.set_num_threads(1)

    def reject_encoded_images(*args, **kwargs):
        pytest.fail("Numerical driving and replay must not decode JPEG/PNG image files")

    monkeypatch.setattr(Image, "open", reject_encoded_images)
    r = run_experiment(
        request, realtime_environment=environment, numeric_actor_factory=lambda: actor
    ).summary["realtime"]
    assert controller.active.is_set(), r["stop_reason"]
    assert controller.commands[-1] == NEUTRAL
    assert controller.closed and capture.closed and telemetry.closed and r["resources_released"]
    assert r["environment"]["mode"] == "numeric_driving"
    assert r["commands_sent_to_game"] is False
    assert r["real_game_validation"] is False
    decisions = [d for d in r["decisions"] if d["status"] == "accepted"]
    assert decisions and any(any(d["actor"]["action_mask"]) for d in decisions)
    assert r["metrics"]["source_to_send_return_ms"]["count"] == len(decisions)
    assert (
        r["metrics"]["source_to_send_return_ms"]["max"]
        >= r["metrics"]["source_to_sendable_ms"]["max"]
    )
    assert all(abs(c.steer_i16) <= round(0.4 * 32767) for c in controller.commands)
    assert all(c.throttle_u8 <= round(0.25 * 255) for c in controller.commands)
    assert all(c.brake_u8 <= round(0.5 * 255) for c in controller.commands)
    replay = run_experiment(
        RealtimeNumericReplay(request.output_dir, tmp_path / "replay.html"),
        numeric_actor=FrozenNumericActor(numeric_driving_model, request.config.pixels),
    ).summary["realtime_numeric_replay"]
    assert replay["verified"], replay["errors"]
    assert (
        main(
            [
                "realtime-replay",
                str(request.output_dir),
                "--model",
                str(numeric_driving_model),
                "--report",
                str(tmp_path / "cli-replay.html"),
            ]
        )
        == 0
    )


@pytest.mark.parametrize("signal", ["stop", "focus"])
def test_desktop_stop_releases_actual_model_commands(tmp_path, numeric_driving_model, signal):
    controller = Controller()

    class Signals:
        def focused(self):
            return not (signal == "focus" and controller.active.is_set())

        def stop_requested(self):
            return signal == "stop" and controller.active.is_set()

    request, env, _, capture, telemetry = setup_drive(
        tmp_path, desktop=Signals(), factory=lambda: controller
    )
    r = run_experiment(
        request,
        realtime_environment=env,
        numeric_actor_factory=lambda: FrozenNumericActor(
            numeric_driving_model, request.config.pixels
        ),
    ).summary["realtime"]
    assert controller.active.is_set()
    assert controller.commands[-1] == NEUTRAL
    assert controller.closed and capture.closed and telemetry.closed and r["resources_released"]
    assert r["stop_reason"] != "time_limit"
    assert sum(c != NEUTRAL for c in controller.commands) == 1


def test_slow_device_creation_never_sends_an_expired_prediction(tmp_path, numeric_driving_model):
    controller = Controller()

    def create():
        time.sleep(0.15)
        return controller

    request, env, _, capture, telemetry = setup_drive(tmp_path, factory=create)
    r = run_experiment(
        request,
        realtime_environment=env,
        numeric_actor_factory=lambda: FrozenNumericActor(
            numeric_driving_model, request.config.pixels
        ),
    ).summary["realtime"]
    assert not controller.active.is_set()
    assert controller.closed and capture.closed and telemetry.closed and r["resources_released"]
    assert "stale_telemetry" in r["stop_reason"]
    assert r["environment"]["controller_sends"] == len(controller.commands) == 1


def test_native_control_requires_explicit_opt_in_before_opening_devices(
    tmp_path, numeric_driving_model
):
    request, env, controller, capture, telemetry = setup_drive(tmp_path, source_kind="native")
    with pytest.raises(ValueError, match="explicit live"):
        run_experiment(
            request,
            realtime_environment=env,
            numeric_actor_factory=lambda: FrozenNumericActor(
                numeric_driving_model, request.config.pixels
            ),
        )
    assert not request.output_dir.exists()
    assert not controller.commands and not capture.closed and not telemetry.closed


def test_synthetic_model_is_rejected_before_native_capture_or_controller(
    tmp_path, numeric_driving_model
):
    request, env, controller, capture, telemetry = setup_drive(tmp_path, source_kind="native")
    request = replace(request, live=True)
    r = run_experiment(
        request,
        realtime_environment=env,
        numeric_actor_factory=lambda: FrozenNumericActor(
            numeric_driving_model, request.config.pixels
        ),
    ).summary["realtime"]
    assert "Native driving requires" in r["stop_reason"]
    assert not controller.commands and not controller.closed and not capture.closed
    assert telemetry.closed and r["commands_sent_to_game"] is False
    assert not r["environment"]["controller_created"]


def test_failed_driver_release_is_not_reported_as_released(tmp_path, numeric_driving_model):
    class FailingController(Controller):
        def send(self, command):
            super().send(command)
            raise OSError("driver write failed")

        def close(self):
            raise OSError("driver detach failed")

    controller = FailingController()
    request, env, _, capture, telemetry = setup_drive(tmp_path, factory=lambda: controller)
    r = run_experiment(
        request,
        realtime_environment=env,
        numeric_actor_factory=lambda: FrozenNumericActor(
            numeric_driving_model, request.config.pixels
        ),
    ).summary["realtime"]
    assert controller.active.is_set()
    assert not r["resources_released"]
    assert not r["environment"]["controller"]["resources_released"]
    assert not r["evidence"]["recording_complete"]
    assert capture.closed and telemetry.closed


def test_native_recording_replay_never_reexecutes_driver_calls(tmp_path, numeric_driving_model):
    request, env, controller, _, _ = setup_drive(tmp_path)
    actor = FrozenNumericActor(numeric_driving_model, request.config.pixels)
    actor.torch.set_num_threads(1)
    run_experiment(request, realtime_environment=env, numeric_actor_factory=lambda: actor)
    # External file fixture: model and recorded predictions remain real, while
    # the container represents a native recording. This is NOT live evidence.
    path = request.output_dir / "report.json"
    report = json.loads(path.read_text())
    report.update(evidence_kind="native", commands_sent_to_game=True)
    payload = json.dumps(report).encode()
    path.write_bytes(payload)
    (request.output_dir / "realtime-manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "report_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    )
    before = list(controller.commands)
    replay = run_experiment(
        RealtimeNumericReplay(request.output_dir, tmp_path / "native-replay.html"),
        numeric_actor=FrozenNumericActor(numeric_driving_model, request.config.pixels),
    ).summary["realtime_numeric_replay"]
    assert replay["verified"], replay["errors"]
    assert replay["source_evidence_kind"] == "native"
    assert replay["commands_sent_to_game"] is False
    assert controller.commands == before and controller.closed


def test_device_acquisition_failure_retains_failed_release_evidence(
    tmp_path, numeric_driving_model, monkeypatch
):
    acquired = threading.Event()
    thread_start = threading.Thread.start

    def unavailable_after_acquisition(thread):
        if acquired.is_set():
            raise OSError("OS thread capacity exhausted")
        thread_start(thread)

    class FailingDetach(Controller):
        def close(self):
            raise OSError("device still attached")

    controller = FailingDetach()

    def acquire():
        acquired.set()
        return controller

    request, env, _, capture, telemetry = setup_drive(tmp_path, factory=acquire)
    monkeypatch.setattr(threading.Thread, "start", unavailable_after_acquisition)
    r = run_experiment(
        request,
        realtime_environment=env,
        numeric_actor_factory=lambda: FrozenNumericActor(
            numeric_driving_model, request.config.pixels
        ),
    ).summary["realtime"]
    assert acquired.is_set() and not controller.active.is_set()
    assert not r["resources_released"]
    assert r["environment"]["controller_created"]
    assert "device still attached" in r["environment"]["controller"]["close_error"]
    assert capture.closed and telemetry.closed
