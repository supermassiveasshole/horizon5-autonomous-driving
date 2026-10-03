"""Actual frozen numerical BC at the experiment seam, with external devices simulated."""

import hashlib
import json
import threading
import time
from dataclasses import replace

import pytest

from fh5.capture.pipeline import CaptureConfig
from fh5.cli import main
from fh5.driving.realtime.driving import NumericDrivingEnvironment
from fh5.driving.realtime.model import RealtimeConfig, RealtimeNumericReplay, RealtimeRun
from fh5.driving.realtime.shadow import LocalTask, ShadowEnvironment
from fh5.driving.windows import NEUTRAL
from fh5.experiment import run_experiment
from fh5.learning.bc.actor import FrozenNumericActor
from fh5.learning.bc.training import TemporalBCTrain
from fh5.observation.numeric import PixelContract
from tests.driving.test_realtime_shadow import Capture, Desktop, Telemetry
from tests.learning.bc.test_temporal_bc import temporal_fixture
from tests.observation.test_route_check import route


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


def setup_drive(
    tmp_path,
    *,
    desktop=None,
    factory=None,
    source_kind="synthetic",
    capture=None,
    telemetry=None,
    seconds=1.5,
):
    task_file = route(tmp_path)
    task = LocalTask(
        task_file, hashlib.sha256(task_file.read_bytes()).hexdigest(), end_margin_m=0.5
    )
    request = RealtimeRun(
        tmp_path / "drive",
        RealtimeConfig(pixels=PixelContract(size=(64, 36)), reference_count=1),
        seconds=seconds,
    )
    capture, telemetry, controller = capture or Capture(), telemetry or Telemetry(), Controller()
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


def test_slow_device_creation_waits_for_fresh_input_then_drives(tmp_path, numeric_driving_model):
    controller = Controller()
    created_ns = []

    def create():
        time.sleep(0.15)
        created_ns.append(time.perf_counter_ns())
        return controller

    request, env, _, capture, telemetry = setup_drive(tmp_path, factory=create)
    r = run_experiment(
        request,
        realtime_environment=env,
        numeric_actor_factory=lambda: FrozenNumericActor(
            numeric_driving_model, request.config.pixels
        ),
    ).summary["realtime"]
    assert r["stop_reason"] == "time_limit"
    assert len(created_ns) == 1 and controller.active.is_set()
    assert controller.closed and capture.closed and telemetry.closed and r["resources_released"]
    accepted = [row for row in r["decisions"] if row["status"] == "accepted"]
    assert accepted
    for row in accepted:
        assert row["telemetry_received_ns"] > created_ns[0]
        assert row["frames"][-1]["source_time_ns"] > created_ns[0]
        assert row["inference_returned_ns"] < row["deadline_ns"]
        assert row["inference_returned_ns"] - row["frames"][-1]["source_time_ns"] <= 100_000_000
    assert all(row["issued_ns"] > created_ns[0] for row in r["commands"])
    assert controller.commands[-1] == NEUTRAL


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
