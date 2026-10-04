"""Device preparation through the experiment seam and real transport wrappers."""

import json
import socket
import threading
import time

import pytest

from fh5.driving.realtime.udp import UDPTelemetry
from fh5.driving.windows import NEUTRAL
from fh5.evaluation.handoff import ReadyHandoff
from fh5.experiment import run_experiment
from fh5.learning.bc.actor import FrozenNumericActor
from fh5.learning.loop.environments import StoppingDrive
from fh5.telemetry.packet import decode_packet
from tests.driving.test_realtime_driving import Controller, setup_drive
from tests.driving.test_realtime_driving import numeric_driving_model as numeric_driving_model
from tests.driving.test_realtime_shadow import Capture, Desktop, Telemetry


def test_ready_handoff_and_loop_stop_wrappers_prepare_the_same_device(
    tmp_path, numeric_driving_model
):
    controller = Controller()
    created = []

    def create():
        time.sleep(0.15)
        created.append(time.perf_counter_ns())
        return controller

    request, environment, _, capture, telemetry = setup_drive(tmp_path, factory=create)
    ready = decode_packet(Telemetry().read(0).packets[0])
    environment = ReadyHandoff(
        StoppingDrive(environment, lambda: False),
        ready,
        {"start_position_m": [0, 2, 0.2], "start_radius_m": 0.5},
        request.config,
    )
    result = run_experiment(
        request,
        realtime_environment=environment,
        numeric_actor_factory=lambda: FrozenNumericActor(
            numeric_driving_model, request.config.pixels
        ),
    ).summary["realtime"]
    assert result["stop_reason"] == "time_limit"
    assert len(created) == 1 and controller.active.is_set()
    assert result["environment"]["handoff_confirmed"]
    assert result["environment"]["handoff_error"] is None
    assert controller.closed and capture.closed and telemetry.closed
    assert result["resources_released"]


def test_device_creation_keeps_udp_and_capture_running_beyond_the_queue_window(
    tmp_path, numeric_driving_model
):
    receiving, creating, enough_packets, finished = (threading.Event() for _ in range(4))
    controller = Controller()
    entered, returned, captured = [], [], []

    class Receiver(UDPTelemetry):
        def read(self, period_s):
            receiving.set()
            return super().read(period_s)

    class Images(Capture):
        def capture(self):
            captured.append(time.perf_counter_ns())
            return super().capture()

    def create():
        entered.append(time.perf_counter_ns())
        creating.set()
        assert enough_packets.wait(3), "External UDP producer did not finish the test window"
        returned.append(time.perf_counter_ns())
        return controller

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    address = receiver.getsockname()
    telemetry, capture = Receiver(receiver=receiver), Images()
    request, environment, _, _, _ = setup_drive(
        tmp_path, factory=create, telemetry=telemetry, capture=capture, seconds=3
    )

    def produce():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            if not receiving.wait(3):
                return
            sent_during_creation = 0
            while not finished.is_set():
                sender.sendto(Telemetry().read(0).packets[0].payload, address)
                if creating.is_set():
                    sent_during_creation += 1
                    # This exceeds the existing 64-packet receiver queue window.
                    if sent_during_creation >= 80:
                        enough_packets.set()
                finished.wait(0.005)

    producer = threading.Thread(target=produce)
    producer.start()
    try:
        result = run_experiment(
            request,
            realtime_environment=environment,
            numeric_actor_factory=lambda: FrozenNumericActor(
                numeric_driving_model, request.config.pixels
            ),
        ).summary["realtime"]
    finally:
        finished.set()
        producer.join(timeout=3)
        receiver.close()
    assert not producer.is_alive()
    assert result["stop_reason"] == "time_limit"
    assert len(entered) == len(returned) == 1
    events = [
        json.loads(line)
        for line in (request.output_dir / "realtime-events.jsonl").read_text().splitlines()
    ]
    received_during_creation = [
        row
        for row in events
        if row["kind"] == "packet"
        and entered[0] < row["data"]["received_monotonic_ns"] < returned[0]
    ]
    assert len(received_during_creation) > 64
    assert any(entered[0] < timestamp < returned[0] for timestamp in captured)
    accepted = [row for row in result["decisions"] if row["status"] == "accepted"]
    assert accepted and controller.active.is_set()
    assert all(row["telemetry_received_ns"] > returned[0] for row in accepted)
    assert all(row["frames"][-1]["source_time_ns"] > returned[0] for row in accepted)
    assert all(row["inference_returned_ns"] < row["deadline_ns"] for row in accepted)
    assert controller.commands[-1] == NEUTRAL
    assert controller.closed and capture.closed and telemetry.closed
    assert result["resources_released"] and result["evidence"]["recording_complete"]


@pytest.mark.parametrize("signal", ["stop", "focus"])
def test_desktop_stop_during_device_creation_is_latched_and_releases_neutral(
    tmp_path, numeric_driving_model, signal
):
    interrupted, observed = threading.Event(), threading.Event()
    controller = Controller()
    created = []

    class Signals(Desktop):
        def focused(self):
            lost = signal == "focus" and interrupted.is_set()
            if lost:
                observed.set()
            return not lost

        def stop_requested(self):
            stop = signal == "stop" and interrupted.is_set()
            if stop:
                observed.set()
            return stop

    def create():
        interrupted.set()
        assert observed.wait(1), "Desktop polling stopped during device creation"
        # Restore the external signal before the factory returns: observing the
        # interruption once must still prevent a later policy command.
        time.sleep(0.025)
        interrupted.clear()
        created.append(time.perf_counter_ns())
        return controller

    request, environment, _, capture, telemetry = setup_drive(
        tmp_path, desktop=Signals(), factory=create
    )
    result = run_experiment(
        request,
        realtime_environment=environment,
        numeric_actor_factory=lambda: FrozenNumericActor(
            numeric_driving_model, request.config.pixels
        ),
    ).summary["realtime"]
    assert len(created) == 1
    assert result["stop_reason"] == ("user_stop" if signal == "stop" else "focus_lost")
    assert not controller.active.is_set()
    assert not any(row["status"] == "accepted" for row in result["decisions"])
    assert controller.commands and all(command == NEUTRAL for command in controller.commands)
    assert controller.closed and capture.closed and telemetry.closed
    assert result["resources_released"]


def test_loop_stop_during_wrapped_device_creation_releases_the_acquired_controller(
    tmp_path, numeric_driving_model
):
    stopping, observed = threading.Event(), threading.Event()
    controller = Controller()
    created = []

    def stopped():
        value = stopping.is_set()
        if value:
            observed.set()
        return value

    def create():
        stopping.set()
        assert observed.wait(1), "Loop stop polling stopped during device creation"
        created.append(time.perf_counter_ns())
        return controller

    request, environment, _, capture, telemetry = setup_drive(tmp_path, factory=create)
    ready = decode_packet(Telemetry().read(0).packets[0])
    environment = StoppingDrive(
        ReadyHandoff(
            environment,
            ready,
            {"start_position_m": [0, 2, 0.2], "start_radius_m": 0.5},
            request.config,
        ),
        stopped,
    )
    result = run_experiment(
        request,
        realtime_environment=environment,
        numeric_actor_factory=lambda: FrozenNumericActor(
            numeric_driving_model, request.config.pixels
        ),
    ).summary["realtime"]
    assert len(created) == 1
    assert (
        result["stop_reason"] == "user_stop" or "Learning stop requested" in result["stop_reason"]
    )
    assert not controller.active.is_set()
    assert not any(row["status"] == "accepted" for row in result["decisions"])
    assert controller.commands and all(command == NEUTRAL for command in controller.commands)
    assert controller.closed and capture.closed and telemetry.closed
    assert result["resources_released"]


def test_focus_loss_in_the_collected_preparation_sample_cannot_be_overwritten_by_a_new_poll(
    tmp_path, numeric_driving_model
):
    creating, observed = threading.Event(), threading.Event()
    local_signal = threading.local()
    controller, created = Controller(), []

    class Signals(Desktop):
        def focused(self):
            if getattr(local_signal, "lose_focus_once", False):
                local_signal.lose_focus_once = False
                observed.set()
                return False
            return True

    class Packets(Telemetry):
        def read(self, period_s):
            result = super().read(period_s)
            # Only the desktop poll following this receive sees the brief loss;
            # later polls have recovered before the same input reaches runtime.
            if creating.is_set() and not observed.is_set():
                local_signal.lose_focus_once = True
            return result

    def create():
        creating.set()
        assert observed.wait(1), "Input capture stopped during device creation"
        created.append(time.perf_counter_ns())
        return controller

    request, environment, _, capture, telemetry = setup_drive(
        tmp_path, factory=create, desktop=Signals(), telemetry=Packets()
    )
    result = run_experiment(
        request,
        realtime_environment=environment,
        numeric_actor_factory=lambda: FrozenNumericActor(
            numeric_driving_model, request.config.pixels
        ),
    ).summary["realtime"]
    assert len(created) == 1
    assert result["stop_reason"] == "focus_lost"
    assert not controller.active.is_set()
    assert not any(row["status"] == "accepted" for row in result["decisions"])
    assert controller.commands and all(command == NEUTRAL for command in controller.commands)
    assert controller.closed and capture.closed and telemetry.closed
    assert result["resources_released"]


@pytest.mark.parametrize("boundary", ["unconfirmed", "expired_before", "expired_during"])
def test_control_preparation_preserves_the_ready_handoff_boundary(
    tmp_path, numeric_driving_model, boundary
):
    controller, created = Controller(), []
    deadline = time.perf_counter_ns() + (1_000_000_000 if boundary == "expired_during" else -1)

    def create():
        if boundary == "expired_during":
            time.sleep(max(0, (deadline - time.perf_counter_ns()) / 1e9) + 0.025)
        created.append(time.perf_counter_ns())
        return controller

    request, environment, _, capture, telemetry = setup_drive(tmp_path, factory=create)
    ready = decode_packet(Telemetry().read(0).packets[0])
    environment = ReadyHandoff(
        environment,
        ready,
        {
            "start_position_m": [100, 2, 0.2] if boundary == "unconfirmed" else [0, 2, 0.2],
            "start_radius_m": 0.5,
        },
        request.config,
        deadline_ns=None if boundary == "unconfirmed" else deadline,
    )
    result = run_experiment(
        request,
        realtime_environment=environment,
        numeric_actor_factory=lambda: FrozenNumericActor(
            numeric_driving_model, request.config.pixels
        ),
    ).summary["realtime"]
    expected = "handoff_state_changed" if boundary == "unconfirmed" else "handoff_expired"
    assert expected in result["stop_reason"]
    assert result["environment"]["handoff_error"] == expected
    assert len(created) == (1 if boundary == "expired_during" else 0)
    assert not controller.active.is_set()
    assert not any(row["status"] == "accepted" for row in result["decisions"])
    if created:
        assert controller.closed
        assert controller.commands and all(command == NEUTRAL for command in controller.commands)
    else:
        assert not controller.closed and not controller.commands
    assert capture.closed and telemetry.closed and result["resources_released"]
