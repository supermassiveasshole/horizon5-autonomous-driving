"""Stop-time inference publication through actual frozen inference and external I/O."""

import threading
import time

import pytest
from test_evaluation import sha
from test_evaluation_execution import PacketGame
from test_sac_learning import warm_start

from fh5.experiment import run_experiment
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig, RealtimeNumericReplay, RealtimeRun
from fh5.sac_actions import ActionBounds
from fh5.sac_evaluation_actor import SACEvaluationActor
from fh5.sac_learning import SACTrain


@pytest.fixture
def stop_model(tmp_path):
    replay = warm_start(
        tmp_path, bounds=ActionBounds(max_steer=0.4, max_throttle=0.25, max_brake=0.5)
    )
    model = tmp_path / "model"
    run_experiment(SACTrain(tmp_path / "warm", replay, model, steps=0))
    return model


def test_result_returned_during_bounded_shutdown_is_recorded_but_never_sent(
    tmp_path, stop_model, monkeypatch
):
    completed_prediction = threading.Event()
    released_command = threading.Event()
    clock = time.perf_counter_ns
    worker_clock_calls = 0

    def scheduled_clock():
        nonlocal worker_clock_calls
        if threading.current_thread().name == "fh5-persistent-inference":
            worker_clock_calls += 1
            if worker_clock_calls == 4:
                # Pause only the external timing boundary after actual inference.
                completed_prediction.set()
                if not released_command.wait(2):
                    raise TimeoutError("Supervisor did not release the action")
                time.sleep(0.05)
        return clock()

    class StopAtPrediction(PacketGame):
        def signals(self):
            return True, completed_prediction.is_set()

        def send(self, command):
            super().send(command)
            if completed_prediction.is_set() and not any(
                (command.steer_i16, command.throttle_u8, command.brake_u8)
            ):
                released_command.set()

    pixels = PixelContract(size=(64, 36))

    def factory():
        return SACEvaluationActor(stop_model, pixels, sha(stop_model / "policy.json"))

    game = StopAtPrediction()
    with monkeypatch.context() as timing:
        timing.setattr(time, "perf_counter_ns", scheduled_clock)
        original = run_experiment(
            RealtimeRun(
                tmp_path / "execution", RealtimeConfig(pixels=pixels, reference_count=1), seconds=1
            ),
            realtime_environment=game,
            numeric_actor_factory=factory,
        ).summary["realtime"]
    assert completed_prediction.is_set() and released_command.is_set()
    assert original["stop_reason"] == "user_stop" and original["resources_released"]
    assert [row["status"] for row in original["decisions"] if "actor" in row] == [
        "accepted",
        "discard_stopped",
    ]
    stopped = original["decisions"][-1]
    assert not any(c["decision_id"] == stopped["decision_id"] for c in original["commands"])
    assert original["commands"][-1]["owner"] == "hard_stop"
    assert original["evidence"]["exact_replay_eligible"]
    verified = run_experiment(
        RealtimeNumericReplay(tmp_path / "execution", tmp_path / "verified.html"),
        numeric_actor=factory(),
    ).summary["realtime_numeric_replay"]
    assert verified["verified"] and verified["verified_predictions"] == 2
