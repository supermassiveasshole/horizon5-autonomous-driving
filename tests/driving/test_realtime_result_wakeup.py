"""Completed inference wakes supervision without waiting for a periodic timer."""

import threading
from types import SimpleNamespace

from fh5.driving.realtime.model import RealtimeConfig, RealtimeRun
from fh5.experiment import run_experiment
from fh5.observation.numeric import PixelContract
from tests.driving.test_realtime import FastActor, ThreadedGame


def test_completed_prediction_wakes_a_sleeping_supervisor(tmp_path, monkeypatch):
    sleeping = threading.Event()
    sent = threading.Event()

    class CoarseTimer(threading.Event):
        def wait(self, timeout=None):
            if threading.current_thread().name == "fh5-action-supervisor":
                # Model a delayed timer wakeup. An event notification must still
                # wake immediately; no production deadline or guard is changed.
                sleeping.set()
                timeout = 0.5
            return super().wait(timeout)

    class ReadyActor(FastActor):
        predictions = 0

        def predict(self, actor, frames):
            self.predictions += 1
            if self.predictions > 1:  # Warmup is outside the running supervisor.
                assert sleeping.wait(2), "Supervisor never reached its timed wait"
            return super().predict(actor, frames)

    class StopAfterSend(ThreadedGame):
        def send(self, command):
            super().send(command)
            if command.throttle_u8:
                sent.set()

        def signals(self):
            return True, sent.is_set()

    # Only replace the runtime's OS timing boundary, leaving its queues, worker,
    # decision state, journal and external observation/command path intact.
    monkeypatch.setattr(
        "fh5.driving.realtime.runtime.threading",
        SimpleNamespace(Event=CoarseTimer, Lock=threading.Lock, Thread=threading.Thread),
    )
    game = StopAfterSend()
    result = run_experiment(
        RealtimeRun(
            tmp_path / "notified-result",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            seconds=1,
        ),
        realtime_environment=game,
        numeric_actor_factory=ReadyActor,
    ).summary["realtime"]

    predictions = [row for row in result["decisions"] if "actor" in row]
    assert predictions and predictions[0]["status"] == "accepted", result
    first = predictions[0]
    assert first["worker_returned_ns"] <= first["inference_returned_ns"] < first["deadline_ns"]
    assert result["stop_reason"] == "user_stop"
    assert result["resources_released"] and game.closed
    assert game.sent[-1][1].throttle_u8 == 0
