"""In-flight storage pressure through the public learning experiment seam."""

import json
import os
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from fh5.driving.control import Command
from fh5.experiment import run_experiment
from fh5.learning.loop.environments import LearningUnavailable
from fh5.learning.loop.runner import LearningContinue, LearningLoop
from tests.evaluation.test_candidate_store import candidates as candidates
from tests.evaluation.test_evaluation import sha
from tests.learning.loop.test_learning_loop import SharedBackend, loop_request
from tests.learning.loop.test_learning_loop import seeded_loop as seeded_loop


def monitored_request(tmp_path, seeded_loop):
    namespace = Path(os.path.commonpath([str(tmp_path), str(seeded_loop[0])]))
    return loop_request(
        tmp_path,
        seeded_loop,
        version=2,
        rounds=1,
        acquisition_retry={"max_retries": 2, "delay_seconds": 0},
        storage_monitor={"interval_seconds": 0.05, "timeout_seconds": 0.2},
        storage={
            "root": str(namespace),
            "budget_bytes": 2**40,
            "min_free_bytes": 0,
            "phase_reserve_bytes": 16 * 1024**2,
            "stop_reserve_bytes": 4 * 1024**2,
        },
    )


def test_pressure_during_acquisition_stops_before_a_retry_or_empty_child(tmp_path, seeded_loop):
    request = monitored_request(tmp_path, seeded_loop)
    pressure = threading.Event()
    observed = threading.Event()
    usage = shutil.disk_usage(tmp_path)
    acquisition_thread = threading.get_ident()
    readers = []

    def disk_usage(_):
        if pressure.is_set():
            readers.append(threading.get_ident())
            observed.set()
            return usage._replace(free=0)
        return usage._replace(free=2**40)

    class WaitingBackend(SharedBackend):
        requests = 0

        def sampling(self, identity):
            self.requests += 1
            pressure.set()
            assert observed.wait(1), "No independent storage observation during acquisition"
            raise LearningUnavailable("Service still connecting", resources_released=True)

    backend = WaitingBackend(seeded_loop[0])
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", disk_usage)
        result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "storage_budget_exhausted", result.get("error")
    assert backend.requests == 1 and backend.closed and not backend.leases
    assert readers and acquisition_thread not in readers
    assert result["resources_released"] and result["learner_updates"] == 0
    assert not (request.output_dir / "round-000/learning").exists()
    monitor = result["storage_monitors"][-1]
    assert monitor["minimum_free_bytes"] == 0
    assert monitor["stop_reason"] == "storage_budget_exhausted"
    assert monitor["resources_released"] and monitor["samples"] >= 2
    assert json.loads((request.output_dir / "state.json").read_bytes())["storage_monitors"]


def test_stalled_disk_observation_is_bounded_and_refuses_an_unreleased_monitor(
    tmp_path, seeded_loop
):
    request = monitored_request(tmp_path, seeded_loop)
    pressure = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    usage = shutil.disk_usage(tmp_path)

    def disk_usage(_):
        if pressure.is_set():
            entered.set()
            assert release.wait(3), "Test did not release its external disk query"
            exited.set()
        return usage._replace(free=2**40)

    class WaitingBackend(SharedBackend):
        requests = 0
        returned_at = None

        def sampling(self, identity):
            self.requests += 1
            pressure.set()
            assert entered.wait(1)
            time.sleep(0.25)
            self.returned_at = time.monotonic()
            raise LearningUnavailable("Connection pending", resources_released=True)

    backend = WaitingBackend(seeded_loop[0])
    try:
        with pytest.MonkeyPatch.context() as disk:
            disk.setattr(shutil, "disk_usage", disk_usage)
            result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
        elapsed = time.monotonic() - backend.returned_at
        assert result["stop_reason"] == "release_fault" and not result["resources_released"]
        assert elapsed < 1.5, "Disk query blocked session shutdown"
        monitor = result["storage_monitors"][-1]
        assert monitor["stop_reason"] == "storage_monitor_stalled"
        assert not monitor["resources_released"] and backend.closed and backend.requests == 1
        assert not (request.output_dir / "round-000/learning").exists()
    finally:
        release.set()
        if entered.is_set():
            assert exited.wait(1)
    state = request.output_dir / "state.json"
    original = state.read_bytes()
    next_backend = SharedBackend(seeded_loop[0])
    with pytest.raises(ValueError, match="unreleased storage monitor"):
        run_experiment(
            LearningContinue(request.output_dir, sha(state)), learning_environment=next_backend
        )
    assert state.read_bytes() == original and not next_backend.leases and next_backend.closed


def test_monitor_thread_start_failure_still_releases_the_session_and_saves_the_reason(
    tmp_path, seeded_loop
):
    request = monitored_request(tmp_path, seeded_loop)
    start = threading.Thread.start

    def start_thread(thread):
        if thread.name == "fh5-learning-storage-monitor":
            raise OSError("Synthetic OS thread quota exhausted")
        return start(thread)

    backend = SharedBackend(seeded_loop[0])
    with pytest.MonkeyPatch.context() as system:
        system.setattr(threading.Thread, "start", start_thread)
        result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "storage_monitor_error"
    assert result["resources_released"] and backend.closed and not backend.leases
    monitor = result["storage_monitors"][-1]
    assert monitor["samples"] == 0 and monitor["resources_released"]
    assert "thread quota exhausted" in monitor["error"]
    assert not (request.output_dir / "round-000").exists()
    state = request.output_dir / "state.json"
    second = SharedBackend(seeded_loop[0])
    (request.output_dir / "stop.request").write_text("Stop the resumed session before driving")
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(state)), learning_environment=second
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "stop_requested" and resumed["resources_released"]
    assert not second.leases and second.closed


def test_stop_observed_before_release_stays_recorded_when_the_file_is_cleared(
    tmp_path, seeded_loop
):
    request = monitored_request(tmp_path, seeded_loop)

    class ClearingBackend(SharedBackend):
        def sampling(self, identity):
            source = super().sampling(identity)
            close = source.close
            stop = request.output_dir / "stop.request"
            stop.write_text("Stop before driving")

            def release():
                result = close()
                stop.unlink()  # External operator clears the request during cleanup.
                return result

            source.close = release
            return source

    backend = ClearingBackend(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "stop_requested"
    assert result["resources_released"] and backend.closed
    assert len(backend.leases) == 1 and backend.leases[0].closed
    assert not (request.output_dir / "round-000/learning").exists()


def test_pressure_during_sampling_keeps_the_attempt_and_complete_prior_learner(
    tmp_path, seeded_loop
):
    request = monitored_request(tmp_path, seeded_loop)
    pressure = threading.Event()
    observed = threading.Event()
    usage = shutil.disk_usage(tmp_path)
    store_before = sha(tmp_path / "versions/state.sqlite")

    def disk_usage(_):
        if pressure.is_set():
            observed.set()
            return usage._replace(free=0)
        return usage._replace(free=2**40)

    class PressureBackend(SharedBackend):
        def sampling(self, identity):
            source = super().sampling(identity)
            step = source.step

            def respond(command):
                value = step(command)
                pressure.set()
                assert observed.wait(1)
                return value

            source.step = respond
            return source

    backend = PressureBackend(seeded_loop[0])
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", disk_usage)
        result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "storage_budget_exhausted", result.get("error")
    assert result["resources_released"] and backend.closed
    assert len(backend.leases) == 1 and backend.leases[0].closed
    assert result["learner_updates"] == result["rounds_completed"] == 0
    assert result["latest_learner"]["sha256"] == result["explorer"]["sha256"]
    assert sha(tmp_path / "versions/state.sqlite") == store_before
    learning = Path(result["rounds"][0]["learning"]["directory"])
    summary = json.loads((learning / "summary.json").read_bytes())
    assert summary["stop_reason"] == "stop_requested"
    assert len(summary["attempts"]) == 1
    trace = json.loads((learning / "attempt-000/trace.json").read_bytes())
    assert len(trace["actions"]) == 1
    assert (learning / "attempt-000/recording/packets.jsonl").is_file()


def test_healthy_monitor_covers_learning_and_evaluation_without_changing_the_budget(
    tmp_path, seeded_loop
):
    request = monitored_request(tmp_path, seeded_loop)
    # Permit a longer scheduling interval for this real CPU learning integration.
    config = json.loads(request.config_file.read_bytes())
    config["storage_monitor"] = {"interval_seconds": 0.1, "timeout_seconds": 2}
    request.config_file.write_text(json.dumps(config))
    backend = SharedBackend(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "budget_completed", result.get("error")
    assert result["rounds_completed"] == 1 and result["learner_updates"] == 3
    assert result["rounds"][0]["evaluation"]["metrics"]["all_attempts"] == 2
    assert len(backend.leases) == 2 and all(lease.closed for lease in backend.leases)
    assert result["resources_released"] and backend.closed
    monitor = result["storage_monitors"][-1]
    assert monitor["samples"] > 2 and monitor["stop_reason"] is None
    assert monitor["resources_released"] and monitor["files_deleted"] == 0


@pytest.fixture
def waiting_monitored_evaluation(tmp_path, seeded_loop):
    request = monitored_request(tmp_path, seeded_loop)
    config = json.loads(request.config_file.read_bytes())
    config["storage_monitor"]["timeout_seconds"] = 2
    request.config_file.write_text(json.dumps(config))

    class OfflineEvaluation(SharedBackend):
        def evaluation(self, identity):
            raise OSError("Evaluation intentionally unavailable for initial handoff")

    result = run_experiment(
        request, learning_environment=OfflineEvaluation(seeded_loop[0])
    ).summary["learning_loop"]
    assert result["stop_reason"] == "interface_error" and result["learner_updates"] == 3
    return request, seeded_loop[0]


@pytest.mark.parametrize("timing", ["cleanup", "menu"])
def test_resource_stop_is_attributed_only_when_evaluation_observed_it(
    tmp_path, waiting_monitored_evaluation, timing
):
    request, source = waiting_monitored_evaluation
    pressure, observed = threading.Event(), threading.Event()
    usage = shutil.disk_usage(tmp_path)

    def disk_usage(_):
        if pressure.is_set():
            observed.set()
            return usage._replace(free=0)
        return usage._replace(free=2**40)

    class UnfocusedBackend(SharedBackend):
        def evaluation(self, identity):
            lease = super().evaluation(identity)
            event, close = lease.event, lease.close

            def unfocused(slot):
                menu = event(slot)
                read = menu.read

                def observation(period):
                    value = read(period)
                    if timing == "menu":
                        pressure.set()
                        assert observed.wait(1)
                    return replace(value, focused=timing == "menu")

                menu.read = observation
                return menu

            def release():
                result = close()
                pressure.set()
                assert observed.wait(1)
                return result

            lease.event, lease.close = unfocused, release
            return lease

    backend = UnfocusedBackend(source)
    state = request.output_dir / "state.json"
    with pytest.MonkeyPatch.context() as disk:
        disk.setattr(shutil, "disk_usage", disk_usage)
        result = run_experiment(
            LearningContinue(request.output_dir, sha(state)), learning_environment=backend
        ).summary["learning_loop"]
    assert result["stop_reason"] == "storage_budget_exhausted", result.get("error")
    row = result["rounds"][0]
    assert row["evaluation_run"]["stop_reason"] == "ready_unconfirmed"
    assert row["evaluation_run"]["preparations"][0]["stop_reason"] == (
        "user_stop" if timing == "menu" else "focus_lost"
    )
    assert row["evaluation_interrupted_by_resource"] is (timing == "menu")
    assert row["evaluation_interrupted_by_stop"] is False
    assert backend.closed and result["resources_released"]
    assert not backend.leases[0].drives
    resumed_backend = SharedBackend(source)
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(state)), learning_environment=resumed_backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == (
        "budget_completed" if timing == "menu" else "evaluation_ready_unconfirmed"
    ), resumed.get("error")
    assert resumed["learner_updates"] == result["learner_updates"] == 3
    assert not resumed_backend.leases and resumed_backend.closed


def test_an_older_file_query_cannot_clear_a_stop_observed_by_another_thread(
    tmp_path, waiting_monitored_evaluation
):
    request, source = waiting_monitored_evaluation
    state = request.output_dir / "state.json"
    stop = request.output_dir / "stop.request"
    reading, released = threading.Event(), threading.Event()
    reader = []
    operators = []
    exists = Path.exists

    def stale_query(path):
        if path == stop and reader == [threading.get_ident()] and not reading.is_set():
            assert not exists(path)
            reading.set()
            assert released.wait(2), "Concurrent stop did not release the external controls"
            return False  # Query started before the stop file was created.
        return exists(path)

    class ConcurrentBackend(SharedBackend):
        def evaluation(self, identity):
            lease = super().evaluation(identity)
            driving = lease.driving

            def connect(slot, ready):
                game = driving(slot, ready)
                read = game.read

                def observation(period):
                    reader[:] = [threading.get_ident()]
                    return read(period)

                def signals():
                    assert reading.wait(2)
                    stop.write_text("External stop during a concurrent file query")
                    observer = threading.current_thread()

                    def clear_after_observer_exits():
                        observer.join(timeout=1)
                        if not observer.is_alive():
                            stop.unlink()
                            released.set()

                    operator = threading.Thread(target=clear_after_observer_exits, daemon=True)
                    operators.append(operator)
                    operator.start()
                    return True, False

                game.read, game.signals = observation, signals
                return game

            lease.driving = connect
            return lease

    backend = ConcurrentBackend(source)
    with pytest.MonkeyPatch.context() as filesystem:
        filesystem.setattr(Path, "exists", stale_query)
        result = run_experiment(
            LearningContinue(request.output_dir, sha(state)), learning_environment=backend
        ).summary["learning_loop"]
    for operator in operators:
        operator.join(timeout=2)
        assert not operator.is_alive()
    assert reading.is_set() and released.is_set()
    assert result["stop_reason"] == "stop_requested", result.get("error")
    assert result["rounds"][0]["evaluation_interrupted_by_stop"] is True
    assert result["learner_updates"] == 3 and result["rounds_completed"] == 0
    assert backend.closed and result["resources_released"]
    assert all(
        command == Command(0, 0, 0) for game in backend.leases[0].drives for _, command in game.sent
    )


@pytest.mark.parametrize(
    "monitor",
    [
        {"interval_seconds": 0, "timeout_seconds": 1},
        {"interval_seconds": 0.5, "timeout_seconds": 0.5},
        {"interval_seconds": True, "timeout_seconds": 2},
        {"interval_seconds": 0.1, "timeout_seconds": float("inf")},
    ],
)
def test_invalid_monitor_configuration_is_rejected_before_creating_resources(tmp_path, monitor):
    config = {
        "version": 2,
        "store": {"directory": "unused", "revision": "unused"},
        "registry": "unused",
        "recording": "unused",
        "task": "unused",
        "reward": "unused",
        "rounds": 1,
        "steps_per_attempt": 1,
        "evaluation_seconds": 1,
        "seed": 0,
        "storage": {
            "root": str(tmp_path),
            "budget_bytes": 1024,
            "min_free_bytes": 0,
            "phase_reserve_bytes": 1,
            "stop_reserve_bytes": 1,
        },
        "storage_monitor": monitor,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    backend = SharedBackend(tmp_path)
    with pytest.raises(ValueError, match="finite interval"):
        run_experiment(LearningLoop(path, tmp_path / "loop"), learning_environment=backend)
    assert not (tmp_path / "loop").exists() and not backend.leases
