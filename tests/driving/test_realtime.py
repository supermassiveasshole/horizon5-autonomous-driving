"""Deadline and action ownership behavior at the experiment-run seam."""

import random
import threading
import time
from dataclasses import asdict, replace

import pytest

from fh5.driving.realtime.model import (
    InferenceReply,
    RealtimeConfig,
    RealtimeObservation,
    RealtimeReplay,
    RealtimeRun,
    SafetyState,
    TimelineInput,
)
from fh5.experiment import run_experiment
from fh5.learning.sac.actions import ActionBounds
from fh5.learning.sac.context import SEND_CONTEXT
from fh5.observation.numeric import NumericFrame, PixelContract

BASE = 1_000_000_000
MS = 1_000_000


@pytest.mark.parametrize("resume", [False, True])
def test_submillisecond_command_gap_waits_without_renewing_and_resumes(tmp_path, resume):
    # A real failed evaluation had only 227900 ns after the last sent action.
    # 4 units/second * that gap cannot span even one 8-bit trigger step.
    points = [sample(250), sample(300), sample(350), sample(400), sample(450), sample(500)]
    points[2] = replace(points[2], at_ns=BASE + 350 * MS + 227900)
    if not resume:
        points[3:] = [replace(p, observation=None) for p in points[3:]]
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "short-gap",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            tuple(points),
            (InferenceReply(50, (2137 / 32767, 18 / 255)), InferenceReply(10)),
            require_command_context=True,
            command_bounds=ActionBounds(max_steer=0.4, max_throttle=0.25, max_brake=0.5),
        )
    ).summary["realtime"]
    assert result["decisions"][2]["status"] == "skip_action_support"
    assert "actor" not in result["decisions"][2]
    assert result["decisions"][2]["command_context"]["returned_ns"] == BASE + 350 * MS
    assert result["decisions"][3]["status"] == ("accepted" if resume else "skip_history")
    assert [c["decision_id"] for c in result["commands"] if c["owner"] == "policy"] == (
        ["d1", "d3"] if resume else ["d1"]
    )
    assert next(c for c in result["commands"] if c["owner"] == "lease_expiry")["issued_ns"] == (
        BASE + (550 if resume else 450) * MS
    )
    assert result["stop_reason"] == "time_limit"


def observation(ms, epoch="e1"):
    frames = tuple(
        NumericFrame(
            epoch,
            str(ms - offset),
            BASE + (ms - offset) * MS,
            BASE + (ms - offset) * MS,
            BASE + (ms - offset) * MS,
            "synthetic",
            0,
            (2, 1),
            memoryview(bytes([51, 17, 34] * 2)),
            {"size": [2, 1], "format": "RGB"},
        )
        for offset in (200, 100, 0)
    )
    return RealtimeObservation(
        epoch,
        frames,
        {"speed_mps": 0, "velocity_car_mps": [0, 0, 0], "angular_velocity_car_radps": [0, 0, 0]},
        BASE + ms * MS,
    )


def sample(ms, *, obs=True, **changes):
    safety = SafetyState(
        "e1",
        BASE + ms * MS,
        ms,
        True,
        True,
        False,
        2941,
        999,
        0,
        None,
    )
    return TimelineInput(
        BASE + ms * MS,
        replace(safety, **changes),
        observation(ms) if obs else None,
    )


def test_skips_do_not_renew_absolute_action_lease_and_recovery_uses_new_images(tmp_path):
    points = tuple(sample(ms, obs=ms not in (350, 400)) for ms in range(250, 551, 50))
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "run",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            points,
            (InferenceReply(10, (0.2, 0.3)),) * 20,
        )
    )
    r = result.summary["realtime"]
    commands = r["commands"]
    assert commands[0]["issued_ns"] == BASE + 260 * MS
    assert commands[0]["valid_until_ns"] == BASE + 400 * MS
    # The second action was issued at 310, so it expires at 450 even with missing ticks.
    release = next(c for c in commands if c["owner"] == "lease_expiry")
    assert release["issued_ns"] == BASE + 450 * MS
    assert release["sent"] == {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0}
    assert release["previous_hold_ns"] == 140 * MS
    resumed = next(
        c for c in commands if c["issued_ns"] > release["issued_ns"] and c["owner"] == "policy"
    )
    assert resumed["decision_id"] == "d4"
    assert r["stop_reason"] == "time_limit"
    assert [d["status"] for d in r["decisions"]][2:4] == ["skip_history", "skip_history"]
    assert r["commands_sent_to_game"] is False
    assert r["metrics"]["source_to_sendable_ms"]["p95"] == 10
    assert r["metrics"]["maximum_consecutive_skip_ms"] == 100
    assert result.report_path.suffix == ".html"
    assert "动作有效期" in result.report_path.read_text(encoding="utf-8")
    assert r["configuration"]["action_lease_ms"] == 150
    assert r["configuration"]["pixels"]["dtype"] == "uint8"
    assert r["decisions"][0]["telemetry_received_ns"] == BASE + 250 * MS
    assert r["decisions"][0]["safety_at_decision"]["car_ordinal"] == 2941


@pytest.mark.parametrize("delay,reason", [(110, "discard_deadline"), (None, "abandoned_inference")])
def test_single_inflight_work_is_not_replaced_and_late_or_stuck_work_cannot_renew(
    tmp_path, delay, reason
):
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "run",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            tuple(sample(ms) for ms in range(250, 851, 50)),
            (InferenceReply(10), InferenceReply(delay), *(InferenceReply(10) for _ in range(20))),
        )
    )
    r = result.summary["realtime"]
    assert r["decisions"][1]["status"] == reason
    assert r["decisions"][2]["status"] == "skip_busy"
    assert not any(c["decision_id"] == "d1" for c in r["commands"])
    if delay is None:
        assert r["stop_reason"] == "inference_watchdog"
        assert not any(
            c["owner"] == "policy" and c["issued_ns"] > BASE + 300 * MS for c in r["commands"]
        )
        assert r["inference_resources_released"] is False
    else:
        assert any(d["status"] == "accepted" for d in r["decisions"][4:])
        assert r["stop_reason"] == "time_limit"
        assert r["inference_resources_released"] is True


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"focused": False}, "focus_lost"),
        ({"active": False}, "inactive"),
        ({"stop_requested": True}, "user_stop"),
        ({"car_ordinal": 7}, "unexpected_vehicle"),
        ({"speed_kmh": 15}, "speed_limit"),
        ({"task_fault": "outside_corridor"}, "outside_corridor"),
        ({"epoch": "restart"}, "session_boundary"),
        ({"epoch": "stop", "stop_requested": True}, "user_stop"),
        ({"epoch": "focus", "focused": False}, "focus_lost"),
        ({"game_timestamp_ms": 1}, "game_clock_discontinuity"),
    ],
)
def test_hard_stop_during_inference_latches_despite_subsequent_good_input(tmp_path, change, reason):
    points = [sample(ms) for ms in range(250, 651, 50)]
    points[2] = sample(350, **change)
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "run",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            tuple(points),
            (InferenceReply(10), InferenceReply(70), *(InferenceReply(10) for _ in range(20))),
        )
    )
    r = result.summary["realtime"]
    assert r["stop_reason"] == reason
    assert r["decisions"][1]["status"] == "discard_stopped"
    assert not any(
        c["owner"] == "policy" and c["issued_ns"] >= BASE + 350 * MS for c in r["commands"]
    )
    assert r["commands"][-1]["owner"] == "hard_stop"
    assert all(d["status"] == "skip_stopped" for d in r["decisions"][2:])


def test_send_rechecks_original_input_age_and_current_capture_epoch(tmp_path):
    points = [sample(ms) for ms in range(250, 551, 50)]
    # First observation was already 90 ms old; a 20 ms inference cannot make it fresh.
    points[0] = replace(points[0], observation=observation(160))
    # During the next inference the capture backend is recreated, not a game restart.
    points[2] = replace(points[2], observation=observation(350, "capture-2"))
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "run",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            tuple(points),
            (InferenceReply(20), InferenceReply(70), *(InferenceReply(10) for _ in range(20))),
        )
    )
    r = result.summary["realtime"]
    assert r["decisions"][0]["status"] == "discard_stale_input"
    assert r["decisions"][1]["status"] == "discard_capture_epoch"
    assert all(c["decision_id"] not in ("d0", "d1") for c in r["commands"])
    assert any(d["status"] == "accepted" for d in r["decisions"][3:])


def test_frozen_old_frame_cannot_renew_lease_or_hide_blackout(tmp_path):
    points = tuple(replace(sample(ms), observation=observation(250)) for ms in range(250, 651, 50))
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "run",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            points,
            (InferenceReply(10),) * 20,
        )
    )
    r = result.summary["realtime"]
    assert len([c for c in r["commands"] if c["owner"] == "policy"]) == 1
    assert r["decisions"][1]["status"] == "skip_repeated_source"
    assert r["stop_reason"] == "decision_watchdog"


@pytest.mark.parametrize("prediction", [(float("nan"), 0.5), (1.01, 0), (0, -1.1)])
def test_invalid_prediction_cannot_reach_simulated_actuator(tmp_path, prediction):
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "run",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            (sample(250), sample(300)),
            (InferenceReply(10, prediction), InferenceReply(10)),
        )
    )
    r = result.summary["realtime"]
    assert r["decisions"][0]["status"] == "discard_invalid_prediction"
    assert r["decisions"][0]["prediction"] is None
    assert all(c["decision_id"] != "d0" for c in r["commands"])


class ThreadedGame:
    source_kind = "synthetic"

    def __init__(self):
        self.sent = []
        self.closed = False

    def read(self, period_s):
        time.sleep(period_s)
        ms = time.perf_counter_ns() // MS - BASE // MS
        return sample(ms)

    def signals(self):
        return True, False

    def send(self, command):
        self.sent.append((time.perf_counter_ns(), command))

    def close(self):
        self.closed = True
        return {"resources_released": True}


class PausingActor:
    kind = "synthetic_delayed_model"
    manifest = {"diagnostic_only": True}

    def __init__(self):
        self.calls = []

    def predict(self, actor, frames):
        self.calls.append(threading.get_ident())
        if len(self.calls) == 3:  # Warmup, first prediction, then one slow kernel.
            time.sleep(0.4)
        return [0.2, 0.3]


class FastActor(PausingActor):
    def predict(self, actor, frames):
        return [0.2, 0.3]


def test_new_decision_settles_expired_action_before_binding_context(tmp_path):
    resumed = threading.Event()

    class DelayedSignals(ThreadedGame):
        def __init__(self):
            super().__init__()
            self.first_policy_ns = None

        def send(self, command):
            super().send(command)
            if command.throttle_u8 and self.first_policy_ns is None:
                self.first_policy_ns = time.perf_counter_ns()

        def signals(self):
            # An external signal read delays supervision while input and the
            # decision schedule remain active. No policy call can renew the lease.
            if self.first_policy_ns is not None:
                assert resumed.wait(2), "Decision did not resume after fresh input"
            return True, False

        def read(self, period_s):
            point = super().read(period_s)
            if self.first_policy_ns is not None and point.at_ns - self.first_policy_ns < 100 * MS:
                return replace(point, observation=None)
            return point

    game = DelayedSignals()

    class ContextActor:
        kind = "synthetic-command-context"
        manifest = {
            "command_context": SEND_CONTEXT,
            "bounds": asdict(ActionBounds(max_steer=0.4, max_throttle=0.25, max_brake=0.5)),
        }
        resumed_decision = None

        def predict_decision(self, decision, command_context):
            if game.first_policy_ns is not None and self.resumed_decision is None:
                self.resumed_decision = decision.decision_id
                resumed.set()
            return [0.2, 0.2]

    actor = ContextActor()
    try:
        result = run_experiment(
            RealtimeRun(
                tmp_path / "expired-context",
                RealtimeConfig(pixels=PixelContract(size=(2, 1)), action_lease_ms=50),
                seconds=0.45,
            ),
            realtime_environment=game,
            numeric_actor_factory=lambda: actor,
        ).summary["realtime"]
    finally:
        resumed.set()
    assert actor.resumed_decision is not None, result
    decision = next(d for d in result["decisions"] if d["decision_id"] == actor.resumed_decision)
    context = decision["command_context"]
    assert context["owner"] == "lease_expiry", decision
    assert context["sent"] == {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0}
    assert context["returned_ns"] < decision["decision_ns"]
    assert decision["status"] == "accepted"
    assert any(c["decision_id"] == actor.resumed_decision for c in result["commands"])
    assert result["resources_released"] and game.closed


def test_persistent_worker_cannot_block_independent_action_expiry(tmp_path):
    game, actor = ThreadedGame(), PausingActor()
    result = run_experiment(
        RealtimeRun(
            tmp_path / "threaded", RealtimeConfig(pixels=PixelContract(size=(2, 1))), seconds=0.65
        ),
        realtime_environment=game,
        numeric_actor_factory=lambda: actor,
    )
    r = result.summary["realtime"]
    assert len(set(actor.calls)) == 1
    assert r["stop_reason"] == "inference_watchdog"
    actions = [c for c in r["commands"] if c["owner"] == "policy"]
    assert len(actions) == 1
    release = next(c for c in r["commands"] if c["owner"] == "lease_expiry")
    assert 0 <= release["issued_ns"] - actions[0]["valid_until_ns"] < 50 * MS
    assert r["inference"]["warmup_completed"] is True
    assert game.sent[-1][1].throttle_u8 == 0 and game.closed
    assert r["commands_sent_to_game"] is False


def test_stalled_preview_does_not_fill_the_exact_input_queue(tmp_path, monkeypatch):
    from pathlib import Path

    preview_started, release_preview = threading.Event(), threading.Event()
    write = Path.write_bytes

    def stalled_preview(path, data):
        if path.suffix == ".png":
            preview_started.set()
            assert release_preview.wait(2), "Preview blocked control shutdown"
        return write(path, data)

    class ClosingGame(ThreadedGame):
        def close(self):
            release_preview.set()
            return super().close()

    game = ClosingGame()
    monkeypatch.setattr(Path, "write_bytes", stalled_preview)
    try:
        result = run_experiment(
            RealtimeRun(
                tmp_path / "preview", RealtimeConfig(pixels=PixelContract(size=(2, 1))), seconds=0.7
            ),
            realtime_environment=game,
            numeric_actor_factory=FastActor,
        ).summary["realtime"]
    finally:
        release_preview.set()
    assert preview_started.is_set()
    assert result["stop_reason"] == "time_limit"
    assert sum(row["status"] == "accepted" for row in result["decisions"]) >= 8
    assert all(row["archive"] for row in result["decisions"] if "actor" in row)
    assert result["evidence"]["exact_replay_eligible"] is True
    assert result["archive"]["previews"]["peak_pending_bytes"] == 6
    assert result["resources_released"] and game.closed
    assert game.sent[-1][1].throttle_u8 == 0
    assert result["ended_ns"] - result["started_ns"] < 900 * MS


def test_one_second_writer_stall_does_not_block_decisions_and_quarantines_gaps(tmp_path):
    writes = []

    def stalled_disk(payload):
        if not writes:
            time.sleep(1)
        writes.append((threading.get_ident(), payload))

    game = ThreadedGame()
    result = run_experiment(
        RealtimeRun(
            tmp_path / "writer",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            seconds=0.7,
            journal_capacity=2,
        ),
        realtime_environment=game,
        numeric_actor_factory=FastActor,
        realtime_journal_sink=stalled_disk,
    )
    r = result.summary["realtime"]
    accepted = [d for d in r["decisions"] if d["status"] == "accepted"]
    assert len(accepted) >= 8
    assert r["stop_reason"] == "time_limit"
    assert r["journal"]["dropped"] > 0
    assert r["journal"]["missing_sequences"]
    assert r["evidence"]["training_eligible"] is False
    assert r["evidence"]["exact_replay_eligible"] is False
    assert r["evidence"]["promotion_eligible"] is False
    assert len({tid for tid, _ in writes}) == 1
    assert writes[0][0] != threading.get_ident()
    assert game.closed and r["resources_released"]
    assert r["ended_ns"] - r["started_ns"] < 900 * MS
    assert r["finalized_ns"] > r["ended_ns"]


def test_capture_native_timestamps_can_change_while_pixels_are_owned_and_layout_is_stable(tmp_path):
    original = observation(250)
    shared = bytearray([51, 17, 34] * 2)
    frames = tuple(
        replace(
            f,
            pixels=memoryview(shared),
            source_layout={
                "size": [3840, 2160],
                "format": "BGRA",
                "stride_bytes": 15360,
                "present_ticks": i * 1_000_000,
                "accumulated_frames": i + 1,
            },
        )
        for i, f in enumerate(original.frames)
    )
    shared[:] = bytes([255] * 6)
    p = replace(sample(250), observation=replace(original, frames=frames))
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "capture",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            (p,),
            (InferenceReply(10),),
        )
    )
    assert result.summary["realtime"]["decisions"][0]["status"] == "accepted"
    assert frames[0].pixels[0, 0, 0] == 51


def test_result_cannot_start_an_already_expired_action_lease(tmp_path):
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "expired",
            RealtimeConfig(pixels=PixelContract(size=(2, 1)), action_lease_ms=20),
            (sample(250),),
            (InferenceReply(40),),
        )
    )
    assert result.summary["realtime"]["commands"] == []
    assert result.summary["realtime"]["decisions"][0]["status"] == "discard_expired_lease"


@pytest.mark.parametrize("probability", [0.05, 0.10])
def test_seeded_source_frame_losses_allow_fresh_decisions_to_continue(tmp_path, probability):
    rng = random.Random(43)
    available = [ms for ms in range(0, 2101, 17) if rng.random() >= probability]
    points = []
    for ms in range(250, 2051, 50):
        selected = [max(t for t in available if t <= ms - offset) for offset in (200, 100, 0)]
        fresh = observation(ms)
        frames = tuple(
            replace(
                f,
                frame_id=str(t),
                source_time_ns=BASE + t * MS,
                capture_received_ns=BASE + t * MS,
                preprocess_ready_ns=BASE + t * MS,
            )
            for t, f in zip(selected, fresh.frames)
        )
        points.append(replace(sample(ms), observation=replace(fresh, frames=frames)))
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "loss",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            tuple(points),
            (InferenceReply(10),) * len(points),
        )
    )
    r = result.summary["realtime"]
    assert r["stop_reason"] == "time_limit"
    assert sum(d["status"] == "accepted" for d in r["decisions"]) >= 30


def test_300_ms_blackout_latches_watchdog_and_cannot_resume_on_its_own(tmp_path):
    points = tuple(sample(ms, obs=not 350 <= ms < 650) for ms in range(250, 801, 50))
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "blackout",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            points,
            (InferenceReply(10),) * len(points),
        )
    )
    r = result.summary["realtime"]
    assert r["stop_reason"] == "decision_watchdog"
    assert not any(
        c["owner"] == "policy" and c["issued_ns"] >= BASE + 350 * MS for c in r["commands"]
    )
    assert r["decisions"][-1]["status"] == "skip_stopped"


def test_first_inflight_prediction_is_cancelled_if_the_prepared_session_loses_focus(tmp_path):
    points = (sample(250), sample(300, focused=False), sample(350), sample(400))
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "first",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            points,
            (InferenceReply(70), InferenceReply(10)),
        )
    )
    r = result.summary["realtime"]
    assert r["stop_reason"] == "focus_lost"
    assert r["commands"] == []


def test_partial_send_failure_attempts_neutral_and_keeps_a_failure_report(tmp_path):
    class FailingDriver(ThreadedGame):
        def send(self, command):
            super().send(command)
            if command.throttle_u8:
                raise OSError("driver reported failure after accepting bytes")

    game = FailingDriver()
    result = run_experiment(
        RealtimeRun(
            tmp_path / "failure", RealtimeConfig(pixels=PixelContract(size=(2, 1))), seconds=0.3
        ),
        realtime_environment=game,
        numeric_actor_factory=FastActor,
    )
    r = result.summary["realtime"]
    assert r["stop_reason"] == "send_failed"
    assert r["commands"][0]["status"] == "failed"
    assert r["commands"][-1]["sent"]["throttle_u8"] == 0
    assert r["evidence"]["exact_replay_eligible"] is False
    assert game.closed


def test_instant_results_do_not_create_extra_decision_ticks(tmp_path):
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "instant",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            (sample(250), sample(300)),
            (InferenceReply(0), InferenceReply(0)),
        )
    )
    assert len(result.summary["realtime"]["decisions"]) == 2


def test_capture_boundary_wins_over_result_arriving_at_the_same_instant(tmp_path):
    result = run_experiment(
        RealtimeReplay(
            tmp_path / "boundary",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            (sample(250), replace(sample(300), observation=observation(300, "new"))),
            (InferenceReply(50), InferenceReply(10)),
        )
    )
    assert result.summary["realtime"]["decisions"][0]["status"] == "discard_capture_epoch"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"active": False}, "inactive"),
        ({"stop_requested": True}, "user_stop"),
        ({"speed_kmh": 15}, "speed_limit"),
    ],
)
def test_input_hard_fault_is_latched_even_if_normal_input_arrives_before_next_guard_poll(
    tmp_path, change, reason
):
    class BurstGame(ThreadedGame):
        def __init__(self):
            super().__init__()
            self.activated = threading.Event()
            self.restored = threading.Event()
            self.phase = 0

        def send(self, command):
            super().send(command)
            if command.throttle_u8:
                self.activated.set()

        def signals(self):
            if self.activated.is_set():
                self.restored.wait(0.2)
            return True, False

        def read(self, period_s):
            point = super().read(period_s)
            if self.activated.is_set() and self.phase == 0:
                self.phase = 1
                return replace(point, safety=replace(point.safety, **change))
            if self.phase == 1:
                self.phase = 2
                self.restored.set()
            return point

    game = BurstGame()
    result = run_experiment(
        RealtimeRun(
            tmp_path / "burst", RealtimeConfig(pixels=PixelContract(size=(2, 1))), seconds=0.4
        ),
        realtime_environment=game,
        numeric_actor_factory=FastActor,
    )
    r = result.summary["realtime"]
    assert r["stop_reason"] == reason
    assert len([c for c in r["commands"] if c["owner"] == "policy"]) == 1
    assert game.closed


def test_action_hold_reports_send_return_intervals_and_call_uncertainty(tmp_path):
    class SlowSink(ThreadedGame):
        def send(self, command):
            time.sleep(0.015)
            super().send(command)

    result = run_experiment(
        RealtimeRun(
            tmp_path / "hold", RealtimeConfig(pixels=PixelContract(size=(2, 1))), seconds=0.35
        ),
        realtime_environment=SlowSink(),
        numeric_actor_factory=FastActor,
    )
    commands = result.summary["realtime"]["commands"]
    first, second = commands[:2]
    assert second["previous_hold_ns"] == second["returned_ns"] - first["returned_ns"]
    assert (
        second["previous_hold_lower_bound_ns"]
        <= second["previous_hold_ns"]
        <= second["previous_hold_upper_bound_ns"]
    )
    assert second["hold_time_basis"] == "send_return_proxy; game_application_time_unverified"


def test_model_startup_failure_never_claims_exact_replay_evidence(tmp_path):
    def fail_model():
        raise ValueError("weights unavailable")

    game = ThreadedGame()
    result = run_experiment(
        RealtimeRun(
            tmp_path / "startup", RealtimeConfig(pixels=PixelContract(size=(2, 1))), seconds=0.2
        ),
        realtime_environment=game,
        numeric_actor_factory=fail_model,
    )
    r = result.summary["realtime"]
    assert r["stop_reason"] == "model_startup_failed"
    assert r["commands"] == []
    assert r["evidence"]["exact_replay_eligible"] is False
    assert game.closed
