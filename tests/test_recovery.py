import json
import subprocess
import sys
from pathlib import Path

import pytest

from fh5.experiment import run_experiment
from fh5.recovery import RecoveryReplay


def replay(tmp_path: Path, inputs: list[dict], *, source_kind="synthetic", **settings):
    config = {
        "version": 1,
        "rewind_available": True,
        "max_rewinds": 2,
        "phase_timeout_s": 2.0,
        "no_progress_timeout_s": 3.0,
        "session_timeout_s": 20.0,
        "max_frame_age_s": 0.25,
        "warmup_frames": 2,
        **settings,
    }
    config_file, trace_file = tmp_path / "config.json", tmp_path / "trace.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    trace_file.write_text(
        json.dumps({"version": 1, "source_kind": source_kind, "inputs": inputs}), encoding="utf-8"
    )
    return run_experiment(RecoveryReplay(config_file, trace_file, tmp_path / "output"))


def event(at: float, kind: str, **values):
    return {"at_s": at, "kind": kind, **values}


def sample(at: float, packet: int, game_ms: int, progress: float, **values):
    return event(
        at,
        "sample",
        observed_s=at,
        packet_index=packet,
        game_time_ms=game_ms,
        progress_m=progress,
        conditions_valid=True,
        route_valid=True,
        neutral=True,
        **values,
    )


def successful_recovery():
    return [
        sample(0, 0, 1000, 5),
        event(0.1, "failure", reason="off_road"),
        event(0.2, "released", request_id=1),
        event(0.3, "rewind_ready", request_id=2, observed_s=0.3),
        event(0.4, "resumed", request_id=3, observed_s=0.4),
        event(0.5, "history_ready", request_id=4, generation=1),
        sample(0.6, 1, 600, 2),
        sample(0.7, 2, 700, 2.1),
        sample(0.8, 3, 800, 2.2),
        event(0.9, "finish"),
    ]


def test_recovery_releases_before_ui_and_keeps_failed_attempt(tmp_path):
    result = replay(tmp_path, successful_recovery())
    recovery = result.summary["recovery"]
    assert [a["command"] for a in recovery["directives"]] == [
        "release",
        "rewind",
        "resume",
        "reset_history",
        "allow_driving",
        "release",
    ]
    assert recovery["attempt"]["outcome"] == "failed"
    assert recovery["attempt"]["finish_observed"] is True
    assert recovery["attempt"]["no_rewind_completion"] is False
    assert recovery["recoveries"][0]["status"] == "recovered"
    assert recovery["fragments"][0]["outcome"] == "failed"
    assert recovery["fragments"][1]["parent_fragment_id"] == "segment-0"
    assert recovery["fragments"][1]["packet_indices"] == [2, 3]
    assert recovery["commands_sent"] is False
    assert result.report_path.is_file()


@pytest.mark.parametrize(
    "index,changes",
    [
        (2, {"request_id": 999}),
        (3, {"observed_s": 0.0}),
        (4, {"observed_s": 0.1}),
        (5, {"generation": 0}),
    ],
)
def test_stale_or_wrong_acknowledgement_never_resumes_driving(tmp_path, index, changes):
    inputs = successful_recovery()
    inputs[index].update(changes)
    result = replay(tmp_path, inputs).summary["recovery"]
    assert result["phase"] == "stopped"
    assert result["reason"] == "invalid_acknowledgement"
    assert not any(d["command"] == "allow_driving" for d in result["directives"])
    assert result["recoveries"][0]["status"] == "failed"


def test_resume_needs_fresh_forward_neutral_samples_and_new_history(tmp_path):
    inputs = successful_recovery()[:6] + [
        sample(0.6, 1, 600, 2),
        sample(0.7, 2, 600, 2),  # Same physics tick cannot complete warmup.
        sample(0.8, 3, 700, 2.1),
        sample(0.9, 4, 800, 2.2),
        sample(1.0, 5, 900, 2.3),
    ]
    inputs[8]["neutral"] = False
    result = replay(tmp_path, inputs).summary["recovery"]
    allowed = [d for d in result["directives"] if d["command"] == "allow_driving"]
    assert [d["at_s"] for d in allowed] == [1.0]
    assert result["fragments"][1]["packet_indices"] == [5]


def test_inverse_time_during_driving_is_not_a_transition(tmp_path):
    result = replay(tmp_path, [sample(0, 0, 1000, 5), sample(0.1, 1, 900, 4)])
    recovery = result.summary["recovery"]
    assert recovery["reason"] == "invalid_forward_sample"
    assert recovery["fragments"][0]["packet_indices"] == [0]
    assert recovery["fragments"][0]["outcome"] == "truncated"


@pytest.mark.parametrize("prefix,deadline", [(2, 2.1), (3, 2.2), (4, 2.3), (5, 2.4)])
def test_each_missing_recovery_acknowledgement_times_out(tmp_path, prefix, deadline):
    result = replay(tmp_path, successful_recovery()[:prefix] + [event(10, "tick")])
    recovery = result.summary["recovery"]
    assert recovery["reason"] == "phase_timeout"
    assert recovery["phase"] == "stopped"
    assert recovery["directives"][-1]["at_s"] == pytest.approx(deadline)
    assert recovery["recoveries"][0]["status"] == "failed"


def test_retries_are_bounded_and_failures_remain_after_two_recoveries(tmp_path):
    inputs = successful_recovery()[:-1] + [
        event(1.0, "failure", reason="missed_checkpoint"),
        event(1.1, "released", request_id=6),
        event(1.2, "rewind_ready", request_id=7, observed_s=1.2),
        event(1.3, "resumed", request_id=8, observed_s=1.3),
        event(1.4, "history_ready", request_id=9, generation=2),
        sample(1.5, 4, 500, 1),
        sample(1.6, 5, 600, 1.1),
        event(1.7, "failure", reason="off_road"),
        event(1.8, "finish"),
    ]
    recovery = replay(tmp_path, inputs).summary["recovery"]
    assert recovery["reason"] == "rewind_limit"
    assert len(recovery["attempt"]["failures"]) == 3
    assert sum(d["command"] == "rewind" for d in recovery["directives"]) == 2
    assert recovery["attempt"]["finish_observed"] is False
    assert recovery["metrics"]["recovered_count"] == 2
    assert recovery["metrics"]["recovery_request_count"] == 3
    assert recovery["metrics"]["recovery_wall_seconds"] == pytest.approx(1.2)


@pytest.mark.parametrize(
    "inputs,settings,reason,last_time",
    [
        (successful_recovery()[:2], {"rewind_available": False}, "rewind_unavailable", 0.1),
        (
            [sample(i / 10, i, 1000 + i * 100, 1) for i in range(30)] + [event(6, "tick")],
            {},
            "phase_timeout",
            5,
        ),
        (
            [
                sample(0, 0, 1000, 1),
                sample(0.5, 1, 1500, 2),
                sample(1, 2, 2000, 3),
                event(6, "tick"),
            ],
            {"session_timeout_s": 1.5, "max_frame_age_s": 1},
            "session_timeout",
            1.5,
        ),
        (successful_recovery()[:3] + [event(0.25, "stop")], {}, "user_stop", 0.25),
        (successful_recovery()[:3] + [event(0.25, "fault")], {}, "interface_fault", 0.25),
        (successful_recovery()[:3], {}, "trace_ended", 0.2),
    ],
)
def test_unavailable_stalled_interrupted_or_incomplete_run_stops(
    tmp_path, inputs, settings, reason, last_time
):
    recovery = replay(tmp_path, inputs, **settings).summary["recovery"]
    assert recovery["phase"] == "stopped"
    assert recovery["reason"] == reason
    assert recovery["directives"][-1]["command"] == "release"
    assert recovery["directives"][-1]["at_s"] == pytest.approx(last_time)
    assert not any(r["status"] == "pending" for r in recovery["recoveries"])
    if reason == "rewind_unavailable":
        assert not any(d["command"] == "rewind" for d in recovery["directives"])
    if last_time == 5:
        assert recovery["attempt"]["failures"] == [{"at_s": 3, "reason": "stalled"}]


@pytest.mark.parametrize(
    "settings",
    [
        {"version": True},
        {"rewind_available": "yes"},
        {"max_rewinds": 1000},
        {"phase_timeout_s": float("nan")},
        {"no_progress_timeout_s": 0},
        {"session_timeout_s": float("inf")},
        {"warmup_frames": 1},
        {"max_frame_age_s": -1},
        {"unused": True},
    ],
)
def test_invalid_limits_are_rejected_before_output(tmp_path, settings):
    with pytest.raises(ValueError):
        replay(tmp_path, successful_recovery(), **settings)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "inputs,source",
    [
        ([], "synthetic"),
        (successful_recovery(), "udp"),
        ([event(0, "invented")], "synthetic"),
        ([sample(1, 0, 1, 0), sample(0, 1, 2, 0)], "synthetic"),
        ([sample(float("nan"), 0, 1, 0)], "synthetic"),
        ([sample(0, True, 1, 0)], "synthetic"),
        ([sample(0, 0, 1, float("inf"))], "synthetic"),
        ([event(0, "failure", reason="invented")], "synthetic"),
        ([event(0, "released", request_id=True)], "synthetic"),
    ],
)
def test_invalid_trace_is_not_simulated_or_promoted_to_game_evidence(tmp_path, inputs, source):
    with pytest.raises(ValueError):
        replay(tmp_path, inputs, source_kind=source)
    assert not (tmp_path / "output").exists()


def test_silence_is_an_interface_fault_not_a_driving_failure(tmp_path):
    recovery = replay(tmp_path, [sample(0, 0, 1000, 1), event(10, "tick")]).summary["recovery"]
    assert recovery["reason"] == "observation_timeout"
    assert recovery["attempt"]["failures"] == []
    assert recovery["directives"][-1]["at_s"] == 0.25


def test_cli_replays_frozen_signals_without_controller_or_game(tmp_path):
    expected = replay(tmp_path, successful_recovery()).summary["recovery"]
    saved = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fh5",
            "recovery-replay",
            "--config",
            str(saved / "config.json"),
            "--trace",
            str(saved / "trace.json"),
            "--output",
            str(tmp_path / "replayed"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["commands_sent"] is False
    assert json.loads((tmp_path / "replayed/recovery.json").read_text(encoding="utf-8")) == expected
