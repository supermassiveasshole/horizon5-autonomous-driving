"""Event lifecycle through the agreed experiment entry, with an external game fixture."""

import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest
from test_experiment import config_file, sample_packet

from fh5.events import EventInput, EventRun, ScreenFrame
from fh5.experiment import Packet, Replay, run_experiment


def event_config(tmp_path: Path) -> Path:
    path = config_file(tmp_path)
    root = json.loads(path.read_text(encoding="utf-8"))
    for name in ("vehicle", "variant", "tune", "assists", "event", "environment"):
        root["snapshot"][name] = {"value": name, "status": "verified", "evidence": "fixture"}
    root["event_run"] = {
        "version": 1,
        "conditions_verified": False,
        "verification_evidence": [],
        "max_attempts": 2,
        "ready_timeout_s": 2,
        "attempt_timeout_s": 1,
        "restart_timeout_s": 2,
        "expected_car_ordinal": 123,
        "expected_pi": 900,
        "start_position_m": [0, 0, 0],
        "start_radius_m": 5,
        "screen_size": [4, 2],
        "signatures": {},
        "start_steps": [],
        "restart_steps": [],
    }
    path.write_text(json.dumps(root), encoding="utf-8")
    return path


class Game:
    source_kind = "synthetic"

    def __init__(self):
        self.time = 1_000_000_000
        self.pulses = []
        self.released = False
        self.closed = False

    def now_ns(self):
        return self.time

    def read(self, period_s):
        self.time += round(period_s * 1e9)
        data = bytearray(sample_packet())
        struct.pack_into("<I", data, 4, self.time // 1_000_000)
        struct.pack_into("<fff", data, 244, 0, 0, 0)
        struct.pack_into("<f", data, 256, 0)
        data[315] = data[316] = data[320] = 0
        return EventInput(
            packets=(Packet(self.time, "2026-09-28T12:00:00+00:00", bytes(data)),),
            frame=ScreenFrame(self.time, 4, 2, bytes(8)),
            focused=True,
        )

    def pulse(self, button):
        self.pulses.append(button)

    def release(self):
        self.released = True

    def close(self):
        self.closed = True


def test_unverified_empty_event_exits_with_evidence_and_never_presses_start(tmp_path):
    game = Game()
    result = run_experiment(
        EventRun(event_config(tmp_path), tmp_path / "run"), event_environment=game
    )
    assert result.summary["event_run"]["stop_reason"] == "conditions_unverified"
    assert result.summary["event_run"]["attempts"] == []
    assert result.summary["event_run"]["unattended_verified"] is False
    assert game.pulses == []
    assert game.released and game.closed
    assert any(e["kind"] == "conditions_unverified" for e in result.events)
    assert (tmp_path / "run" / "frames" / "000000.pgm").is_file()


def test_visual_templates_are_frozen_before_the_first_observation(tmp_path):
    path = verified_config(tmp_path)

    class ChangedSource(RestartGame):
        def read(self, period_s):
            # An editor changes the source template while this experiment is running.
            (tmp_path / "ready.pgm").write_bytes(b"P5\n4 2\n255\n" + bytes(8))
            return super().read(period_s)

    game = ChangedSource()
    result = run_experiment(EventRun(path, tmp_path / "run"), event_environment=game)
    assert len(result.summary["event_run"]["attempts"]) == 2
    assert result.summary["event_run"]["stop_reason"] == "attempt_limit"


def test_restart_probe_can_check_menus_without_claiming_verified_conditions(tmp_path):
    path = verified_config(tmp_path)
    root = json.loads(path.read_text(encoding="utf-8"))
    root["event_run"].update(conditions_verified=False, purpose="restart_probe")
    root["snapshot"]["tune"]["status"] = "user_reported"
    path.write_text(json.dumps(root), encoding="utf-8")
    game = RestartGame()
    result = run_experiment(EventRun(path, tmp_path / "run"), event_environment=game)
    assert len(result.summary["event_run"]["attempts"]) == 2
    assert result.summary["event_run"]["purpose"] == "restart_probe"
    assert result.summary["event_run"]["conditions_verified"] is False
    assert result.summary["event_run"]["unattended_verified"] is False
    assert game.pulses == ["A", "START", "X", "A", "A"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_attempts", 4),
        ("attempt_timeout_s", 11),
        ("ready_timeout_s", 31),
        ("restart_timeout_s", 31),
    ],
)
def test_restart_probe_rejects_extended_sampling_before_input(tmp_path, field, value):
    path = verified_config(tmp_path)
    root = json.loads(path.read_text(encoding="utf-8"))
    root["event_run"].update(purpose="restart_probe", conditions_verified=False)
    root["event_run"][field] = value
    path.write_text(json.dumps(root), encoding="utf-8")
    game = RestartGame()
    with pytest.raises(ValueError, match="three short stationary"):
        run_experiment(EventRun(path, tmp_path / "run"), event_environment=game)
    assert game.pulses == []


def test_finish_page_accepts_zeroed_inactive_telemetry_without_inventing_a_clock_fault(tmp_path):
    class FinishedGame(RestartGame):
        driving_frames = 0

        def read(self, period_s):
            observation = super().read(period_s)
            if self.screen == "driving":
                self.driving_frames += 1
            if self.driving_frames > 3:
                observation = replace(
                    observation,
                    packets=(Packet(self.time, "2026-09-28T12:00:00+00:00", bytes(324)),),
                    frame=ScreenFrame(self.time, 4, 2, PATTERNS["finish"]),
                )
            return observation

    path = verified_config(tmp_path)
    root = json.loads(path.read_text(encoding="utf-8"))
    root["event_run"]["max_attempts"] = 1
    path.write_text(json.dumps(root), encoding="utf-8")
    result = run_experiment(EventRun(path, tmp_path / "run"), event_environment=FinishedGame())
    assert result.summary["event_run"]["attempts"][0]["outcome"] == "completion_observed"


@pytest.mark.parametrize("transient", [False, True])
def test_restart_probe_stops_if_someone_operates_the_car(tmp_path, transient):
    class DrivenGame(RestartGame):
        def read(self, period_s):
            observation = super().read(period_s)
            data = bytearray(observation.packets[0].payload)
            data[315] = 50
            driven = replace(observation.packets[0], payload=bytes(data))
            packets = (driven,)
            if transient:
                packets = (
                    replace(driven, received_monotonic_ns=self.time - 1),
                    observation.packets[0],
                )
            return replace(observation, packets=packets)

    path = verified_config(tmp_path)
    root = json.loads(path.read_text(encoding="utf-8"))
    root["event_run"].update(purpose="restart_probe", conditions_verified=False)
    path.write_text(json.dumps(root), encoding="utf-8")
    game = DrivenGame()
    result = run_experiment(EventRun(path, tmp_path / "run"), event_environment=game)
    assert result.summary["event_run"]["stop_reason"] == "probe_vehicle_moving_or_input"
    assert game.pulses == []


PATTERNS = {
    "ready": bytes([0, 255, 0, 255, 0, 255, 0, 255]),
    "driving": bytes([255, 0, 255, 0, 255, 0, 255, 0]),
    "pause": bytes([0, 0, 255, 255, 0, 0, 255, 255]),
    "confirm": bytes([255, 255, 0, 0, 255, 255, 0, 0]),
    "finish": bytes([255, 0, 0, 255, 255, 0, 0, 255]),
}


def verified_config(tmp_path):
    path = event_config(tmp_path)
    root = json.loads(path.read_text(encoding="utf-8"))
    config = root["event_run"]
    evidence = tmp_path / "conditions.txt"
    evidence.write_text("Synthetic fixture: empty Goliath, unchanged tune and assists.")
    config.update(
        conditions_verified=True,
        verification_evidence=["conditions.txt"],
        start_steps=[{"screen": "ready", "button": "A"}],
        restart_steps=[
            {"screen": "driving", "button": "START"},
            {"screen": "pause", "button": "X"},
            {"screen": "confirm", "button": "A"},
            {"screen": "ready", "button": "A"},
        ],
        finish_steps=[{"screen": "finish", "button": "X"}, {"screen": "ready", "button": "A"}],
    )
    for name, pixels in PATTERNS.items():
        (tmp_path / f"{name}.pgm").write_bytes(b"P5\n4 2\n255\n" + pixels)
        config["signatures"][name] = [
            {"box": [0, 0, 4, 2], "template": f"{name}.pgm", "max_error": 0.01}
        ]
    path.write_text(json.dumps(root), encoding="utf-8")
    return path


class RestartGame(Game):
    def __init__(self):
        super().__init__()
        self.screen = "ready"

    def read(self, period_s):
        observation = super().read(period_s)
        from dataclasses import replace

        return replace(observation, frame=ScreenFrame(self.time, 4, 2, PATTERNS[self.screen]))

    def pulse(self, button):
        super().pulse(button)
        self.screen = {
            ("ready", "A"): "driving",
            ("driving", "START"): "pause",
            ("pause", "X"): "confirm",
            ("confirm", "A"): "ready",
        }[self.screen, button]


def test_timeout_restart_creates_a_new_attempt_only_after_start_is_verified(tmp_path):
    game = RestartGame()
    result = run_experiment(
        EventRun(verified_config(tmp_path), tmp_path / "run"), event_environment=game
    )
    replay = run_experiment(Replay(tmp_path / "run", tmp_path / "replay.html"))
    lifecycle = result.summary["event_run"]
    assert lifecycle["stop_reason"] == "attempt_limit"
    assert len(lifecycle["attempts"]) == 2
    assert len({a["attempt_id"] for a in lifecycle["attempts"]}) == 2
    assert [a["outcome"] for a in lifecycle["attempts"]] == ["failed", "failed"]
    assert game.pulses == ["A", "START", "X", "A", "A"]
    assert [e["kind"] for e in result.events].count("recovery_verified") == 1
    assert lifecycle == replay.summary["event_run"]
    assert game.released and game.closed


@pytest.mark.parametrize(
    "fault", ["user_stop", "focus_lost", "screen_stale", "telemetry_stale", "capture_failed"]
)
def test_external_faults_release_without_pressing_a_menu_button(tmp_path, fault):
    class FaultyGame(RestartGame):
        def read(self, period_s):
            observation = super().read(period_s)
            if fault == "user_stop":
                return replace(observation, stop_requested=True)
            if fault == "focus_lost":
                return replace(observation, focused=False)
            if fault == "screen_stale":
                return replace(
                    observation, frame=replace(observation.frame, received_monotonic_ns=0)
                )
            if fault == "telemetry_stale":
                return replace(observation, packets=())
            return replace(observation, fault=fault)

    game = FaultyGame()
    result = run_experiment(
        EventRun(verified_config(tmp_path), tmp_path / "run"), event_environment=game
    )
    assert result.summary["event_run"]["stop_reason"] == fault
    assert game.pulses == []
    assert game.released and game.closed


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", 2),
        ("max_attempts", 0),
        ("attempt_timeout_s", float("inf")),
        ("conditions_verified", "yes"),
        ("screen_size", [0, 2]),
    ],
)
def test_invalid_lifecycle_config_is_rejected_before_interaction(tmp_path, key, value):
    path = verified_config(tmp_path)
    root = json.loads(path.read_text(encoding="utf-8"))
    root["event_run"][key] = value
    path.write_text(json.dumps(root), encoding="utf-8")
    game = RestartGame()
    with pytest.raises(ValueError):
        run_experiment(EventRun(path, tmp_path / "run"), event_environment=game)
    assert game.pulses == []
    assert game.closed


def test_failed_button_send_and_release_are_preserved_in_the_report(tmp_path):
    class BrokenGame(RestartGame):
        def pulse(self, button):
            raise OSError("controller detached during pulse")

        def release(self):
            if self.released:
                raise OSError("release failed")
            super().release()

    result = run_experiment(
        EventRun(verified_config(tmp_path), tmp_path / "run"), event_environment=BrokenGame()
    )
    assert result.summary["event_run"]["stop_reason"] == "interface_error"
    assert result.summary["event_run"]["release_sent"] is False
    assert any(e["kind"] == "menu_action" and e["status"] == "failed" for e in result.events)
    assert any(e["kind"] == "release_failed" for e in result.events)


@pytest.mark.parametrize("damage", ["summary", "journal", "frame", "telemetry"])
def test_damaged_event_evidence_cannot_replay_as_a_complete_run(tmp_path, damage):
    directory = tmp_path / "run"
    run_experiment(EventRun(verified_config(tmp_path), directory), event_environment=RestartGame())
    target = {
        "summary": directory / "event-run.json",
        "journal": directory / "event-journal.jsonl",
        "frame": directory / "frames" / "000000.pgm",
        "telemetry": directory / "packets.jsonl",
    }[damage]
    target.write_bytes(b"truncated")
    result = run_experiment(Replay(directory, tmp_path / "damaged.html"))
    assert result.summary["event_run"]["evidence_status"] == "incomplete"
    assert result.summary["event_run"]["release_sent"] is False
    assert any(e["kind"] == "event_evidence_incomplete" for e in result.events)
    if damage == "summary":
        assert [a["outcome"] for a in result.summary["event_run"]["attempts"]] == [
            "failed",
            "failed",
        ]


@pytest.mark.parametrize("fault", ["screen_repeated", "screen_malformed", "clock_stalled"])
def test_invalid_or_reused_observations_never_trigger_menu_input(tmp_path, fault):
    class BadObservation(RestartGame):
        def read(self, period_s):
            observation = super().read(period_s)
            if fault == "clock_stalled":
                self.time = 1_000_000_000
            frame = observation.frame
            if fault == "screen_repeated":
                frame = replace(frame, received_monotonic_ns=1_100_000_000)
            elif fault == "screen_malformed":
                frame = replace(frame, grayscale=b"")
            return replace(observation, frame=frame)

    game = BadObservation()
    # Guard the external fixture so a broken implementation cannot hang this test.
    original_read = game.read
    reads = 0

    def bounded_read(period_s):
        nonlocal reads
        reads += 1
        if reads > 30:
            raise OSError("test fixture exhausted")
        return original_read(period_s)

    game.read = bounded_read
    result = run_experiment(
        EventRun(verified_config(tmp_path), tmp_path / "run"), event_environment=game
    )
    assert result.summary["event_run"]["stop_reason"] == fault
    assert game.pulses == []
    assert game.released and game.closed


@pytest.mark.parametrize("screen", ["finish", "pause", "confirm", "ready"])
def test_only_a_stable_finish_screen_ends_an_attempt_as_completion(tmp_path, screen):
    class EndScreen(RestartGame):
        def read(self, period_s):
            if self.time >= 1_500_000_000:
                self.screen = screen
            return super().read(period_s)

    path = verified_config(tmp_path)
    root = json.loads(path.read_text())
    root["event_run"]["max_attempts"] = 1
    path.write_text(json.dumps(root))
    result = run_experiment(EventRun(path, tmp_path / "run"), event_environment=EndScreen())
    attempts = result.summary["event_run"]["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["outcome"] == ("completion_observed" if screen == "finish" else "failed")


def test_stuck_restart_stops_without_repeating_buttons_or_inventing_an_attempt(tmp_path):
    class Stuck(RestartGame):
        def pulse(self, button):
            if button == "START":
                Game.pulse(self, button)
            else:
                super().pulse(button)

    game = Stuck()
    result = run_experiment(
        EventRun(verified_config(tmp_path), tmp_path / "run"), event_environment=game
    )
    lifecycle = result.summary["event_run"]
    assert lifecycle["stop_reason"] == "restart_timeout"
    assert len(lifecycle["attempts"]) == 1
    assert game.pulses == ["A", "START"]
    assert game.released and game.closed


@pytest.mark.parametrize("fault", ["vehicle_changed", "game_time_jump", "position_jump"])
def test_running_telemetry_discontinuities_invalidate_attempt_before_finish(tmp_path, fault):
    class Discontinuity(RestartGame):
        def read(self, period_s):
            observation = super().read(period_s)
            if self.time >= 1_600_000_000:
                packet = observation.packets[0]
                data = bytearray(packet.payload)
                if fault == "vehicle_changed":
                    struct.pack_into("<i", data, 212, 456)
                elif fault == "game_time_jump":
                    struct.pack_into("<I", data, 4, 0)
                else:
                    struct.pack_into("<fff", data, 244, 1000, 0, 0)
                observation = replace(
                    observation,
                    packets=(replace(packet, payload=bytes(data)),),
                    frame=replace(observation.frame, grayscale=PATTERNS["finish"]),
                )
            return observation

    game = Discontinuity()
    result = run_experiment(
        EventRun(verified_config(tmp_path), tmp_path / "run"), event_environment=game
    )
    lifecycle = result.summary["event_run"]
    assert lifecycle["stop_reason"] == fault
    assert lifecycle["attempts"][0]["outcome"] == "interrupted"
    assert game.pulses == ["A"]
    assert game.released and game.closed


def test_wrong_start_position_does_not_create_an_attempt(tmp_path):
    path = verified_config(tmp_path)
    root = json.loads(path.read_text())
    root["event_run"]["start_position_m"] = [1000, 0, 0]
    path.write_text(json.dumps(root))
    result = run_experiment(EventRun(path, tmp_path / "run"), event_environment=RestartGame())
    assert result.summary["event_run"]["stop_reason"] == "prepare_timeout"
    assert result.summary["event_run"]["attempts"] == []


def test_changed_summary_cannot_override_recorded_attempt_outcomes(tmp_path):
    directory = tmp_path / "run"
    run_experiment(EventRun(verified_config(tmp_path), directory), event_environment=RestartGame())
    path = directory / "event-run.json"
    saved = json.loads(path.read_text())
    saved["summary"]["attempts"][0]["outcome"] = "completion_observed"
    path.write_text(json.dumps(saved))
    replay = run_experiment(Replay(directory, tmp_path / "replay.html"))
    assert replay.summary["event_run"]["evidence_status"] == "incomplete"
    assert replay.summary["event_run"]["attempts"][0]["outcome"] == "failed"
