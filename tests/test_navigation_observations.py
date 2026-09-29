"""Optional reference observations at the agreed experiment-run seam."""

import hashlib
import json
import struct
from dataclasses import replace

import pytest
from test_experiment import config_file
from test_observations import motion_packet, observation_fixture
from test_vision import Observations, frame

from fh5.experiment import run_experiment
from fh5.observations import ObservationReplay
from fh5.vision import VisionInput, VisionRecord


def navigation_fixture(tmp_path, **overrides):
    return observation_fixture(
        tmp_path,
        [
            VisionInput(packets=(motion_packet(990),), frame=frame()),
            VisionInput(packets=(motion_packet(1190),)),
        ],
        **{
            "version": 2,
            "reference_mode": "optional",
            "history_offsets_ms": [0],
            "max_image_age_ms": 250,
            "action_history_offsets_ms": [100, 0],
            "max_action_age_ms": 250,
            "navigation": {"display": "unknown", "visibility": "unknown", "evidence": []},
            **overrides,
        },
    )


def test_optional_absence_preserves_constructible_rgb_and_ego_observation(tmp_path):
    request = replace(navigation_fixture(tmp_path), route_file=None)
    obs = run_experiment(request).summary["observations"]
    decision = obs["decisions"][-1]
    assert obs["version"] == 2
    assert decision["usable"] is True
    assert decision["route"]["status"] == "absent"
    assert decision["actor"]["reference"]["waypoints_m"] == [None, None, None]
    assert decision["actor"]["reference"]["mask"] == [False, False, False]
    assert decision["actor"]["ego"]["velocity_car_mps"] == [1.0, 2.0, 10.0]
    assert set(decision["actor"]["ego"]) == {
        "speed_mps",
        "velocity_car_mps",
        "angular_velocity_car_radps",
    }
    assert obs["policy_support"] == "not_evaluated"
    assert obs["task_assessment"]["scorable"] is False


@pytest.mark.parametrize("mode", ["required", "optional", "disabled"])
@pytest.mark.parametrize("asset", ["valid", "absent", "corrupt"])
def test_reference_modes_distinguish_missing_and_quarantined_assets(tmp_path, mode, asset):
    request = navigation_fixture(tmp_path, reference_mode=mode)
    if asset == "absent":
        request = replace(request, route_file=None)
    if asset == "corrupt":
        # A directory cannot be read as a manifest: disabled must never open it.
        request = replace(request, route_file=tmp_path)
    if mode == "required" and asset != "valid":
        with pytest.raises(ValueError, match="Required reference"):
            run_experiment(request)
        return
    obs = run_experiment(request).summary["observations"]
    decision = obs["decisions"][-1]
    expected = (
        "disabled"
        if mode == "disabled"
        else {"valid": "located", "absent": "absent", "corrupt": "invalid_asset"}[asset]
    )
    assert decision["route"]["status"] == expected
    assert decision["usable"]
    assert decision["actor"]["reference"]["mask"] == [expected == "located"] * 3
    assert bool(obs["reference"]["errors"]) == (mode == "optional" and asset == "corrupt")
    if mode == "disabled":
        assert obs["source"]["route_sha256"] is None


def test_legacy_missing_reference_still_rejects(tmp_path):
    request = observation_fixture(tmp_path, [])
    with pytest.raises(ValueError, match="Required reference"):
        run_experiment(replace(request, route_file=None))


def test_evaluation_reference_and_future_telemetry_do_not_change_past_actor_input(tmp_path):
    setup = navigation_fixture(tmp_path)
    request = replace(setup, route_file=None)
    first = run_experiment(request).summary["observations"]
    evaluated = run_experiment(
        replace(
            request, report_path=tmp_path / "evaluated.html", evaluation_route_file=setup.route_file
        )
    ).summary["observations"]
    assert [d["actor"] for d in first["decisions"]] == [d["actor"] for d in evaluated["decisions"]]
    assert first["task_assessment"] != evaluated["task_assessment"]
    assert evaluated["task_assessment"]["status"] == "reference_evidence_only"
    assert not evaluated["task_assessment"]["scorable"]
    assert first["decisions"][-1]["actor"]["images"][0]["path"] == "frames/000000.png"
    serialized = json.dumps(first["decisions"][-1]["actor"])
    for forbidden in (
        "position_m",
        "reference_s_m",
        "packet_index",
        "recording",
        "decision_ns",
        "yaw_rad",
    ):
        assert forbidden not in serialized
    # A future sample on a different road must not rewrite any past actor view.
    journal = request.recording_dir / "packets.jsonl"
    rows = journal.read_text().splitlines()
    extra = json.loads(rows[-1])
    extra["received_monotonic_ns"] += 200_000_000
    # Same payload intentionally creates a later stalled/repeated-game-time diagnostic.
    with journal.open("a") as stream:
        stream.write(json.dumps(extra) + "\n")
    session_file = request.recording_dir / "vision-session.json"
    session = json.loads(session_file.read_text())
    session["hashes"]["packets.jsonl"] = hashlib.sha256(journal.read_bytes()).hexdigest()
    session_file.write_text(json.dumps(session))
    future = run_experiment(replace(request, report_path=tmp_path / "future.html")).summary[
        "observations"
    ]
    assert [d["actor"] for d in future["decisions"][:3]] == [d["actor"] for d in first["decisions"]]


def save_actions(request, rows):
    journal = request.recording_dir / "actions.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (request.recording_dir / "action-history.json").write_text(
        json.dumps(
            {
                "version": 1,
                "mapping_version": "dual-axis-v1",
                "actions_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
                "packets_sha256": hashlib.sha256(
                    (request.recording_dir / "packets.jsonl").read_bytes()
                ).hexdigest(),
            }
        )
    )


def action(occurred_ms, available_ms, steer, segment=0):
    return {
        "occurred_ns": occurred_ms * 1_000_000,
        "available_ns": available_ms * 1_000_000,
        "telemetry_segment": segment,
        "source": "human_input",
        "steer": steer,
        "longitudinal": 0.4,
    }


def test_action_history_excludes_current_labels_future_commands_and_late_old_actions(tmp_path):
    request = replace(
        navigation_fixture(tmp_path, action_history_offsets_ms=[100, 0], max_action_age_ms=250),
        route_file=None,
    )
    save_actions(
        request,
        [
            action(1050, 1060, 0.2),
            action(1080, 1090, 0.3),
            action(1040, 1180, -0.9),
            action(1190, 1190, 0.8),
            action(1200, 1200, 1.0),
        ],
    )
    obs = run_experiment(request).summary["observations"]
    first, middle, last = obs["decisions"]
    assert first["actor"]["actions"] == [None, None]
    assert first["actor"]["action_mask"] == [False, False]
    assert middle["actor"]["actions"] == [None, [0.2, 0.4]]
    assert last["actor"]["actions"] == [[0.2, 0.4], [0.3, 0.4]]
    assert last["actor"]["action_age_ms"] == [140, 110]
    assert last["action_history"][1]["source"] == "human_input"
    assert last["action_history"][1]["available_ns"] == 1_090_000_000


@pytest.mark.parametrize("mode", ["optional", "disabled"])
def test_passive_no_reference_capture_freezes_navigation_conditions_and_replays_identically(
    tmp_path, mode
):
    navigation = {
        "display": "full",
        "visibility": "unknown",
        "evidence": ["User selected full navigation display; no online detector"],
    }
    setup = navigation_fixture(tmp_path, reference_mode=mode, navigation=navigation)
    directory = tmp_path / "capture-v2"
    environment = Observations(
        [VisionInput(packets=(motion_packet(990), motion_packet(1090)), frame=frame())]
    )
    result = run_experiment(
        VisionRecord(
            config_file(tmp_path),
            directory,
            observation_config=setup.config_file,
            route_file=tmp_path / "never-read" if mode == "disabled" else None,
        ),
        vision_environment=environment,
    )
    obs = result.summary["observations"]
    replay = run_experiment(
        ObservationReplay(
            directory, directory / "again.html", None, directory / "observation-config.json"
        )
    )
    assert obs == replay.summary["observations"]
    assert obs["navigation"] == navigation
    assert obs["commands_sent"] is False
    assert environment.closed
    assert not (directory / "observation-route").exists()


def test_cli_replays_optional_reference_and_separate_evaluation_evidence(tmp_path):
    import subprocess
    import sys

    request = navigation_fixture(tmp_path)
    command = subprocess.run(
        [
            sys.executable,
            "-m",
            "fh5",
            "observe",
            str(request.recording_dir),
            "--config",
            str(request.config_file),
            "--evaluation-route",
            str(request.route_file),
            "--report",
            str(request.report_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert command.returncode == 0, command.stderr
    saved = json.loads(request.report_path.with_suffix(".json").read_text(encoding="utf-8"))
    obs = saved["summary"]["observations"]
    assert obs["reference"]["status"] == "absent"
    assert obs["task_assessment"]["status"] == "reference_evidence_only"


@pytest.mark.parametrize("break_kind", ["pause", "restart", "rewind", "gap", "focus_lost"])
def test_actions_are_masked_across_discontinuities_even_if_delivered_afterwards(
    tmp_path, break_kind
):
    setup = navigation_fixture(tmp_path)
    settings = json.loads(setup.config_file.read_text())
    after = motion_packet(1190 if break_kind != "gap" else 1890)
    events = ()
    if break_kind == "pause":
        raw = bytearray(after.payload)
        struct.pack_into("<i", raw, 0, 0)
        after = replace(after, payload=bytes(raw))
    elif break_kind in ("restart", "rewind", "focus_lost"):
        events = ({"kind": break_kind, "observed_ns": 1_150_000_000},)
    second = tmp_path / "boundary"
    second.mkdir()
    request = observation_fixture(
        second,
        [
            VisionInput(packets=(motion_packet(990),), frame=frame()),
            VisionInput(packets=(after,), events=events),
        ],
        **settings,
    )
    request = replace(request, route_file=None)
    save_actions(request, [action(1050, 1060, 0.2), action(1080, 1170, 0.3)])
    last = run_experiment(request).summary["observations"]["decisions"][-1]
    assert last["actor"]["actions"] == [None, None]
    assert last["actor"]["action_mask"] == [False, False]


@pytest.mark.parametrize(
    "fault", ["bad_hash", "nonfinite", "noncausal", "unknown_source", "orphan"]
)
def test_bad_actions_are_quarantined_without_faking_neutral_truth(tmp_path, fault):
    request = replace(navigation_fixture(tmp_path), route_file=None)
    row = action(1050, 1060, 0.2)
    if fault == "nonfinite":
        row["steer"] = float("nan")
    elif fault == "noncausal":
        row["available_ns"] = 1
    elif fault == "unknown_source":
        row["source"] = "telemetry_steer"
    save_actions(request, [row])
    if fault == "bad_hash":
        with (request.recording_dir / "actions.jsonl").open("a") as stream:
            stream.write("{}\n")
    if fault == "orphan":
        (request.recording_dir / "action-history.json").unlink()
    obs = run_experiment(request).summary["observations"]
    assert obs["action_source"]["status"] == "invalid_asset"
    assert obs["action_source"]["errors"]
    assert obs["decisions"][-1]["actor"]["actions"] == [None, None]
    assert obs["policy_support"] == "not_evaluated"


def test_optional_corrupt_reference_hash_is_not_reported_as_normal_absence(tmp_path):
    request = navigation_fixture(tmp_path)
    manifest = json.loads(request.route_file.read_text())
    asset = request.route_file.parent / manifest["assets"]["reference"]["path"]
    asset.write_text("{}")
    obs = run_experiment(request).summary["observations"]
    assert obs["reference"]["status"] == "invalid_asset"
    assert "hash mismatch" in obs["reference"]["errors"][0]
    assert obs["decisions"][-1]["usable"]
    assert obs["decisions"][-1]["actor"]["reference"]["mask"] == [False] * 3


@pytest.mark.parametrize("shape", ["outside", "ambiguous"])
def test_unlocated_optional_reference_masks_waypoints_without_invalidating_rgb(tmp_path, shape):
    from fh5.experiment import Record
    from fh5.routes import BuildRoute

    request = navigation_fixture(tmp_path)
    source = tmp_path / "other-source"
    points = (
        [(100, 0, 0), (100, 0, 10)]
        if shape == "outside"
        else [(0.5, 0, 0), (0.5, 0, 10), (-0.5, 0, 10), (-0.5, 0, 0)]
    )
    run_experiment(
        Record(config_file(tmp_path), source),
        packets=[motion_packet(1000 + i * 100, p) for i, p in enumerate(points)],
    )
    output = tmp_path / "other-route"
    run_experiment(BuildRoute(source, output, 0, len(points) - 1, spacing_m=0.5))
    obs = run_experiment(replace(request, route_file=output / "route.json")).summary["observations"]
    decision = obs["decisions"][-1]
    assert decision["route"]["status"] == (
        "outside_reference" if shape == "outside" else "ambiguous"
    )
    assert decision["usable"]
    assert decision["actor"]["reference"]["waypoints_m"] == [None] * 3


def test_waiting_observation_keeps_fixed_masks_before_first_telemetry(tmp_path):
    setup = navigation_fixture(tmp_path)
    result = run_experiment(
        VisionRecord(
            config_file(tmp_path),
            tmp_path / "waiting",
            observation_config=setup.config_file,
        ),
        vision_environment=Observations(
            [
                VisionInput(),
                VisionInput(packets=(motion_packet(1290),), frame=frame(1200, 1220, 1230)),
            ]
        ),
    )
    first = result.summary["observations"]["decisions"][0]
    assert first["reasons"] == ["missing_telemetry"]
    assert first["actor"]["ego"] is None
    assert first["actor"]["reference"]["mask"] == [False] * 3
    assert first["actor"]["actions"] == [None, None]


def test_v1_does_not_silently_ignore_v2_evaluation_request(tmp_path):
    request = observation_fixture(tmp_path, [])
    with pytest.raises(ValueError, match="Evaluation reference requires observation v2"):
        run_experiment(replace(request, evaluation_route_file=request.route_file))


def test_no_reference_history_rebuilds_after_game_clock_stalls_then_resumes(tmp_path):
    setup = navigation_fixture(
        tmp_path, reference_mode="disabled", max_image_age_ms=1000, max_action_age_ms=1000
    )
    settings = json.loads(setup.config_file.read_text())
    case = tmp_path / "stalled-clock"
    case.mkdir()
    packets = [
        replace(motion_packet(game_ms), received_monotonic_ns=receipt_ms * 1_000_000)
        for receipt_ms, game_ms in [(990, 990), (1090, 990), (1190, 990), (1290, 990), (1390, 1000)]
    ]
    request = observation_fixture(
        case,
        [
            VisionInput(packets=tuple(packets[:2]), frame=frame()),
            VisionInput(packets=tuple(packets[2:4])),
            VisionInput(packets=(packets[-1],)),
        ],
        **settings,
    )
    save_actions(request, [action(1050, 1060, 0.2)])
    decisions = run_experiment(request).summary["observations"]["decisions"]
    assert "stalled_game_clock" in decisions[-2]["reasons"]
    resumed = decisions[-1]
    assert "stalled_game_clock" not in resumed["reasons"]
    assert resumed["history_mask"] == [False]
    assert resumed["actor"]["action_mask"] == [False, False]
    assert not resumed["usable"]
