"""Causal driving observations through the agreed experiment-run seam."""

import hashlib
import json
import math
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from fh5.capture.legacy import VisionInput, VisionRecord
from fh5.experiment import Packet, Record, Replay, run_experiment
from fh5.observation.routes import BuildRoute
from tests.capture.test_vision import Observations, frame
from tests.telemetry.test_experiment import changed_packet, config_file


def motion_packet(ms=1000, position=(0.0, 0.0, 0.0), yaw=0.0):
    raw = bytearray(changed_packet(ms, 10))
    struct.pack_into("<fff", raw, 32, 1.0, 2.0, 10.0)
    struct.pack_into("<fff", raw, 44, 0.1, 0.2, 0.3)
    struct.pack_into("<f", raw, 56, yaw)
    struct.pack_into("<fff", raw, 244, *position)
    return Packet(ms * 1_000_000, "2026-09-29T00:00:00+00:00", bytes(raw))


def test_motion_fields_are_decoded_without_turning_telemetry_into_action_labels(tmp_path: Path):
    result = run_experiment(
        Record(config_file(tmp_path), tmp_path / "record"),
        packets=[motion_packet(yaw=math.pi / 2)],
    )
    motion = result.samples[0]["motion"]
    assert motion["yaw_rad"] == pytest.approx(math.pi / 2)
    assert motion["velocity_car_mps"] == [1.0, 2.0, 10.0]
    assert motion["angular_velocity_car_radps"] == pytest.approx([0.1, 0.2, 0.3])
    assert result.samples[0]["command"] is None
    replay = run_experiment(Replay(tmp_path / "record", tmp_path / "replay.html"))
    assert replay.samples[0]["motion"] == motion


def observation_fixture(tmp_path, batches, **overrides):
    from fh5.observation.multimodal import ObservationReplay

    config = config_file(tmp_path)
    source = tmp_path / "reference-source"
    run_experiment(
        Record(config, source),
        packets=[motion_packet(1000 + i * 100, (0, 0, i * 10)) for i in range(6)],
    )
    route = tmp_path / "route"
    run_experiment(BuildRoute(source, route, 0, 5))
    directory = tmp_path / "rgb"
    run_experiment(VisionRecord(config, directory), vision_environment=Observations(batches))
    settings = tmp_path / "observation-settings.json"
    settings.write_text(
        json.dumps(
            {
                "version": 1,
                "period_ms": 100,
                "history_offsets_ms": [100, 0],
                "max_image_age_ms": 150,
                "max_telemetry_age_ms": 100,
                "waypoint_distances_m": [5, 10, 20],
                **overrides,
            }
        ),
        encoding="utf-8",
    )
    return ObservationReplay(
        directory, tmp_path / "observations.html", route / "route.json", settings
    )


def test_observation_ticks_use_only_delivered_history_and_received_telemetry(tmp_path):
    request = observation_fixture(
        tmp_path,
        [
            VisionInput(packets=(motion_packet(990), motion_packet(1090)), frame=frame()),
            VisionInput(
                packets=(motion_packet(1190), motion_packet(1290)), frame=frame(1200, 1220, 1230)
            ),
        ],
    )
    result = run_experiment(request)
    obs = result.summary["observations"]
    first = next(o for o in obs["decisions"] if o["decision_ns"] == 1_190_000_000)
    assert first["telemetry"]["packet_index"] == 2
    assert first["history_mask"] == [False, False]
    assert first["images"][1]["capture_start_ns"] == 1_000_000_000
    assert first["images"][1]["age_ms"] == 190
    assert first["images"][1]["valid"] is False  # Delivery does not refresh capture time.
    assert first["route"]["waypoints_m"] == [[0, 5], [0, 10], [0, 20]]
    assert obs["preprocessing"]["normalization"] == "none_rgb_uint8"


@pytest.mark.parametrize("break_kind", ["focus_lost", "pause", "restart", "gap"])
def test_image_history_is_cut_at_observation_discontinuities(tmp_path, break_kind):
    after = motion_packet(1290 if break_kind != "gap" else 1790)
    events = ()
    if break_kind == "focus_lost":
        events = ({"kind": "focus_lost", "observed_ns": 1_200_000_000},)
    elif break_kind in ("pause", "restart"):
        raw = bytearray(after.payload)
        struct.pack_into(
            "<i" if break_kind == "pause" else "<I", raw, 0 if break_kind == "pause" else 4, 0
        )
        after = Packet(after.received_monotonic_ns, after.received_utc, bytes(raw))
    request = observation_fixture(
        tmp_path,
        [
            VisionInput(packets=(motion_packet(990), motion_packet(1090)), frame=frame()),
            VisionInput(packets=(after,), events=events),
        ],
        history_offsets_ms=[0],
        max_image_age_ms=1000,
    )
    decision = run_experiment(request).summary["observations"]["decisions"][-1]
    assert decision["history_mask"] == [False]
    assert "history_discontinuity" in decision["images"][0]["reasons"]


def test_route_jump_is_not_silently_relocalized_to_future_reference(tmp_path):
    request = observation_fixture(
        tmp_path,
        [
            VisionInput(
                packets=(
                    motion_packet(990),
                    motion_packet(1090, (0, 0, 30)),
                )
            )
        ],
        history_offsets_ms=[0],
    )
    decision = run_experiment(request).summary["observations"]["decisions"][-1]
    assert decision["route"]["status"] == "discontinuity"
    assert decision["route"]["reference_s_m"] is None
    assert decision["route"]["waypoints_m"] == [None, None, None]


def test_passive_capture_and_replay_share_recorded_observation_ticks(tmp_path):
    from fh5.observation.multimodal import ObservationReplay

    setup = observation_fixture(tmp_path, [], history_offsets_ms=[0])
    directory = tmp_path / "new-capture"
    result = run_experiment(
        VisionRecord(
            config_file(tmp_path),
            directory,
            observation_config=setup.config_file,
            route_file=setup.route_file,
        ),
        vision_environment=Observations(
            [
                VisionInput(packets=(motion_packet(990), motion_packet(1090)), frame=frame()),
            ]
        ),
    )
    obs = result.summary["observations"]
    assert obs["clock"] == "recorded_passive_checks_not_policy_calls"
    assert obs["decisions"][0]["decision_ns"] == 1_100_000_000
    assert obs["decisions"][0]["usable"] is True
    replay = run_experiment(
        ObservationReplay(
            directory,
            directory / "again.html",
            directory / "observation-route/route.json",
            directory / "observation-config.json",
        )
    )
    assert replay.summary["observations"] == obs


@pytest.mark.parametrize("fault", ["bad_pixels", "wrong_color", "wrong_size", "unknown_delivery"])
def test_untrustworthy_image_is_not_a_complete_model_observation(tmp_path, fault):
    picture = frame()
    if fault == "bad_pixels":
        picture = replace(picture, encoded=b"not an image")
    elif fault == "wrong_color":
        picture = replace(picture, color="BGR")
    elif fault == "wrong_size":
        picture = replace(picture, size=(2, 2))
    request = observation_fixture(
        tmp_path,
        [
            VisionInput(
                packets=(motion_packet(990), motion_packet(1190)),
                frame=picture,
            )
        ],
        history_offsets_ms=[0],
        max_image_age_ms=250,
    )
    if fault == "unknown_delivery":
        import hashlib

        journal = request.recording_dir / "vision.jsonl"
        row = json.loads(journal.read_text())
        del row["delivered_ns"]
        journal.write_text(json.dumps(row) + "\n")
        session = request.recording_dir / "vision-session.json"
        data = json.loads(session.read_text())
        data["hashes"]["vision.jsonl"] = hashlib.sha256(journal.read_bytes()).hexdigest()
        session.write_text(json.dumps(data))
    decision = run_experiment(request).summary["observations"]["decisions"][-1]
    assert decision["usable"] is False
    assert decision["history_mask"] == [False]


@pytest.mark.parametrize(
    "yaw,expected",
    [
        (0, [0, 5]),
        (math.pi / 2, [-5, 0]),
        (-math.pi / 2, [5, 0]),
        (math.pi, [0, -5]),
        (-math.pi, [0, -5]),
    ],
)
def test_waypoint_right_forward_axes_and_yaw_wrap_have_independent_expected_values(
    tmp_path, yaw, expected
):
    request = observation_fixture(tmp_path, [VisionInput(packets=(motion_packet(990, yaw=yaw),))])
    point = run_experiment(request).summary["observations"]["decisions"][0]["route"]["waypoints_m"][
        0
    ]
    assert point == pytest.approx(expected, abs=1e-5)


def test_source_recording_cannot_supply_its_own_future_navigation_input(tmp_path):
    from fh5.observation.multimodal import ObservationReplay

    request = observation_fixture(tmp_path, [VisionInput(packets=(motion_packet(990),))])
    with pytest.raises(ValueError, match="independent historical"):
        run_experiment(
            ObservationReplay(
                tmp_path / "reference-source",
                tmp_path / "leak.html",
                request.route_file,
                request.config_file,
            )
        )


def test_pure_telemetry_and_invalid_motion_remain_replayable_but_not_complete_inputs(tmp_path):
    request = observation_fixture(
        tmp_path, [VisionInput(packets=(motion_packet(990, yaw=float("nan")),))]
    )
    result = run_experiment(request)
    assert result.summary["valid_packets"] == 1
    assert result.samples[0]["motion"] is None
    assert result.summary["observations"]["usable_decisions"] == 0
    assert result.summary["observations"]["decisions"][0]["route"]["status"] == "unknown_heading"


def test_frozen_game_clock_is_not_fresh_just_because_packets_keep_arriving(tmp_path):
    packets = tuple(
        replace(motion_packet(990), received_monotonic_ns=t * 1_000_000)
        for t in (990, 1090, 1190, 1290)
    )
    request = observation_fixture(
        tmp_path,
        [VisionInput(packets=packets, frame=frame())],
        history_offsets_ms=[0],
        max_image_age_ms=1000,
    )
    decision = run_experiment(request).summary["observations"]["decisions"][-1]
    assert "stalled_game_clock" in decision["reasons"]


def test_route_discontinuity_also_cuts_image_history(tmp_path):
    request = observation_fixture(
        tmp_path,
        [
            VisionInput(packets=(motion_packet(990), motion_packet(1090)), frame=frame()),
            VisionInput(packets=(motion_packet(1190, (0, 0, 30)),)),
        ],
        history_offsets_ms=[0],
        max_image_age_ms=1000,
    )
    decision = run_experiment(request).summary["observations"]["decisions"][-1]
    assert decision["history_mask"] == [False]
    assert "history_discontinuity" in decision["images"][0]["reasons"]


def test_reference_validation_failure_still_closes_passive_environment(tmp_path):
    setup = observation_fixture(tmp_path, [])
    env = Observations([])
    with pytest.raises(FileNotFoundError):
        run_experiment(
            VisionRecord(
                config_file(tmp_path),
                tmp_path / "not-created",
                observation_config=setup.config_file,
                route_file=tmp_path / "missing.json",
            ),
            vision_environment=env,
        )
    assert env.closed


def test_replay_of_v1_telemetry_without_rgb_is_not_a_training_input(tmp_path):
    setup = observation_fixture(tmp_path, [])
    directory = tmp_path / "legacy"
    run_experiment(Record(config_file(tmp_path), directory), packets=[motion_packet(990)])
    session = directory / "session.json"
    metadata = json.loads(session.read_text())
    metadata["decoder_version"] = "fh5-dash-324-v1"
    session.write_text(json.dumps(metadata))
    result = run_experiment(replace(setup, recording_dir=directory))
    assert result.metadata["decoder_version"] == "fh5-dash-324-v1"
    assert result.metadata["analysis_decoder_version"] == "fh5-dash-324-v2"
    assert result.samples[0]["motion"]["velocity_car_mps"] == [1, 2, 10]
    assert result.summary["observations"]["usable_decisions"] == 0
    assert result.summary["observations"]["decisions"][0]["images"] == [None, None]


def test_corrupted_journal_cannot_produce_a_complete_observation(tmp_path):
    request = observation_fixture(
        tmp_path,
        [
            VisionInput(
                packets=(motion_packet(990), motion_packet(1190)),
                frame=frame(),
            )
        ],
        history_offsets_ms=[0],
        max_image_age_ms=250,
    )
    journal = request.recording_dir / "vision.jsonl"
    with journal.open("a") as f:
        f.write('{"kind":"unbound_event"}\n')
    obs = run_experiment(request).summary["observations"]
    assert obs["usable_decisions"] == 0
    assert "artifact_integrity" in obs["decisions"][-1]["reasons"]


def test_future_telemetry_cannot_change_prefix_observations(tmp_path):
    request = observation_fixture(
        tmp_path,
        [
            VisionInput(packets=(motion_packet(990), motion_packet(1090)), frame=frame()),
            VisionInput(
                packets=(motion_packet(1190), motion_packet(1290)), frame=frame(1200, 1220, 1230)
            ),
        ],
        history_offsets_ms=[0],
        max_image_age_ms=250,
    )
    before = run_experiment(request).summary["observations"]["decisions"]
    packets = request.recording_dir / "packets.jsonl"
    rows = [json.loads(line) for line in packets.read_text().splitlines()]
    rows[-1]["payload_hex"] = motion_packet(1290, (200, 0, -100), math.pi).payload.hex()
    packets.write_text("".join(json.dumps(r) + "\n" for r in rows))
    session = request.recording_dir / "vision-session.json"
    meta = json.loads(session.read_text())
    meta["hashes"]["packets.jsonl"] = hashlib.sha256(packets.read_bytes()).hexdigest()
    session.write_text(json.dumps(meta))
    after = run_experiment(replace(request, report_path=tmp_path / "after.html")).summary[
        "observations"
    ]["decisions"]
    assert [d for d in before if d["decision_ns"] < 1_290_000_000] == [
        d for d in after if d["decision_ns"] < 1_290_000_000
    ]


def test_observe_cli_publishes_manifest_without_game_adapters(tmp_path):
    import subprocess
    import sys

    request = observation_fixture(tmp_path, [VisionInput(packets=(motion_packet(990),))])
    command = subprocess.run(
        [
            sys.executable,
            "-m",
            "fh5",
            "observe",
            str(request.recording_dir),
            "--config",
            str(request.config_file),
            "--route",
            str(request.route_file),
            "--report",
            str(request.report_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert command.returncode == 0, command.stderr
    assert json.loads(command.stdout)["observations"]["decision_count"] == 1
    saved = json.loads(request.report_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["summary"]["observations"]["commands_sent"] is False


def test_nearby_return_road_remains_ambiguous_until_position_disambiguates(tmp_path):
    request = observation_fixture(
        tmp_path,
        [
            VisionInput(
                packets=(
                    motion_packet(990, (0.5, 0, 5)),
                    motion_packet(1090, (0.5, 0, 5)),
                )
            )
        ],
    )
    directory = tmp_path / "hairpin-source"
    run_experiment(
        Record(config_file(tmp_path), directory),
        packets=[
            motion_packet(1000 + i * 100, pos)
            for i, pos in enumerate(
                [
                    (0, 0, 0),
                    (0, 0, 10),
                    (1, 0, 10),
                    (1, 0, 0),
                ]
            )
        ],
    )
    route_dir = tmp_path / "hairpin-route"
    run_experiment(BuildRoute(directory, route_dir, 0, 3, spacing_m=0.5))
    result = run_experiment(replace(request, route_file=route_dir / "route.json"))
    assert [o["route"]["status"] for o in result.summary["observations"]["decisions"]] == [
        "ambiguous",
        "ambiguous",
    ]
    assert all(
        o["route"]["waypoints_m"] == [None, None, None]
        for o in result.summary["observations"]["decisions"]
    )


def test_new_frame_after_restart_cannot_use_pre_restart_telemetry(tmp_path):
    request = observation_fixture(
        tmp_path,
        [
            VisionInput(
                packets=(motion_packet(990), motion_packet(1090)),
                frame=frame(1096, 1097, 1098),
                events=({"kind": "restart", "observed_ns": 1_095_000_000},),
            )
        ],
        history_offsets_ms=[0],
    )
    journal = request.recording_dir / "vision.jsonl"
    with journal.open("a") as f:
        f.write(json.dumps({"kind": "observation_tick", "observed_ns": 1_100_000_000}) + "\n")
    session = request.recording_dir / "vision-session.json"
    value = json.loads(session.read_text())
    value["hashes"]["vision.jsonl"] = hashlib.sha256(journal.read_bytes()).hexdigest()
    session.write_text(json.dumps(value))
    decision = run_experiment(request).summary["observations"]["decisions"][0]
    assert not decision["usable"]
    assert decision["telemetry_age_ms"] == 10
    assert "telemetry_discontinuity" in decision["reasons"]
    assert decision["route"]["waypoints_m"] == [None, None, None]


def test_duplicate_history_prefers_latest_slot_without_refreshing_old_image(tmp_path):
    request = observation_fixture(
        tmp_path,
        [
            VisionInput(packets=(motion_packet(990), motion_packet(1090)), frame=frame()),
            VisionInput(packets=(motion_packet(1290),)),
        ],
        max_image_age_ms=1000,
    )
    decisions = run_experiment(request).summary["observations"]["decisions"]
    assert decisions[-2]["images"][-1] is not None
    assert decisions[-1]["images"][-1] is not None  # The display must not blank on reuse.
    assert decisions[-1]["history_mask"] == [False, True]
    assert decisions[-1]["images"][-1]["age_ms"] == 290
    assert decisions[-1]["images"][0] is None  # Never duplicate it into another history slot.
