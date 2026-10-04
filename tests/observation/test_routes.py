"""Route construction and localization through the agreed experiment-run seam."""

import json
import struct

import pytest

from fh5.experiment import Packet, Record, Replay, run_experiment
from fh5.observation.routes import BuildRoute


def recording(tmp_path, points, name="source"):
    config = tmp_path / f"{name}.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "control_source": "human",
                "snapshot": {
                    k: {"value": None, "status": "unverified"}
                    for k in ("vehicle", "variant", "tune", "assists", "event", "environment")
                },
            }
        )
    )
    packets = []
    for i, (x, z) in enumerate(points):
        raw = bytearray(324)
        struct.pack_into("<iI", raw, 0, 1, 1000 + i * 100)
        struct.pack_into("<iii", raw, 212, 2941, 5, 999)
        struct.pack_into("<ffff", raw, 244, x, 2, z, 10)
        packets.append(
            Packet(1_000_000_000 + i * 100_000_000, "2026-09-29T00:00:00+00:00", bytes(raw))
        )
    path = tmp_path / name
    run_experiment(Record(config, path), packets=packets)
    return path


def test_recorded_reference_is_separate_from_unknown_road_boundaries(tmp_path):
    source = recording(tmp_path, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    result = run_experiment(BuildRoute(source, tmp_path / "route", 0, 4))
    route = result.summary["route"]
    assert route["length_m"] == 4
    assert route["reference_points"] == 3
    assert route["verified_corridor_length_m"] == 0
    assert route["low_speed_ready"] is False
    assert route["corridors"] == []
    assert route["checkpoints"] == []
    replay = run_experiment(
        Replay(source, tmp_path / "route-replay.html", route_file=tmp_path / "route/route.json")
    )
    assert [s["route"]["reference_s_m"] for s in replay.samples] == [0, 1, 2, 3, 4]
    assert {s["route"]["status"] for s in replay.samples} == {"unverified_corridor"}
    assert replay.samples[-1]["route"]["confirmed_progress_m"] == 0
    assert replay.summary["route"]["source"]["first_packet"] == 0
    assert replay.summary["route"]["source"]["last_packet"] == 4


@pytest.mark.parametrize(
    "fault", ["rewind", "pause", "gap", "car", "stationary", "bounds", "spacing"]
)
def test_route_build_rejects_discontinuous_or_unusable_reference_without_publishing(
    tmp_path, fault
):
    source = recording(tmp_path, [(0, 0), (1, 0), (2, 0)])
    raw_path = source / "packets.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    raw = bytearray.fromhex(rows[1]["payload_hex"])
    if fault == "rewind":
        struct.pack_into("<I", raw, 4, 900)
    elif fault == "pause":
        struct.pack_into("<i", raw, 0, 0)
    elif fault == "gap":
        rows[1]["received_monotonic_ns"] += 700_000_000
    elif fault == "car":
        struct.pack_into("<i", raw, 212, 100)
    rows[1]["payload_hex"] = raw.hex()
    if fault == "stationary":
        for row in rows:
            payload = bytearray.fromhex(row["payload_hex"])
            struct.pack_into("<f", payload, 244, 0)
            row["payload_hex"] = payload.hex()
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="reference|spacing|packet"):
        run_experiment(
            BuildRoute(
                source,
                tmp_path / "bad-route",
                0,
                10 if fault == "bounds" else 2,
                spacing_m=0 if fault == "spacing" else 2,
            )
        )
    assert not (tmp_path / "bad-route").exists()


def annotations(tmp_path):
    (tmp_path / "survey.txt").write_text(
        "Synthetic surveyed strip and gate; not real game evidence."
    )
    review = {"status": "verified", "evidence": ["survey.txt"]}
    value = {
        "version": 1,
        "reference_review": review,
        "checkpoints_review": review,
        "corridors": [
            {
                "id": "strip",
                "s_start_m": 0,
                "s_end_m": 4,
                "polygon_xz": [[-1, -1], [5, -1], [5, 1], [-1, 1]],
                "y_min_m": 1,
                "y_max_m": 3,
                **review,
            }
        ],
        "checkpoints": [
            {
                "id": "gate-1",
                "s_m": 2,
                "left_xz": [2, -1],
                "right_xz": [2, 1],
                "y_min_m": 1,
                "y_max_m": 3,
                **review,
            }
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(value))
    return path


def test_surveyed_corridor_and_ordered_gate_allow_confirmed_progress_and_freeze_evidence(tmp_path):
    source = recording(tmp_path, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    note = annotations(tmp_path)
    result = run_experiment(BuildRoute(source, tmp_path / "route", 0, 4, annotations_file=note))
    assert result.summary["route"]["verified_corridor_length_m"] == 4
    assert result.summary["route"]["low_speed_ready"] is True
    assert [s["route"]["confirmed_progress_m"] for s in result.samples] == [0, 1, 2, 3, 4]
    assert result.samples[-1]["route"]["next_checkpoint"] is None
    assert any(
        e["kind"] == "route_checkpoint_passed" and e["checkpoint"] == "gate-1"
        for e in result.events
    )
    (tmp_path / "survey.txt").write_text("Changed after export")
    replay = run_experiment(
        Replay(source, tmp_path / "again.html", route_file=tmp_path / "route/route.json")
    )
    assert replay.summary["route"] == result.summary["route"]
    assert replay.samples == result.samples


@pytest.mark.parametrize(
    "case", ["leave-and-rejoin", "miss-gate", "unknown-gap", "wrong-height", "late-start"]
)
def test_invalid_or_unobserved_path_cannot_earn_progress_after_rejoining(tmp_path, case):
    source = recording(tmp_path, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    notes = annotations(tmp_path)
    cfg = json.loads(notes.read_text())
    drive = [(0, 0), (1, 0), (2, 2), (3, 0), (4, 0)]
    if case == "miss-gate":
        cfg["checkpoints"][0].update(left_xz=[2, -0.2], right_xz=[2, 0.2])
        drive = [(0, 0), (1, 0), (2, 0.8), (3, 0), (4, 0)]
    elif case == "unknown-gap":
        first = cfg["corridors"][0]
        cfg["corridors"] = [{**first, "s_end_m": 1}, {**first, "id": "second", "s_start_m": 3}]
        drive = [(0, 0), (1, 0), (3, 0), (4, 0)]
    elif case == "wrong-height":
        cfg["corridors"][0].update(y_min_m=10, y_max_m=12)
        drive = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    elif case == "late-start":
        drive = [(3, 0), (4, 0)]
    notes.write_text(json.dumps(cfg))
    run_experiment(BuildRoute(source, tmp_path / "route", 0, 4, annotations_file=notes))
    attempt = recording(tmp_path, drive, "attempt")
    result = run_experiment(
        Replay(attempt, tmp_path / "checked.html", tmp_path / "route/route.json")
    )
    assert result.samples[-1]["route"]["confirmed_progress_m"] == (
        0 if case in ("wrong-height", "late-start") else 1
    )
    assert not any(e["kind"] == "route_checkpoint_passed" for e in result.events)


def test_nearby_return_road_cannot_jump_forward_in_route_index(tmp_path):
    points = (
        [(x, 0) for x in range(11)]
        + [(10, z) for z in range(1, 5)]
        + [(x, 4) for x in range(9, -1, -1)]
    )
    source = recording(tmp_path, points)
    run_experiment(BuildRoute(source, tmp_path / "route", 0, len(points) - 1, spacing_m=1))
    attempt = recording(tmp_path, [(0, 0), (1, 0), (2, 3.8)], "adjacent")
    result = run_experiment(
        Replay(attempt, tmp_path / "nearby.html", tmp_path / "route/route.json")
    )
    assert result.samples[-1]["route"]["reference_s_m"] <= 4.5
    assert result.samples[-1]["route"]["confirmed_progress_m"] == 0


def test_repeated_distance_and_restart_do_not_duplicate_confirmed_progress(tmp_path):
    source = recording(tmp_path, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    run_experiment(
        BuildRoute(source, tmp_path / "route", 0, 4, annotations_file=annotations(tmp_path))
    )
    attempt = recording(
        tmp_path, [(0, 0), (1, 0), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 0), (1, 0)], "repeat"
    )
    raw_path = attempt / "packets.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    for i in (7, 8):
        raw = bytearray.fromhex(rows[i]["payload_hex"])
        struct.pack_into("<I", raw, 4, 100 + (i - 7) * 100)
        rows[i]["payload_hex"] = raw.hex()
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = run_experiment(
        Replay(attempt, tmp_path / "repeat-analysis.html", tmp_path / "route/route.json")
    )
    assert [s["route"]["new_progress_m"] for s in result.samples] == [0, 1, 0, 0, 1, 1, 1, 0, 1]
    assert result.samples[7]["route"]["confirmed_progress_m"] == 0
    assert result.samples[7]["route"]["next_checkpoint"] == "gate-1"


@pytest.mark.parametrize("damage", ["reference", "evidence", "version"])
def test_route_replay_rejects_changed_or_unsupported_bundle_before_reporting(tmp_path, damage):
    source = recording(tmp_path, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    run_experiment(
        BuildRoute(source, tmp_path / "route", 0, 4, annotations_file=annotations(tmp_path))
    )
    if damage == "reference":
        (tmp_path / "route/reference.json").write_text("{}")
    elif damage == "evidence":
        next((tmp_path / "route/evidence").iterdir()).write_text("changed")
    else:
        manifest = tmp_path / "route/route.json"
        value = json.loads(manifest.read_text())
        value["version"] = 99
        manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash|version"):
        run_experiment(Replay(source, tmp_path / "damaged.html", tmp_path / "route/route.json"))
    assert not (tmp_path / "damaged.html").exists()


@pytest.mark.parametrize(
    "fault", ["no-evidence", "concave", "nonfinite", "gate-order", "range", "unknown-status"]
)
def test_untrustworthy_annotations_are_rejected_without_publishing(tmp_path, fault):
    source = recording(tmp_path, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    notes = annotations(tmp_path)
    value = json.loads(notes.read_text())
    if fault == "no-evidence":
        value["reference_review"]["evidence"] = []
    elif fault == "concave":
        value["corridors"][0]["polygon_xz"] = [[-1, -1], [5, -1], [0, 0], [5, 1], [-1, 1]]
    elif fault == "nonfinite":
        value["corridors"][0]["y_max_m"] = float("nan")
    elif fault == "gate-order":
        value["checkpoints"] *= 2
    elif fault == "range":
        value["corridors"][0]["s_end_m"] = 100
    else:
        value["corridors"][0]["status"] = "trust-me"
    notes.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        run_experiment(BuildRoute(source, tmp_path / "route", 0, 4, annotations_file=notes))
    assert not (tmp_path / "route").exists()


@pytest.mark.parametrize("fault", ["height", "gate"])
def test_verified_labels_cannot_override_geometry_in_route_quality(tmp_path, fault):
    source = recording(tmp_path, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    notes = annotations(tmp_path)
    data = json.loads(notes.read_text())
    if fault == "height":
        data["corridors"][0].update(y_min_m=10, y_max_m=12)
    else:
        data["checkpoints"][0].update(left_xz=[100, -1], right_xz=[100, 1])
    notes.write_text(json.dumps(data))
    result = run_experiment(BuildRoute(source, tmp_path / "route", 0, 4, annotations_file=notes))
    assert result.summary["route"]["low_speed_ready"] is False
    if fault == "height":
        assert result.summary["route"]["verified_corridor_length_m"] == 0


def test_route_localization_stays_on_the_current_pass_at_a_self_crossing(tmp_path):
    points = [
        (-2, -2),
        (-1, -1),
        (0, 0),
        (1, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (-1, 2),
        (-2, 2),
        (-1, 1),
        (0, 0),
        (1, -1),
        (2, -2),
    ]
    source = recording(tmp_path, points)
    result = run_experiment(BuildRoute(source, tmp_path / "route", 0, 12, spacing_m=0.25))
    assert result.samples[2]["route"]["reference_s_m"] == pytest.approx(2.828427)
    assert result.samples[10]["route"]["reference_s_m"] > 12


def test_quantized_game_clock_allows_brief_repeats_but_rejects_a_stalled_reference(tmp_path):
    source = recording(tmp_path, [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    raw_path = source / "packets.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    raw = bytearray.fromhex(rows[1]["payload_hex"])
    struct.pack_into("<I", raw, 4, 1000)
    rows[1]["payload_hex"] = raw.hex()
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = run_experiment(
        BuildRoute(source, tmp_path / "brief", 0, 4, annotations_file=annotations(tmp_path))
    )
    assert result.samples[-1]["route"]["confirmed_progress_m"] == 4
    for row in rows:
        raw = bytearray.fromhex(row["payload_hex"])
        struct.pack_into("<I", raw, 4, 1000)
        row["payload_hex"] = raw.hex()
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="clock"):
        run_experiment(BuildRoute(source, tmp_path / "stalled", 0, 4))


@pytest.mark.parametrize("z", [0.8, -0.8])
def test_slanted_gate_is_tracked_when_crossed_before_or_after_reference_station(tmp_path, z):
    source = recording(tmp_path, [(x, 0) for x in range(5)])
    notes = annotations(tmp_path)
    data = json.loads(notes.read_text())
    data["checkpoints"][0].update(left_xz=[1, 1], right_xz=[3, -1])
    notes.write_text(json.dumps(data))
    run_experiment(BuildRoute(source, tmp_path / "route", 0, 4, annotations_file=notes))
    drive = recording(tmp_path, [(x / 10, z) for x in range(41)], "slanted")
    result = run_experiment(
        Replay(drive, tmp_path / "slanted-review.html", tmp_path / "route/route.json")
    )
    assert result.samples[-1]["route"]["confirmed_progress_m"] == 4
    assert [e["checkpoint"] for e in result.events if e["kind"] == "route_checkpoint_passed"] == [
        "gate-1"
    ]


def test_wrong_gate_station_on_a_coarse_reference_edge_cannot_pass_quality_check(tmp_path):
    source = recording(tmp_path, [(x, 0) for x in range(5)])
    notes = annotations(tmp_path)
    data = json.loads(notes.read_text())
    data["checkpoints"][0].update(left_xz=[0.5, -1], right_xz=[0.5, 1], s_m=3)
    notes.write_text(json.dumps(data))
    result = run_experiment(
        BuildRoute(source, tmp_path / "route", 0, 4, spacing_m=4, annotations_file=notes)
    )
    assert result.summary["route"]["checkpoint_geometry_consistent"] is False
    assert result.summary["route"]["low_speed_ready"] is False


def test_sideways_correction_can_cross_a_slanted_gate_in_the_forward_direction(tmp_path):
    source = recording(tmp_path, [(x, 0) for x in range(5)])
    notes = annotations(tmp_path)
    data = json.loads(notes.read_text())
    data["checkpoints"][0].update(left_xz=[1, 1], right_xz=[3, -1])
    notes.write_text(json.dumps(data))
    run_experiment(BuildRoute(source, tmp_path / "route", 0, 4, annotations_file=notes))
    drive = recording(
        tmp_path, [(0, -0.8), (1, -0.8), (2, -0.8), (2, 0.8), (3, 0.8), (4, 0.8)], "lateral"
    )
    result = run_experiment(
        Replay(drive, tmp_path / "lateral-review.html", tmp_path / "route/route.json")
    )
    assert result.samples[-1]["route"]["confirmed_progress_m"] == 4
    assert [e["packet_index"] for e in result.events if e["kind"] == "route_checkpoint_passed"] == [
        3
    ]


def test_crossing_two_gates_in_reverse_order_between_samples_is_not_legal_progress(tmp_path):
    source = recording(tmp_path, [(x, 0) for x in range(5)])
    notes = annotations(tmp_path)
    data = json.loads(notes.read_text())
    gate = data["checkpoints"][0]
    data["checkpoints"] = [
        {**gate, "s_m": 1.5, "left_xz": [0.5, -1], "right_xz": [2.5, 1]},
        {**gate, "id": "gate-2", "s_m": 2.5, "left_xz": [1.5, 1], "right_xz": [3.5, -1]},
    ]
    notes.write_text(json.dumps(data))
    run_experiment(BuildRoute(source, tmp_path / "route", 0, 4, annotations_file=notes))
    drive = recording(tmp_path, [(0, 0.8), (4, 0.8)], "reverse-gates")
    path = drive / "packets.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["received_monotonic_ns"] = 1_400_000_000
    raw = bytearray.fromhex(rows[1]["payload_hex"])
    struct.pack_into("<I", raw, 4, 1400)
    rows[1]["payload_hex"] = raw.hex()
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = run_experiment(
        Replay(drive, tmp_path / "reverse-gates-review.html", tmp_path / "route/route.json")
    )
    assert result.samples[-1]["route"]["confirmed_progress_m"] == 0
    assert not any(e["kind"] == "route_checkpoint_passed" for e in result.events)
