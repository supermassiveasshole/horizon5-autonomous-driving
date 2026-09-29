"""Independent local-data checks through the agreed experiment-run boundary."""

import json
import struct

import pytest

from fh5.experiment import Packet, Record, run_experiment
from fh5.routes import BuildRoute, RouteCheck


def record(tmp_path, name, positions, speed=4):
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
    for i, (x, z) in enumerate(positions):
        raw = bytearray(324)
        struct.pack_into("<iI", raw, 0, 1, 1000 + 100 * i)
        struct.pack_into("<iii", raw, 212, 2941, 6, 999)
        struct.pack_into("<ffff", raw, 244, x, 2, z, speed)
        packets.append(
            Packet(1_000_000_000 + i * 100_000_000, "2026-09-29T00:00:00+00:00", bytes(raw))
        )
    directory = tmp_path / name
    run_experiment(Record(config, directory), packets=packets)
    return directory


def route(tmp_path):
    source = record(tmp_path, "reference", [(0, 0), (1, 0), (2, 0), (3, 0)])
    (tmp_path / "evidence.txt").write_text("Synthetic reviewed area, no game acceptance.")
    review = {"status": "verified", "evidence": ["evidence.txt"]}
    notes = tmp_path / "annotations.json"
    notes.write_text(
        json.dumps(
            {
                "version": 1,
                "reference_review": review,
                "checkpoints_review": review,
                "checkpoints": [],
                "corridors": [
                    {
                        "id": "interior",
                        "s_start_m": 0,
                        "s_end_m": 3,
                        "polygon_xz": [[-1, -1], [4, -1], [4, 1], [-1, 1]],
                        "y_min_m": 1,
                        "y_max_m": 3,
                        **review,
                    }
                ],
            }
        )
    )
    run_experiment(BuildRoute(source, tmp_path / "route", 0, 3, annotations_file=notes))
    return tmp_path / "route/route.json"


def test_independent_short_record_reports_data_gate_without_rewriting_full_attempt(tmp_path):
    bundle = route(tmp_path)
    source = record(
        tmp_path, "independent", [(-5, 0), (0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2), (5, 2)]
    )
    original = (source / "packets.jsonl").read_bytes()
    result = run_experiment(RouteCheck(source, tmp_path / "check.html", bundle, 1, 4, 20))
    check = result.summary["route_check"]
    assert check["passed"] is True
    assert check["reasons"] == []
    assert check["max_speed_kmh"] == 14.4
    assert check["confirmed_progress_m"] == 3
    assert check["recording_packet_count"] == 6
    assert check["selected_packet_count"] == 4
    assert check["source_kind"] == "synthetic"
    assert check["formal_validity"] == "not_evaluated"
    assert result.metadata["analysis_packet_range"] == [1, 4]
    assert [s["packet_index"] for s in result.samples] == [1, 2, 3, 4]
    assert len(check["recording_sha256"]) == 64
    assert len(check["route_sha256"]) == 64
    assert (source / "packets.jsonl").read_bytes() == original


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("fast", "speed_limit_exceeded"),
        ("leave", "unconfirmed_path"),
        ("gap", "multiple_segments"),
        ("bad_packet", "missing_packets"),
        ("end_early", "route_end_missing"),
        ("start_late", "route_start_missing"),
    ],
)
def test_local_gate_cannot_hide_speed_or_route_failures(tmp_path, fault, reason):
    bundle = route(tmp_path)
    points = [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)]
    if fault == "leave":
        points[1] = (1, 2)
    if fault == "end_early":
        points[-1] = (2.5, 0.2)
    if fault == "start_late":
        points[0] = (0.5, 0.2)
    source = record(tmp_path, "drive", points, speed=10 if fault == "fast" else 4)
    path = source / "packets.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == "gap":
        for row in rows[2:]:
            row["received_monotonic_ns"] += 1_000_000_000
    if fault == "bad_packet":
        rows[1]["payload_hex"] = "00"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = run_experiment(RouteCheck(source, tmp_path / "check.html", bundle, 0, 3, 20))
    check = result.summary["route_check"]
    assert not check["passed"]
    assert reason in check["reasons"]
    assert check["formal_validity"] == "not_evaluated"
    if fault == "bad_packet":
        assert result.summary["packet_count"] == 4
        assert result.summary["valid_packets"] == 3
        assert result.summary["invalid_packets"] == 1
        assert check["selected_packet_count"] == 4


def test_reference_self_replay_is_not_independent_acceptance(tmp_path):
    bundle = route(tmp_path)
    source = tmp_path / "reference"
    result = run_experiment(RouteCheck(source, tmp_path / "check.html", bundle, 0, 3, 20))
    assert "reference_source_reused" in result.summary["route_check"]["reasons"]
    assert not result.summary["route_check"]["passed"]


def test_unreviewed_geometry_and_unfinished_recording_stay_unaccepted(tmp_path):
    source = record(tmp_path, "reference", [(0, 0), (1, 0), (2, 0), (3, 0)])
    run_experiment(BuildRoute(source, tmp_path / "unknown", 0, 3))
    independent = record(tmp_path, "drive", [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)])
    session = independent / "session.json"
    data = json.loads(session.read_text())
    data["capture_status"] = "interrupted"
    session.write_text(json.dumps(data))
    result = run_experiment(
        RouteCheck(
            independent,
            tmp_path / "check.html",
            tmp_path / "unknown/route.json",
            0,
            3,
        )
    )
    assert not result.summary["route_check"]["passed"]
    assert {"incomplete_recording", "route_not_reviewed", "unconfirmed_path"} <= set(
        result.summary["route_check"]["reasons"]
    )


def test_invalid_range_or_unbounded_speed_cannot_produce_a_quality_report(tmp_path):
    bundle = route(tmp_path)
    source = tmp_path / "reference"
    for first, last, limit in [
        (2, 1, 20),
        (0, 3, 0),
        (0, 3, 41),
        (0, 3, float("nan")),
        (0, 30, 20),
        (30, 31, 20),
    ]:
        with pytest.raises(ValueError):
            run_experiment(RouteCheck(source, tmp_path / "check.html", bundle, first, last, limit))
        assert not (tmp_path / "check.html").exists()
