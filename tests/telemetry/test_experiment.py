"""Exercise the agreed experiment-run seam with synthetic external telemetry."""

import json
import struct
from collections.abc import Iterator
from pathlib import Path

import pytest

from fh5.experiment import Packet, Record, Replay, run_experiment


def sample_packet() -> bytes:
    # Synthetic, not game evidence. FH5 offsets: go-forza-telemetry v1.2.0 v2.go.
    payload = bytearray(324)
    struct.pack_into("<iI", payload, 0, 1, 1000)
    struct.pack_into("<iii", payload, 212, 123, 5, 900)
    struct.pack_into("<ffff", payload, 244, 10.0, 2.0, -30.0, 12.5)
    struct.pack_into("<BB", payload, 315, 128, 10)
    struct.pack_into("<b", payload, 320, -32)
    return bytes(payload)


def config_file(tmp_path: Path) -> Path:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "control_source": "human",
                "snapshot": {
                    name: {"value": None, "status": "unverified"}
                    for name in ("vehicle", "variant", "tune", "assists", "event", "environment")
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def test_recording_can_be_replayed_with_raw_times_and_human_control(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    raw = sample_packet()
    result = run_experiment(
        Record(config_file(tmp_path), run_dir, source_kind="synthetic"),
        packets=[Packet(1_000_000_000, "2026-09-28T10:00:00+00:00", raw)],
    )
    replay = run_experiment(Replay(run_dir, tmp_path / "replay.html"))

    assert result.summary["valid_packets"] == replay.summary["valid_packets"] == 1
    assert replay.samples[0]["game_timestamp_ms"] == 1000
    assert replay.samples[0]["received_monotonic_ns"] == 1_000_000_000
    assert replay.samples[0]["position_m"] == [10.0, 2.0, -30.0]
    assert replay.samples[0]["speed_kmh"] == 45.0
    assert replay.samples[0]["telemetry_controls"] == {"accel": 128, "brake": 10, "steer": -32}
    assert replay.samples[0]["command"] is None
    assert replay.metadata["control_source"] == "human"
    assert replay.metadata["source_kind"] == "synthetic"
    assert replay.metadata["game_validation"] == "unverified"
    assert replay.metadata["snapshot"]["variant"]["status"] == "unverified"
    assert raw.hex() in (run_dir / "packets.jsonl").read_text(encoding="utf-8")
    assert replay.report_path.is_file()


def changed_packet(timestamp: int, x: float) -> bytes:
    payload = bytearray(sample_packet())
    struct.pack_into("<I", payload, 4, timestamp)
    struct.pack_into("<f", payload, 244, x)
    return bytes(payload)


def test_replay_keeps_bad_packets_and_breaks_trajectory_at_discontinuities(tmp_path: Path) -> None:
    run_dir = tmp_path / "anomalies"
    run_experiment(
        Record(config_file(tmp_path), run_dir),
        packets=[
            Packet(1_000_000_000, "2026-09-28T10:00:00+00:00", sample_packet()),
            Packet(1_100_000_000, "2026-09-28T10:00:00.1+00:00", b"unknown format"),
            Packet(2_000_000_000, "2026-09-28T10:00:01+00:00", changed_packet(2000, 20)),
            Packet(2_100_000_000, "2026-09-28T10:00:01.1+00:00", changed_packet(900, 21)),
            Packet(2_200_000_000, "2026-09-28T10:00:01.2+00:00", changed_packet(1000, 1000)),
            Packet(
                2_300_000_000, "2026-09-28T10:00:01.3+00:00", changed_packet(1100, float("nan"))
            ),
        ],
    )
    replay = run_experiment(Replay(run_dir, tmp_path / "anomaly-replay.html"))

    assert replay.summary["packet_count"] == 6
    assert replay.summary["valid_packets"] == 4
    assert replay.summary["invalid_packets"] == 2
    kinds = {event["kind"] for event in replay.events}
    assert {
        "unsupported_packet",
        "receive_gap",
        "game_time_discontinuity",
        "position_jump",
        "invalid_value",
    } <= kinds
    assert [sample["segment"] for sample in replay.samples] == [0, 1, 2, 3]
    assert len((run_dir / "packets.jsonl").read_text(encoding="utf-8").splitlines()) == 6


def test_stopping_a_run_preserves_packets_and_a_replayable_interruption(tmp_path: Path) -> None:
    def interrupted_source() -> Iterator[Packet]:
        yield Packet(1_000_000_000, "2026-09-28T10:00:00+00:00", sample_packet())
        raise KeyboardInterrupt

    run_dir = tmp_path / "stopped"
    stopped = run_experiment(Record(config_file(tmp_path), run_dir), packets=interrupted_source())
    replay = run_experiment(Replay(run_dir, tmp_path / "stopped-replay.html"))

    assert stopped.summary["capture_status"] == replay.summary["capture_status"] == "interrupted"
    assert replay.summary["valid_packets"] == 1
    assert any(event["kind"] == "capture_interrupted" for event in replay.events)


def test_verified_configuration_requires_evidence_before_recording(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["snapshot"]["variant"] = {"value": "an unverified guess", "status": "verified"}
    path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "invalid-config"

    with pytest.raises(ValueError, match="evidence"):
        run_experiment(Record(path, output), packets=[])
    assert not output.exists()


def test_report_exposes_replay_data_without_executing_snapshot_text(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["snapshot"]["event"] = {
        "value": "</script><script>window.injected=true</script>",
        "status": "user_reported",
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    result = run_experiment(
        Record(path, tmp_path / "report-run"),
        packets=[Packet(1_000_000_000, "2026-09-28T10:00:00+00:00", sample_packet())],
    )
    page = result.report_path.read_text(encoding="utf-8")
    assert '<canvas id="trajectory"' in page
    assert '<canvas id="speed"' in page
    assert 'type="range"' in page
    assert 'type="application/json"' in page
    assert "</script><script>window.injected=true</script>" not in page
    summary = json.loads((tmp_path / "report-run" / "report.json").read_text(encoding="utf-8"))
    assert summary["summary"]["valid_packets"] == 1
    assert summary["metadata"]["game_validation"] == "unverified"


def test_replay_cannot_overwrite_recording_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "preserved"
    run_experiment(Record(config_file(tmp_path), run_dir), packets=[])
    original = (run_dir / "session.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(Replay(run_dir, run_dir / "session.html"))
    assert (run_dir / "session.json").read_bytes() == original


def test_replay_salvages_records_around_corruption_without_editing_source(tmp_path: Path) -> None:
    run_dir = tmp_path / "damaged"
    run_experiment(
        Record(config_file(tmp_path), run_dir),
        packets=[
            Packet(1_000_000_000, "2026-09-28T10:00:00+00:00", sample_packet()),
            Packet(1_100_000_000, "2026-09-28T10:00:00.1+00:00", changed_packet(1100, 11)),
        ],
    )
    raw_path = run_dir / "packets.jsonl"
    lines = raw_path.read_bytes().splitlines(keepends=True)
    damaged = lines[0] + b'{"received_monotonic_ns":"bad"}\n' + lines[1] + b'{"payload_'
    raw_path.write_bytes(damaged)
    replay = run_experiment(Replay(run_dir, tmp_path / "salvaged.html"))
    assert replay.summary["valid_packets"] == 2
    assert replay.summary["invalid_packets"] == 2
    assert [s["segment"] for s in replay.samples] == [0, 1]
    assert {"corrupt_record", "incomplete_tail"} <= {e["kind"] for e in replay.events}
    assert raw_path.read_bytes() == damaged


@pytest.mark.parametrize("field,value", [("format_version", 99), ("decoder_version", "unknown")])
def test_replay_rejects_unknown_versions(tmp_path: Path, field: str, value: object) -> None:
    run_dir = tmp_path / "future-format"
    run_experiment(Record(config_file(tmp_path), run_dir), packets=[])
    metadata_path = run_dir / "session.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        run_experiment(Replay(run_dir, tmp_path / "future.html"))


def test_pause_and_clock_wrap_are_visible_without_claiming_rewind(tmp_path: Path) -> None:
    paused = bytearray(changed_packet(84, 11))
    struct.pack_into("<i", paused, 0, 0)
    result = run_experiment(
        Record(config_file(tmp_path), tmp_path / "paused"),
        packets=[
            Packet(1_000_000_000, "2026-09-28T10:00:00+00:00", changed_packet(0xFFFFFFF0, 10)),
            Packet(1_100_000_000, "2026-09-28T10:00:00.1+00:00", bytes(paused)),
            Packet(1_200_000_000, "2026-09-28T10:00:00.2+00:00", changed_packet(184, 12)),
        ],
    )
    kinds = {event["kind"] for event in result.events}
    assert {"paused", "resumed", "game_clock_wrap"} <= kinds
    assert "game_time_discontinuity" not in kinds
    assert [s["segment"] for s in result.samples] == [0, 1, 2]
    assert result.summary["receive_span_seconds"] == pytest.approx(0.2)
    assert result.metadata["diagnostics"]["receive_gap_seconds"] == 0.5


def test_inactive_zero_fields_do_not_count_as_driving_position_jumps(tmp_path: Path) -> None:
    result = run_experiment(
        Record(config_file(tmp_path), tmp_path / "activity"),
        packets=[
            Packet(1_000_000_000, "2026-09-28T10:00:00+00:00", bytes(324)),
            Packet(1_100_000_000, "2026-09-28T10:00:00.1+00:00", changed_packet(100, 5700)),
            Packet(1_200_000_000, "2026-09-28T10:00:00.2+00:00", bytes(324)),
        ],
    )
    assert result.summary["valid_packets"] == 3
    assert result.summary["active_packets"] == 1
    assert [s["segment"] for s in result.samples] == [0, 1, 2]
    assert "position_jump" not in {e["kind"] for e in result.events}
