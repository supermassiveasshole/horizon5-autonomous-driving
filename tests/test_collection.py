"""Continuous passive collection through the agreed experiment entry point."""

import hashlib
import json
import struct
import threading
import time
from dataclasses import replace

import pytest
from test_demonstrations import profile_file, raw_input
from test_observations import motion_packet
from test_realtime import observation

from fh5.collection import (
    CollectionConfig,
    CollectionControl,
    CollectionInput,
    CollectionReview,
    CollectionRun,
)
from fh5.experiment import run_experiment
from fh5.numeric_images import PixelContract


class Stream:
    source_kind = "synthetic"

    def __init__(self, points):
        self.points = iter(points)
        self.closed = False

    def read(self, period_s):
        return next(self.points, None)

    def close(self):
        self.closed = True
        return {"resources_released": True}


def input_at(ms, **changes):
    now = 1_000 + ms
    return replace(
        CollectionInput(
            now * 1_000_000,
            packets=(motion_packet(now - 1),),
            human_input=raw_input(now - 1),
            frames=observation(ms).frames,
            capture_epoch="e1",
            focused=True,
        ),
        **changes,
    )


def request(tmp_path, **changes):
    profile = profile_file(tmp_path)
    data = json.loads(profile.read_bytes())
    data["calibration"] = {"status": "verified", "evidence": ["synthetic-calibration"]}
    profile.write_text(json.dumps(data))
    return CollectionRun(
        tmp_path / "session",
        profile,
        CollectionConfig(
            **{
                "pixels": PixelContract(size=(2, 1)),
                "block_rows": 3,
                "min_free_bytes": 0,
                "expected_car_ordinal": 123,
                "expected_pi": 900,
                **changes,
            }
        ),
    )


def test_sealed_blocks_preserve_numeric_frames_raw_packets_and_human_input(tmp_path):
    req = request(tmp_path)
    source = Stream([input_at(ms) for ms in range(250, 851, 50)])
    collected = run_experiment(req, collection_environment=source).summary["collection"]
    assert source.closed and collected["commands_sent"] is False
    assert collected["stop_reason"] == "source_end"
    assert collected["sealed_blocks"] == 5
    assert collected["seen_rows"] == collected["written_rows"] == 13
    review = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html"))
    r = review.summary["collection"]
    assert r["complete"] and r["verified_blocks"] == 5 and r["errors"] == []
    assert r["missing_rows"] == 0
    assert r["training_eligible"] is False
    rows = [
        json.loads(line)
        for b in sorted((req.output_dir / "blocks").iterdir())
        for line in (b / "rows.jsonl").read_bytes().splitlines()
    ]
    assert rows[0]["human_input"]["raw"]["thumb_lx"] == -32768
    assert rows[0]["mapped_input"]["mapped"] == [-1, 128 / 255]
    assert rows[0]["packets"][0]["payload_hex"] == motion_packet(1249).payload.hex()
    image_row = next(row for row in rows if row["frames"])
    first_block = next(
        b
        for b in (req.output_dir / "blocks").iterdir()
        if image_row["sequence"]
        in [json.loads(line)["sequence"] for line in (b / "rows.jsonl").read_bytes().splitlines()]
    )
    pixels = first_block / image_row["frames"][-1]["path"]
    assert pixels.read_bytes() == bytes([51, 17, 34] * 2)


def saved_rows(root):
    return [
        json.loads(line)
        for b in sorted((root / "blocks").iterdir())
        for line in (b / "rows.jsonl").read_bytes().splitlines()
    ]


@pytest.mark.parametrize("kind", ["focus", "input", "restart", "capture"])
def test_boundaries_preserve_raw_evidence_and_rebuild_history_before_accepting_images(
    tmp_path, kind
):
    points = [input_at(ms) for ms in range(250, 1251, 50)]
    p = points[7]
    if kind == "focus":
        points[7] = replace(p, focused=False)
    elif kind == "input":
        points[7] = replace(p, human_input=raw_input(1599, connected=False))
    elif kind == "restart":
        points[7] = replace(p, packets=(replace(p.packets[0], payload=motion_packet(100).payload),))
    else:
        for i in range(7, len(points)):
            points[i] = replace(
                points[i],
                capture_epoch="e2",
                frames=tuple(replace(f, epoch="e2") for f in points[i].frames),
            )
    req = request(tmp_path)
    result = run_experiment(req, collection_environment=Stream(points)).summary["collection"]
    assert result["seen_rows"] == result["written_rows"] == 21
    rows = saved_rows(req.output_dir)
    assert rows[7]["boundary"] and rows[7]["frames"] == []
    assert any(row["frames"] for row in rows[:7])
    assert any(row["frames"] for row in rows[13:])
    assert all(
        f["source_time_ns"] >= row["segment_start_ns"] for row in rows for f in row["frames"]
    )
    assert all(not row["training_eligible"] for row in rows)
    assert result["confirmed_attempts"] == 0


def test_wrong_vehicle_does_not_become_valid_when_next_poll_has_no_datagram(tmp_path):
    points = [input_at(ms) for ms in range(250, 951, 50)]
    raw = bytearray(points[7].packets[0].payload)
    struct.pack_into("<i", raw, 212, 999)
    points[7] = replace(points[7], packets=(replace(points[7].packets[0], payload=bytes(raw)),))
    points[8] = replace(points[8], packets=())
    req = request(tmp_path)
    run_experiment(req, collection_environment=Stream(points))
    rows = saved_rows(req.output_dir)
    assert rows[8]["input_usable"] is False
    assert "unexpected_vehicle" in rows[8]["reasons"]


@pytest.mark.parametrize("clock", ["stale", "repeated", "overlapping"])
def test_new_delivery_time_cannot_make_an_old_handset_poll_valid(tmp_path, clock):
    points = [input_at(ms) for ms in range(250, 1251, 50)]
    human = dict(points[7].human_input)
    human["observed_ns"] = {
        "stale": 1_000_000,
        "repeated": points[6].human_input["observed_ns"],
        "overlapping": points[6].human_input["available_ns"] - 100,
    }[clock]
    points[7] = replace(points[7], human_input=human)
    req = request(tmp_path)
    run_experiment(req, collection_environment=Stream(points))
    rows = saved_rows(req.output_dir)
    bad = rows[7]
    assert not bad["input_usable"] and not bad["synchronized"] and bad["boundary"]
    assert bad["human_input"]["observed_ns"] == human["observed_ns"]
    assert "stale_human_input" in bad["reasons"] or "input_clock_overlap" in bad["reasons"]
    assert rows[8]["input_usable"] and rows[8]["boundary"]
    assert rows[-1]["synchronized"]


def test_slow_disk_drops_visibly_without_blocking_source_and_keeps_sealed_blocks(tmp_path):
    entered = threading.Event()
    finished_reads = []

    def write(path, data):
        if not entered.is_set():
            entered.set()
            time.sleep(0.6)
        path.write_bytes(data)

    def points():
        yield input_at(250)
        assert entered.wait(1)
        start = time.monotonic()
        for ms in range(300, 5251, 50):
            yield input_at(ms)
        finished_reads.append(time.monotonic() - start)

    req = request(tmp_path, block_rows=1, queue_items=2)
    result = run_experiment(
        req, collection_environment=Stream(points()), collection_write=write
    ).summary["collection"]
    assert finished_reads[0] < 0.3
    assert result["seen_rows"] == 101 and result["dropped_rows"] > 90
    assert not result["complete"] and result["archive_released"]
    review = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html")).summary[
        "collection"
    ]
    assert review["verified_blocks"] > 0
    assert review["missing_rows"] == result["unsealed_rows"]
    assert not review["complete"]


def test_crash_like_missing_final_and_partial_tail_keep_prior_blocks_readable(tmp_path):
    req = request(tmp_path)
    run_experiment(req, collection_environment=Stream(input_at(ms) for ms in range(250, 951, 50)))
    (req.output_dir / "final.json").rename(req.output_dir / "final-before-crash.json")
    index_path = req.output_dir / "index.json"
    index = json.loads(index_path.read_bytes())
    index["blocks"].pop()  # Last seal reached disk just before the index update.
    index_path.write_text(json.dumps(index))
    active = req.output_dir / ".partial/999999"
    active.mkdir()
    (active / "rows.jsonl").write_text("unfinished")
    result = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html")).summary[
        "collection"
    ]
    assert result["verified_blocks"] == 5 and result["errors"] == []
    assert result["active_blocks_ignored"] == 1 and not result["complete"]
    assert result["recovery"] == "sealed_blocks_only; final tail size unknown"


def test_status_and_stop_work_while_stream_is_running_and_flush_the_tail(tmp_path):
    def points():
        for ms in range(250, 50000, 50):
            time.sleep(0.005)
            yield input_at(ms)

    req = request(tmp_path)
    results = []
    thread = threading.Thread(
        target=lambda: results.append(run_experiment(req, collection_environment=Stream(points())))
    )
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while not (req.output_dir / "index.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        status = run_experiment(CollectionControl(req.output_dir)).summary["collection"]
        assert status["seen_rows"] > 0 and not status["final_status_present"]
        assert "not_checked" in status["process_liveness"]
        live_review = run_experiment(
            CollectionReview(req.output_dir, tmp_path / "during.html")
        ).summary["collection"]
        assert live_review["verified_blocks"] > 0 and live_review["errors"] == []
        assert not live_review["final_status_present"] and not live_review["complete"]
        stopped = run_experiment(CollectionControl(req.output_dir, stop=True)).summary["collection"]
        assert stopped["stop_requested"]
    finally:
        (req.output_dir / "stop.request").touch()
        thread.join(timeout=3)
    assert not thread.is_alive()
    final = results[0].summary["collection"]
    assert final["stop_reason"] == "requested_stop" and final["complete"]
    status = run_experiment(CollectionControl(req.output_dir)).summary["collection"]
    assert status["final_status_present"] and status["state"] == "stopped"


def test_corrupted_sealed_pixels_are_reported_without_losing_other_blocks(tmp_path):
    req = request(tmp_path)
    run_experiment(req, collection_environment=Stream(input_at(ms) for ms in range(250, 951, 50)))
    pixel = next((req.output_dir / "blocks").rglob("*.rgb"))
    pixel.write_bytes(bytes([0] * 6))
    result = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html")).summary[
        "collection"
    ]
    assert result["errors"] and not result["complete"]
    assert result["verified_blocks"] == 4


def test_invalid_block_cannot_poison_recovery_of_later_sealed_blocks(tmp_path):
    req = request(tmp_path)
    run_experiment(req, collection_environment=Stream(input_at(ms) for ms in range(250, 951, 50)))
    # Recover without a final/index, as after an interrupted index write. The block's
    # rows remain internally hashed, but its declared sequence bounds disagree.
    (req.output_dir / "final.json").unlink()
    (req.output_dir / "index.json").unlink()
    block = req.output_dir / "blocks" / "000001"
    rows = [json.loads(line) for line in (block / "rows.jsonl").read_bytes().splitlines()]
    for row in rows:
        row["sequence"] += 10000
    payload = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    (block / "rows.jsonl").write_bytes(payload)
    manifest = json.loads((block / "manifest.json").read_bytes())
    manifest["rows_sha256"] = hashlib.sha256(payload).hexdigest()
    (block / "manifest.json").write_text(json.dumps(manifest))
    result = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html")).summary[
        "collection"
    ]
    assert len(result["errors"]) == 1 and result["errors"][0]["block"] == "000001"
    assert result["verified_blocks"] == 4 and result["rows"] == 12
    assert result["missing_rows"] == 3 and not result["complete"]


@pytest.mark.parametrize("reference", ["index", "final"])
def test_sealed_manifest_must_match_each_existing_stable_reference(tmp_path, reference):
    req = request(tmp_path)
    run_experiment(req, collection_environment=Stream(input_at(ms) for ms in range(250, 951, 50)))
    if reference == "index":
        (req.output_dir / "final.json").unlink()
    else:
        (req.output_dir / "index.json").unlink()
    block = req.output_dir / "blocks" / "000001"
    manifest = json.loads((block / "manifest.json").read_bytes())
    manifest["bytes"] += 1
    (block / "manifest.json").write_text(json.dumps(manifest))
    result = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html")).summary[
        "collection"
    ]
    assert not result["complete"] and result["verified_blocks"] == 4
    assert result["rows"] == 12 and result["missing_rows"] == 3
    assert any("manifest reference" in e["error"] for e in result["errors"])


def test_final_cannot_claim_completeness_after_omitting_a_sealed_reference(tmp_path):
    req = request(tmp_path)
    run_experiment(req, collection_environment=Stream(input_at(ms) for ms in range(250, 951, 50)))
    final_path = req.output_dir / "final.json"
    final = json.loads(final_path.read_bytes())
    final["blocks"].pop()
    final_path.write_text(json.dumps(final))
    result = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html")).summary[
        "collection"
    ]
    assert not result["complete"] and result["errors"]
    assert result["verified_blocks"] == 5 and result["rows"] == 15


@pytest.mark.parametrize("fault", ["source_error", "archive_error", "archive_busy", "source_busy"])
def test_final_completeness_cannot_override_its_own_failure_evidence(tmp_path, fault):
    req = request(tmp_path)
    run_experiment(req, collection_environment=Stream(input_at(ms) for ms in range(250, 951, 50)))
    final_path = req.output_dir / "final.json"
    final = json.loads(final_path.read_bytes())
    if fault == "source_error":
        final.update(stop_reason="source_error", error="synthetic source lost")
    elif fault == "archive_error":
        final["archive_error"] = "synthetic disk error"
    elif fault == "archive_busy":
        final["archive_released"] = False
    else:
        final["environment"]["resources_released"] = False
    final_path.write_text(json.dumps(final))
    result = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html")).summary[
        "collection"
    ]
    assert not result["complete"] and result["errors"]
    assert result["verified_blocks"] == 5 and result["rows"] == 15


def test_disk_budget_stops_with_visible_error_and_keeps_earlier_seals(tmp_path):
    req = request(tmp_path, max_disk_bytes=15000, block_rows=1)
    result = run_experiment(
        req, collection_environment=Stream(input_at(ms) for ms in range(250, 2001, 50))
    ).summary["collection"]
    assert result["stop_reason"] == "archive_failure" and not result["complete"]
    assert "disk_budget" in result["archive_error"]
    reviewed = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html")).summary[
        "collection"
    ]
    assert reviewed["verified_blocks"] > 0 and reviewed["missing_rows"] > 0


def test_review_and_status_cli_never_open_input_devices(tmp_path, capsys):
    from fh5.cli import main

    req = request(tmp_path)
    run_experiment(req, collection_environment=Stream([input_at(250), input_at(300)]))
    assert main(["collection-status", str(req.output_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["final_status_present"]
    assert (
        main(["collection-review", str(req.output_dir), "--report", str(tmp_path / "review.html")])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["verified_blocks"] == 1


def test_input_stream_error_retains_sealed_evidence_and_marks_incomplete(tmp_path):
    def points():
        for ms in range(250, 1001, 50):
            yield input_at(ms)
        raise OSError("synthetic source lost")

    req = request(tmp_path)
    source = Stream(points())
    result = run_experiment(req, collection_environment=source).summary["collection"]
    assert source.closed and result["stop_reason"] == "source_error"
    assert "synthetic source lost" in result["error"] and not result["complete"]
    review = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html")).summary[
        "collection"
    ]
    assert review["verified_blocks"] == 6 and not review["complete"]
    assert review["source_status"]["stop_reason"] == "source_error"
    assert "synthetic source lost" in review["source_status"]["error"]


def test_queue_byte_budget_limits_memory_and_reports_every_missing_row(tmp_path):
    req = request(tmp_path, queue_bytes=1)
    result = run_experiment(
        req, collection_environment=Stream(input_at(ms) for ms in range(250, 1001, 50))
    ).summary["collection"]
    assert result["seen_rows"] == result["dropped_rows"] == 16
    assert result["peak_pending_bytes"] == result["written_rows"] == 0
    assert not result["complete"]
    review = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html")).summary[
        "collection"
    ]
    assert review["missing_rows"] == 16 and review["verified_blocks"] == 0


@pytest.mark.parametrize("field", ["human", "frame"])
def test_oversized_source_metadata_stops_before_it_can_bypass_queue_budget(tmp_path, field):
    def points():
        for ms in range(250, 851, 50):
            yield input_at(ms)
        point = input_at(900)
        extra = "x" * 1024**2
        if field == "human":
            yield replace(point, human_input={**point.human_input, "diagnostics": extra})
        else:
            yield replace(
                point,
                frames=tuple(
                    replace(frame, source_layout={**frame.source_layout, "diagnostics": extra})
                    for frame in point.frames
                ),
            )

    req = request(tmp_path, queue_bytes=70000)
    source = Stream(points())
    result = run_experiment(req, collection_environment=source).summary["collection"]
    assert source.closed and result["stop_reason"] == "source_error"
    assert "metadata" in result["error"] and not result["complete"]
    assert result["peak_pending_bytes"] <= 70000
    assert all("diagnostics" not in row["human_input"] for row in saved_rows(req.output_dir))


def test_long_stream_spans_many_seals_with_bounded_pending_memory(tmp_path):
    def points():
        for ms in range(250, 60250, 50):
            time.sleep(0.001)
            yield input_at(ms)

    req = request(tmp_path, block_rows=100, queue_items=512)
    result = run_experiment(req, collection_environment=Stream(points())).summary["collection"]
    assert result["seen_rows"] == result["written_rows"] == 1200
    assert result["sealed_blocks"] == 12 and result["complete"]
    assert result["peak_pending_bytes"] <= req.config.queue_bytes
    review = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html")).summary[
        "collection"
    ]
    assert review["complete"] and review["rows"] == 1200 and review["errors"] == []
