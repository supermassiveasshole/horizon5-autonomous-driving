"""Presentation growth cannot veto sealed real-time numerical evidence."""

import json
import tracemalloc
from pathlib import Path

import pytest

from fh5.artifacts.io import sha256_file
from fh5.driving.realtime.model import RealtimeConfig, RealtimeNumericReplay, RealtimeRun
from fh5.experiment import run_experiment
from fh5.observation.numeric import PixelContract
from tests.driving.test_realtime import ThreadedGame
from tests.driving.test_realtime_numeric_replay import PixelTimeActor


def record(root, environment):
    return run_experiment(
        RealtimeRun(root, RealtimeConfig(pixels=PixelContract(size=(2, 1))), seconds=0.3),
        realtime_environment=environment,
        numeric_actor_factory=PixelTimeActor,
    )


def replay(root, destination):
    return run_experiment(RealtimeNumericReplay(root, destination), numeric_actor=PixelTimeActor())


def test_report_beyond_old_256_mib_limit_keeps_exact_replay(tmp_path, monkeypatch):
    # The external environment's retained sample diagnostics are genuine report
    # data. Repeated immutable strings keep construction cheap; all bytes must
    # still pass through the public run's writer, manifest and replay reader.
    block = "synthetic</script>" + "x" * (256 * 1024)

    class SampleHistory(ThreadedGame):
        def close(self):
            return {
                **super().close(),
                "source_samples": {
                    "records": [{"index": i, "diagnostic": block} for i in range(1025)]
                },
            }

    root = tmp_path / "recorded"
    original_open = Path.open
    retained_at_html = []

    def observe_html(path, mode="r", *args, **kwargs):
        if path == root / "report.html" and "w" in mode:
            manifest = json.loads((root / "realtime-manifest.json").read_text())
            assert sha256_file(root / "report.json") == manifest["report_sha256"]
            retained_at_html.append(tracemalloc.get_traced_memory()[0])
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", observe_html)
        tracemalloc.start()
        try:
            recorded = record(root, SampleHistory())
        finally:
            tracemalloc.stop()
    original = recorded.summary["realtime"]
    assert (root / "report.json").stat().st_size > 256 * 1024**2
    assert original["evidence"]["recording_complete"]
    assert original["evidence"]["exact_replay_eligible"]
    assert retained_at_html and max(retained_at_html) < (root / "report.json").stat().st_size
    tracemalloc.start()
    try:
        restored = replay(root, tmp_path / "verified.html").summary["realtime_numeric_replay"]
        _, read_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    report_bytes = (root / "report.json").stat().st_size
    (tmp_path / "resource-use.json").write_text(
        json.dumps(
            {
                "report_bytes": report_bytes,
                "retained_at_html_bytes": max(retained_at_html),
                "replay_peak_bytes": read_peak,
            }
        )
    )
    assert read_peak < report_bytes, "Replay retains the complete unused diagnostic history"
    assert restored["verified"] and not restored["errors"]
    assert restored["verified_predictions"] >= 3
    assert restored["decisions"] == original["decisions"]
    assert restored["commands"] == original["commands"]


@pytest.mark.parametrize("failure", [OSError, MemoryError])
def test_html_failure_retains_exact_replay_and_still_rejects_tampering(
    tmp_path, monkeypatch, failure
):
    root = tmp_path / "recorded"
    original_open = Path.open

    def fail_html(path, mode="r", *args, **kwargs):
        if path.suffix == ".html" and "w" in mode:
            raise failure("injected presentation exhaustion")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_html)
        recorded = record(root, ThreadedGame())
        assert recorded.report_path == root / "report.json"
        assert recorded.summary["realtime"]["evidence"]["exact_replay_eligible"]
        assert recorded.summary["realtime"]["presentation"]["status"] == "unavailable"
        restored = replay(root, tmp_path / "verified.html")
        assert restored.report_path == tmp_path / "verified.json"
        assert restored.summary["realtime_numeric_replay"]["verified"]
        assert (
            restored.summary["realtime_numeric_replay"]["presentation"]["status"] == "unavailable"
        )
    bound = sha256_file(root / "report.json")
    assert json.loads((root / "realtime-manifest.json").read_text())["report_sha256"] == bound
    with (root / "report.json").open("ab") as stream:
        stream.write(b" ")
    rejected = replay(root, tmp_path / "rejected.html").summary["realtime_numeric_replay"]
    assert not rejected["verified"]
    assert any("hash mismatch" in row["error"] for row in rejected["errors"])


def test_metric_memory_failure_does_not_lose_completed_decisions(tmp_path, monkeypatch):
    import fh5.reporting.realtime

    def exhausted(values):
        raise MemoryError("injected optional percentile exhaustion")

    with monkeypatch.context() as patch:
        patch.setattr(fh5.reporting.realtime, "percentiles", exhausted)
        recorded = record(tmp_path / "recorded", ThreadedGame())
    summary = recorded.summary["realtime"]
    assert summary["metrics"]["status"] == "unavailable"
    assert summary["evidence"]["exact_replay_eligible"]
    checked = replay(tmp_path / "recorded", tmp_path / "verified.html").summary[
        "realtime_numeric_replay"
    ]
    assert checked["verified"] and checked["verified_predictions"] >= 3


@pytest.mark.parametrize("required", ["report.json", "realtime-manifest.json"])
def test_required_seal_failure_is_reported_and_keeps_original_journal(
    tmp_path, monkeypatch, required
):
    root = tmp_path / "recorded"
    original_open = Path.open

    def fail_required(path, mode="r", *args, **kwargs):
        if path == root / required and any(flag in mode for flag in ("w", "x")):
            raise OSError("injected required evidence failure")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_required)
        with pytest.raises(OSError, match="required evidence failure"):
            record(root, ThreadedGame())
    journal = root / "realtime-events.jsonl"
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert sum(row["kind"] == "stop" for row in events) == 1
    assert any(row["kind"] == "decision_result" for row in events)
    assert any((root / "inputs").glob("*.json"))
    assert not (root / "realtime-manifest.json").exists()
    original_hash = sha256_file(journal)
    checked = replay(root, tmp_path / "rejected.html").summary["realtime_numeric_replay"]
    assert not checked["verified"] and checked["errors"]
    assert sha256_file(journal) == original_hash
