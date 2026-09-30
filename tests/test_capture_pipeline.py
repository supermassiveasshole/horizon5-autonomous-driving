"""Capture and history behavior through the agreed experiment-run interface."""

import hashlib
import threading
import time
from dataclasses import replace

import pytest

from fh5.capture import (
    CaptureConfig,
    CaptureEvent,
    CaptureReplay,
    CaptureRun,
    QpcMapping,
    RawCapture,
)
from fh5.cli import main
from fh5.experiment import run_experiment
from fh5.numeric_images import PixelContract

ANCHOR = 2**60


def frame(index, *, duration_ms=5, layout=None):
    return CaptureEvent(
        received_ns=1_002_000_000 + index * 100_000_000,
        frame=RawCapture(
            present_ticks=ANCHOR + index * 1_000_000,
            mapping=QpcMapping(ANCHOR, 1_000_000_000, 10_000_000, 200),
            size=(2, 1),
            bgra=bytes([30, 20, 10 + index, 255] * 2),
            layout=layout or {"client_size": [2, 1], "crop": [0, 0, 2, 1]},
        ),
        processing_ns=duration_ms * 1_000_000,
    )


def config(**kwargs):
    return CaptureConfig(pixels=PixelContract(size=(2, 1)), **kwargs)


def test_capture_uses_raw_qpc_delta_and_only_completed_unique_images(tmp_path):
    result = run_experiment(
        CaptureReplay(
            tmp_path / "probe",
            config(),
            tuple(frame(i) for i in range(3)),
            (1_204_000_000, 1_210_000_000),
        )
    )
    capture = result.summary["capture"]
    assert capture["commands_sent"] is False
    assert capture["timing_kind"] == "simulated"
    first, ready = capture["decisions"]
    assert first["status"] == "skip_not_ready"
    assert ready["status"] == "ready"
    assert ready["timing"]["adjacent_delta_s"] == [0.1, 0.1]
    assert ready["timing"]["image_age_s"] == [0.21, 0.11, 0.01]
    assert ready["frames"][-1]["source_time_ns"] == 1_200_000_000
    path = tmp_path / "probe" / ready["stored_frames"][-1]["path"]
    assert path.read_bytes() == bytes([12, 20, 30] * 2)
    assert capture["new_frames"] == 3


def test_slow_preprocess_finishes_inflight_and_overwrites_only_pending(tmp_path):
    result = run_experiment(
        CaptureReplay(
            tmp_path / "slow",
            CaptureConfig(
                pixels=PixelContract(size=(2, 1), history_offsets_ms=(200, 0)), max_age_ms=300
            ),
            tuple(frame(i, duration_ms=150) for i in range(6)),
            (1_610_000_000,),
        )
    )
    capture = result.summary["capture"]
    row = capture["decisions"][0]
    assert row["status"] == "ready"
    assert [f["frame_id"] for f in row["frames"]] == ["f3", "f5"]
    assert capture["pending_overwritten"] == 1
    assert capture["preprocessed"] == 4
    assert capture["pending_capacity"] == 1
    assert capture["resources_released"] is True


def test_mouse_only_duplicate_and_future_frames_do_not_become_new_images(tmp_path):
    base = frame(0)
    assert base.frame is not None
    events = (
        base,
        replace(base, received_ns=1_020_000_000),
        replace(base, received_ns=1_030_000_000, frame=replace(base.frame, present_ticks=0)),
        replace(frame(2), received_ns=1_040_000_000),
        frame(1),
        frame(2),
    )
    result = run_experiment(
        CaptureReplay(tmp_path / "duplicates", config(), events, (1_210_000_000,))
    )
    capture = result.summary["capture"]
    assert capture["new_frames"] == 3
    assert capture["no_present"] == 1
    assert capture["nonforward_present"] == 1
    assert capture["future_present"] == 1
    assert capture["decisions"][0]["status"] == "ready"


def test_layout_change_and_access_loss_end_old_history_and_inflight_work(tmp_path):
    result = run_experiment(
        CaptureReplay(
            tmp_path / "epochs",
            config(),
            (
                frame(0),
                frame(1),
                frame(2, duration_ms=150),
                frame(3, layout={"client_size": [2, 1], "crop": [10, 0, 12, 1]}),
                CaptureEvent(1_360_000_000, boundary="access_lost"),
                frame(4),
                frame(5),
                frame(6),
            ),
            (1_355_000_000, 1_610_000_000, 1_900_000_000),
        )
    )
    capture = result.summary["capture"]
    before, after, stale = capture["decisions"]
    assert before["status"] == "skip_not_ready"
    assert after["status"] == "ready"
    assert {f["epoch"] for f in after["frames"]} == {after["epoch"]}
    assert [f["source_time_ns"] for f in after["frames"]] == [
        1_400_000_000,
        1_500_000_000,
        1_600_000_000,
    ]
    assert stale["reason"] == "stale_latest_image"
    assert capture["old_epoch_completion"] == 1
    assert capture["epoch_changes"] == 2


def test_live_workers_capture_numeric_history_and_release_source(tmp_path):
    class Source:
        source_kind = "synthetic"
        closed = False

        def capture(self):
            now = time.perf_counter_ns()
            raw = RawCapture(
                now - 1_000_000,
                QpcMapping(0, 0, 1_000_000_000, 0),
                (2, 1),
                bytes([30, 20, 10, 255] * 2),
                {"client_size": [2, 1], "crop": [0, 0, 2, 1]},
            )
            return CaptureEvent(now, frame=raw)

        def close(self):
            self.closed = True

    source = Source()
    result = run_experiment(
        CaptureRun(tmp_path / "workers", config(), seconds=0.5),
        capture_source_factory=lambda: source,
    )
    capture = result.summary["capture"]
    assert source.closed
    assert capture["resources_released"] is True
    assert capture["commands_sent"] is False
    assert capture["timing_kind"] == "measured"
    assert any(row["status"] == "ready" for row in capture["decisions"])
    assert capture["archive"]["resources_released"] is True
    assert capture["pipeline"]["history_peak"] <= 32


def test_dxgi_adapter_waits_for_target_without_capturing_desktop(tmp_path):
    from fh5.dxgi_capture import DXGIFrames, DXGISettings

    def camera_factory(settings):
        raise AssertionError("No camera may open before a foreground FH5 client exists")

    source = DXGIFrames(DXGISettings(), target=lambda: None, camera_factory=camera_factory)
    result = run_experiment(
        CaptureRun(tmp_path / "absent", config(), seconds=0.12),
        capture_source_factory=lambda: source,
    )
    capture = result.summary["capture"]
    assert capture["pipeline"]["target_unavailable"] > 0
    assert all(row["status"] == "skip_not_ready" for row in capture["decisions"])
    assert capture["resources_released"] is True


def test_dxgi_crop_is_physical_client_and_mouse_only_updates_are_excluded(tmp_path):
    from fh5.dxgi_capture import ClientArea, DesktopImage, DXGIFrames, DXGISettings

    class NativeCamera:
        output_rect = (-1920, 0, 0, 1080)
        monitor = 99
        mapping = QpcMapping(0, 0, 1_000_000_000, 0)
        identity = {"adapter": "synthetic SDK boundary", "output": 1}
        count = 0
        closed = False

        def grab(self, region):
            assert region == (100, 50, 102, 51)
            self.count += 1
            return DesktopImage(
                present_ticks=0 if self.count % 2 else time.perf_counter_ns() - 1_000_000,
                accumulated_frames=0 if self.count % 2 else 1,
                protected=False,
                size=(2, 1),
                bgra=bytes([30, 20, 10, 255] * 2),
                source_texture_size=(1920, 1080),
            )

        def release(self):
            self.closed = True

    camera = NativeCamera()
    source = DXGIFrames(
        DXGISettings(expected_client_size=(2, 1)),
        target=lambda: ClientArea(10, 99, (-1820, 50, -1818, 51), 144),
        camera_factory=lambda _: camera,
    )
    result = run_experiment(
        CaptureRun(tmp_path / "dxgi", config(), seconds=0.5),
        capture_source_factory=lambda: source,
    )
    capture = result.summary["capture"]
    assert camera.closed
    assert capture["pipeline"]["no_present"] > 0
    assert capture["pipeline"]["new_frames"] < camera.count
    row = next(r for r in capture["decisions"] if r["status"] == "ready")
    assert row["frames"][-1]["source_layout"]["dpi"] == 144
    assert row["frames"][-1]["source_layout"]["source_texture_size"] == [1920, 1080]
    assert row["frames"][-1]["time_quality"] == "dxgi_qpc"


def test_capture_command_only_validates_until_live_is_explicit(tmp_path, capsys):
    assert (
        main(
            [
                "capture-dxgi",
                "--config",
                "configs/capture-dxgi.example.json",
                "--output",
                str(tmp_path / "never-open"),
            ]
        )
        == 0
    )
    assert not (tmp_path / "never-open").exists()
    assert "3840" in capsys.readouterr().out


def test_capture_failure_retries_are_bounded_and_never_report_dynamic_evidence(tmp_path):
    attempts = []

    def unavailable():
        attempts.append(True)
        raise OSError("synthetic access loss")

    result = run_experiment(
        CaptureRun(tmp_path / "failure", config(), seconds=1), capture_source_factory=unavailable
    )
    capture = result.summary["capture"]
    assert len(attempts) == 3
    assert capture["resources_released"] is True
    assert capture["pipeline"]["fault"].startswith("capture_retry_limit")
    assert capture["dynamic_game_validation"] is False
    assert capture["moving_observations"] == 0


def test_capture_rejects_unbounded_numerical_history_before_output(tmp_path):
    with pytest.raises(ValueError, match="pixel budget"):
        request = CaptureReplay(
            tmp_path / "oversized", CaptureConfig(pixels=PixelContract(size=(4096, 4096))), (), ()
        )
        run_experiment(request)
    assert not (tmp_path / "oversized").exists()


def test_interrupted_capture_preserves_result_and_releases_workers(tmp_path):
    class Idle:
        source_kind = "synthetic"

        def capture(self):
            return CaptureEvent(time.perf_counter_ns(), reason="idle")

        def close(self):
            pass

    def interrupt():
        raise KeyboardInterrupt

    result = run_experiment(
        CaptureRun(tmp_path / "interrupted", config(), seconds=0.5),
        capture_source_factory=Idle,
        capture_activity=interrupt,
    )
    capture = result.summary["capture"]
    assert capture["stop_reason"] == "interrupted"
    assert capture["resources_released"] is True
    assert result.report_path.exists()


def test_replay_honors_preprocess_rate_and_preserves_latest_pending(tmp_path):
    events = []
    for i in range(7):
        event = frame(i, duration_ms=1)
        assert event.frame is not None
        events.append(
            replace(
                event,
                received_ns=1_002_000_000 + i * 16_000_000,
                frame=replace(event.frame, present_ticks=ANCHOR + i * 160_000),
            )
        )
    result = run_experiment(
        CaptureReplay(
            tmp_path / "rate",
            CaptureConfig(
                pixels=PixelContract(size=(2, 1), history_offsets_ms=(64, 32, 0)), preprocess_hz=30
            ),
            tuple(events),
            (1_110_000_000,),
        )
    )
    capture = result.summary["capture"]
    assert capture["preprocessed"] == 4
    assert capture["pending_overwritten"] == 3
    assert [f["frame_id"] for f in capture["decisions"][0]["frames"]] == ["f3", "f5", "f7"]


def test_frame_rate_excludes_duplicates_and_epoch_gaps_and_reports_stale_age(tmp_path):
    events = [frame(0), frame(1)]
    events.append(replace(frame(1), received_ns=1_120_000_000))
    events.append(CaptureEvent(1_400_000_000, boundary="focus_lost"))
    events.extend([frame(5), frame(6), frame(9)])
    events[-1] = replace(events[-1], frame=replace(events[-1].frame, accumulated_frames=3))
    result = run_experiment(
        CaptureReplay(tmp_path / "cadence", config(capture_hz=10), tuple(events), (2_200_000_000,))
    )
    capture = result.summary["capture"]
    cadence = capture["cadence"]["synthetic_qpc"]
    assert cadence["accepted_frames"] == 5
    assert cadence["interval_ms"]["count"] == 3
    assert cadence["interval_ms"]["p50"] == 100
    assert cadence["interval_ms"]["max"] == 300
    assert cadence["within_epoch_rate_hz"] == 6
    assert cadence["long_gap_count"] == 1
    assert capture["decisions"][0]["latest_age_ms"] == 300
    assert capture["decisions"][0]["reason"] == "stale_latest_image"
    assert capture["game_frame_time"]["status"] == "unavailable"
    assert capture["desktop_presentations_coalesced"] == 2


def test_slow_resource_probe_cannot_stall_capture_and_is_reported_unreleased(tmp_path):
    release = threading.Event()

    class Source:
        source_kind = "synthetic"

        def capture(self):
            now = time.perf_counter_ns()
            return replace(
                frame(0),
                received_ns=now,
                frame=RawCapture(
                    now - 1_000_000,
                    QpcMapping(0, 0, 1_000_000_000, 0),
                    (2, 1),
                    bytes([0, 0, 0, 255] * 2),
                    {},
                ),
            )

        def close(self):
            pass

    def blocked_os_call():
        release.wait(10)
        return {"process_working_set_bytes": 1234}

    try:
        result = run_experiment(
            CaptureRun(tmp_path / "resources", config(), seconds=0.35),
            capture_source_factory=Source,
            capture_resources=blocked_os_call,
        )
        capture = result.summary["capture"]
        assert capture["pipeline"]["new_frames"] > 3
        assert capture["pipeline"]["resources_released"] is True
        assert capture["resource_monitor"]["resources_released"] is False
        assert capture["resources_released"] is False
    finally:
        release.set()


def test_high_resolution_samples_keep_exact_source_and_stop_at_byte_budget(tmp_path):
    class Source:
        source_kind = "synthetic"

        def capture(self):
            now = time.perf_counter_ns()
            return CaptureEvent(
                now,
                frame=RawCapture(
                    now - 1_000_000,
                    QpcMapping(0, 0, 1_000_000_000, 0),
                    (2, 1),
                    bytes([30, 20, 10, 255] * 2),
                    {},
                ),
            )

        def close(self):
            pass

    result = run_experiment(
        CaptureRun(
            tmp_path / "source-samples",
            config(),
            seconds=0.35,
            raw_sample_limit=3,
            raw_sample_interval_s=0,
            raw_sample_bytes=16,
        ),
        capture_source_factory=Source,
    )
    capture = result.summary["capture"]
    samples = capture["raw_samples"]
    assert samples["accepted"] == 2
    assert len(samples["records"]) == 2
    assert samples["accepted_bytes"] == 16
    assert samples["resources_released"] is True
    for sample in samples["records"]:
        raw = (result.report_path.parent / sample["path"]).read_bytes()
        assert raw == bytes([30, 20, 10, 255] * 2)
        assert hashlib.sha256(raw).hexdigest() == sample["sha256"]
        assert sample["size"] == [2, 1]
    assert capture["pipeline"]["new_frames"] > 3


def test_mss_numerical_comparison_uses_same_crop_but_marks_proxy_clock(tmp_path):
    from types import SimpleNamespace

    from fh5.dxgi_capture import ClientArea, DXGISettings
    from fh5.mss_probe import MSSFrames

    class MSS:
        closed = False

        def grab(self, region):
            assert region == {"left": 100, "top": 50, "width": 2, "height": 1}
            return SimpleNamespace(size=(2, 1), bgra=bytes([30, 20, 10, 255] * 2))

        def close(self):
            self.closed = True

    backend = MSS()
    result = run_experiment(
        CaptureRun(tmp_path / "mss", config(), seconds=0.35),
        capture_source_factory=lambda: MSSFrames(
            DXGISettings(expected_client_size=(2, 1)),
            target=lambda: ClientArea(1, 2, (100, 50, 102, 51), 144),
            grabber_factory=lambda: backend,
        ),
    )
    capture = result.summary["capture"]
    row = next(row for row in capture["decisions"] if row["status"] == "ready")
    assert row["frames"][-1]["time_quality"] == "capture_start_proxy"
    assert row["frames"][-1]["uncertainty_ns"] is None
    assert (
        "not native new-frame FPS"
        in capture["pipeline"]["cadence"]["capture_start_proxy"]["meaning"]
    )
    assert backend.closed
    assert capture["pipeline"]["source_kind"] == "mss_numeric_diagnostic"


def test_legacy_comparison_roundtrips_jpeg_with_explicit_lossy_provenance(tmp_path):
    from io import BytesIO

    from PIL import Image

    from fh5.dxgi_capture import ClientArea, DXGISettings
    from fh5.mss_probe import LegacyJPEGFrames
    from fh5.vision import ColorFrame

    buffer = BytesIO()
    Image.new("RGB", (2, 1), (10, 20, 30)).save(buffer, format="JPEG", quality=90)
    encoded = buffer.getvalue()

    class Source:
        def capture(self):
            stamp = time.perf_counter_ns()
            return ColorFrame(stamp, stamp, stamp, encoded, "jpeg", (2, 1), (4, 2))

        def close(self):
            pass

    output = tmp_path / "legacy"
    result = run_experiment(
        CaptureRun(
            output,
            CaptureConfig(
                pixels=PixelContract(
                    size=(2, 1),
                    origin="legacy_offline",
                    resize="legacy-jpeg-roundtrip-then-bilinear-diagnostic-v1",
                )
            ),
            seconds=0.35,
        ),
        capture_source_factory=lambda: LegacyJPEGFrames(
            DXGISettings(expected_client_size=(4, 2)),
            output,
            target=lambda: ClientArea(1, 2, (0, 0, 4, 2), 96),
            source_factory=Source,
        ),
    )
    capture = result.summary["capture"]
    assert (output / "diagnostic-latest.jpg").read_bytes() == encoded
    row = next(row for row in capture["decisions"] if row["status"] == "ready")
    assert row["frames"][-1]["source_layout"]["physical_client_size"] == [4, 2]
    assert row["frames"][-1]["source_layout"]["lossy_jpeg"] is True
    assert capture["pipeline"]["metrics"]["legacy_disk_write_read_ms"]["count"] > 0
    assert capture["pipeline"]["metrics"]["legacy_decode_ms"]["count"] > 0


def test_game_frame_trace_is_filtered_by_process_chain_and_capture_time(tmp_path):
    from fh5.capture_trace import CaptureTraceReview

    events = tuple(
        replace(frame(i), frame=replace(frame(i).frame, time_quality="dxgi_qpc")) for i in range(3)
    )
    run = run_experiment(
        CaptureReplay(tmp_path / "trace-run", config(), events, (1_210_000_000, 1_400_000_000))
    )
    trace = tmp_path / "presentmon.csv"
    trace.write_text(
        "Application,ProcessID,SwapChainAddress,CPUStartQPC,MsBetweenPresents,DisplayedTime\n"
        f"ForzaHorizon5.exe,42,0xA,{ANCHOR + 500_000},10,10\n"
        f"ForzaHorizon5.exe,42,0xA,{ANCHOR + 1_500_000},20,NA\n"
        f"ForzaHorizon5.exe,42,0xA,{ANCHOR + 2_500_000},30,30\n"
        f"ForzaHorizon5.exe,42,0xB,{ANCHOR + 1_500_000},100,100\n"
        f"ForzaHorizon5.exe,99,0xA,{ANCHOR + 1_500_000},100,100\n"
        f"other.exe,42,0xA,{ANCHOR + 1_500_000},100,100\n"
        f"ForzaHorizon5.exe,42,0xA,{ANCHOR + 8_000_000},100,100\n",
        encoding="utf-8",
    )
    result = run_experiment(
        CaptureTraceReview(run.report_path.parent, trace, tmp_path / "trace-report.html", 42, "0xA")
    )
    timing = result.summary["capture"]["game_frame_time"]
    assert timing["frame_count"] == 3
    assert timing["ms_between_presents"]["p50"] == 20
    assert timing["displayed_time_ms"]["count"] == 2
    assert timing["trace_sha256"] == hashlib.sha256(trace.read_bytes()).hexdigest()
    assert result.summary["capture"]["dynamic_game_validation"] is False
