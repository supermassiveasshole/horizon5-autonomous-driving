"""Capture and history behavior through the agreed experiment-run interface."""

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
