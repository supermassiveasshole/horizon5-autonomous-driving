"""DXGI menu conversion through EventRun, using only external raw pixel fixtures."""

from dataclasses import replace

import pytest
from test_event_run import PATTERNS, RestartGame, verified_config

from fh5.capture import CaptureEvent, QpcMapping, RawCapture
from fh5.evaluation_native_io import DXGIEventFrames
from fh5.events import EventRun
from fh5.experiment import run_experiment


@pytest.mark.parametrize("fault", [None, "window_changed", "duplicate"])
def test_event_recipe_uses_fresh_raw_pixels_and_stops_on_unusable_capture(tmp_path, fault):
    class RawDesktop:
        source_kind = "simulated_dxgi"

        def __init__(self):
            self.closed = False
            self.calls = 0
            self.stamp = 0

        def capture(self):
            self.calls += 1
            now = game.time
            if fault == "window_changed":
                return CaptureEvent(now, boundary="window_changed")
            if self.calls == 1:
                return CaptureEvent(now, reason="no_new_frame")
            if not self.stamp or fault != "duplicate":
                self.stamp = now
            raw = bytes(
                channel for value in PATTERNS[game.screen] for channel in (value, value, value, 255)
            )
            return CaptureEvent(
                now, RawCapture(self.stamp, QpcMapping(now, now, 1_000_000_000, 0), (4, 2), raw, {})
            )

        def close(self):
            self.closed = True

    class MenuWithRawFrames(RestartGame):
        def read(self, period_s):
            observed = super().read(period_s)
            return replace(observed, frame=frames.capture())

        def close(self):
            frames.close()
            super().close()

    game = MenuWithRawFrames()
    source = RawDesktop()
    frames = DXGIEventFrames(source, (4, 2))
    result = run_experiment(
        EventRun(verified_config(tmp_path), tmp_path / "ready", operation="start_ready"),
        event_environment=game,
    )
    summary = result.summary["event_run"]
    assert source.closed and game.closed and game.released
    if fault is None:
        assert summary["ready_verified"] and game.pulses == ["A"]
        assert source.calls > 2
    else:
        assert not summary["ready_verified"] and not game.pulses
