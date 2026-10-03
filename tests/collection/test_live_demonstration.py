"""Competing physical inputs are quarantined through the experiment seam."""

import json
import time
from pathlib import Path

import pytest

from fh5.capture.legacy import VisionInput, VisionRecord
from fh5.collection.demonstration_windows import HumanInputEnvironment, XInputReader
from fh5.collection.demonstrations import DemonstrationRecord
from fh5.experiment import run_experiment
from tests.collection.test_demonstrations import profile_file
from tests.observation.test_observations import motion_packet
from tests.telemetry.test_experiment import config_file


@pytest.mark.parametrize("peer_axis", ["thumb_ly", "thumb_rx", "thumb_ry"])
def test_other_controller_motion_cannot_be_a_trusted_input(tmp_path: Path, monkeypatch, peer_axis):
    # Fake only the external XInput ABI, desktop and passive telemetry clock.
    class Function:
        def __init__(self, read):
            self.read = read

        def __call__(self, *args):
            return self.read(*args)

    def state(index, pointer):
        if index > 1:
            return 1167
        if index == 0:
            pointer._obj.gamepad.right_trigger = 128
        else:
            setattr(pointer._obj.gamepad, peer_axis, 32767)
        return 0

    class API:
        XInputGetState = Function(state)
        XInputGetCapabilities = Function(lambda index, flags, pointer: 0)

    class Keys:
        def GetAsyncKeyState(self, key):
            return 0

    class Desktop:
        user32 = Keys()

        def focused(self):
            return True

    class Vision:
        source_kind = "synthetic"
        read_once = False

        def now_ns(self):
            return time.perf_counter_ns()

        def read(self, period):
            if self.read_once:
                return VisionInput(stop_requested=True)
            self.read_once = True
            return VisionInput(packets=(motion_packet(self.now_ns() // 1_000_000),))

        def close(self):
            return True

    monkeypatch.setattr(
        "fh5.collection.demonstration_windows.ctypes.WinDLL", lambda name: API(), raising=False
    )
    reader = XInputReader(0, Desktop())
    profile = profile_file(tmp_path)
    contents = json.loads(profile.read_text())
    contents["device"] = reader.identity()
    profile.write_text(json.dumps(contents))
    result = run_experiment(
        DemonstrationRecord(VisionRecord(config_file(tmp_path), tmp_path / "demo"), profile),
        vision_environment=HumanInputEnvironment(Vision(), reader, profile),
    )
    row = result.summary["demonstration"]["inputs"][0]
    assert row["raw"]["right_trigger"] == 128
    assert row["mapped"] is None
    assert "other_input" in row["reasons"]
