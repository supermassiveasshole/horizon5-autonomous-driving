"""Exercise pinned native bindings at the experiment seam with a fake external SDK."""

import ctypes
import hashlib
import importlib.metadata
import json
import struct
import sys
import time
from types import SimpleNamespace

import pytest

from fh5.capture.dxgi import ClientArea, DXGIFrames, DXGISettings
from fh5.capture.pipeline import CaptureConfig, CaptureRun
from fh5.capture.windows import DXcamSession
from fh5.experiment import run_experiment
from fh5.observation.numeric import PixelContract


class NativeFunction:
    def __init__(self, value):
        self.value = value

    def __call__(self, pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int64)).contents.value = self.value()
        return 1


class TextureDescription(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ("Width", "Height", "Format")]


class NativeTexture:
    def __init__(self, size=(1920, 1080)):
        self.size = size

    def GetDesc(self, pointer):
        desc = ctypes.cast(pointer, ctypes.POINTER(TextureDescription)).contents
        desc.Width, desc.Height = self.size
        desc.Format = 87


class NativeArray:
    ndim, dtype = 3, "uint8"

    def __init__(self, size=(2, 1)):
        self.size = size
        self.shape = (size[1], size[0], 4)

    def tobytes(self, order):
        assert order == "C"
        return bytes((30, 20, 10, 255)) * (self.size[0] * self.size[1])


class NativeDuplication:
    def __init__(self):
        self.count = 0

    def AcquireNextFrame(self, wait, info, resource):
        self.count += 1
        ticks = time.perf_counter_ns() - 1_000_000
        # Published Win32 layout: two LARGE_INTEGER values, eight 32-bit fields.
        # A fresh mouse update with zero present must never become a new image.
        raw = struct.pack(
            "<qq8I",
            0 if self.count % 2 else ticks,
            ticks + 100,
            0 if self.count % 2 else 1,
            0,
            0,
            123,
            456,
            1,
            0,
            0,
        )
        ctypes.memmove(info, raw, len(raw))


@pytest.mark.parametrize("source_size", ((2, 1), (7681, 1), (4097, 4096)))
def test_native_frame_info_matches_pixels_and_rejects_mouse_timestamp_fallback(
    tmp_path, monkeypatch, source_size
):
    native = NativeDuplication()
    desc = SimpleNamespace(
        Description="Test adapter", AdapterLuid=SimpleNamespace(HighPart=0, LowPart=42)
    )
    output_size = (max(1920, source_size[0] + 100), max(1080, source_size[1] + 50))
    region = (100, 50, 100 + source_size[0], 50 + source_size[1])

    class Camera:
        is_capturing, backend, rotation_angle = False, "dxgi", 0
        _device = SimpleNamespace(desc=desc)
        _output = SimpleNamespace(
            hmonitor=99,
            devicename="Test output",
            desc=SimpleNamespace(
                DesktopCoordinates=SimpleNamespace(
                    left=0, top=0, right=output_size[0], bottom=output_size[1]
                )
            ),
        )
        _duplicator = SimpleNamespace(
            performance_frequency=1_000_000_000,
            duplicator=native,
            texture=NativeTexture(output_size),
        )
        closed = False

        def grab(self, *, region: tuple[int, ...], copy, new_frame_only):
            assert region == (100, 50, 100 + source_size[0], 50 + source_size[1])
            assert copy is True and new_frame_only is True
            # One genuine large presentation suffices; do not allocate it at
            # capture frequency just to exercise a dimension admission gate.
            if source_size != (2, 1) and native.count >= 2:
                return None
            info = ctypes.create_string_buffer(48)
            self._duplicator.duplicator.AcquireNextFrame(0, ctypes.byref(info), None)
            return NativeArray(source_size)

        def release(self):
            self.closed = True

    camera = Camera()

    def create(**kwargs):
        assert kwargs == {
            "device_idx": 0,
            "output_idx": 0,
            "backend": "dxgi",
            "output_color": "BGRA",
            "processor_backend": "numpy",
            "max_buffer_len": 2,
        }
        return camera

    monkeypatch.setitem(sys.modules, "dxcam", SimpleNamespace(create=create))
    monkeypatch.setitem(
        sys.modules, "dxcam._libs.d3d11", SimpleNamespace(D3D11_TEXTURE2D_DESC=TextureDescription)
    )
    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.3.0" if name == "dxcam" else original_version(name),
    )
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda *args, **kwargs: SimpleNamespace(
            QueryPerformanceCounter=NativeFunction(time.perf_counter_ns),
            QueryPerformanceFrequency=NativeFunction(lambda: 1_000_000_000),
        ),
        raising=False,
    )
    source = DXGIFrames(
        DXGISettings(expected_client_size=source_size),
        target=lambda: ClientArea(10, 99, region, 96),
        camera_factory=DXcamSession,
    )
    result = run_experiment(
        CaptureRun(
            tmp_path / "native",
            CaptureConfig(
                pixels=PixelContract(
                    size=(2, 1), history_offsets_ms=(200, 100, 0) if source_size == (2, 1) else (0,)
                ),
                max_age_ms=1000,
            ),
            seconds=0.5 if source_size == (2, 1) else 1.0,
        ),
        capture_source_factory=lambda: source,
    )
    capture = result.summary["capture"]
    assert capture["pipeline"]["fault"] is None
    assert capture["pipeline"]["no_present"] > 0
    ready = [row for row in capture["decisions"] if row["status"] == "ready"]
    assert ready
    assert ready[-1]["frames"][-1]["source_layout"]["adapter_luid"] == [0, 42]
    archived = ready[-1]["archive"]
    assert archived is not None
    recorded = json.loads((tmp_path / "native" / archived["path"]).read_bytes())
    saved = recorded["frames"][-1]
    assert saved["source_layout"]["client_size"] == list(source_size)
    assert saved["source_layout"]["source_texture_size"] == list(output_size)
    expected = bytes((10, 20, 30)) * 2
    assert saved["sha256"] == hashlib.sha256(expected).hexdigest()
    assert (tmp_path / "native" / saved["path"]).read_bytes() == expected
    assert capture["resources_released"] is True and camera.closed
