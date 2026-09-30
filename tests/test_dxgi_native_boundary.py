"""Exercise pinned native bindings at the experiment seam with a fake external SDK."""

import ctypes
import importlib.metadata
import struct
import sys
import time
from types import SimpleNamespace

from fh5.capture import CaptureConfig, CaptureRun
from fh5.dxgi_capture import ClientArea, DXGIFrames, DXGISettings
from fh5.dxgi_windows import DXcamSession
from fh5.experiment import run_experiment
from fh5.numeric_images import PixelContract


class NativeFunction:
    def __init__(self, value):
        self.value = value

    def __call__(self, pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int64)).contents.value = self.value()
        return 1


class TextureDescription(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ("Width", "Height", "Format")]


class NativeTexture:
    def GetDesc(self, pointer):
        desc = ctypes.cast(pointer, ctypes.POINTER(TextureDescription)).contents
        desc.Width, desc.Height, desc.Format = 1920, 1080, 87


class NativeArray:
    shape, ndim, dtype = (1, 2, 4), 3, "uint8"

    def tobytes(self, order):
        assert order == "C"
        return bytes([30, 20, 10, 255] * 2)


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


def test_native_frame_info_matches_pixels_and_rejects_mouse_timestamp_fallback(
    tmp_path, monkeypatch
):
    native = NativeDuplication()
    desc = SimpleNamespace(
        Description="Test adapter", AdapterLuid=SimpleNamespace(HighPart=0, LowPart=42)
    )

    class Camera:
        is_capturing, backend, rotation_angle = False, "dxgi", 0
        _device = SimpleNamespace(desc=desc)
        _output = SimpleNamespace(
            hmonitor=99,
            devicename="Test output",
            desc=SimpleNamespace(
                DesktopCoordinates=SimpleNamespace(left=0, top=0, right=1920, bottom=1080)
            ),
        )
        _duplicator = SimpleNamespace(
            performance_frequency=1_000_000_000, duplicator=native, texture=NativeTexture()
        )
        closed = False

        def grab(self, region, copy, new_frame_only):
            assert region == (100, 50, 102, 51) and copy is True and new_frame_only is True
            info = ctypes.create_string_buffer(48)
            self._duplicator.duplicator.AcquireNextFrame(0, ctypes.byref(info), None)
            return NativeArray()

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
        DXGISettings(expected_client_size=(2, 1)),
        target=lambda: ClientArea(10, 99, (100, 50, 102, 51), 96),
        camera_factory=DXcamSession,
    )
    result = run_experiment(
        CaptureRun(
            tmp_path / "native", CaptureConfig(pixels=PixelContract(size=(2, 1))), seconds=0.5
        ),
        capture_source_factory=lambda: source,
    )
    capture = result.summary["capture"]
    assert capture["pipeline"]["fault"] is None
    assert capture["pipeline"]["no_present"] > 0
    ready = [row for row in capture["decisions"] if row["status"] == "ready"]
    assert ready
    assert ready[-1]["frames"][-1]["source_layout"]["adapter_luid"] == [0, 42]
    assert capture["resources_released"] is True and camera.closed
