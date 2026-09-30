"""Windows and pinned DXcam bindings; imported only for explicit passive capture."""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import time
from ctypes import wintypes
from typing import Any

from fh5.capture import CaptureEvent, QpcMapping
from fh5.dxgi_capture import ClientArea, DesktopImage, DXGIFrames, DXGISettings
from fh5.live import WindowsDesktop


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_int32)]


class _Pointer(ctypes.Structure):
    _fields_ = [("position", _Point), ("visible", ctypes.c_int32)]


class _FrameInfo(ctypes.Structure):
    _fields_ = [
        ("present", ctypes.c_int64),
        ("mouse", ctypes.c_int64),
        ("accumulated", ctypes.c_uint32),
        ("coalesced", ctypes.c_int32),
        ("protected", ctypes.c_int32),
        ("pointer", _Pointer),
        ("metadata_bytes", ctypes.c_uint32),
        ("shape_bytes", ctypes.c_uint32),
    ]


class _FrameTap:
    """Read info from the same AcquireNextFrame call whose texture is copied.

    DXcam 0.3.0 falls back to LastMouseUpdateTime and hides this distinction.
    Intercept only its native boundary; leave acquisition/texture ownership intact.
    """

    def __init__(self, native: Any) -> None:
        self.native = native
        self.info: tuple[int, int, bool] | None = None

    def AcquireNextFrame(self, wait: int, info: Any, resource: Any) -> Any:
        result = self.native.AcquireNextFrame(wait, info, resource)
        record = ctypes.cast(info, ctypes.POINTER(_FrameInfo)).contents
        self.info = (int(record.present), int(record.accumulated), bool(record.protected))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)


def _mapping() -> QpcMapping:
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.QueryPerformanceCounter.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    api.QueryPerformanceFrequency.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    frequency = ctypes.c_int64()
    if not api.QueryPerformanceFrequency(ctypes.byref(frequency)) or frequency.value <= 0:
        raise OSError("Cannot read QPC frequency")
    samples = []
    for _ in range(9):
        before = time.perf_counter_ns()
        ticks = ctypes.c_int64()
        if not api.QueryPerformanceCounter(ctypes.byref(ticks)):
            raise OSError("Cannot read QPC")
        after = time.perf_counter_ns()
        samples.append((after - before, ticks.value, (before + after) // 2))
    span, ticks_value, midpoint = min(samples)
    error = (span + 1) // 2 + (1_000_000_000 + frequency.value - 1) // frequency.value
    return QpcMapping(ticks_value, midpoint, frequency.value, error)


def _access_lost() -> None:
    # Default DXcam recovery loops indefinitely. Our caller closes and recreates
    # at most three times, isolating each attempt into a new observation epoch.
    raise OSError("dxgi_access_lost")


class DXcamSession:
    def __init__(self, settings: DXGISettings) -> None:
        if importlib.metadata.version("dxcam") != "0.3.0":
            raise OSError("This adapter requires the audited dxcam==0.3.0")
        dxcam = importlib.import_module("dxcam")
        self.camera = dxcam.create(
            device_idx=settings.device_idx,
            output_idx=settings.output_idx,
            backend="dxgi",
            output_color="BGRA",
            processor_backend="numpy",
            max_buffer_len=2,
        )
        camera = self.camera
        try:
            if camera.is_capturing or camera.backend != "dxgi" or camera.rotation_angle != 0:
                raise OSError("Expected an idle DXGI camera on an unrotated output")
            self.mapping = _mapping()
            duplicator = camera._duplicator
            if duplicator.performance_frequency != self.mapping.frequency:
                raise OSError("DXGI and Windows QPC frequency differ")
            self.tap = _FrameTap(duplicator.duplicator)
            duplicator.duplicator = self.tap
            camera._recover_output = _access_lost
            self.texture_type = importlib.import_module("dxcam._libs.d3d11").D3D11_TEXTURE2D_DESC
            output = camera._output
            rect = output.desc.DesktopCoordinates
            self.output_rect = (rect.left, rect.top, rect.right, rect.bottom)
            self.monitor = int(output.hmonitor)
            description = camera._device.desc
            self.identity = {
                "backend": "dxgi",
                "dxcam_version": "0.3.0",
                "device_idx": settings.device_idx,
                "output_idx": settings.output_idx,
                "adapter": str(description.Description),
                "output": str(output.devicename),
                "adapter_luid": [description.AdapterLuid.HighPart, description.AdapterLuid.LowPart],
                "output_rect": list(self.output_rect),
                "rotation": camera.rotation_angle,
                "capture_mode": "one-shot-owned-bgra-no-video-mode",
            }
        except Exception:
            camera.release()
            raise

    def grab(self, region: tuple[int, int, int, int]) -> DesktopImage | None:
        self.tap.info = None
        # One caller thread, no camera.start(): pixels and raw ticks cannot race
        # a separate DXcam ring-buffer producer. copy=True gives owned storage.
        pixels = self.camera.grab(region=region, copy=True, new_frame_only=True)
        if pixels is None:
            return None
        if self.tap.info is None:
            raise OSError("DXcam frame lacks matching native presentation evidence")
        present, accumulated, protected = self.tap.info
        desc = self.texture_type()
        self.camera._duplicator.texture.GetDesc(ctypes.byref(desc))
        if (
            desc.Format != 87
            or pixels.ndim != 3
            or pixels.shape[2] != 4
            or str(pixels.dtype) != "uint8"
        ):
            raise OSError("Expected DXGI B8G8R8A8_UNORM uint8 capture")
        return DesktopImage(
            present,
            accumulated,
            protected,
            (int(pixels.shape[1]), int(pixels.shape[0])),
            pixels.tobytes(order="C"),
            (int(desc.Width), int(desc.Height)),
        )

    def release(self) -> None:
        self.camera.release()


class WindowsClient:
    """Physical coordinates on the capture thread, with DPI context restored on close."""

    def __init__(self) -> None:
        self.desktop = WindowsDesktop()
        api = self.desktop.user32
        api.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        api.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        self.prior_dpi = api.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        if not self.prior_dpi:
            raise OSError("Cannot select per-monitor physical DPI coordinates")
        api.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        api.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        api.MonitorFromWindow.restype = wintypes.HMONITOR
        api.GetDpiForWindow.argtypes = [wintypes.HWND]
        api.GetDpiForWindow.restype = wintypes.UINT

    def __call__(self) -> ClientArea | None:
        api = self.desktop.user32
        window = api.GetForegroundWindow()
        if not window or not self.desktop.focused():
            return None
        rect, origin = wintypes.RECT(), wintypes.POINT()
        if not api.GetClientRect(window, ctypes.byref(rect)) or not api.ClientToScreen(
            window, ctypes.byref(origin)
        ):
            raise OSError("Cannot read physical FH5 client geometry")
        monitor = api.MonitorFromWindow(window, 0)
        dpi = api.GetDpiForWindow(window)
        if not monitor or not dpi or window != api.GetForegroundWindow():
            return None
        return ClientArea(
            int(window),
            int(monitor),
            (origin.x, origin.y, origin.x + rect.right, origin.y + rect.bottom),
            int(dpi),
        )

    def close(self) -> None:
        self.desktop.user32.SetThreadDpiAwarenessContext(self.prior_dpi)


class WindowsDXGIFrames(DXGIFrames):
    def __init__(self, settings: DXGISettings) -> None:
        self.client = WindowsClient()
        super().__init__(settings, target=self.client, camera_factory=DXcamSession)

    def capture(self) -> CaptureEvent:
        if self.client.desktop.stop_requested():
            return CaptureEvent(time.perf_counter_ns(), reason="user_stop")
        return super().capture()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self.client.close()
