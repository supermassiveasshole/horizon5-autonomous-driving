"""Timer precondition for real-time tests with simulated Windows devices."""

import ctypes
import sys

import pytest


@pytest.fixture(scope="module", autouse=True)
def native_device_timing():
    if sys.platform != "win32":
        yield
        return

    class TimerCaps(ctypes.Structure):
        _fields_ = [("minimum_ms", ctypes.c_uint), ("maximum_ms", ctypes.c_uint)]

    timer = ctypes.WinDLL("winmm")
    timer.timeGetDevCaps.argtypes = [ctypes.POINTER(TimerCaps), ctypes.c_uint]
    timer.timeGetDevCaps.restype = ctypes.c_uint
    for method in (timer.timeBeginPeriod, timer.timeEndPeriod):
        method.argtypes = [ctypes.c_uint]
        method.restype = ctypes.c_uint
    caps = TimerCaps()
    assert timer.timeGetDevCaps(ctypes.byref(caps), ctypes.sizeof(caps)) == 0
    # Use the interface's supported period, not a relaxed decision deadline.
    # This affects waits throughout this test process, not only fake devices.
    # https://learn.microsoft.com/en-us/windows/win32/api/timeapi/nf-timeapi-timebeginperiod
    assert timer.timeBeginPeriod(caps.minimum_ms) == 0
    try:
        yield
    finally:
        assert timer.timeEndPeriod(caps.minimum_ms) == 0
