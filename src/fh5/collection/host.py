"""Read process identity from the OS; never infer liveness from a PID file."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any


def process_identity(pid: int) -> dict[str, Any]:
    if type(pid) is not int or pid <= 0:
        return {"state": "unknown", "birth": None}
    if os.name != "nt":
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return {"state": "exited" if fields[0] == "Z" else "running", "birth": fields[19]}
        except FileNotFoundError:
            return {"state": "exited", "birth": None}
        except (OSError, IndexError):
            return {"state": "unknown", "birth": None}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        return {"state": "exited" if error in (87, 1168) else "unknown", "birth": None}
    try:
        times = [ctypes.c_uint64() for _ in range(4)]
        code = ctypes.c_uint32()
        if not kernel.GetProcessTimes(
            handle, *(ctypes.byref(t) for t in times)
        ) or not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return {"state": "unknown", "birth": None}
        return {"state": "running" if code.value == 259 else "exited", "birth": str(times[0].value)}
    finally:
        kernel.CloseHandle(handle)
