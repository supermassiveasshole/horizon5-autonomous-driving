"""Optional sampled resource diagnostics, isolated from capture and observation."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any


class ResourceMonitor:
    def __init__(self, sample: Callable[[], dict[str, Any]] | None) -> None:
        self.sample = sample
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.rows: deque[dict[str, Any]] = deque(maxlen=601)
        self.worker: threading.Thread | None = None
        if sample is not None:
            self.worker = threading.Thread(target=self._work, name="fh5-resources", daemon=True)
            self.worker.start()

    def _work(self) -> None:
        assert self.sample is not None
        while not self.done.is_set():
            start = time.perf_counter_ns()
            try:
                payload = json.dumps(self.sample(), allow_nan=False)
                if len(payload) > 16_384:
                    raise ValueError("Resource diagnostic sample exceeds metadata budget")
                value = json.loads(payload)
                if not isinstance(value, dict):
                    raise ValueError("Resource diagnostic must be an object")
            except Exception as error:
                value = {"error": f"{type(error).__name__}: {error}"}
            end = time.perf_counter_ns()
            with self.lock:
                if not self.done.is_set():
                    self.rows.append(
                        dict(value, observed_ns=end, sample_duration_ms=(end - start) / 1e6)
                    )
            self.done.wait(max(0, 1 - (time.perf_counter_ns() - start) / 1e9))

    def close(self) -> dict[str, Any]:
        self.done.set()
        if self.worker is not None:
            self.worker.join(timeout=0.25)
        with self.lock:
            rows = list(self.rows)
        peaks = {}
        for key in ("process_working_set_bytes", "process_private_bytes", "process_cpu_cores"):
            values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
            peaks[key] = max(values) if values else None
        gpu_peaks: dict[str, float] = {}
        for row in rows:
            for gpu in row.get("gpus", []):
                if gpu.get("memory_used_mib") is not None:
                    identity = gpu["uuid"]
                    gpu_peaks[identity] = max(gpu_peaks.get(identity, 0), gpu["memory_used_mib"])
        return {
            "status": "not_requested" if self.sample is None else "sampled",
            "resources_released": self.worker is None or not self.worker.is_alive(),
            "period_s": 1,
            "samples": rows,
            "sampled_peaks": peaks,
            "device_total_memory_sampled_peak_mib": gpu_peaks,
            "scope": "this process RAM/CPU; GPU memory is whole-device, not attributable to capture",
        }


class _Memory(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("faults", ctypes.c_uint32)] + [
        (name, ctypes.c_size_t)
        for name in (
            "peak_working",
            "working",
            "peak_paged",
            "paged",
            "peak_nonpaged",
            "nonpaged",
            "pagefile",
            "peak_pagefile",
            "private",
        )
    ]


class WindowsResources:
    """Read current-process counters and optional NVIDIA whole-device samples."""

    def __init__(self, *, include_gpu: bool = True) -> None:
        self.previous: tuple[int, int] | None = None
        self.include_gpu = include_gpu
        self.nvidia = shutil.which("nvidia-smi")

    def __call__(self) -> dict[str, Any]:
        if os.name != "nt":
            return {"error": "Windows resource counters unavailable on this platform"}
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.K32GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Memory),
            ctypes.c_uint32,
        ]
        memory = _Memory()
        memory.cb = ctypes.sizeof(memory)
        row: dict[str, Any] = {}
        if kernel.K32GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb
        ):
            row.update(
                process_working_set_bytes=memory.working,
                process_private_bytes=memory.private,
                process_lifetime_peak_working_set_bytes=memory.peak_working,
            )
        else:
            row["memory_error"] = ctypes.get_last_error()
        now, cpu = time.perf_counter_ns(), time.process_time_ns()
        row["process_cpu_cores"] = (
            (cpu - self.previous[1]) / (now - self.previous[0]) if self.previous else None
        )
        self.previous = (now, cpu)
        if not self.include_gpu:
            row["gpu_status"] = "not_requested"
            return row
        if self.nvidia is None:
            row["gpu_status"] = "nvidia-smi unavailable"
            return row
        try:
            output = subprocess.run(
                [
                    self.nvidia,
                    "--query-gpu=uuid,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=0.8,
                check=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            ).stdout
            gpus = []
            for line in output.splitlines():
                uuid, memory_mib, utilization = [value.strip() for value in line.split(",")]
                gpus.append(
                    {
                        "uuid": uuid,
                        "memory_used_mib": float(memory_mib)
                        if memory_mib.replace(".", "", 1).isdigit()
                        else None,
                        "utilization_percent": float(utilization)
                        if utilization.replace(".", "", 1).isdigit()
                        else None,
                    }
                )
            row["gpus"] = gpus
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            row["gpu_error"] = f"{type(error).__name__}: {error}"
        return row
