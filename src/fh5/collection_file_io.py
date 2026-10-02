"""Read replaceable control files without blocking their atomic publication."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO


def _windows_path(path: Path) -> str:
    value = os.path.abspath(path)
    if value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def _windows_kernel() -> Any:
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.ReplaceFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel.ReplaceFileW.restype = wintypes.BOOL
    return kernel


@contextmanager
def control_file_reader(path: Path) -> Iterator[BinaryIO]:
    """One file identity, with Windows sharing compatible with ReplaceFileW."""
    if os.name != "nt":
        with path.open("rb") as stream:
            yield stream
        return
    import msvcrt

    kernel = _windows_kernel()
    # GENERIC_READ; FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    # OPEN_EXISTING; FILE_ATTRIBUTE_NORMAL. No delete-on-close or write access.
    handle = kernel.CreateFileW(_windows_path(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    # The CRT owns the handle after open_osfhandle; never CloseHandle it again.
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        yield stream


def replace_control_file(source: Path, target: Path) -> None:
    """Replace existing Windows files while shared readers retain the old bytes."""
    if os.name != "nt" or not target.exists():
        source.replace(target)
        return
    kernel = _windows_kernel()
    # Flags=0 preserves ACL/attribute errors. In particular, 1175/1176/1177
    # can describe a partial replacement and must propagate without blind retry
    # or deleting the surviving source. atomic_json retries only 5/32/33.
    if not kernel.ReplaceFileW(_windows_path(target), _windows_path(source), None, 0, None, None):
        raise ctypes.WinError(ctypes.get_last_error())
