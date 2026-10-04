"""OS-backed cooperative acquisition ownership; a stale file is not a lock."""

from __future__ import annotations

import importlib
import os
from pathlib import Path


class CollectionLease:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def acquire(self) -> None:
        if self.fd is not None:
            raise RuntimeError("Collection lease is already held")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
            os.lseek(fd, 0, os.SEEK_SET)
            if os.name == "nt":
                api = importlib.import_module("msvcrt")
                api.locking(fd, api.LK_NBLCK, 1)
            else:
                api = importlib.import_module("fcntl")
                api.flock(fd, api.LOCK_EX | api.LOCK_NB)
        except OSError as error:
            os.close(fd)
            raise OSError("Passive collection resource is already owned or unavailable") from error
        self.fd = fd

    def close(self) -> None:
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)
