"""External fault fixtures shared by public experiment tests."""

from contextlib import contextmanager
from pathlib import Path

import pytest


@pytest.fixture
def corrupt_private_reads(monkeypatch):
    """Change copy reads while leaving the preceding readinto-based hash intact."""

    @contextmanager
    def corrupt(target, original, replacement):
        opening = Path.open
        changed = []

        class ChangingStream:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.wrapped.close()

            def read(self, size=-1):
                raw = self.wrapped.read(size)
                if original in raw:
                    changed.append(True)
                    return raw.replace(original, replacement)
                return raw

        def changing(path, mode="r", *args, **kwargs):
            stream = opening(path, mode, *args, **kwargs)
            return ChangingStream(stream) if path == target and mode == "rb" else stream

        with monkeypatch.context() as fault:
            fault.setattr(Path, "open", changing)
            yield changed

    return corrupt
