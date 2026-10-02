"""External synthetic sampling and file inspection for public recovery experiments."""

import hashlib
from pathlib import Path

from test_learning_loop import SharedBackend


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def originals(root):
    return {str(path.resolve()): digest(path) for path in root.rglob("*") if path.is_file()}


class FailedOriginals(SharedBackend):
    """Fail external sampling; never replace a learner or fabricate its result."""

    def __init__(self, source, extra_files=256):
        super().__init__(source)
        self.extra_files = extra_files

    def sampling(self, identity):
        lease = super().sampling(identity)
        lease.fail_at = 0
        finish = lease.finish

        def review(recording):
            if len(self.leases) == 1:
                folder = recording.parent / "external-diagnostics"
                folder.mkdir()
                for number in range(self.extra_files):
                    (folder / f"capture-{number:05d}-原件.bin").write_bytes(
                        number.to_bytes(4, "little")
                    )
                # Outside the per-attempt declaration, but inside the failed
                # sampling directory: only the whole-run archive binds this.
                (recording.parent.parent / "failure-note.bin").write_bytes(b"closed receiver")
            return finish(recording)

        lease.finish = review
        return lease
