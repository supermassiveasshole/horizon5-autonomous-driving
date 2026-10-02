"""Optional BC loss journals never own optimizer progress or candidate publication."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fh5.artifact_io import VerifiedFile, sha256_file
from fh5.learning_diagnostics import RecordJournal
from fh5.replay_document import read_document_projection


@dataclass(frozen=True)
class _LegacyLosses:
    records: int
    valid: bool


def _legacy_losses(values: Iterator[Any]) -> _LegacyLosses:
    count, valid = 0, True
    for value in values:
        count += 1
        try:
            valid = valid and type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
    return _LegacyLosses(count, valid)


def read_bc_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Project old loss arrays without rewriting or rebinding frozen model bytes."""
    digest = sha256_file(path)
    value = read_document_projection(
        VerifiedFile(path, digest), {("training", "losses"): _legacy_losses}
    )
    training = value.get("training")
    if isinstance(training, dict) and "losses" in training:
        losses = training.pop("losses")
        if not isinstance(losses, _LegacyLosses) or not losses.valid:
            raise ValueError("Invalid legacy BC loss history")
        training["loss_history"] = {
            "format": "bc-loss-history-embedded-v1",
            "status": "legacy_embedded",
            "path": str(path.resolve()),
            "manifest_sha256": digest,
            "field": ["training", "losses"],
            "records": losses.records,
            "role": "optional_local_diagnostic; not required for actor replay",
        }
    return value, digest


class BCLossHistory:
    """Stage diagnostics independently until the trained actor has been saved."""

    def __init__(self) -> None:
        self.temporary: tempfile.TemporaryDirectory[str] | None = None
        self.journal: RecordJournal | None = None
        self.descriptor: dict[str, Any] = {
            "format": "bc-loss-history-jsonl-v1",
            "path": "diagnostics/losses.jsonl",
            "status": "unavailable",
            "records": 0,
            "sha256": None,
            "error": None,
            "role": "optional_local_diagnostic; not required for actor replay",
        }
        try:
            self.temporary = tempfile.TemporaryDirectory(prefix="fh5-bc-losses-")
            self.journal = RecordJournal(
                Path(self.temporary.name), self.descriptor["path"], self.descriptor["format"]
            )
        except (OSError, MemoryError) as error:
            self.descriptor["error"] = f"{type(error).__name__}: {error}"
            self.close()

    def record(self, completed: int, loss: Any) -> None:
        if self.journal is None or self.journal.error is not None:
            return
        try:
            self.journal.append({"step": completed, "loss": float(loss.item())})
        except (OSError, MemoryError) as error:
            self.journal.unavailable(error)
        if self.journal.error is not None:
            self.descriptor.update(records=self.journal.records, error=self.journal.error)
            # A failed optional prefix must not occupy disk needed for actor sealing.
            self.close()

    def publish(self, output: Path) -> dict[str, Any]:
        if self.journal is None or self.temporary is None:
            return self.descriptor
        pending: Path | None = None
        try:
            self.descriptor = self.journal.finish()
            self.descriptor["role"] = "optional_local_diagnostic; not required for actor replay"
            if self.descriptor["status"] != "complete":
                return self.descriptor
            destination = output / self.descriptor["path"]
            destination.parent.mkdir(exist_ok=True)
            pending = destination.with_name(".pending-" + destination.name)
            VerifiedFile(
                Path(self.temporary.name) / self.descriptor["path"], self.descriptor["sha256"]
            ).copy_to(pending)
            pending.replace(destination)
        except (OSError, MemoryError, ValueError) as error:
            # A failed integrity check on this optional copied journal also
            # cannot discard the separately saved actor or its required evidence.
            self.descriptor.update(
                status="unavailable",
                records=self.journal.records,
                sha256=None,
                error=f"{type(error).__name__}: {error}",
            )
            if pending is not None:
                try:
                    pending.unlink(missing_ok=True)
                except (OSError, MemoryError) as cleanup_error:
                    self.descriptor["cleanup_error"] = (
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
        finally:
            # Persist cleanup failures in the manifest, before serialization.
            self.close()
        return self.descriptor

    def close(self) -> None:
        if self.journal is not None:
            self.journal.close()
        temporary, self.temporary = self.temporary, None
        if temporary is not None:
            try:
                temporary.cleanup()
            except (OSError, MemoryError) as error:
                self.descriptor["cleanup_error"] = f"{type(error).__name__}: {error}"
