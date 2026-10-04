"""Keep recovery phases inline; stream optional stage history by invocation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from fh5.learning.diagnostics import RecordJournal


def _event(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("phase"), str)
        and type(value.get("at_ns")) is int
        and type(value.get("round")) is int
    )


class StageHistory:
    """Full logs are optional; current and preceding phases are required state.

    Each invocation writes a fresh segment linking its predecessor's committed
    prefix. Recovery never needs to open old logs, and never truncates crash tails.
    """

    def __init__(self, root: Path, saved: Any = None) -> None:
        self.root = root
        self.journal: RecordJournal | None = None
        self.legacy: list[dict[str, Any]] = []
        self.previous: dict[str, Any] | None = None
        if saved is None:
            saved = []
        if isinstance(saved, list):
            if not all(_event(row) for row in saved):
                raise ValueError("Invalid legacy learning stages")
            self.legacy = saved
            self.events = len(saved)
            self.tail = saved[-2:]
        elif (
            isinstance(saved, dict)
            and saved.get("format") == "learning-stages-v1"
            and type(saved.get("events")) is int
            and saved["events"] >= 0
            and isinstance(saved.get("tail"), list)
            and len(saved["tail"]) == min(2, saved["events"])
            and all(_event(row) for row in saved["tail"])
            and isinstance(saved.get("diagnostic"), dict)
        ):
            self.events = saved["events"]
            self.tail = list(saved["tail"])
            self.previous = saved["diagnostic"]
        else:
            raise ValueError("Invalid learning stage summary")

    def before_stop(self, fallback: str) -> str:
        """The preceding phase distinguishes a pending update from other stops."""
        phase: str = self.tail[-2]["phase"] if len(self.tail) == 2 else fallback
        return phase

    def record(self, phase: str, at_ns: int, round_number: int, *, final: bool) -> dict[str, Any]:
        if self.journal is None:
            self.journal = RecordJournal(
                self.root, f"diagnostics/stages-{uuid4().hex}.jsonl", "learning-stage-segment-v1"
            )
            self.journal.append(
                {
                    "kind": "segment",
                    "previous": self.previous,
                    "base_events": self.events - len(self.legacy),
                }
            )
            for row in self.legacy:
                self.journal.append(row)
            self.legacy = []
        event = {"phase": phase, "at_ns": at_ns, "round": round_number}
        self.journal.append(event)
        self.events += 1
        # Two records have recovery semantics: stopped and the phase before it.
        # This does not limit the retained history in the external segments.
        self.tail = (self.tail + [event])[-2:]
        return {
            "format": "learning-stages-v1",
            "events": self.events,
            "tail": self.tail,
            "diagnostic": self.journal.checkpoint(close=final),
        }

    def close(self) -> None:
        if self.journal is not None:
            self.journal.close()
