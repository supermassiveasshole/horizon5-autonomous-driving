"""Shared result of an experiment operation, with no workflow dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunResult:
    metadata: dict[str, Any]
    samples: list[dict[str, Any]]
    events: list[dict[str, Any]]
    summary: dict[str, Any]
    report_path: Path
