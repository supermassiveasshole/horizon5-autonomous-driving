"""Replay request for archived v1 BC models; new training uses temporal BC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BCReplay:
    model_dir: Path
    dataset_file: Path
    report_path: Path
    device: str = "cpu"
