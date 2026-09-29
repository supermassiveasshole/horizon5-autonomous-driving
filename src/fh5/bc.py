"""Offline behavior cloning at the experiment seam; optional tensor runtime is lazy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class BCTrain:
    config_file: Path
    output_dir: Path


@dataclass(frozen=True)
class BCReplay:
    model_dir: Path
    dataset_file: Path
    report_path: Path
    device: str = "cpu"


def run_bc(request: BCTrain | BCReplay) -> RunResult:
    from fh5.bc_learning import run_offline

    return run_offline(request)
