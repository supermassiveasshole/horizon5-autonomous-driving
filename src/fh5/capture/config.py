"""Shared source/transform declaration for passive probes and read-only inference."""

from __future__ import annotations

from typing import Any

from fh5.capture.dxgi import DXGISettings
from fh5.capture.pipeline import CaptureConfig
from fh5.observation.numeric import PixelContract


def parse_capture_config(document: dict[str, Any]) -> tuple[CaptureConfig, DXGISettings]:
    if (
        set(document) != {"version", "pixels", "pipeline", "target", "input_conditions"}
        or document["version"] != 1
    ):
        raise ValueError("Unsupported capture configuration")
    config = CaptureConfig(
        pixels=PixelContract.from_metadata(document["pixels"]), **document["pipeline"]
    )
    target_options = dict(document["target"])
    target_options["expected_client_size"] = tuple(target_options["expected_client_size"])
    target = DXGISettings(**target_options)
    conditions = document["input_conditions"]
    if conditions.get("version") != 1 or conditions.get("id") != target.condition_id:
        raise ValueError("Capture input condition identity must match target")
    return config, target
