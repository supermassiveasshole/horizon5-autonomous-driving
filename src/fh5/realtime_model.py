"""Frozen Δt actor for read-only shadow work, with explicit source provenance."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import NumericFrame, PixelContract


def shadow_model_contract(
    directory: Path,
    source: PixelContract,
    expected_model_sha256: str,
    allow_legacy_source_diagnostic: bool = False,
) -> tuple[dict[str, Any], PixelContract]:
    original = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    if original.get("version") != 2:
        raise ValueError("Shadow runtime requires a frozen temporal BC model")
    if original.get("weights_sha256") != expected_model_sha256:
        raise ValueError("Shadow expected model hash differs from selected model")
    trained = PixelContract.from_metadata(original["numeric_contract"])
    if trained != source and not (
        allow_legacy_source_diagnostic
        and trained.origin == "legacy_offline"
        and source.origin == "direct_numeric"
        and replace(trained, origin="direct_numeric") == source
    ):
        raise ValueError("Explicit source diagnostic required; all other pixel fields must match")
    return original, trained


class ShadowNumericActor:
    kind = "frozen-temporal-bc-read-only-shadow-v1"

    def __init__(
        self,
        directory: Path,
        source: PixelContract,
        expected_model_sha256: str,
        device: str = "cpu",
        *,
        allow_legacy_source_diagnostic: bool = False,
        expected_manifest_sha256: str | None = None,
    ) -> None:
        _, trained = shadow_model_contract(
            directory, source, expected_model_sha256, allow_legacy_source_diagnostic
        )
        self.actor = FrozenNumericActor(
            directory, trained, device, expected_manifest_sha256=expected_manifest_sha256
        )
        self.device = self.actor.device
        if self.actor.manifest["weights_sha256"] != expected_model_sha256:
            raise ValueError("Shadow expected model changed while loading")
        self.manifest = {
            **self.actor.manifest,
            "numeric_contract": source.metadata(),
            "training_numeric_contract": trained.metadata(),
            "source_compatibility": "matching_pixel_contract; live_conditions_unverified"
            if trained == source
            else "explicit_unvalidated_legacy_to_direct_shadow",
            "diagnostic_only": True,
            "new_capture_distribution_validated": False,
        }

    def predict(self, actor: dict[str, Any], frames: tuple[NumericFrame, ...]) -> list[float]:
        return self.actor.predict(actor, frames)

    def input_features(
        self, actor: dict[str, Any], frames: tuple[NumericFrame, ...]
    ) -> list[float]:
        return self.actor.input_features(actor, frames)

    def clear_input_cache(self) -> None:
        self.actor.clear_input_cache()
