"""Frozen Δt actor for read-only shadow work, with explicit source provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from fh5.artifact_io import VerifiedFile
from fh5.bc_losses import read_bc_manifest
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import NumericFrame, PixelContract
from fh5.replay_document import read_document_fields


def sac_model_contract(
    directory: Path, source: PixelContract, expected_sha256: str | None = None
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Bind the SAC policy and its BC pixel metadata before creating native resources."""
    raw = (directory / "policy.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Frozen SAC policy manifest changed")
    policy = json.loads(raw)
    if (
        policy.get("version") not in (2, 3, 4)
        or policy.get("architecture") != "conditional-temporal-sac-v1"
        or policy.get("stage") != "sac_updates"
    ):
        raise ValueError("Numerical inference requires a sealed SAC policy checkpoint")
    if policy.get("source_kind") in ("native", "mixed"):
        metadata, bc_digest = read_bc_manifest(directory / "bc/model.json")
        if bc_digest != policy["bc_manifest_sha256"]:
            raise ValueError("Frozen SAC parent BC manifest changed")
    else:
        metadata = read_document_fields(
            VerifiedFile(directory / "bc/model.json", policy["bc_manifest_sha256"]),
            {"version", "numeric_contract", "contract"},
        )
    if (
        metadata.get("version") != 2
        or PixelContract.from_metadata(metadata["numeric_contract"]) != source
    ):
        raise ValueError("SAC requires its exact numerical pixel contract")
    return metadata, policy, digest


def shadow_model_contract(
    directory: Path,
    source: PixelContract,
    expected_model_sha256: str | None = None,
    allow_legacy_source_diagnostic: bool = False,
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], PixelContract, str]:
    raw = (directory / "model.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError("Frozen numerical model manifest changed")
    original = json.loads(raw)
    if original.get("version") != 2:
        raise ValueError("Shadow runtime requires a frozen temporal BC model")
    if not isinstance(original.get("weights_sha256"), str):
        raise ValueError("Frozen model manifest requires its weights hash")
    if (
        expected_model_sha256 is not None
        and original.get("weights_sha256") != expected_model_sha256
    ):
        raise ValueError("Shadow expected model hash differs from selected model")
    trained = PixelContract.from_metadata(original["numeric_contract"])
    if trained != source and not (
        allow_legacy_source_diagnostic
        and trained.origin == "legacy_offline"
        and source.origin == "direct_numeric"
        and replace(trained, origin="direct_numeric") == source
    ):
        raise ValueError("Explicit source diagnostic required; all other pixel fields must match")
    return original, trained, digest


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
        _, trained, manifest_sha256 = shadow_model_contract(
            directory,
            source,
            expected_model_sha256,
            allow_legacy_source_diagnostic,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        self.actor = FrozenNumericActor(
            directory, trained, device, expected_manifest_sha256=manifest_sha256
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
