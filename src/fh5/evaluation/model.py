"""Versioned model assets for frozen BC and SAC evaluation batches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fh5.artifacts.io import read_bounded
from fh5.learning.bc.actor import FrozenNumericActor
from fh5.learning.sac.actor import FrozenSAC
from fh5.learning.sac.imitation import checkpoint_imitation, imitation_evidence
from fh5.observation.numeric import DecisionActor, PixelContract


def asset_limit(name: str) -> int:
    return (256 if name == "model/policy.pt" else 128) * 1024**2


def model_payloads(
    directory: Path, binding: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, bytes]]:
    sac = binding.get("kind") == "sac"
    names = (
        ["policy.json", "policy.pt", "training-report.json", "bc/model.json", "bc/actor.pt"]
        if sac
        else ["model.json", "actor.pt"]
    )
    payloads = {
        "model/" + name: read_bounded(directory / name, asset_limit("model/" + name))
        for name in names
    }
    manifest = payloads["model/policy.json" if sac else "model/model.json"]
    if hashlib.sha256(manifest).hexdigest() != binding["manifest_sha256"]:
        raise ValueError("Evaluation model manifest changed")
    if sac:
        payloads.update(
            {
                "model/" + name: raw
                for name, raw in imitation_evidence(
                    directory, checkpoint_imitation(json.loads(manifest))
                ).items()
            }
        )
    model = json.loads(payloads["model/bc/model.json" if sac else "model/model.json"])
    return model, payloads


def validate_model(
    torch: Any, directory: Path, binding: dict[str, Any], runtime: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    pixels = PixelContract.from_metadata(runtime["pixels"])
    if binding.get("kind") == "sac":
        actor = FrozenSAC(torch, directory)
        if actor.sha != binding["manifest_sha256"] or actor.pixels != pixels:
            raise ValueError("Frozen SAC evaluation contract differs")
        if runtime["action_offsets_ms"] != [200, 100, 0] or any(
            runtime[key] != getattr(actor.bounds, key)
            for key in ("max_steer", "max_throttle", "max_brake")
        ):
            raise ValueError("SAC execution bounds or history differ from the trained contract")
        if "device" in binding:
            if (
                binding["device"] != "cpu"
                or actor.manifest.get("source_kind") not in ("native", "mixed")
                or actor.bc.manifest["diagnostic_only"]
            ):
                raise ValueError(
                    "Native SAC evaluation requires native/mixed experience and non-diagnostic BC"
                )
            return actor.bc.original_contract, False
        return actor.bc.original_contract, True
    bc = FrozenNumericActor(directory, pixels, expected_manifest_sha256=binding["manifest_sha256"])
    return bc.original_contract, bc.manifest["diagnostic_only"]


def evaluation_actor(
    directory: Path, binding: dict[str, Any], runtime: dict[str, Any], *, device: str = "cpu"
) -> DecisionActor:
    pixels = PixelContract.from_metadata(runtime["pixels"])
    if binding.get("kind") == "sac":
        from fh5.learning.sac.evaluation_actor import SACEvaluationActor

        return SACEvaluationActor(directory, pixels, binding["manifest_sha256"])
    return FrozenNumericActor(
        directory, pixels, device, expected_manifest_sha256=binding["manifest_sha256"]
    )
