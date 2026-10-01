"""Versioned temporary guidance from the original, frozen BC actor."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


def initial_imitation(
    weights: tuple[float, ...] | list[float],
    teacher_sha256: str,
    protocol_sha256: str | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(weights, (tuple, list))
        or not 1 <= len(weights) <= 16
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10
            for value in weights
        )
        or weights[-1] != 0
        or any(a <= b for a, b in zip(weights, weights[1:]))
    ):
        raise ValueError("Imitation weights must strictly decrease to zero, within [0, 10]")
    return {
        "version": 1,
        "weights": list(weights),
        "index": 0,
        "weight": weights[0],
        "phase": "exited" if weights[0] == 0 else "guided",
        "teacher_manifest_sha256": teacher_sha256,
        "protocol_sha256": protocol_sha256,
        "objective": "mean squared deterministic command distance / per-axis interval width",
        "transitions": [],
    }


def freeze_imitation_protocol(batch_dir: Path) -> str:
    from fh5.candidate_selection import evaluation_conditions
    from fh5.collection_store import encode, read_bounded
    from fh5.evaluation import read_evaluation_batch

    digest = hashlib.sha256(read_bounded(batch_dir / "batch.json", 4 * 1024**2)).hexdigest()
    batch, _, _ = read_evaluation_batch(batch_dir, digest)
    if batch["config"]["purpose"] != "development":
        raise ValueError("Imitation requires a development evaluation protocol")
    return hashlib.sha256(encode(evaluation_conditions(batch))).hexdigest()


def guidance_loss(
    torch: Any, teacher: Any, inputs: tuple[Any, Any], context: Any, mean: Any
) -> Any:
    lower, upper = context[:, 3:5], context[:, 5:7]
    grid = mean.new_tensor([32767, 255])
    with torch.no_grad():
        target = torch.round(teacher(*inputs).clamp(lower, upper) * grid) / grid
    continuous = (lower + upper) / 2 + (upper - lower) / 2 * torch.tanh(mean)
    # The same straight-through approximation as SAC Q actions; this branch guides the mean.
    rounded = torch.round(continuous * grid) / grid
    command = continuous + (rounded - continuous).detach()
    return ((command - target) / (upper - lower)).square().mean()


def checkpoint_imitation(manifest: dict[str, Any]) -> dict[str, Any] | None:
    weights = manifest["configuration"].get("imitation_weights")
    if manifest["version"] != 4:
        if weights is not None or "imitation" in manifest:
            raise ValueError("Legacy SAC checkpoint cannot adopt unversioned imitation")
        return None
    if not isinstance(weights, list):
        raise ValueError("Imitation checkpoint lacks its frozen schedule")
    state = manifest.get("imitation", {})
    expected = initial_imitation(
        weights, manifest["bc_manifest_sha256"], state.get("protocol_sha256")
    )
    transitions = state.get("transitions")
    if not isinstance(transitions, list) or len(transitions) > 64:
        raise ValueError("Invalid imitation evaluation history")
    for transition in transitions:
        before, after = transition.get("from_index"), transition.get("to_index")
        if (
            before != expected["index"]
            or type(after) is not int
            or after not in (before, before + 1)
            or after >= len(weights)
            or expected["phase"] == "exited"
        ):
            raise ValueError("Invalid imitation phase transition")
        expected["index"], expected["weight"] = after, weights[after]
        expected["phase"] = "exited" if weights[after] == 0 else "guided"
    expected["transitions"] = transitions
    if manifest.get("imitation") != expected:
        raise ValueError("Imitation checkpoint phase or teacher differs from its contract")
    return expected


def imitation_evidence(root: Path, state: dict[str, Any] | None) -> dict[str, bytes]:
    import json

    from fh5.collection_store import read_bounded
    from fh5.numeric_images import asset

    blobs: dict[str, bytes] = {}
    if state is None:
        return blobs
    for item in state["transitions"]:
        raw = read_bounded(asset(root, item["review"]), 128 * 1024**2)
        proof = json.loads(raw)
        if (
            hashlib.sha256(raw).hexdigest() != item["review_sha256"]
            or any(
                proof[key] != item[key]
                for key in (
                    "from_index",
                    "to_index",
                    "parent_checkpoint_sha256",
                    "recording_origins",
                )
            )
            or proof["advanced"] != (item["to_index"] > item["from_index"])
            or proof["protocol_sha256"] != state["protocol_sha256"]
            or proof["scope"] != "synthetic_development_only"
        ):
            raise ValueError("Imitation evaluation evidence changed")
        blobs[item["review"]] = raw
        if sum(map(len, blobs.values())) > 128 * 1024**2:
            raise ValueError("Imitation evidence exceeds 128 MiB")
    return blobs
