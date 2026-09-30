"""Shared numerical features for temporal BC training and frozen inference."""

from __future__ import annotations

import math
from typing import Any

from fh5.bc_learning import MODEL_METADATA_KEYS, _numeric
from fh5.numeric_images import NumericFrame, PixelContract

TEMPORAL_ARCHITECTURE = "rgb-history-conv4-64-state64-fusion128-tanh2-dt-v2"
TEMPORAL_METADATA_KEYS = (
    *MODEL_METADATA_KEYS,
    "numeric_contract",
    "dataset_sha256",
    "provenance",
    "groups",
)
ACTOR_FIELDS = {
    "ego",
    "ego_mask",
    "ego_age_ms",
    "images",
    "image_mask",
    "image_age_ms",
    "actions",
    "action_mask",
    "action_age_ms",
    "reference",
}


def time_contract(mode: str, pixels: PixelContract) -> dict[str, Any]:
    if mode not in ("actual", "fixed") or len(pixels.history_offsets_ms) < 2:
        raise ValueError("Temporal BC requires actual/fixed time mode and at least two images")
    return {
        "version": 1,
        "mode": mode,
        "units": "seconds",
        "order": "oldest_first",
        "delta": "adjacent integer source_ns subtraction before float conversion",
        "numeric_suffix": "(adjacent_delta_s, valid_mask) per pair",
        "fixed_image_age_ms": list(pixels.history_offsets_ms),
        "fixed_mode_fields": ["all_image_ages", "all_adjacent_image_deltas"],
        "incomplete_history": "skip",
    }


def actor_shape(actor: dict[str, Any], images: int) -> dict[str, int]:
    if set(actor) != ACTOR_FIELDS:
        raise ValueError("Unexpected actor fields; supervision must remain outside actor")
    if any(len(actor[k]) != images for k in ("images", "image_mask", "image_age_ms")):
        raise ValueError("Temporal image history shape mismatch")
    actions, reference = len(actor["actions"]), len(actor["reference"]["mask"])
    if (
        not 1 <= actions <= 64
        or not 1 <= reference <= 256
        or any(len(actor[k]) != actions for k in ("action_mask", "action_age_ms"))
        or len(actor["reference"]["waypoints_m"]) != reference
        or set(actor["reference"]) != {"waypoints_m", "mask"}
        or set(actor["ego"]) != {"speed_mps", "velocity_car_mps", "angular_velocity_car_radps"}
        or len(actor["ego"]["velocity_car_mps"]) != 3
        or len(actor["ego"]["angular_velocity_car_radps"]) != 3
    ):
        raise ValueError("Temporal numerical history shape mismatch")
    for action, mask, age in zip(actor["actions"], actor["action_mask"], actor["action_age_ms"]):
        if type(mask) is not bool or (
            mask
            and (
                not isinstance(action, list)
                or len(action) != 2
                or any(type(v) not in (int, float) or not -1 <= v <= 1 for v in action)
                or type(age) not in (int, float)
                or not math.isfinite(age)
                or age <= 0
            )
        ):
            raise ValueError("Action history must be bounded and strictly before decision")
    if not actor["ego_mask"] or not math.isfinite(actor["ego_age_ms"]) or actor["ego_age_ms"] < 0:
        raise ValueError("Ego state must be available by the decision")
    for point, mask in zip(actor["reference"]["waypoints_m"], actor["reference"]["mask"]):
        if type(mask) is not bool or (mask and (not isinstance(point, list) or len(point) != 2)):
            raise ValueError("Invalid reference shape")
    return {"action_count": actions, "reference_count": reference}


def temporal_features(
    actor: dict[str, Any], frames: tuple[NumericFrame, ...], timing: dict[str, Any]
) -> list[float]:
    state = dict(actor)
    deltas = [(b.source_time_ns - a.source_time_ns) / 1e9 for a, b in zip(frames, frames[1:])]
    if any(v <= 0 for v in deltas):
        raise ValueError("Temporal features require distinct forward source times")
    if timing["mode"] == "fixed":
        ages = timing["fixed_image_age_ms"]
        state["image_age_ms"] = ages
        deltas = [(a - b) / 1000 for a, b in zip(ages, ages[1:])]
    values = _numeric(state)
    for delta in deltas:
        values.extend([delta, 1.0])
    return values


def describe_time(actor: dict[str, Any], frames: tuple[NumericFrame, ...]) -> dict[str, Any]:
    return {
        "image_age_s": [age / 1000 for age in actor["image_age_ms"]],
        "adjacent_delta_s": [
            (b.source_time_ns - a.source_time_ns) / 1e9 for a, b in zip(frames, frames[1:])
        ],
        "time_quality": [f.time_quality for f in frames],
        "uncertainty_ns": [f.uncertainty_ns for f in frames],
    }
