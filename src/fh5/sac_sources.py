"""Bounded replay provenance and explicit demonstration/online batch sampling."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from fh5.sac_checkpoint import source_replays


def check_compatible(replay: dict[str, Any], anchor: dict[str, Any]) -> None:
    for key in ("pixel_contract", "task_context", "task_state_role"):
        if replay[key] != anchor[key]:
            raise ValueError("Incompatible SAC experience: " + key)
    for key in ("route", "reward"):
        if replay["source_hashes"][key] != anchor["source_hashes"][key]:
            raise ValueError("Incompatible SAC experience: " + key)
    current_task, prior_task = replay.get("task_contract"), anchor.get("task_contract")
    if current_task is not None and prior_task is not None:
        if {k: v for k, v in current_task.items() if k != "control_owner"} != {
            k: v for k, v in prior_task.items() if k != "control_owner"
        }:
            raise ValueError("Incompatible SAC experience: task")
    elif replay["source_hashes"]["task"] != anchor["source_hashes"]["task"]:
        raise ValueError("Incompatible SAC experience: task")


def _leaf_roles(replay: dict[str, Any]) -> list[str]:
    if replay["version"] == 1:
        if (
            "source_role" in replay
            or "task_contract" in replay
            or any("control_owner" in row for row in replay["transitions"])
        ):
            raise ValueError("Legacy SAC replay cannot declare new role or owner fields")
        return ["online"] * len(replay["transitions"])
    owner = {"demonstration": "human", "online": "policy"}.get(replay.get("source_role", ""))
    if (
        owner is None
        or replay.get("task_contract", {}).get("control_owner") != owner
        or any(row.get("control_owner") != owner for row in replay["transitions"])
    ):
        raise ValueError("SAC source role differs from its actual control owner")
    return [replay["source_role"]] * len(replay["transitions"])


def _signature(row: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(row)
    value.pop("id", None)
    value.pop("provenance", None)
    for observation in (value["current"], value["next"]):
        if observation is not None:
            for frame in observation["frames"]:
                frame.pop("path", None)  # Packing may relocate identical content-addressed RGB.
    return value


def replay_roles(root: Path, replay: dict[str, Any]) -> list[str]:
    if (
        (replay.get("version"), replay.get("kind"))
        not in ((1, "sac-numeric-replay-v1"), (2, "sac-numeric-replay-v2"))
        or replay.get("source_kind") != "synthetic"
        or not isinstance(replay.get("transitions"), list)
        or not 1 <= len(replay["transitions"]) <= 10_000
        or replay.get("version") == 2
        and not isinstance(replay.get("task_contract"), dict)
    ):
        raise ValueError("SAC requires a bounded prepared synthetic replay")
    required = {"id", "current", "next", "action", "reward", "discount", "bootstrap", "terminated"}
    if any(not required <= row.keys() for row in replay["transitions"]):
        raise ValueError("SAC Q learning requires complete transitions, not action-only labels")
    if len({row["id"] for row in replay["transitions"]}) != len(replay["transitions"]):
        raise ValueError("Duplicate SAC transition")
    sources = source_replays(root, replay)
    if not sources:
        return _leaf_roles(replay)
    originals = {}
    for entry in replay["source_inventory"]:
        source = json.loads(sources[entry["path"]])
        if source.get("source_inventory"):
            raise ValueError("SAC source inventory must contain original leaf replays")
        roles = replay_roles(root, source)
        check_compatible(replay, source)
        for row, role in zip(source["transitions"], roles):
            originals[(entry["replay_sha256"], row["id"])] = (row, role)
    result = []
    seen = set()
    for row in replay["transitions"]:
        provenance = row.get("provenance", {})
        key = (provenance.get("replay_sha256"), provenance.get("transition_id"))
        if key not in originals or key in seen:
            raise ValueError("SAC transition lacks a unique original source")
        original, role = originals[key]
        if _signature(row) != _signature(original):
            raise ValueError("SAC transition differs from its sealed original source")
        seen.add(key)
        result.append(role)
    return result


class ReplaySampling:
    """Fixed quotas, no replacement within a batch and no hidden small-pool backfill."""

    def __init__(self, roles: list[str], batch_size: int, fraction: float | None) -> None:
        if fraction is not None and (
            type(fraction) not in (int, float)
            or not math.isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise ValueError("SAC demonstration fraction must be finite and in [0, 1]")
        self.roles, self.batch_size, self.fraction = roles, batch_size, fraction
        self.pools = {
            role: [i for i, value in enumerate(roles) if value == role]
            for role in ("demonstration", "online")
        }
        self.quotas = None
        if fraction is not None:
            demo_count = math.floor(batch_size * fraction + 0.5)
            self.quotas = {"demonstration": demo_count, "online": batch_size - demo_count}
            if any(count and not self.pools[role] for role, count in self.quotas.items()):
                raise ValueError("SAC sampling quota requires a nonempty source pool")
        self.sampled = dict.fromkeys(self.pools, 0)

    def sample(self, torch: Any) -> list[int]:
        if self.quotas is None:
            indices: list[int] = torch.randperm(len(self.roles))[: self.batch_size].tolist()
        else:
            indices = []
            for role, count in self.quotas.items():
                if count:
                    pool = self.pools[role]
                    indices.extend(pool[i] for i in torch.randperm(len(pool))[:count].tolist())
            indices = [indices[i] for i in torch.randperm(len(indices)).tolist()]
        for index in indices:
            self.sampled[self.roles[index]] += 1
        return indices

    def report(self) -> dict[str, Any]:
        return {
            "version": 1,
            "mode": "uniform" if self.quotas is None else "fixed_source_quotas",
            "demonstration_fraction": self.fraction,
            "requested_batch_size": self.batch_size,
            "quotas": self.quotas,
            "available": {role: len(pool) for role, pool in self.pools.items()},
            "sampled": dict(self.sampled),
            "small_pool_policy": "shrink_without_replacement_or_backfill",
        }
