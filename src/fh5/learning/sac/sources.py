"""Indexed replay provenance and explicit demonstration/online batch sampling."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import closing, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from fh5.artifacts.document import ReplayArray
from fh5.learning.sac.provenance_index import ReplayRoles, provenance_index
from fh5.learning.sac.source_files import source_replays
from fh5.learning.sac.timing import next_action_elapsed


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


def _leaf_role(replay: dict[str, Any]) -> str:
    if replay["version"] == 1:
        if (
            "source_role" in replay
            or "task_contract" in replay
            or any("control_owner" in row for row in replay["transitions"])
        ):
            raise ValueError("Legacy SAC replay cannot declare new role or owner fields")
        return "online"
    owner = {"demonstration": "human", "online": "policy"}.get(replay.get("source_role", ""))
    if (
        owner is None
        or replay.get("task_contract", {}).get("control_owner") != owner
        or any(row.get("control_owner") != owner for row in replay["transitions"])
    ):
        raise ValueError("SAC source role differs from its actual control owner")
    return str(replay["source_role"])


def _signature(row: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(row)
    value.pop("id", None)
    value.pop("provenance", None)
    for observation in (value["current"], value["next"]):
        if observation is not None:
            for frame in observation["frames"]:
                frame.pop("path", None)  # Packing may relocate identical content-addressed RGB.
    return value


def _validate_document(replay: dict[str, Any], roles: ReplayRoles) -> None:
    if (
        (replay.get("version"), replay.get("kind"))
        not in (
            (1, "sac-numeric-replay-v1"),
            (2, "sac-numeric-replay-v2"),
            (3, "sac-numeric-replay-v3"),
        )
        or replay.get("source_kind") not in ("synthetic", "native", "mixed")
        or not isinstance(replay.get("transitions"), (list, ReplayArray))
        or not len(replay["transitions"])
        or replay.get("version") in (2, 3)
        and not isinstance(replay.get("task_contract"), dict)
    ):
        raise ValueError("SAC requires a nonempty prepared replay with known sources")
    if not replay.get("source_inventory") and (
        replay["source_kind"] == "mixed"
        or replay["source_kind"] == "native"
        and (
            replay["version"] != 3
            or replay.get("source_role") != "online"
            or not replay.get("source_hashes", {}).get("execution")
        )
    ):
        raise ValueError("Native SAC experience requires an asynchronous execution source")
    required = {"id", "current", "next", "action", "reward", "discount", "bootstrap", "terminated"}
    roles.begin_document()
    for row in replay["transitions"]:
        if not required <= row.keys():
            raise ValueError("SAC Q learning requires complete transitions, not action-only labels")
        roles.check_identity(row["id"])
        next_action_elapsed(row)
        if (
            replay["version"] == 3
            and not replay.get("source_inventory")
            and row.get("action_time_basis") != "asynchronous_send_return_proxy_v1"
        ):
            raise ValueError("Version 3 requires explicit asynchronous transition timing")


@contextmanager
def replay_roles(root: Path, replay: dict[str, Any]) -> Iterator[ReplayRoles]:
    with provenance_index() as roles:
        _validate_document(replay, roles)
        if not replay.get("source_inventory"):
            role = _leaf_role(replay)
            for _ in replay["transitions"]:
                roles.add_role(role)
        else:
            source_kinds = set()
            with closing(source_replays(root, replay)) as sources:
                for entry in sources:
                    source = entry.document
                    if source.get("source_inventory"):
                        raise ValueError("SAC source inventory must contain original leaf replays")
                    _validate_document(source, roles)
                    source_kinds.add(source["source_kind"])
                    role = _leaf_role(source)
                    check_compatible(replay, source)
                    for row in source["transitions"]:
                        roles.add_original(entry.file.sha256, row, role)
            expected_kind = next(iter(source_kinds)) if len(source_kinds) == 1 else "mixed"
            if replay["source_kind"] != expected_kind:
                raise ValueError("SAC replay source kind differs from its sealed originals")
            for row in replay["transitions"]:
                provenance = row.get("provenance", {})
                original, role = roles.take_original(
                    provenance.get("replay_sha256"), provenance.get("transition_id")
                )
                if _signature(row) != _signature(original):
                    raise ValueError("SAC transition differs from its sealed original source")
                roles.add_role(role)
        yield roles


class ReplaySampling:
    """Fixed quotas, no replacement within a batch and no hidden small-pool backfill."""

    def __init__(self, roles: ReplayRoles, batch_size: int, fraction: float | None) -> None:
        if fraction is not None and (
            type(fraction) not in (int, float)
            or not math.isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise ValueError("SAC demonstration fraction must be finite and in [0, 1]")
        self.roles, self.batch_size, self.fraction = roles, batch_size, fraction
        self.quotas = None
        if fraction is not None:
            demo_count = math.floor(batch_size * fraction + 0.5)
            self.quotas = {"demonstration": demo_count, "online": batch_size - demo_count}
            if any(count and not roles.counts[role] for role, count in self.quotas.items()):
                raise ValueError("SAC sampling quota requires a nonempty source pool")
        self.sampled = dict.fromkeys(roles.counts, 0)

    def sample(self, torch: Any) -> list[int]:
        if self.quotas is None:
            indices: list[int] = torch.randperm(len(self.roles))[: self.batch_size].tolist()
        else:
            indices = []
            for role, count in self.quotas.items():
                if count:
                    indices.extend(
                        self.roles.position(role, i)
                        for i in torch.randperm(self.roles.counts[role])[:count].tolist()
                    )
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
            "available": dict(self.roles.counts),
            "sampled": dict(self.sampled),
            "small_pool_policy": "shrink_without_replacement_or_backfill",
        }
