"""Stream explicit additions into immutable SAC experience."""

from __future__ import annotations

from contextlib import ExitStack, closing
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Any

from fh5.artifact_io import VerifiedFile, sha256_file
from fh5.numeric_actor import FrozenNumericActor
from fh5.replay_document import ReplayArray, write_replay_document
from fh5.sac_actions import ActionBounds
from fh5.sac_data import LearningReplay
from fh5.sac_experience_assets import ExperienceAssets, experience_assets
from fh5.sac_experience_index import ExperienceUnion, experience_union
from fh5.sac_source_files import source_replays
from fh5.sac_sources import check_compatible


def expand_experience(
    torch: Any,
    parent: Path,
    parent_sha: str,
    additions: tuple[tuple[Path, str], ...],
    output: Path,
    bc: FrozenNumericActor,
    bounds: ActionBounds,
) -> tuple[Path, str, int, dict[str, Any]]:
    with (
        experience_assets() as assets,
        experience_union() as union,
        ExitStack() as parent_resources,
    ):
        return _expand(
            torch,
            parent,
            parent_sha,
            additions,
            output,
            bc,
            bounds,
            assets,
            union,
            parent_resources,
        )


def _expand(
    torch: Any,
    parent: Path,
    parent_sha: str,
    additions: tuple[tuple[Path, str], ...],
    output: Path,
    bc: FrozenNumericActor,
    bounds: ActionBounds,
    assets: ExperienceAssets,
    union: ExperienceUnion,
    parent_resources: ExitStack,
) -> tuple[Path, str, int, dict[str, Any]]:
    if not additions:
        raise ValueError("SAC expansion requires sealed replay additions")
    sources = chain(((parent, parent_sha),), additions)
    added = 0
    combined: dict[str, Any] | None = None
    for number, (path, sha) in enumerate(sources):
        with ExitStack() as resources:
            data = LearningReplay(
                torch,
                path,
                sha,
                bc,
                bounds,
                resources=parent_resources if number == 0 else resources,
            )
            replay = data.replay
            if combined is None:
                replaced = {"transitions", "source_inventory", "excluded", "observation_errors"}
                # Keep the parent's index alive for extension arrays copied to
                # the union; don't deep-copy the database or load those arrays.
                combined = {
                    k: v if isinstance(v, ReplayArray) else deepcopy(v)
                    for k, v in replay.items()
                    if k not in replaced
                }
            check_compatible(replay, combined)
            entries = replay.get("source_inventory") or [
                {
                    "replay_sha256": sha,
                    "source_hashes": replay["source_hashes"],
                    "path": f"sources/{sha}.json",
                }
            ]
            if replay.get("source_inventory"):
                with closing(source_replays(path.parent, replay)) as originals:
                    for original_source in originals:
                        assets.add_source(original_source.file)
            else:
                assets.add_source(VerifiedFile(path, sha))
            for entry in entries:
                union.add_source(number, entry)
            for original in replay["transitions"]:
                row = deepcopy(original)
                provenance = row.get("provenance") or {
                    "replay_sha256": sha,
                    "transition_id": row["id"],
                }
                if not union.contains_source(number, provenance["replay_sha256"]):
                    raise ValueError("SAC transition lacks its source inventory")
                row["provenance"] = provenance
                row["id"] = provenance["replay_sha256"] + ":" + provenance["transition_id"]
                for observation in (row["current"], row["next"]):
                    if observation is None:
                        continue
                    for entry in observation["frames"]:
                        entry["path"] = assets.add_frame(path.parent, entry)
                union.add_transition(row)
                added += int(number > 0)
    assert combined is not None
    combined.update(
        transitions=union.array("transitions"), source_inventory=union.array("source_inventory")
    )
    if combined["version"] in (2, 3):
        combined["source_role"] = "mixed"
    # Recording-specific identities belong to each inventory entry, not to the union.
    combined["source_hashes"] = {
        k: combined["source_hashes"][k] for k in ("task", "route", "reward")
    }
    combined["excluded"] = []
    combined["observation_errors"] = []
    output.mkdir(parents=True, exist_ok=False)
    statistics = assets.copy_into(output)
    path = output / "replay.json"
    write_replay_document(path, combined)
    return (
        path,
        sha256_file(path),
        added,
        statistics,
    )
