"""Explicit, bounded additions to immutable SAC experience."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from fh5.collection_store import encode, write_file
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_recording import read_numeric_frame
from fh5.sac_actions import ActionBounds
from fh5.sac_checkpoint import source_replays
from fh5.sac_data import LearningReplay
from fh5.sac_sources import check_compatible


def expand_experience(
    torch: Any,
    parent: Path,
    parent_sha: str,
    additions: tuple[tuple[Path, str], ...],
    output: Path,
    bc: FrozenNumericActor,
    bounds: ActionBounds,
) -> tuple[Path, str, int]:
    if not 1 <= len(additions) <= 10:
        raise ValueError("SAC expansion requires 1..10 sealed replay additions")
    sources = [(parent, parent_sha), *additions]
    inventory: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    hashes, source_ids, transition_ids = set(), set(), set()
    pixels: dict[str, bytes] = {}
    manifests: dict[str, bytes] = {}
    byte_count, added = 0, 0
    combined: dict[str, Any] | None = None
    for number, (path, sha) in enumerate(sources):
        data = LearningReplay(torch, path, sha, bc, bounds)
        replay = json.loads(data.raw)
        if combined is None:
            combined = deepcopy(replay)
        check_compatible(replay, combined)
        entries = replay.get("source_inventory") or [
            {
                "replay_sha256": sha,
                "source_hashes": replay["source_hashes"],
                "path": f"sources/{sha}.json",
            }
        ]
        manifests.update(
            source_replays(path.parent, replay)
            if replay.get("source_inventory")
            else {f"sources/{sha}.json": data.raw}
        )
        for entry in entries:
            # Re-reviewing or reformatting metadata does not create another interaction.
            identity = entry["source_hashes"]["packets"]
            if entry["replay_sha256"] in hashes or identity in source_ids:
                raise ValueError("Duplicate SAC experience cannot earn new update credit")
            hashes.add(entry["replay_sha256"])
            source_ids.add(identity)
            inventory.append(entry)
        for original in replay["transitions"]:
            row = deepcopy(original)
            provenance = row.get("provenance") or {"replay_sha256": sha, "transition_id": row["id"]}
            if provenance["replay_sha256"] not in {e["replay_sha256"] for e in entries}:
                raise ValueError("SAC transition lacks its source inventory")
            row["provenance"] = provenance
            row["id"] = provenance["replay_sha256"] + ":" + provenance["transition_id"]
            if row["id"] in transition_ids:
                raise ValueError("Duplicate SAC transition")
            transition_ids.add(row["id"])
            for observation in (row["current"], row["next"]):
                if observation is None:
                    continue
                for entry in observation["frames"]:
                    frame = read_numeric_frame(path.parent, entry)
                    name = "frames/" + entry["sha256"] + ".rgb"
                    if name not in pixels:
                        byte_count += frame.pixels.nbytes
                        if byte_count > 512 * 1024**2:
                            raise ValueError("Expanded SAC experience exceeds 512 MiB")
                        pixels[name] = bytes(frame.pixels)
                    entry["path"] = name
            rows.append(row)
            added += int(number > 0)
            if len(rows) > 10_000:
                raise ValueError("Expanded SAC replay exceeds 10000 transitions")
    assert combined is not None
    combined.update(transitions=rows, source_inventory=inventory)
    if combined["version"] == 2:
        combined["source_role"] = "mixed"
    # Recording-specific identities belong to each inventory entry, not to the union.
    combined["source_hashes"] = {
        k: combined["source_hashes"][k] for k in ("task", "route", "reward")
    }
    combined["excluded"] = []
    combined["observation_errors"] = []
    raw = encode(combined)
    if len(raw) > 128 * 1024**2:
        raise ValueError("Expanded SAC replay exceeds 128 MiB")
    output.mkdir(parents=True, exist_ok=False)
    if sum(map(len, manifests.values())) > 128 * 1024**2:
        raise ValueError("Expanded SAC source manifests exceed 128 MiB")
    for name, payload in {**pixels, **manifests}.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_file(target, payload)
    path = output / "replay.json"
    write_file(path, raw)
    return path, hashlib.sha256(raw).hexdigest(), added
