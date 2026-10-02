"""Verified numerical BC snapshots with indexed rows and per-observation pixels."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, overload

from fh5.artifact_io import VerifiedFile, sha256_file
from fh5.bc_learning import VIEWS
from fh5.numeric_images import NumericDecision, PixelContract, validate_decision
from fh5.numeric_recording import read_numeric_frame
from fh5.replay_document import replay_document
from fh5.sac_data_index import LearningDataIndex, learning_data_index
from fh5.temporal_features import actor_shape


class TemporalRows(Sequence[dict[str, Any]]):
    """Reconstruct only selected observations while the snapshot context is open."""

    def __init__(
        self,
        root: Path,
        entries: Sequence[Any],
        groups: dict[str, str],
        index: LearningDataIndex,
        counts: dict[str, int],
        role: str | None = None,
    ) -> None:
        self.root, self.entries, self.groups = root, entries, groups
        self.store, self.counts, self.role = index, counts, role

    def __len__(self) -> int:
        return 2 * (len(self.entries) if self.role is None else self.counts.get(self.role, 0))

    def select(self, split: str, *, eligible: bool = False) -> TemporalRows:
        role = ("eligible:" if eligible else "split:") + split
        return TemporalRows(self.root, self.entries, self.groups, self.store, self.counts, role)

    def _pair(self, position: int) -> Iterator[dict[str, Any]]:
        source = position if self.role is None else self.store.get(self.role, position)
        entry = self.entries[source]
        # Revalidate every selected asset; preflight does not grant trust to
        # pixels that change before an update or a later verification pass.
        frames = tuple(read_numeric_frame(self.root, metadata) for metadata in entry["frames"])
        for view in VIEWS:
            yield {
                "entry": entry,
                "decision": NumericDecision(
                    entry["decision_id"] + ":" + view,
                    entry["epoch"],
                    entry["decision_ns"],
                    frames,
                    entry["views"][view],
                    entry["supervision"],
                ),
                "view": view,
                "split": self.groups[entry["group"]],
            }

    @overload
    def __getitem__(self, key: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, key: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, key: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        if key < 0:
            key += len(self)
        if not 0 <= key < len(self):
            raise IndexError(key)
        return tuple(self._pair(key // 2))[key % 2]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for position in range(len(self) // 2):
            yield from self._pair(position)


@contextmanager
def temporal_snapshot(
    path: Path, *, expected_sha256: str | None = None
) -> Iterator[tuple[dict[str, Any], PixelContract, TemporalRows]]:
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Numerical dataset changed before input reconstruction")
    with (
        replay_document(VerifiedFile(path, digest)) as data,
        learning_data_index() as index,
    ):
        if (
            data.get("version") != 1
            or data.get("kind") != "numeric-bc-snapshot-v1"
            or data.get("action_contract") != "xinput-lx-rt-lt-v1"
        ):
            raise ValueError("Unsupported numerical training snapshot")
        pixels = PixelContract.from_metadata(data["pixel_contract"])
        # Group/provenance metadata is still part of the existing Torch model
        # contract. Do not persist SQLite objects into that frozen contract.
        data["groups"] = list(data["groups"])
        groups = {g["id"]: g["split"] for g in data["groups"]}
        source_ranges = {g["id"]: g.get("source_ranges") for g in data["groups"]}
        if (
            len(groups) != len(data["groups"])
            or not groups
            or any(v not in ("train", "development", "evaluation") for v in groups.values())
        ):
            raise ValueError("Invalid independent attempt groups")
        evidence = [g["evidence_id"] for g in data["groups"]]
        if len(set(evidence)) != len(evidence):
            raise ValueError("One source cannot be split between independent groups")
        if not data["decisions"]:
            raise ValueError("Numerical snapshot requires decisions")
        counts: dict[str, int] = {}
        shape = None
        for position, entry in enumerate(data["decisions"]):
            if (
                not index.add("decisions", entry["decision_id"], True)
                or entry["group"] not in groups
            ):
                raise ValueError("Duplicate decision or unknown attempt group")
            ranges = source_ranges[entry["group"]]
            if ranges is not None:
                sequence = entry.get("source_sequence")
                if (
                    not isinstance(sequence, int)
                    or isinstance(sequence, bool)
                    or not any(
                        entry["decision_id"] == f"{r['session_sha256']}:{sequence}"
                        and r["start_sequence"] <= sequence < r["end_sequence"]
                        and r["first_ns"] <= entry["decision_ns"] <= r["last_ns"]
                        and all(
                            r["first_ns"] <= f["source_time_ns"] <= entry["decision_ns"]
                            and f["epoch"].startswith(r["session_sha256"] + ":")
                            and f["frame_id"].startswith(r["session_sha256"] + ":")
                            for f in entry["frames"]
                        )
                        for r in ranges
                    )
                ):
                    raise ValueError("Numerical sample contradicts its declared source range")
            frames = []
            for metadata in entry["frames"]:
                key = json.dumps([metadata["epoch"], metadata["frame_id"]])
                identity = [metadata, entry["group"]]
                prior = index.get("frames", key)
                if prior is not None and prior != identity:
                    raise ValueError("Numerical frame identity or attempt group changed")
                index.add("frames", key, identity)
                frames.append(read_numeric_frame(path.parent, metadata))
            if set(entry["views"]) != set(VIEWS):
                raise ValueError("Snapshot requires paired reference views")
            plain, assisted = (entry["views"][v] for v in VIEWS)
            if any(plain["reference"]["mask"]) or {
                k: v for k, v in plain.items() if k != "reference"
            } != {k: v for k, v in assisted.items() if k != "reference"}:
                raise ValueError("Reference views differ outside the reference branch")
            for view in VIEWS:
                actor = entry["views"][view]
                current = actor_shape(actor, len(pixels.history_offsets_ms))
                if shape is not None and shape != current:
                    raise ValueError("Numerical actor dimensions changed in snapshot")
                shape = current
                decision = NumericDecision(
                    entry["decision_id"] + ":" + view,
                    entry["epoch"],
                    entry["decision_ns"],
                    tuple(frames),
                    actor,
                    entry["supervision"],
                )
                reason = validate_decision(decision, pixels)
                if reason:
                    raise ValueError("Invalid numerical training observation: " + reason)
                supervision = entry["supervision"]
                if entry["bc_eligible"] and (
                    not supervision.get("action_mask")
                    or supervision.get("quality") != "trusted"
                    or supervision.get("reasons")
                ):
                    raise ValueError("Untrusted action cannot enter BC training")
                if entry["bc_eligible"] and (
                    len(supervision["action"]) != 2
                    or any(
                        not isinstance(a, (int, float)) or not -1 <= a <= 1
                        for a in supervision["action"]
                    )
                ):
                    raise ValueError("Invalid bounded demonstration action")
            split = groups[entry["group"]]
            for role in ("split:" + split, "eligible:" + split):
                if role.startswith("eligible:") and not entry["bc_eligible"]:
                    continue
                index.add(role, counts.get(role, 0), position)
                counts[role] = counts.get(role, 0) + 1
        # The generator stays suspended during learning: release the last
        # validation observation rather than pinning its pixels in the frame.
        del frames, decision
        yield data, pixels, TemporalRows(path.parent, data["decisions"], groups, index, counts)
