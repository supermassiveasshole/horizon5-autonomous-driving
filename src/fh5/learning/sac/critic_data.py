"""Frozen critic features on disk; materialize only a selected numerical batch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fh5.artifacts.io import encode
from fh5.learning.bc.actor import FrozenNumericActor
from fh5.learning.sac.data_index import LearningDataIndex
from fh5.observation.numeric import PixelContract
from fh5.observation.recording import read_numeric_frame


@dataclass(frozen=True)
class CriticBatch:
    current: Any
    following: Any
    actions: Any
    next_actions: Any
    rewards: Any
    discounts: Any


class CriticData:
    def __init__(
        self, torch: Any, index: LearningDataIndex, pixels: PixelContract, root: Path
    ) -> None:
        self.torch, self.index, self.pixels, self.root = torch, index, pixels, root
        self.count = 0
        self.width = 0
        self.total_bytes = 0

    def predict(
        self, actor: FrozenNumericActor, root: Path, row: dict[str, Any]
    ) -> tuple[Any, list[float]]:
        frame_bytes = self.pixels.size[0] * self.pixels.size[1] * 3
        try:
            frames = tuple(read_numeric_frame(root, item, frame_bytes) for item in row["frames"])
            return actor.predict_with_features(row["actor"], frames)
        finally:
            actor.clear_input_cache()

    def observation(
        self, actor: FrozenNumericActor, row: dict[str, Any]
    ) -> tuple[Any, list[float]]:
        identity = hashlib.sha256(encode(row)).hexdigest()
        cached = self.index.get("features", identity)
        if cached is None:
            features, prediction = self.predict(actor, self.root, row)
            cached = {"features": features.tolist(), "prediction": prediction}
            self.index.add("features", identity, cached)
            self.index.add("observations", identity, row)
            for metadata in row["frames"]:
                # Preserve the existing full-metadata source accounting.
                key = json.dumps(metadata, sort_keys=True)
                if self.index.add("frames", key, True):
                    self.total_bytes += self.pixels.size[0] * self.pixels.size[1] * 3
        return self.torch.tensor(cached["features"], dtype=self.torch.float32), cached["prediction"]

    def add(self, record: dict[str, Any]) -> None:
        self.index.add("transitions", self.count, record)
        self.count += 1
        self.width = len(record["current"])

    def record(self, position: int) -> dict[str, Any]:
        row: dict[str, Any] | None = self.index.get("transitions", position)
        if row is None:
            raise ValueError("Missing indexed critic transition")
        return row

    def batch(self, positions: list[int]) -> CriticBatch:
        rows = [self.record(position) for position in positions]
        return CriticBatch(
            **{
                field: self.torch.tensor([row[field] for row in rows], dtype=self.torch.float32)
                for field in (
                    "current",
                    "following",
                    "actions",
                    "next_actions",
                    "rewards",
                    "discounts",
                )
            }
        )

    def reload_error(self, actor: FrozenNumericActor, root: Path) -> float:
        maximum = 0.0
        for identity, row in self.index.items("observations"):
            prediction = self.index.get("features", identity)["prediction"]
            maximum = max(
                maximum,
                *(abs(a - b) for a, b in zip(prediction, self.predict(actor, root, row)[1])),
            )
        return maximum
