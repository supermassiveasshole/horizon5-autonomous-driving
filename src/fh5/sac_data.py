"""Raw numerical SAC replay; trainable representations are never cached."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fh5.artifact_io import VerifiedFile
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import NumericDecision, validate_decision
from fh5.numeric_recording import read_numeric_frame
from fh5.replay_document import replay_document
from fh5.sac import _task_features
from fh5.sac_actions import ActionBounds
from fh5.sac_data_index import learning_data_index
from fh5.sac_sources import replay_roles
from fh5.sac_timing import next_action_elapsed


def validate_cache_budget(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("SAC raw frame cache budget must be nonnegative integer bytes")


@dataclass(frozen=True)
class LearningBatch:
    context: Any
    next_context: Any
    task: Any
    next_task: Any
    actions: Any
    rewards: Any
    discounts: Any


class LearningReplay:
    def __init__(
        self,
        torch: Any,
        path: Path,
        expected: str,
        actor: FrozenNumericActor,
        bounds: ActionBounds,
        *,
        resources: ExitStack,
        cache_bytes: int = 0,
    ) -> None:
        validate_cache_budget(cache_bytes)
        self.cache_bytes, self.cache_hits = cache_bytes, 0
        self.cache_misses = self.cache_evictions = self.peak_bytes = 0
        self.cache_bypasses = 0
        self.root = path.parent
        self.torch, self.actor, self.bounds = torch, actor, bounds
        self.file = VerifiedFile(path, expected)
        self.replay = replay = resources.enter_context(replay_document(self.file))
        self.roles = resources.enter_context(replay_roles(path.parent, replay))
        if replay["pixel_contract"] != actor.contract.metadata():
            raise ValueError("Unsupported SAC learning replay contract")
        self.rows = replay["transitions"]
        self.images: OrderedDict[str, Any] = OrderedDict()
        self.index = resources.enter_context(learning_data_index())
        self.raw_bytes = self.source_bytes = 0

        def observation(row: dict[str, Any]) -> str:
            identity = hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()
            if self.index.get("observations", identity) is not None:
                return identity
            frame_bytes = actor.contract.size[0] * actor.contract.size[1] * 3
            frames = tuple(read_numeric_frame(path.parent, f, frame_bytes) for f in row["frames"])
            decision = NumericDecision(
                row["decision_id"], row["epoch"], row["decision_ns"], frames, row["actor"]
            )
            reason = validate_decision(decision, actor.contract)
            if reason:
                raise ValueError("Invalid SAC observation: " + reason)
            numeric = actor.input_features(row["actor"], frames)
            references = []
            for frame, entry in zip(frames, row["frames"]):
                digest = entry["sha256"]
                if self.index.add("sources", digest, entry):
                    self.source_bytes += frame.pixels.nbytes
                references.append(digest)
            self.index.add("observations", identity, [references, numeric])
            return identity

        for position, row in enumerate(self.rows):
            current = observation(row["current"])
            context = bounds.context(row["previous_action"], row["action_elapsed_s"])
            if len(row["action"]) != 2 or any(
                not lo - 1e-9 <= value <= hi + 1e-9
                for lo, hi, value in zip(context[3:5], context[5:7], row["action"])
            ):
                raise ValueError("SAC replay action outside executable command support")
            task = _task_features(row["task_state"], replay["task_context"])
            if row["bootstrap"]:
                if row["next"] is None or row["terminated"]:
                    raise ValueError("SAC bootstrap requires a real nonterminal final observation")
                following = observation(row["next"])
                next_context = bounds.context(row["action"], next_action_elapsed(row))
                next_task = _task_features(row["next_task_state"], replay["task_context"])
                discount = row["discount"]
            else:
                if not row["terminated"]:
                    raise ValueError("Missing bootstrap is not a true SAC terminal")
                following, next_context, next_task, discount = current, context, task, 0.0
            values = {
                "context": context,
                "next_context": next_context,
                "task": task,
                "next_task": next_task,
                "actions": row["action"],
                "rewards": row["reward"],
                "discounts": discount,
            }
            if not all(
                torch.isfinite(torch.tensor(values[key], dtype=torch.float32)).all()
                for key in ("actions", "rewards", "discounts")
            ):
                raise ValueError("Non-finite SAC training data")
            self.index.add("transitions", position, [current, following, values])

    def batch(self, indices: list[int]) -> LearningBatch:
        rows = [self.index.get("transitions", i)[2] for i in indices]
        return LearningBatch(
            **{
                key: self.torch.tensor([row[key] for row in rows], dtype=self.torch.float32)
                for key in LearningBatch.__dataclass_fields__
            }
        )

    def cache_summary(self) -> dict[str, Any]:
        return {
            "budget_bytes": self.cache_bytes,
            "peak_bytes": self.peak_bytes,
            "retained_bytes": self.raw_bytes,
            "retained_frames": len(self.images),
            "unique_frames": self.index.source_count,
            "source_bytes": self.source_bytes,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "evictions": self.cache_evictions,
            "bypassed_frames": self.cache_bypasses,
            "storage_dtype": "uint8",
            "files_deleted": 0,
        }

    def _image(self, digest: str) -> Any:
        if digest in self.images:
            self.cache_hits += 1
            self.images.move_to_end(digest)
            return self.images[digest]
        self.cache_misses += 1
        frame_bytes = self.actor.contract.size[0] * self.actor.contract.size[1] * 3
        frame = read_numeric_frame(self.root, self.index.get("sources", digest), frame_bytes)
        size = frame.pixels.nbytes
        width, height = frame.size
        value = (
            self.torch.frombuffer(bytearray(frame.pixels), dtype=self.torch.uint8)
            .reshape(height, width, 3)
            .permute(2, 0, 1)
        )
        if size > self.cache_bytes:
            self.cache_bypasses += 1
            return value
        while self.raw_bytes + size > self.cache_bytes:
            _, old = self.images.popitem(last=False)
            self.raw_bytes -= old.numel() * old.element_size()
            self.cache_evictions += 1
            del old
        self.images[digest] = value
        self.raw_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.raw_bytes)
        return value

    def inputs(self, indices: list[int], *, following: bool = False) -> tuple[Any, Any]:
        width, height = self.actor.contract.size
        history = self.actor.original_contract["image_count"]
        entries = [
            self.index.get("observations", self.index.get("transitions", i)[int(following)])
            for i in indices
        ]
        images = self.torch.empty(
            (len(indices), history, 3, height, width), dtype=self.torch.float32
        )
        for index, (references, _) in enumerate(entries):
            for slot, digest in enumerate(references):
                images[index, slot].copy_(self._image(digest))
        return (
            images.div_(255),
            self.torch.tensor([e[1] for e in entries], dtype=self.torch.float32),
        )
