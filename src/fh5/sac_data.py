"""Raw numerical SAC replay; trainable representations are never cached."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fh5.collection_store import read_bounded
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import NumericDecision, validate_decision
from fh5.numeric_recording import read_numeric_frame
from fh5.sac import _task_features
from fh5.sac_actions import ActionBounds
from fh5.sac_sources import replay_roles
from fh5.sac_timing import next_action_elapsed


def validate_cache_budget(value: int) -> None:
    if type(value) is not int or not 1 <= value <= 512 * 1024**2:
        raise ValueError("SAC raw frame cache budget must be 1..512 MiB in bytes")


class LearningReplay:
    def __init__(
        self,
        torch: Any,
        path: Path,
        expected: str,
        actor: FrozenNumericActor,
        bounds: ActionBounds,
        *,
        cache_bytes: int = 512 * 1024**2,
    ) -> None:
        validate_cache_budget(cache_bytes)
        self.cache_bytes, self.cache_hits = cache_bytes, 0
        self.cache_misses = self.cache_evictions = self.peak_bytes = 0
        self.root = path.parent
        self.torch, self.actor, self.bounds = torch, actor, bounds
        self.raw = read_bounded(path, 128 * 1024**2)
        if hashlib.sha256(self.raw).hexdigest() != expected:
            raise ValueError("SAC replay changed from its frozen digest")
        replay = json.loads(self.raw)
        self.roles = replay_roles(path.parent, replay)
        if replay["pixel_contract"] != actor.contract.metadata():
            raise ValueError("Unsupported SAC learning replay contract")
        self.rows = replay["transitions"]
        if not 1 <= len(self.rows) <= 10_000:
            raise ValueError("SAC replay must contain 1..10000 transitions")
        self.images: OrderedDict[str, Any] = OrderedDict()
        self.sources: dict[str, dict[str, Any]] = {}
        self.observations: dict[str, tuple[list[str], list[float]]] = {}
        self.current: list[str] = []
        self.following: list[str] = []
        self.raw_bytes = self.source_bytes = 0

        def observation(row: dict[str, Any]) -> str:
            identity = json.dumps(row, sort_keys=True)
            if identity in self.observations:
                return identity
            frames = tuple(read_numeric_frame(path.parent, f) for f in row["frames"])
            decision = NumericDecision(
                row["decision_id"], row["epoch"], row["decision_ns"], frames, row["actor"]
            )
            reason = validate_decision(decision, actor.contract)
            if reason:
                raise ValueError("Invalid SAC observation: " + reason)
            numeric = actor.input_features(row["actor"], frames)
            references = []
            for frame, entry in zip(frames, row["frames"]):
                if frame.pixels.nbytes > self.cache_bytes:
                    raise ValueError("SAC raw frame exceeds its cache byte budget")
                digest = entry["sha256"]
                if digest not in self.sources:
                    self.source_bytes += frame.pixels.nbytes
                self.sources.setdefault(digest, dict(entry))
                references.append(digest)
            self.observations[identity] = references, numeric
            return identity

        contexts, next_contexts, tasks, next_tasks, discounts = [], [], [], [], []
        for row in self.rows:
            self.current.append(observation(row["current"]))
            context = bounds.context(row["previous_action"], row["action_elapsed_s"])
            if len(row["action"]) != 2 or any(
                not lo - 1e-9 <= value <= hi + 1e-9
                for lo, hi, value in zip(context[3:5], context[5:7], row["action"])
            ):
                raise ValueError("SAC replay action outside executable command support")
            contexts.append(context)
            tasks.append(_task_features(row["task_state"], replay["task_context"]))
            if row["bootstrap"]:
                if row["next"] is None or row["terminated"]:
                    raise ValueError("SAC bootstrap requires a real nonterminal final observation")
                self.following.append(observation(row["next"]))
                next_contexts.append(bounds.context(row["action"], next_action_elapsed(row)))
                next_tasks.append(_task_features(row["next_task_state"], replay["task_context"]))
                discounts.append(row["discount"])
            else:
                if not row["terminated"]:
                    raise ValueError("Missing bootstrap is not a true SAC terminal")
                self.following.append(self.current[-1])
                next_contexts.append(context)
                next_tasks.append(tasks[-1])
                discounts.append(0.0)
        self.context = torch.tensor(contexts, dtype=torch.float32)
        self.next_context = torch.tensor(next_contexts, dtype=torch.float32)
        self.task = torch.tensor(tasks, dtype=torch.float32)
        self.next_task = torch.tensor(next_tasks, dtype=torch.float32)
        self.actions = torch.tensor([r["action"] for r in self.rows], dtype=torch.float32)
        self.rewards = torch.tensor([r["reward"] for r in self.rows], dtype=torch.float32)
        self.discounts = torch.tensor(discounts, dtype=torch.float32)
        if not all(torch.isfinite(t).all() for t in (self.actions, self.rewards, self.discounts)):
            raise ValueError("Non-finite SAC training data")

    def cache_summary(self) -> dict[str, Any]:
        return {
            "budget_bytes": self.cache_bytes,
            "peak_bytes": self.peak_bytes,
            "retained_bytes": self.raw_bytes,
            "retained_frames": len(self.images),
            "unique_frames": len(self.sources),
            "source_bytes": self.source_bytes,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "evictions": self.cache_evictions,
            "storage_dtype": "uint8",
            "files_deleted": 0,
        }

    def _image(self, digest: str) -> Any:
        if digest in self.images:
            self.cache_hits += 1
            self.images.move_to_end(digest)
            return self.images[digest]
        self.cache_misses += 1
        frame = read_numeric_frame(self.root, self.sources[digest], self.cache_bytes)
        size = frame.pixels.nbytes
        while self.raw_bytes + size > self.cache_bytes:
            _, old = self.images.popitem(last=False)
            self.raw_bytes -= old.numel() * old.element_size()
            self.cache_evictions += 1
            del old
        width, height = frame.size
        value = (
            self.torch.frombuffer(bytearray(frame.pixels), dtype=self.torch.uint8)
            .reshape(height, width, 3)
            .permute(2, 0, 1)
        )
        self.images[digest] = value
        self.raw_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.raw_bytes)
        return value

    def inputs(self, indices: list[int], *, following: bool = False) -> tuple[Any, Any]:
        width, height = self.actor.contract.size
        history = self.actor.original_contract["image_count"]
        if len(indices) * history * width * height * 3 * 4 > 256 * 1024**2:
            raise ValueError("SAC image batch exceeds 256 MiB; reduce batch size")
        keys = self.following if following else self.current
        entries = [self.observations[keys[i]] for i in indices]
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
