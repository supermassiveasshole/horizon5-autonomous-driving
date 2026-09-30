"""Raw numerical SAC replay; trainable representations are never cached."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fh5.collection_store import read_bounded
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import NumericDecision, validate_decision
from fh5.numeric_recording import read_numeric_frame
from fh5.sac import _task_features
from fh5.sac_actions import ActionBounds
from fh5.sac_checkpoint import source_replays


class LearningReplay:
    def __init__(
        self, torch: Any, path: Path, expected: str, actor: FrozenNumericActor, bounds: ActionBounds
    ) -> None:
        self.torch, self.actor, self.bounds = torch, actor, bounds
        self.raw = read_bounded(path, 128 * 1024**2)
        if hashlib.sha256(self.raw).hexdigest() != expected:
            raise ValueError("SAC replay changed from its frozen digest")
        replay = json.loads(self.raw)
        source_replays(path.parent, replay)
        if (replay.get("version"), replay.get("kind"), replay.get("source_kind")) != (
            1,
            "sac-numeric-replay-v1",
            "synthetic",
        ) or replay["pixel_contract"] != actor.contract.metadata():
            raise ValueError("Unsupported SAC learning replay contract")
        self.rows = replay["transitions"]
        if not 1 <= len(self.rows) <= 10_000:
            raise ValueError("SAC replay must contain 1..10000 transitions")
        self.images: dict[str, Any] = {}
        self.observations: dict[str, tuple[list[Any], list[float]]] = {}
        self.current: list[str] = []
        self.following: list[str] = []
        self.raw_bytes = 0

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
            tensors = []
            for frame in frames:
                digest = hashlib.sha256(frame.pixels).hexdigest()
                if digest not in self.images:
                    self.raw_bytes += frame.pixels.nbytes
                    if self.raw_bytes > 512 * 1024**2:
                        raise ValueError("SAC raw image cache exceeds 512 MiB")
                    width, height = frame.size
                    self.images[digest] = (
                        torch.frombuffer(bytearray(frame.pixels), dtype=torch.uint8)
                        .reshape(height, width, 3)
                        .permute(2, 0, 1)
                    )
                tensors.append(self.images[digest])
            self.observations[identity] = tensors, numeric
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
                next_contexts.append(bounds.context(row["action"], row["hold_dt_s"]))
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

    def inputs(self, indices: list[int], *, following: bool = False) -> tuple[Any, Any]:
        width, height = self.actor.contract.size
        history = self.actor.original_contract["image_count"]
        if len(indices) * history * width * height * 3 * 4 > 256 * 1024**2:
            raise ValueError("SAC image batch exceeds 256 MiB; reduce batch size")
        keys = self.following if following else self.current
        entries = [self.observations[keys[i]] for i in indices]
        return (
            self.torch.stack([self.torch.stack(e[0]) for e in entries]).float() / 255,
            self.torch.tensor([e[1] for e in entries], dtype=self.torch.float32),
        )
