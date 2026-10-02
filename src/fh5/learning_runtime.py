"""Restore caller-owned Torch state after isolated offline learning experiments."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol


class TrainingStopped(Exception):
    """A cooperative stop raised only at a complete work-unit boundary."""


class TrainingBudget(Protocol):
    def checkpoint(
        self,
        phase: str,
        completed: int,
        suspend: Callable[[], None] | None = None,
        resume: Callable[[], None] | None = None,
    ) -> None: ...


def move_learning_state(torch: Any, models: list[Any], optimizer: Any, device: str) -> None:
    """Move live parameter/optimizer storage at a completed work-unit boundary."""
    for model in models:
        model.zero_grad(set_to_none=True)
        model.to(device)
    if optimizer is not None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    # Adam's scalar step stays on CPU for its non-capturable path.
                    state[key] = value.to("cpu" if key == "step" else device)
    if device == "cpu" and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


@contextmanager
def preserve_torch_state(torch: Any) -> Iterator[None]:
    prior = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    try:
        yield
    finally:
        torch.set_num_threads(prior[0])
        torch.use_deterministic_algorithms(prior[1], warn_only=prior[2])
        torch.backends.cudnn.benchmark = prior[3]
        torch.set_rng_state(prior[4])
        if prior[5] is not None:
            torch.cuda.set_rng_state_all(prior[5])
        if prior[6] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prior[6]
