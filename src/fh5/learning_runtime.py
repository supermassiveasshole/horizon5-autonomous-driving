"""Restore caller-owned Torch state after isolated offline learning experiments."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


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
