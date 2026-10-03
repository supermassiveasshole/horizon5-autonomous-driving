"""Frozen stochastic SAC on the asynchronous numerical decision runner."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from fh5.numeric_images import NumericDecision, PixelContract
from fh5.sac_evaluation_actor import SACEvaluationActor


class SACSamplingActor(SACEvaluationActor):
    """Decision-keyed exploration; no mutable RNG state on the inference path."""

    kind = "frozen-numeric-sac-sampling-v1"

    def __init__(
        self,
        directory: Path,
        pixels: PixelContract,
        expected_sha256: str,
        *,
        exploration_seed: int,
        counterfactual: bool = False,
    ) -> None:
        if type(exploration_seed) is not int or not 0 <= exploration_seed < 2**64:
            raise ValueError("SAC exploration seed must be an unsigned 64-bit integer")
        self._exploration_seed = exploration_seed
        super().__init__(directory, pixels, expected_sha256, counterfactual=counterfactual)
        self.device = self.frozen.bc.device
        self.manifest.update(
            inference_device=self.device,
            exploration=True,
            noise={
                "scheme": "sha256-box-muller-decision-v1",
                "seed": exploration_seed,
                "key": ["epoch", "decision_id", "decision_ns"],
            },
        )

    def _noise(self, decision: NumericDecision) -> list[float]:
        # Use open-interval uniforms with 52 bits, avoiding log(0) or a rounded
        # endpoint. Recorded decision identity makes warmup/skips/order irrelevant.
        key = json.dumps(
            [
                "sha256-box-muller-decision-v1",
                self._exploration_seed,
                decision.epoch,
                decision.decision_id,
                decision.decision_ns,
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest = hashlib.sha256(key).digest()
        u1 = ((int.from_bytes(digest[:8], "big") >> 12) + 0.5) / 2**52
        u2 = ((int.from_bytes(digest[8:16], "big") >> 12) + 0.5) / 2**52
        radius = math.sqrt(-2 * math.log(u1))
        angle = 2 * math.pi * u2
        return [radius * math.cos(angle), radius * math.sin(angle)]
