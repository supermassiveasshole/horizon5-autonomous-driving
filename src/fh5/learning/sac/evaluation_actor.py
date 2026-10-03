"""Deterministic SAC inference with explicit successful-send context."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from fh5.learning.sac.actor import FrozenSAC
from fh5.learning.sac.context import PROPOSAL_CONTEXT, SEND_CONTEXT, context_action
from fh5.observation.numeric import NumericDecision, NumericFrame, PixelContract

CONTEXT_KIND = SEND_CONTEXT


class SACEvaluationActor:
    kind = "frozen-numeric-sac-v1"

    def __init__(
        self,
        directory: Path,
        pixels: PixelContract,
        expected_sha256: str,
        *,
        counterfactual: bool = False,
    ) -> None:
        self.frozen = FrozenSAC(importlib.import_module("torch"), directory)
        if self.frozen.sha != expected_sha256 or self.frozen.pixels != pixels:
            raise ValueError("Frozen SAC evaluation model changed")
        self.device = self.frozen.bc.device
        self.manifest = {
            "sac_manifest_sha256": self.frozen.sha,
            "numeric_contract": pixels.metadata(),
            "model_contract": self.frozen.bc.original_contract,
            "command_context": PROPOSAL_CONTEXT if counterfactual else CONTEXT_KIND,
            "bounds": self.frozen.manifest["bounds"],
            "exploration": False,
            "noise": [0.0, 0.0],
            "diagnostic_only": True,
        }
        if counterfactual:
            self.manifest["inference_device"] = self.device
        if self.frozen.manifest.get("source_kind") in ("native", "mixed"):
            self.manifest.update(
                source_kind=self.frozen.manifest["source_kind"],
                provenance=self.frozen.bc.manifest["provenance"],
                explicit_dt_model=self.frozen.bc.manifest["explicit_dt_model"],
                diagnostic_only=self.frozen.bc.manifest["diagnostic_only"],
            )

    def input_features(
        self, actor: dict[str, Any], frames: tuple[NumericFrame, ...]
    ) -> list[float]:
        return self.frozen.bc.input_features(actor, frames)

    def predict_decision(
        self, decision: NumericDecision, command_context: dict[str, Any]
    ) -> list[float]:
        previous, at = context_action(
            command_context, self.manifest["command_context"], decision.decision_ns
        )
        elapsed = (decision.decision_ns - at) / 1e9
        return list(
            self.frozen.predict(decision, previous, elapsed, self._noise(decision))["command"]
        )

    def _noise(self, decision: NumericDecision) -> list[float]:
        return [0.0, 0.0]
