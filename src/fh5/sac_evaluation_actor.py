"""Deterministic SAC inference with explicit successful-send context."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from fh5.numeric_images import NumericDecision, NumericFrame, PixelContract
from fh5.sac_actor import FrozenSAC
from fh5.sac_replay import command_action

CONTEXT_KIND = "successful-send-return-proxy-v1"


class SACEvaluationActor:
    kind = "frozen-numeric-sac-v1"

    def __init__(self, directory: Path, pixels: PixelContract, expected_sha256: str) -> None:
        self.frozen = FrozenSAC(importlib.import_module("torch"), directory)
        if self.frozen.sha != expected_sha256 or self.frozen.pixels != pixels:
            raise ValueError("Frozen SAC evaluation model changed")
        self.manifest = {
            "sac_manifest_sha256": self.frozen.sha,
            "numeric_contract": pixels.metadata(),
            "model_contract": self.frozen.bc.original_contract,
            "command_context": CONTEXT_KIND,
            "bounds": self.frozen.manifest["bounds"],
            "exploration": False,
            "noise": [0.0, 0.0],
            "diagnostic_only": True,
        }

    def input_features(
        self, actor: dict[str, Any], frames: tuple[NumericFrame, ...]
    ) -> list[float]:
        return self.frozen.bc.input_features(actor, frames)

    def predict_decision(
        self, decision: NumericDecision, command_context: dict[str, Any]
    ) -> list[float]:
        context = command_context
        if (
            set(context)
            != {"version", "command_index", "issued_ns", "returned_ns", "sent", "owner"}
            or context["version"] != 1
            or type(context["command_index"]) is not int
            or context["command_index"] < 0
            or context["owner"] not in ("initial_neutral", "policy", "lease_expiry", "warmup")
            or type(context["issued_ns"]) is not int
            or type(context["returned_ns"]) is not int
            or not 0 <= context["issued_ns"] <= context["returned_ns"] < decision.decision_ns
        ):
            raise ValueError("Invalid successful command context")
        previous = command_action(context["sent"])
        elapsed = (decision.decision_ns - context["returned_ns"]) / 1e9
        return list(self.frozen.predict(decision, previous, elapsed, [0.0, 0.0])["command"])
