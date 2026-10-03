"""Shared frozen SAC loading for sampling and offline diagnostics."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fh5.driving.control import Command
from fh5.learning.bc.actor import FrozenNumericActor
from fh5.learning.sac.actions import ActionBounds
from fh5.learning.sac.checkpoint import read_checkpoint
from fh5.learning.sac.policy import encode_history, make_policy
from fh5.learning.sac.replay import command_action
from fh5.observation.numeric import NumericDecision, validate_decision


class FrozenSAC:
    """Complete immutable per-attempt policy, distinct from the learner modules."""

    def __init__(self, torch: Any, checkpoint: Path, *, allow_legacy: bool = False) -> None:
        self.torch = torch
        manifest, saved, raw = read_checkpoint(torch, checkpoint)
        if manifest["version"] not in (2, 3, 4) and not allow_legacy:
            raise ValueError("SAC sampling requires a sealed version 2, 3 or 4 checkpoint")
        self.sha = hashlib.sha256(raw).hexdigest()
        self.manifest = manifest
        model_dir = checkpoint / "bc"
        self.bc = FrozenNumericActor(
            model_dir, expected_manifest_sha256=manifest["bc_manifest_sha256"]
        )
        self.pixels = self.bc.contract
        self.bounds = ActionBounds(**manifest["bounds"])
        self.encoder = torch.nn.ModuleDict(
            {"images": deepcopy(self.bc.model.encoder), "state": deepcopy(self.bc.model.state)}
        )
        self.policy = make_policy(
            torch,
            self.bc.model.fusion,
            64 * (self.bc.original_contract["image_count"] + 1),
            manifest["configuration"]["initial_log_std"],
        )
        self.encoder.load_state_dict(saved["encoder"], strict=True)
        self.policy.load_state_dict(saved["policy"], strict=True)
        for module in (self.encoder, self.policy):
            module.eval()
            for parameter in module.parameters():
                if not torch.isfinite(parameter).all():
                    raise ValueError("Non-finite frozen SAC parameter")
                parameter.requires_grad_(False)

    def predict(
        self, decision: NumericDecision, previous: list[float], elapsed: float, noise: list[float]
    ) -> dict[str, Any]:
        reason = validate_decision(decision, self.pixels)
        if reason:
            raise ValueError("Invalid SAC sampler observation: " + reason)
        torch = self.torch
        values = self.bc.input_features(decision.actor, decision.frames)
        images = [
            torch.frombuffer(bytearray(f.pixels), dtype=torch.uint8)
            .reshape(f.size[1], f.size[0], 3)
            .permute(2, 0, 1)
            for f in decision.frames
        ]
        context = self.bounds.context(previous, elapsed)
        with torch.inference_mode():
            features = encode_history(
                torch,
                self.encoder,
                torch.stack(images).unsqueeze(0).float() / 255,
                torch.tensor([values], dtype=torch.float32),
            )
            output = self.policy(features, torch.tensor([context]), features.new_tensor([noise]))
        result = {k: v[0].tolist() for k, v in output.items()}
        steer, longitudinal = result["command"]
        command = Command(
            round(steer * 32767),
            max(0, round(longitudinal * 255)),
            max(0, round(-longitudinal * 255)),
        )
        result.update(
            sent=asdict(command), command=command_action(asdict(command)), context=context
        )
        return result
