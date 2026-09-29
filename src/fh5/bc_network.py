"""Small trainable RGB-history actor, shared by offline BC and later control adapters."""

from __future__ import annotations

import importlib
import math
from typing import Any


def make_actor(contract: dict[str, Any]) -> Any:
    torch = importlib.import_module("torch")
    nn = torch.nn
    width, height = contract["image_size"]
    fw, fh = math.ceil(width / 16), math.ceil(height / 16)
    pool_w, pool_h = math.ceil(fw / 5), math.ceil(fh / 3)

    class Actor(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 24, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(24, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 48, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.ZeroPad2d((0, pool_w * 5 - fw, 0, pool_h * 3 - fh)),
                nn.AvgPool2d((pool_h, pool_w)),
                nn.Flatten(),
                nn.Linear(48 * 3 * 5, 64),
                nn.ReLU(),
            )
            self.state = nn.Sequential(nn.Linear(contract["numeric_size"], 64), nn.ReLU())
            self.fusion = nn.Sequential(
                nn.Linear(64 * contract["image_count"] + 64, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Tanh(),
            )

        def forward(self, rgb: Any, state: Any) -> Any:
            batch, history = rgb.shape[:2]
            features = self.encoder(rgb.flatten(0, 1)).reshape(batch, history * 64)
            return self.fusion(torch.cat([features, self.state(state)], dim=1))

    return Actor()
