"""BC-preserving conditional Gaussian policy in executable command coordinates."""

import math
from copy import deepcopy
from typing import Any


def command_values(torch: Any, latent: Any, context: Any) -> dict[str, Any]:
    """Map a latent action to the executable grid and its declared gradient surrogate."""
    context = context.to(dtype=latent.dtype)
    lower, upper = context[:, 3:5], context[:, 5:7]
    continuous = (lower + upper) / 2 + (upper - lower) / 2 * torch.tanh(latent)
    grid = latent.new_tensor([32767, 255])
    rounded = torch.round(continuous * grid) / grid
    return {
        "continuous": continuous,
        "rounded": rounded,
        "straight_through": continuous + (rounded - continuous).detach(),
    }


def make_policy(torch: Any, fusion: Any, feature_width: int, initial_log_std: float) -> Any:
    nn = torch.nn

    class Policy(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.fusion = deepcopy(fusion)
            self.context_mean = nn.Linear(7, 2)
            self.log_std = nn.Linear(feature_width + 7, 2)
            nn.init.zeros_(self.context_mean.weight)
            nn.init.zeros_(self.context_mean.bias)
            nn.init.zeros_(self.log_std.weight)
            nn.init.constant_(self.log_std.bias, initial_log_std)

        def forward(self, features: Any, context: Any, noise: Any = None) -> dict[str, Any]:
            # Preserve the sender's integer command at half-cell boundaries:
            # float32 inverse-tanh round trips and grid multiplication can each
            # move an unchanged BC prediction into the adjacent command cell.
            action_context = context.to(dtype=torch.float64)
            lower, upper = action_context[:, 3:5], action_context[:, 5:7]
            center, scale = (lower + upper) / 2, (upper - lower) / 2
            if not bool((scale > 0).all()):
                raise ValueError("SAC policy requires nondegenerate support")
            desired = self.fusion(features).to(dtype=torch.float64)
            normalized = (desired.clamp(lower, upper) - center) / scale
            # A quarter command cell keeps the finite inverse tanh in the same
            # rounded command cell as BC even at a saturated bound.
            grid = desired.new_tensor([32767, 255])
            margin = (0.25 / (grid * scale)).clamp(min=1e-6, max=0.25)
            mean = torch.atanh(normalized.clamp(-1 + margin, 1 - margin))
            mean = mean + self.context_mean(context)
            log_std = self.log_std(torch.cat([features, context], dim=1)).clamp(-5, -1)
            sigma = log_std.exp()
            epsilon = torch.randn_like(log_std) if noise is None else noise.expand_as(mean)
            latent = mean + sigma * epsilon
            # Forward Q values always see executable commands. Backprop uses
            # the explicitly declared straight-through quantization surrogate.
            values = command_values(torch, latent, action_context)
            normal_logp = -0.5 * (epsilon.square() + math.log(2 * math.pi)) - log_std
            log_tanh = 2 * (math.log(2) - latent - torch.nn.functional.softplus(-2 * latent))
            log_probability = (normal_logp - log_tanh - scale.log()).sum(dim=1)
            deterministic = command_values(torch, mean, action_context)["rounded"]
            return {
                "command": values["straight_through"].to(dtype=features.dtype),
                "continuous": values["continuous"].to(dtype=features.dtype),
                "deterministic": deterministic.to(dtype=features.dtype),
                # Guidance recomputes commands from this mean; retain the
                # coordinate precision while network inputs stay float32.
                "mean": mean,
                "log_std": log_std,
                "latent": latent,
                "log_probability": log_probability.to(dtype=features.dtype),
                "log_scale": scale.log().sum(dim=1).to(dtype=features.dtype),
            }

    return Policy()


def encode_history(torch: Any, encoder: Any, rgb: Any, state: Any) -> Any:
    batch, history = rgb.shape[:2]
    features = encoder["images"](rgb.flatten(0, 1)).reshape(batch, history * 64)
    return torch.cat([features, encoder["state"](state)], dim=1)


def soft_update(torch: Any, target: Any, source: Any, tau: float) -> None:
    with torch.no_grad():
        for destination, current in zip(target.parameters(), source.parameters()):
            destination.lerp_(current, tau)
        for destination, current in zip(target.buffers(), source.buffers()):
            if destination.is_floating_point():
                destination.lerp_(current, tau)
            else:
                destination.copy_(current)
