"""Command-coordinate support shared by critic targets and subsequent SAC sampling."""

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ActionBounds:
    max_steer: float = 0.5
    max_throttle: float = 0.35
    max_brake: float = 0.4
    steer_rate: float = 4.0
    longitudinal_rate: float = 4.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not 0 < value <= (100 if "rate" in name else 1):
                raise ValueError("Invalid SAC action bounds: " + name)

    def interval(self, previous: list[float], elapsed_s: float) -> tuple[list[float], list[float]]:
        if len(previous) != 2 or any(not -1 <= v <= 1 for v in previous) or not 0 < elapsed_s <= 60:
            raise ValueError(
                "SAC action interval needs previous sent command and positive actual time"
            )
        lows, highs = [-self.max_steer, -self.max_brake], [self.max_steer, self.max_throttle]
        for i, rate in enumerate((self.steer_rate, self.longitudinal_rate)):
            lows[i] = max(lows[i], previous[i] - rate * elapsed_s)
            highs[i] = min(highs[i], previous[i] + rate * elapsed_s)
        if any(hi - lo <= 1e-8 for lo, hi in zip(lows, highs)):
            raise ValueError("Degenerate SAC support requires a supervisor boundary")
        return lows, highs

    def context(self, previous: list[float], elapsed_s: float) -> list[float]:
        lo, hi = self.interval(previous, elapsed_s)
        return [*previous, elapsed_s, *lo, *hi]

    def deterministic(
        self, prediction: Any, previous: list[float], elapsed_s: float
    ) -> list[float]:
        lo, hi = self.interval(previous, elapsed_s)
        command = []
        for a, b, value, scale in zip(lo, hi, prediction, (32767, 255)):
            lower, upper = math.ceil(a * scale), math.floor(b * scale)
            if lower > upper:
                raise ValueError("SAC support contains no executable command")
            command.append(max(lower, min(upper, round(float(value) * scale))) / scale)
        return command
