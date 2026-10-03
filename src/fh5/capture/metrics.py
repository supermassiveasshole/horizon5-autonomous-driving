"""Bounded diagnostics; a desktop presentation interval is not game frame time."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any


def percentiles(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)

    def percentile(p: float) -> float | None:
        if not ordered:
            return None
        index = (len(ordered) - 1) * p
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return {
        "count": len(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(values) if values else None,
    }


class SourceCadence:
    def __init__(self, capture_hz: int) -> None:
        self.gap_ms = 1500 / capture_hz
        self.counts: Counter[str] = Counter()
        self.intervals: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=72_000))
        self.previous: tuple[str, int] | None = None

    def observe(self, quality: str, source_ns: int) -> None:
        self.counts[quality] += 1
        if self.previous is not None and self.previous[0] == quality:
            self.intervals[quality].append((source_ns - self.previous[1]) / 1e6)
        self.previous = (quality, source_ns)

    def report(self) -> dict[str, Any]:
        result = {}
        for quality, count in self.counts.items():
            intervals = list(self.intervals[quality])
            result[quality] = {
                "accepted_frames": count,
                "interval_ms": percentiles(intervals),
                "within_epoch_rate_hz": (
                    len(intervals) * 1000 / sum(intervals) if intervals else None
                ),
                "long_gap_threshold_ms": self.gap_ms,
                "long_gap_count": sum(value > self.gap_ms for value in intervals),
                "meaning": (
                    "unique acquired desktop presentations; not game rendering FPS"
                    if quality == "dxgi_qpc"
                    else "accepted samples in this time convention; not native new-frame FPS"
                ),
            }
        return result


def game_frame_time_unavailable() -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": "No independently captured FH5 presentation trace; desktop intervals are not game frame times",
    }
