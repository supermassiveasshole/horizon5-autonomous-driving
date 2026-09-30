"""Explicit opt-in for passive capture; validation never opens a screen source."""

from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fh5.capture import CaptureConfig, CaptureRun
from fh5.dxgi_capture import DXGISettings
from fh5.numeric_images import PixelContract


class PassiveActivity:
    """Drain telemetry for capture diagnostics, not a synchronized training source."""

    def __init__(self, receiver: socket.socket) -> None:
        self.receiver = receiver
        receiver.setblocking(False)
        self.latest: dict[str, Any] | None = None

    def __call__(self) -> dict[str, Any] | None:
        from fh5.experiment import Packet, _decode

        for _ in range(256):
            try:
                payload, _ = self.receiver.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                self.latest = _decode(
                    Packet(time.perf_counter_ns(), datetime.now(UTC).isoformat(), payload)
                )
            except ValueError:
                continue
        if self.latest is None:
            return None
        age = time.perf_counter_ns() - self.latest["received_monotonic_ns"]
        return {
            **self.latest,
            "fresh": 0 <= age <= 250_000_000,
            "timing_kind": "diagnostic_receive_poll; not frame-physics synchronization",
        }


def capture_command(args: argparse.Namespace) -> int:
    from fh5.experiment import run_experiment

    document = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if (
        set(document) != {"version", "pixels", "pipeline", "target", "input_conditions"}
        or document["version"] != 1
    ):
        raise ValueError("Unsupported capture configuration")
    config = CaptureConfig(
        pixels=PixelContract.from_metadata(document["pixels"]), **document["pipeline"]
    )
    target_doc = dict(document["target"])
    target_doc["expected_client_size"] = tuple(target_doc["expected_client_size"])
    target = DXGISettings(**target_doc)
    conditions = document["input_conditions"]
    if conditions.get("version") != 1 or conditions.get("id") != target.condition_id:
        raise ValueError("Capture input condition identity must match target")
    request = CaptureRun(args.output, config, args.seconds, input_conditions=conditions)
    if not args.live:
        print(json.dumps({"validated": document, "capture_started": False}, ensure_ascii=False))
        return 0
    from fh5.dxgi_windows import WindowsDXGIFrames

    if not 1024 <= args.port <= 65535:
        raise ValueError("Invalid passive telemetry port")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", args.port))
        result = run_experiment(
            request,
            capture_source_factory=lambda: WindowsDXGIFrames(target),
            capture_activity=PassiveActivity(receiver),
        )
    summary = result.summary["capture"]
    print(
        json.dumps(
            {
                "report": str(result.report_path),
                "pipeline": summary["pipeline"],
                "resources_released": summary["resources_released"],
            },
            ensure_ascii=False,
        )
    )
    return int(
        not summary["resources_released"]
        or bool(summary["pipeline"]["fault"])
        or summary["pipeline"].get("new_frames", 0) == 0
    )
