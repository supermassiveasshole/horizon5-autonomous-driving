"""Explicit opt-in for passive capture; validation never opens a screen source."""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fh5.capture.config import parse_capture_config
from fh5.capture.pipeline import CaptureRun
from fh5.capture.runtime import CaptureSource


class PassiveActivity:
    """Drain telemetry for capture diagnostics, not a synchronized training source."""

    def __init__(self, receiver: socket.socket) -> None:
        self.receiver = receiver
        receiver.setblocking(False)
        self.latest: dict[str, Any] | None = None

    def __call__(self) -> dict[str, Any] | None:
        from fh5.telemetry.packet import Packet
        from fh5.telemetry.packet import decode_packet as _decode

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
    config, target = parse_capture_config(document)
    if args.diagnostic_mss == "jpeg":
        if args.raw_samples:
            raise ValueError("Legacy JPEG comparison cannot supply high-resolution raw samples")
        config = replace(
            config,
            pixels=replace(
                config.pixels,
                origin="legacy_offline",
                resize="legacy-jpeg-roundtrip-then-bilinear-diagnostic-v1",
            ),
        )
    conditions = document["input_conditions"]
    request = CaptureRun(
        args.output,
        config,
        args.seconds,
        input_conditions=conditions,
        raw_sample_limit=args.raw_samples,
    )
    if not args.live:
        print(
            json.dumps(
                {
                    "validated": document,
                    "capture_started": False,
                    "diagnostic_mss": args.diagnostic_mss,
                },
                ensure_ascii=False,
            )
        )
        return 0
    from fh5.capture.resources import WindowsResources
    from fh5.capture.windows import WindowsDXGIFrames

    def factory() -> CaptureSource:
        if args.diagnostic_mss == "numeric":
            from fh5.capture.mss_probe import WindowsMSSFrames

            return WindowsMSSFrames(target)
        if args.diagnostic_mss == "jpeg":
            from fh5.capture.mss_probe import WindowsLegacyJPEGFrames

            return WindowsLegacyJPEGFrames(target, args.output)
        return WindowsDXGIFrames(target)

    if not 1024 <= args.port <= 65535:
        raise ValueError("Invalid passive telemetry port")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", args.port))
        result = run_experiment(
            request,
            capture_source_factory=factory,
            capture_activity=PassiveActivity(receiver),
            capture_resources=WindowsResources(),
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
        or summary["execution_error"] is not None
        or summary["stop_reason"] == "interrupted"
        or bool(summary["pipeline"]["fault"])
        or summary["pipeline"].get("new_frames", 0) == 0
    )
