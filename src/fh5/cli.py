"""Command-line adapter for the experiment-run interface."""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

from fh5.control import Control, validate_control_file
from fh5.events import EventRun, validate_event_file
from fh5.experiment import Packet, Record, Replay, run_experiment
from fh5.routes import BuildRoute


def _udp_packets(receiver: socket.socket, seconds: float) -> Iterator[Packet]:
    deadline = time.monotonic() + seconds
    while (remaining := deadline - time.monotonic()) > 0:
        receiver.settimeout(min(0.2, remaining))
        try:
            payload, _ = receiver.recvfrom(65535)
        except TimeoutError:
            continue
        yield Packet(time.perf_counter_ns(), datetime.now(UTC).isoformat(), payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record and replay FH5 Data Out experiments")
    commands = parser.add_subparsers(dest="mode", required=True)
    record = commands.add_parser("record", help="Receive UDP; stop on Ctrl+C or the time limit")
    record.add_argument("--config", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--bind", default="127.0.0.1")
    record.add_argument("--port", type=int, default=5300)
    record.add_argument("--seconds", type=float, default=60.0)
    replay = commands.add_parser("replay", help="Replay a saved recording without the game")
    replay.add_argument("recording", type=Path)
    replay.add_argument("--report", type=Path, required=True)
    replay.add_argument("--route", type=Path, help="Overlay a frozen local route bundle")
    route = commands.add_parser(
        "route", help="Build a local reference from a continuous recording span"
    )
    route.add_argument("recording", type=Path)
    route.add_argument("--output", type=Path, required=True)
    route.add_argument("--first-packet", type=int, required=True)
    route.add_argument("--last-packet", type=int, required=True)
    route.add_argument("--spacing", type=float, default=2.0)
    route.add_argument("--annotations", type=Path)
    control = commands.add_parser(
        "control", help="Validate a bounded calibration; --live sends game input"
    )
    control.add_argument("--config", type=Path, required=True)
    control.add_argument("--output", type=Path, required=True)
    control.add_argument("--port", type=int, default=5300)
    control.add_argument(
        "--live", action="store_true", help="Send actual input; F8 or Ctrl+C releases it"
    )
    event = commands.add_parser("event", help="Validate event recipes; --live enables menu pulses")
    event.add_argument("--config", type=Path, required=True)
    event.add_argument("--output", type=Path, required=True)
    event.add_argument("--port", type=int, default=5300)
    event.add_argument("--live", action="store_true", help="Run with FH5 foreground; F8 stops")
    args = parser.parse_args(argv)
    try:
        if args.mode == "record":
            if not math.isfinite(args.seconds) or args.seconds <= 0:
                raise ValueError("--seconds must be finite and greater than zero")
            if not 0 <= args.port <= 65535:
                raise ValueError("--port must be between 0 and 65535")
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
                receiver.bind((args.bind, args.port))
                print(
                    json.dumps(
                        {
                            "status": "listening",
                            "bind": args.bind,
                            "port": receiver.getsockname()[1],
                        }
                    ),
                    flush=True,
                )
                result = run_experiment(
                    Record(args.config, args.output, source_kind="udp"),
                    packets=_udp_packets(receiver, args.seconds),
                )
        elif args.mode == "control":
            config = validate_control_file(args.config)
            if not args.live:
                print(
                    json.dumps(
                        {
                            "status": "validated_only",
                            "duration_s": sum(step["seconds"] for step in config["steps"]),
                        }
                    )
                )
                return 0
            if not 0 <= args.port <= 65535:
                raise ValueError("--port must be between 0 and 65535")
            if args.output.exists():
                raise FileExistsError(f"Output directory already exists: {args.output}")
            from fh5.live import LiveEnvironment, WindowsDesktop, XboxController

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
                receiver.bind(("127.0.0.1", args.port))
                desktop = WindowsDesktop()
                environment = LiveEnvironment(receiver, XboxController(), desktop)
                print(
                    json.dumps(
                        {
                            "status": "waiting_for_focused_game_and_stationary_car",
                            "port": receiver.getsockname()[1],
                            "stop_key": "F8",
                        }
                    ),
                    flush=True,
                )
                result = run_experiment(Control(args.config, args.output), environment=environment)
        elif args.mode == "event":
            root = validate_event_file(args.config)
            event_config = root["event_run"]
            if not args.live:
                print(
                    json.dumps(
                        {
                            "status": "validated_only",
                            "purpose": event_config["purpose"],
                            "conditions_verified": event_config["conditions_verified"],
                        }
                    )
                )
                return 0
            if not 0 <= args.port <= 65535:
                raise ValueError("--port must be between 0 and 65535")
            if args.output.exists():
                raise FileExistsError(f"Output directory already exists: {args.output}")
            from fh5.live import WindowsDesktop, XboxController
            from fh5.live_event import BoundedFrames, LiveEventEnvironment, WindowsFrames

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
                receiver.bind(("127.0.0.1", args.port))
                desktop = WindowsDesktop()
                frames = BoundedFrames(
                    lambda: WindowsFrames(desktop, tuple(event_config["screen_size"]))
                )
                try:
                    controller = XboxController()
                except BaseException:
                    frames.close()
                    raise
                environment_event = LiveEventEnvironment(receiver, controller, desktop, frames)
                try:
                    print(
                        json.dumps(
                            {
                                "status": "event_started",
                                "stop_key": "F8",
                                "purpose": event_config["purpose"],
                            }
                        ),
                        flush=True,
                    )
                    result = run_experiment(
                        EventRun(args.config, args.output), event_environment=environment_event
                    )
                finally:
                    environment_event.close()
        elif args.mode == "route":
            result = run_experiment(
                BuildRoute(
                    args.recording,
                    args.output,
                    args.first_packet,
                    args.last_packet,
                    args.spacing,
                    args.annotations,
                )
            )
        else:
            result = run_experiment(Replay(args.recording, args.report, args.route))
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "error", "message": str(error)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                **{key: value for key, value in result.summary.items() if key != "route"},
                **(
                    {
                        "route": {
                            key: result.summary["route"][key]
                            for key in (
                                "length_m",
                                "reference_points",
                                "verified_corridor_length_m",
                                "low_speed_ready",
                            )
                        }
                    }
                    if "route" in result.summary
                    else {}
                ),
                "report": str(result.report_path),
                "game_validation": result.metadata["game_validation"],
            }
        )
    )
    if result.summary["capture_status"] == "source_error":
        return 2
    if args.mode == "control" and result.summary["control"]["stop_reason"] != "completed":
        return 4
    if args.mode == "event" and (
        result.summary["event_run"]["stop_reason"] != "attempt_limit"
        or not result.summary["event_run"]["release_sent"]
    ):
        return 4
    if result.summary["capture_status"] == "interrupted" and args.mode == "record":
        return 130
    if result.summary["valid_packets"] == 0:
        return 3
    return 0
