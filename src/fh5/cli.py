"""Command-line adapter for the experiment-run interface."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import socket
import sys
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

from fh5.bc import BCReplay, BCTrain
from fh5.control import Control, validate_control_file
from fh5.demonstration_dataset import DemonstrationDataset
from fh5.demonstrations import DemonstrationRecord, DemonstrationReplay
from fh5.events import EventRun, validate_event_file
from fh5.experiment import Packet, Record, Replay, run_experiment
from fh5.observations import ObservationReplay
from fh5.perception import Perception, PerceptionReplay
from fh5.routes import BuildRoute
from fh5.vision import VisionRecord


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
    bc = commands.add_parser("bc-train", help="Train a bounded offline multimodal BC actor")
    bc.add_argument("--config", type=Path, required=True)
    bc.add_argument("--output", type=Path, required=True)
    bc_replay = commands.add_parser("bc-replay", help="Replay a frozen BC actor; no game input")
    bc_replay.add_argument("--model", type=Path, required=True)
    bc_replay.add_argument("--dataset", type=Path, required=True)
    bc_replay.add_argument("--report", type=Path, required=True)
    bc_replay.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
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
    vision = commands.add_parser(
        "vision", help="Record RGB and telemetry without sending game input"
    )
    commands.add_parser("input-devices", help="List connected XInput logical slots, read-only")
    demo_replay = commands.add_parser(
        "demonstration-replay", help="Verify and inspect raw human inputs"
    )
    demo_replay.add_argument("recording", type=Path)
    demo_replay.add_argument("--report", type=Path, required=True)
    dataset = commands.add_parser(
        "demonstration-dataset", help="Export reviewed train and holdout demonstrations"
    )
    dataset.add_argument("--config", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    vision.add_argument("--config", type=Path, required=True)
    vision.add_argument("--output", type=Path, required=True)
    vision.add_argument(
        "--camera",
        choices=["chase-far"],
        required=True,
        help="Declare the camera mode you have selected in game",
    )
    vision.add_argument("--port", type=int, default=5300)
    vision.add_argument("--seconds", type=float, default=60)
    vision.add_argument("--period", type=float, default=0.1)
    vision.add_argument("--max-age-ms", type=float, default=100)
    vision.add_argument("--max-mib", type=int, default=256)
    vision.add_argument("--observations", type=Path, help="Record passive observation check times")
    vision.add_argument("--route", type=Path, help="Independent frozen navigation reference")
    demo = commands.add_parser(
        "demonstrate",
        parents=[vision],
        add_help=False,
        help="Record physical XInput, RGB and telemetry; no commands sent",
    )
    demo.add_argument("--input-profile", type=Path, required=True)
    observe = commands.add_parser("observe", help="Replay causal RGB history, state and navigation")
    observe.add_argument("recording", type=Path)
    observe.add_argument("--config", type=Path, required=True)
    observe.add_argument("--route", type=Path, help="Actor reference; required by v1/required mode")
    observe.add_argument(
        "--evaluation-route", type=Path, help="V2 diagnostic evidence, never actor input"
    )
    observe.add_argument("--report", type=Path, required=True)
    perceive = commands.add_parser("perceive", help="Estimate pixels in a frozen RGB dataset")
    perceive.add_argument("--dataset", type=Path, required=True)
    perceive.add_argument("--protocol", type=Path, required=True)
    perceive.add_argument("--model-dir", type=Path, required=True)
    perceive.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    perceive.add_argument("--output", type=Path, required=True)
    perception_replay = commands.add_parser(
        "perception-replay", help="Replay pixel candidates and independent errors"
    )
    perception_replay.add_argument("result", type=Path)
    perception_replay.add_argument("--report", type=Path, required=True)
    perception_replay.add_argument("--labels", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.mode == "input-devices":
            from fh5.live_demonstration import input_devices

            print(json.dumps({"devices": input_devices(), "commands_sent": False}))
            return 0
        if args.mode == "bc-train":
            result = run_experiment(BCTrain(args.config, args.output))
        elif args.mode == "bc-replay":
            result = run_experiment(BCReplay(args.model, args.dataset, args.report, args.device))
        elif args.mode == "demonstration-replay":
            result = run_experiment(DemonstrationReplay(args.recording, args.report))
        elif args.mode == "demonstration-dataset":
            result = run_experiment(DemonstrationDataset(args.config, args.output))
        elif args.mode == "observe":
            result = run_experiment(
                ObservationReplay(
                    args.recording, args.report, args.route, args.config, args.evaluation_route
                )
            )
        elif args.mode == "perceive":
            from fh5.segformer import SegformerRoadModel

            model = SegformerRoadModel(args.model_dir, args.protocol, args.device)
            result = run_experiment(
                Perception(args.dataset, args.protocol, args.output), road_model=model
            )
        elif args.mode == "perception-replay":
            result = run_experiment(PerceptionReplay(args.result, args.report, args.labels))
        elif args.mode in ("vision", "demonstrate"):
            request = VisionRecord(
                args.config,
                args.output,
                args.seconds,
                args.period,
                args.max_age_ms,
                args.max_mib * 1024**2,
                args.observations,
                args.route,
            )
            if not 0 <= args.port <= 65535:
                raise ValueError("--port must be between 0 and 65535")
            if args.output.exists():
                raise FileExistsError(f"Output directory already exists: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(args.output.parent).free < request.max_bytes + 128 * 1024**2:
                raise OSError("Insufficient free space for capture budget and report reserve")
            from fh5.live import WindowsDesktop
            from fh5.live_vision import LiveVisionEnvironment, WindowsColorFrames

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
                receiver.bind(("127.0.0.1", args.port))
                desktop = WindowsDesktop()
                visual_environment = LiveVisionEnvironment(
                    receiver,
                    desktop,
                    args.period,
                    frame_factory=lambda: WindowsColorFrames(desktop),
                )
                print(
                    json.dumps(
                        {
                            "status": "passive_visual_recording",
                            "port": args.port,
                            "output": str(args.output),
                            "seconds": args.seconds,
                            "stop_key": "F8",
                            "stop_file": str(args.output / "STOP"),
                        }
                    ),
                    flush=True,
                )
                try:
                    if args.mode == "demonstrate":
                        from fh5.demonstrations import _profile
                        from fh5.live_demonstration import HumanInputEnvironment, XInputReader

                        profile = _profile(args.input_profile.read_bytes())
                        reader = XInputReader(profile["device"]["index"], desktop)
                        human = HumanInputEnvironment(
                            visual_environment, reader, args.input_profile
                        )
                        result = run_experiment(
                            DemonstrationRecord(request, args.input_profile),
                            vision_environment=human,
                        )
                    else:
                        result = run_experiment(request, vision_environment=visual_environment)
                finally:
                    visual_environment.close()
        elif args.mode == "record":
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
    except (OSError, ValueError, ImportError) as error:
        print(json.dumps({"status": "error", "message": str(error)}), file=sys.stderr)
        return 2
    if "bc" in result.summary:
        print(
            json.dumps(
                {
                    "status": "offline_complete",
                    "report": str(result.report_path),
                    "game_validation": "unverified",
                    "commands_sent": False,
                    "weights_sha256": result.summary["bc"]["weights_sha256"],
                }
            )
        )
        return 0
    print(
        json.dumps(
            {
                **{
                    key: value
                    for key, value in result.summary.items()
                    if key
                    not in (
                        "route",
                        "vision",
                        "perception",
                        "observations",
                        "demonstration",
                        "demonstration_dataset",
                    )
                },
                **(
                    {
                        "observations": {
                            key: result.summary["observations"][key]
                            for key in ("version", "clock", "decision_count", "usable_decisions")
                        }
                    }
                    if "observations" in result.summary
                    else {}
                ),
                **(
                    {
                        "perception": {
                            "frames": len(result.summary["perception"]["frames"]),
                            "evaluation": result.summary["perception"]["evaluation"]["status"],
                            "geometry_ready": result.summary["perception"]["geometry_ready"],
                        }
                    }
                    if "perception" in result.summary
                    else {}
                ),
                **(
                    {
                        "vision": {
                            k: v
                            for k, v in result.summary["vision"].items()
                            if k in ("frame_count", "usable_frames", "timing", "integrity_errors")
                        },
                        "vision_stop_reason": result.summary["vision"]["session"]["stop_reason"],
                    }
                    if "vision" in result.summary
                    else {}
                ),
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
    if "perception" in result.summary:
        return 4 if result.summary["perception"]["evaluation"]["invalid_frames"] else 0
    if args.mode in ("vision", "demonstrate"):
        visual = result.summary["vision"]
        if not visual["session"]["resources_released"] or visual["integrity_errors"]:
            return 4
        if visual["session"]["stop_reason"] not in (
            "time_limit",
            "user_stop",
            "stop_file",
            "interrupted",
        ):
            return 4
        if not visual["frame_count"]:
            return 3
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
