"""Explicit read-only shadow entry point; default validation opens no devices."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_drive_config import NumericDriveConfiguration
from fh5.numeric_images import DecisionActor, PixelContract
from fh5.realtime import RealtimeNumericReplay
from fh5.realtime_model import ShadowNumericActor
from fh5.realtime_numeric_replay import read_realtime_recording
from fh5.realtime_shadow import ShadowEnvironment


def replay_command(args: argparse.Namespace) -> int:
    from fh5.experiment import run_experiment
    from fh5.sac_evaluation_actor import SACEvaluationActor

    recording = read_realtime_recording(args.recording)
    actor: DecisionActor
    pixels = PixelContract.from_metadata(recording["configuration"]["pixels"])
    if recording["actor_kind"] == SACEvaluationActor.kind:
        if args.device != "cpu" or args.allow_legacy_source_diagnostic:
            raise ValueError("SAC replay requires CPU and its exact numerical source contract")
        actor = SACEvaluationActor(args.model, pixels, recording["model"]["sac_manifest_sha256"])
    elif recording["actor_kind"] == ShadowNumericActor.kind:
        actor = ShadowNumericActor(
            args.model,
            pixels,
            recording["model"]["weights_sha256"],
            args.device,
            allow_legacy_source_diagnostic=args.allow_legacy_source_diagnostic,
        )
    elif recording["actor_kind"] == "frozen-numeric-temporal-bc-v2":
        if args.allow_legacy_source_diagnostic:
            raise ValueError("Driving replay requires the exact numerical source contract")
        actor = FrozenNumericActor(args.model, pixels, args.device)
    else:
        raise ValueError("CLI replay requires a supported frozen numerical actor")
    result = run_experiment(
        RealtimeNumericReplay(args.recording, args.report, args.tolerance),
        numeric_actor=actor,
    )
    summary = result.summary["realtime_numeric_replay"]
    print(
        json.dumps(
            {
                "verified": summary["verified"],
                "verified_predictions": summary["verified_predictions"],
                "errors": summary["errors"],
                "report": str(result.report_path),
                "commands_sent_to_game": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["verified"] else 4


def shadow_command(args: argparse.Namespace) -> int:
    from fh5.experiment import run_experiment

    configuration = NumericDriveConfiguration(
        args.config,
        args.output,
        args.seconds,
        args.live,
        mode="legacy-shadow" if args.allow_legacy_source_diagnostic else "shadow",
        hz=args.hz,
    )
    request = configuration.request
    if not args.live:
        print(
            json.dumps(
                {
                    "status": "validated_only",
                    "devices_opened": False,
                    "commands_sent_to_game": False,
                    "weights_loading_and_hash_verification": "performed_by_worker_on_live_start",
                    "configuration": asdict(request.config),
                    "bindings": configuration.bindings,
                    "training_pixel_contract": configuration.training_pixels.metadata(),
                    "source_compatibility_validated": False,
                    "legacy_source_diagnostic": args.allow_legacy_source_diagnostic,
                },
                ensure_ascii=False,
            )
        )
        return 0
    from fh5.capture_resources import WindowsResources
    from fh5.dxgi_windows import WindowsDXGIFrames
    from fh5.live import WindowsDesktop

    environment = ShadowEnvironment(
        request,
        configuration.capture,
        lambda: WindowsDXGIFrames(configuration.target),
        configuration.telemetry,
        WindowsDesktop(),
        configuration.task,
        input_conditions=configuration.bindings,
        resources=WindowsResources(),
    )
    print("只读影子运行：模型加载与预热后接收；不会连接手柄或发送输入。F8 停止。", flush=True)
    result = run_experiment(
        request,
        realtime_environment=environment,
        numeric_actor_factory=lambda: ShadowNumericActor(
            configuration.model_dir,
            configuration.capture.pixels,
            configuration.metadata["weights_sha256"],
            configuration.device,
            allow_legacy_source_diagnostic=args.allow_legacy_source_diagnostic,
            expected_manifest_sha256=configuration.model_hash,
        ),
    )
    summary = result.summary["realtime"]
    print(
        json.dumps(
            {
                "report": str(result.report_path),
                "stop_reason": summary["stop_reason"],
                "metrics": summary["metrics"],
                "resources_released": summary["resources_released"],
                "commands_sent_to_game": False,
            },
            ensure_ascii=False,
        )
    )
    return int(
        not summary["resources_released"]
        or not any(d["status"] == "accepted" for d in summary["decisions"])
        or summary["stop_reason"] not in ("time_limit", "local_end", "user_stop")
    )
