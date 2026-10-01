"""Explicit read-only shadow entry point; default validation opens no devices."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from typing import Any

from fh5.capture_config import parse_capture_config
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import DecisionActor, PixelContract
from fh5.realtime import RealtimeConfig, RealtimeNumericReplay, RealtimeRun
from fh5.realtime_model import ShadowNumericActor, shadow_model_contract
from fh5.realtime_numeric_replay import read_realtime_recording
from fh5.realtime_shadow import LocalTask, ShadowEnvironment
from fh5.realtime_udp import UDPTelemetry


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

    raw = args.config.read_bytes()
    root = json.loads(raw)
    if (
        set(root) != {"version", "capture_config", "model", "task", "decision", "port"}
        or root["version"] != 1
    ):
        raise ValueError("Unsupported read-only shadow configuration")
    capture_bytes = (args.config.parent / root["capture_config"]).read_bytes()
    document = json.loads(capture_bytes)
    capture, target = parse_capture_config(document)
    pixels = capture.pixels
    if pixels.origin != "direct_numeric":
        raise ValueError("Shadow DXGI capture requires direct numerical origin")
    conditions = document["input_conditions"]
    task_options = dict(root["task"])
    task_options["route_file"] = args.config.parent / task_options["route_file"]
    task = LocalTask(**task_options)
    task.load()
    settings = dict(root["decision"])
    if "action_offsets_ms" in settings:
        settings["action_offsets_ms"] = tuple(settings["action_offsets_ms"])
    if args.hz is not None:
        settings["decision_hz"] = args.hz
    request = RealtimeRun(
        args.output, RealtimeConfig(pixels=pixels, **settings), seconds=args.seconds
    )
    model = root["model"]
    if set(model) != {"directory", "expected_sha256", "device"} or model["device"] not in (
        "cpu",
        "cuda",
    ):
        raise ValueError("Invalid shadow model configuration")
    model_dir = args.config.parent / model["directory"]
    metadata, trained = shadow_model_contract(
        model_dir, pixels, model["expected_sha256"], args.allow_legacy_source_diagnostic
    )
    if metadata["contract"]["actor_shape"] != {
        "action_count": len(request.config.action_offsets_ms),
        "reference_count": request.config.reference_count,
    }:
        raise ValueError("Shadow actor feature counts differ from frozen model")
    telemetry = UDPTelemetry(root["port"])
    bindings: dict[str, Any] = {
        "shadow_config_sha256": hashlib.sha256(raw).hexdigest(),
        "capture_config_sha256": hashlib.sha256(capture_bytes).hexdigest(),
        "capture_target": asdict(target),
        "conditions": conditions,
    }
    if not args.live:
        print(
            json.dumps(
                {
                    "status": "validated_only",
                    "devices_opened": False,
                    "commands_sent_to_game": False,
                    "weights_loading_and_hash_verification": "performed_by_worker_on_live_start",
                    "configuration": asdict(request.config),
                    "bindings": bindings,
                    "training_pixel_contract": trained.metadata(),
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
        capture,
        lambda: WindowsDXGIFrames(target),
        telemetry,
        WindowsDesktop(),
        task,
        input_conditions=bindings,
        resources=WindowsResources(),
    )
    print("只读影子运行：模型加载与预热后接收；不会连接手柄或发送输入。F8 停止。", flush=True)
    result = run_experiment(
        request,
        realtime_environment=environment,
        numeric_actor_factory=lambda: ShadowNumericActor(
            model_dir,
            pixels,
            model["expected_sha256"],
            model["device"],
            allow_legacy_source_diagnostic=args.allow_legacy_source_diagnostic,
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
