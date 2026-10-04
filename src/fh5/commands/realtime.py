"""Explicit read-only shadow entry point; default validation opens no devices."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from fh5.driving.config import NumericDriveConfiguration
from fh5.driving.realtime.model import RealtimeNumericReplay
from fh5.driving.realtime.numeric_replay import read_realtime_recording
from fh5.driving.realtime.observation import ShadowNumericActor
from fh5.driving.realtime.shadow import ShadowEnvironment
from fh5.learning.bc.actor import FrozenNumericActor
from fh5.learning.sac.context import PROPOSAL_CONTEXT, SEND_CONTEXT
from fh5.observation.numeric import DecisionActor, PixelContract


def replay_command(args: argparse.Namespace) -> int:
    from fh5.experiment import run_experiment
    from fh5.learning.sac.evaluation_actor import SACEvaluationActor
    from fh5.learning.sac.sampling_actor import SACSamplingActor

    recording = read_realtime_recording(args.recording)
    actor: DecisionActor
    pixels = PixelContract.from_metadata(recording["configuration"]["pixels"])
    if recording["actor_kind"] in (SACEvaluationActor.kind, SACSamplingActor.kind):
        if args.device != "cpu" or args.allow_legacy_source_diagnostic:
            raise ValueError("SAC replay requires CPU and its exact numerical source contract")
        context_kind = recording["model"]["command_context"]
        if context_kind not in (SEND_CONTEXT, PROPOSAL_CONTEXT):
            raise ValueError("Unsupported recorded SAC command context")
        if recording["actor_kind"] == SACSamplingActor.kind:
            actor = SACSamplingActor(
                args.model,
                pixels,
                recording["model"]["sac_manifest_sha256"],
                exploration_seed=recording["model"]["noise"]["seed"],
                counterfactual=context_kind == PROPOSAL_CONTEXT,
            )
        else:
            actor = SACEvaluationActor(
                args.model,
                pixels,
                recording["model"]["sac_manifest_sha256"],
                counterfactual=context_kind == PROPOSAL_CONTEXT,
            )
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
    environment = ShadowEnvironment.from_native(configuration)
    print("只读影子运行：模型加载与预热后接收；不会连接手柄或发送输入。F8 停止。", flush=True)
    result = run_experiment(
        request,
        realtime_environment=environment,
        numeric_actor_factory=configuration.actor,
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
