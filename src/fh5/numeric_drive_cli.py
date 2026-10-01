"""Explicit numerical driving entry; validation never initializes native devices."""

from __future__ import annotations

import argparse
import json

from fh5.numeric_drive_config import NumericDriveConfiguration
from fh5.realtime import RealtimeEnvironment


def native_driving_environment(configuration: NumericDriveConfiguration) -> RealtimeEnvironment:
    """Construct adapters only after qualification; actual capture starts on read."""
    configuration.require_eligible()
    from fh5.capture_resources import WindowsResources
    from fh5.dxgi_windows import WindowsDXGIFrames
    from fh5.live import WindowsDesktop, XboxController
    from fh5.realtime_driving import NumericDrivingEnvironment
    from fh5.realtime_shadow import ShadowEnvironment

    observations = ShadowEnvironment(
        configuration.request,
        configuration.capture,
        lambda: WindowsDXGIFrames(configuration.target),
        configuration.telemetry,
        WindowsDesktop(),
        configuration.task,
        input_conditions=configuration.bindings,
        resources=WindowsResources(),
    )
    return NumericDrivingEnvironment(observations, XboxController, configuration=configuration)


def drive_command(args: argparse.Namespace) -> int:
    from fh5.experiment import run_experiment

    configuration = NumericDriveConfiguration(args.config, args.output, args.seconds, args.live)
    if not args.live:
        print(
            json.dumps(
                {
                    "status": "validated_only",
                    "devices_opened": False,
                    "commands_sent_to_game": False,
                    "qualification": configuration.qualification,
                    "bindings": configuration.bindings,
                },
                ensure_ascii=False,
            )
        )
        return 0
    environment = native_driving_environment(configuration)
    print("数值短段驾驶：模型预热后核对起点，再连接手柄。F8 解除输入。", flush=True)
    result = run_experiment(
        configuration.request,
        realtime_environment=environment,
        numeric_actor_factory=configuration.actor,
    )
    r = result.summary["realtime"]
    print(
        json.dumps(
            {
                "report": str(result.report_path),
                "stop_reason": r["stop_reason"],
                "commands_sent_to_game": r["commands_sent_to_game"],
                "resources_released": r["resources_released"],
                "qualification": configuration.qualification,
                "real_game_validation": False,
            },
            ensure_ascii=False,
        )
    )
    return int(
        not r["resources_released"]
        or not any(d["status"] == "accepted" for d in r["decisions"])
        or r["stop_reason"] not in ("time_limit", "local_end", "user_stop")
    )
