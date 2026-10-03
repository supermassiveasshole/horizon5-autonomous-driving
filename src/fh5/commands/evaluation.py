"""Explicit native evaluation command with a device-free preflight default."""

from __future__ import annotations

import argparse
import json

from fh5.commands.numeric_drive import native_driving_environment
from fh5.evaluation.native import NativeEvaluationEnvironment
from fh5.evaluation.native_io import native_event_environment
from fh5.evaluation.run import EvaluationRun, evaluation_inputs


def evaluation_command(args: argparse.Namespace) -> int:
    from fh5.experiment import run_experiment

    request = EvaluationRun(
        args.batch,
        args.batch_sha256,
        args.event_config,
        args.output,
        seconds=args.seconds,
        registry_file=args.registry,
        initial_operation=args.initial_operation,
        live=True,
    )
    environment = NativeEvaluationEnvironment(
        args.driving_config,
        menu_factory=native_event_environment,
        driving_factory=native_driving_environment,
    )
    batch, _, _ = evaluation_inputs(request)
    qualification = environment.prepare(request, batch)
    if not args.live:
        print(
            json.dumps(
                {
                    "status": "validated_only",
                    "devices_opened": False,
                    "planned_runs": len(batch["config"]["plan"]),
                    "native": qualification,
                },
                ensure_ascii=False,
            )
        )
        return 0
    result = run_experiment(request, evaluation_environment=environment)
    summary = result.summary["evaluation_run"]
    print(json.dumps({"report": str(result.report_path), **summary}, ensure_ascii=False))
    return int(
        summary["stop_reason"] != "plan_complete"
        or not summary["resources_released"]
        or result.summary.get("evaluation", {}).get("quarantined_executions", 0) > 0
    )
