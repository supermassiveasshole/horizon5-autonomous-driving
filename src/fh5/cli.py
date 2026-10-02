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

from fh5.attempts import AttemptReplay
from fh5.bc import BCReplay, BCTrain
from fh5.control import Control, validate_control_file
from fh5.demonstration_dataset import DemonstrationDataset
from fh5.demonstrations import DemonstrationRecord, DemonstrationReplay
from fh5.events import EventRun, validate_event_file
from fh5.experiment import Packet, Record, Replay, run_experiment
from fh5.observations import ObservationReplay
from fh5.perception import Perception, PerceptionReplay
from fh5.recovery import RecoveryReplay
from fh5.reward_audit import RewardAudit
from fh5.rewards import RewardReplay
from fh5.routes import BuildRoute, RouteCheck
from fh5.tracking import TrackingDrive, validate_tracking_file
from fh5.vision import VisionRecord


def _sac(args: argparse.Namespace) -> int:
    from fh5.sac import SACCriticReplay, SACCriticResume, SACCriticWarmup
    from fh5.sac_actions import ActionBounds
    from fh5.sac_learning import SACPolicyReplay, SACResume, SACTrain
    from fh5.sac_replay import SACReplayPrepare

    if args.mode == "sac-resume":
        resumed = run_experiment(
            SACResume(
                args.checkpoint,
                args.output,
                steps=args.steps,
                additions=tuple((Path(path), sha) for path, sha in args.add_replay),
                expected_checkpoint_sha256=args.checkpoint_sha256,
                demonstration_fraction=args.demonstration_fraction,
                imitation_comparison=args.imitation_comparison,
                imitation_registry=args.imitation_registry,
                raw_cache_bytes=args.raw_cache_bytes,
            )
        )
        print(json.dumps(resumed.summary["sac_learning"], ensure_ascii=False))
        return 0 if resumed.summary["sac_learning"]["stop_reason"] == "budget_completed" else 4
    if args.mode == "sac-train":
        options = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if not isinstance(options, dict) or options.pop("version", None) != 1:
            raise ValueError("SAC training requires a version 1 configuration")
        try:
            if options.get("imitation_protocol_batch") is not None:
                options["imitation_protocol_batch"] = (
                    args.config.parent / options["imitation_protocol_batch"]
                )
            request = SACTrain(
                args.config.parent / options.pop("warmup"),
                args.config.parent / options.pop("replay"),
                args.output,
                **options,
            )
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid SAC training fields") from error
        summary = run_experiment(request).summary["sac_learning"]
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if summary["stop_reason"] == "budget_completed" else 4
    if args.mode == "sac-policy-replay":
        replayed = run_experiment(
            SACPolicyReplay(
                args.checkpoint, args.replay, args.report, raw_cache_bytes=args.raw_cache_bytes
            )
        )
        print(json.dumps(replayed.summary["sac_policy"], ensure_ascii=False))
        return 0

    if args.mode == "sac-prepare":
        summary = run_experiment(
            SACReplayPrepare(
                args.recording, args.trace, args.task, args.reward, args.output, args.evidence
            )
        ).summary["sac_replay"]
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if summary["eligible_transitions"] else 4
    if args.mode == "sac-warmup-resume":
        result = run_experiment(
            SACCriticResume(args.checkpoint, args.output, args.steps, args.checkpoint_sha256)
        )
    elif args.mode == "sac-warmup":
        bounds = (
            ActionBounds(**json.loads(args.bounds.read_text(encoding="utf-8-sig")))
            if args.bounds
            else ActionBounds()
        )
        result = run_experiment(
            SACCriticWarmup(
                args.model,
                args.replay,
                args.replay_sha256,
                args.output,
                args.steps,
                args.batch_size,
                args.learning_rate,
                args.seed,
                bounds,
            )
        )
    else:
        result = run_experiment(SACCriticReplay(args.checkpoint, args.replay, args.report))
    print(json.dumps(result.summary["sac"], ensure_ascii=False))
    return (
        4
        if result.summary["sac"]["stop_reason"] in {"stop_requested", "training_data_unavailable"}
        else 0
    )


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
    candidate_archive = commands.add_parser(
        "candidate-archive", help="Retain an exact, complete SAC continuation package"
    )
    candidate_archive.add_argument("--checkpoint", type=Path, required=True)
    candidate_archive.add_argument("--checkpoint-sha256", required=True)
    candidate_restore = commands.add_parser(
        "candidate-restore",
        help="Restore an archived candidate into a new directory; no activation",
    )
    candidate_restore.add_argument("--archive", type=Path, required=True)
    candidate_restore.add_argument("--archive-sha256", required=True)
    for command in (candidate_archive, candidate_restore):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--reason", required=True)
    candidate_compare = commands.add_parser(
        "candidate-compare", help="Compare frozen candidates from source evidence; no activation"
    )
    candidate_compare.add_argument("--config", type=Path, required=True)
    candidate_compare.add_argument("--output", type=Path, required=True)
    candidate_compare.add_argument("--registry", type=Path)
    candidate_record = commands.add_parser(
        "candidate-record", help="Retain candidate roles and evidence in a synthetic version store"
    )
    candidate_record.add_argument("--config", type=Path, required=True)
    candidate_record.add_argument("--expected-revision")
    candidate_rollback = commands.add_parser(
        "candidate-rollback", help="Re-audit a retained synthetic default; keep exploration state"
    )
    candidate_rollback.add_argument("--expected-revision", required=True)
    candidate_rollback.add_argument("--target-revision", required=True)
    candidate_rollback.add_argument("--reason", required=True)
    candidate_history = commands.add_parser(
        "candidate-history", help="Read committed synthetic candidate roles and history"
    )
    candidate_history.add_argument("--after-sequence", type=int, default=0)
    candidate_history.add_argument(
        "--limit", type=int, help="Return at most this many events; 0 returns current roles only"
    )
    storage_plan = commands.add_parser(
        "learning-storage-plan", help="Account for retained learning dependencies without deletion"
    )
    storage_plan.add_argument("--config", type=Path, required=True)
    storage_plan.add_argument("--output", type=Path, required=True)
    for command in (candidate_record, candidate_rollback, candidate_history):
        command.add_argument("--store", type=Path, required=True)
    for command in (candidate_record, candidate_rollback):
        command.add_argument("--registry", type=Path, required=True)
    evaluation_prepare = commands.add_parser(
        "evaluation-prepare", help="Freeze a policy, local task and evaluation protocol; no devices"
    )
    evaluation_prepare.add_argument("--config", type=Path, required=True)
    evaluation_prepare.add_argument("--output", type=Path, required=True)
    evaluation_prepare.add_argument("--registry", type=Path)
    evaluation_review = commands.add_parser(
        "evaluation-review", help="Account for every recorded attempt under a frozen protocol"
    )
    evaluation_review.add_argument("--batch", type=Path, required=True)
    evaluation_review.add_argument("--ledger", type=Path, required=True)
    evaluation_review.add_argument("--output", type=Path, required=True)
    evaluation_review.add_argument("--registry", type=Path)
    evaluation_run = commands.add_parser(
        "evaluation-run", help="Validate a native frozen batch; --live executes it"
    )
    evaluation_run.add_argument("--batch", type=Path, required=True)
    evaluation_run.add_argument("--batch-sha256", required=True)
    evaluation_run.add_argument("--event-config", type=Path, required=True)
    evaluation_run.add_argument("--driving-config", type=Path, required=True)
    evaluation_run.add_argument("--output", type=Path, required=True)
    evaluation_run.add_argument("--registry", type=Path)
    evaluation_run.add_argument("--seconds", type=float, default=15)
    evaluation_run.add_argument(
        "--initial-operation", choices=("start_ready", "restart_ready"), default="start_ready"
    )
    evaluation_run.add_argument("--live", action="store_true")
    evidence_use = commands.add_parser(
        "evidence-use", help="Register recording use for training or selection; no devices"
    )
    evidence_use.add_argument("--registry", type=Path, required=True)
    evidence_use.add_argument("--role", choices=("training", "selection"), required=True)
    evidence_use.add_argument("--recording", type=Path, action="append", required=True)
    evidence_use.add_argument("--model-sha256")
    evidence_use.add_argument("--output", type=Path, required=True)
    sac_train = commands.add_parser("sac-train", help="Bounded synthetic SAC updates; no devices")
    sac_train.add_argument("--config", type=Path, required=True)
    sac_train.add_argument("--output", type=Path, required=True)
    sac_resume = commands.add_parser(
        "sac-resume", help="Continue sealed CPU SAC learning; no devices"
    )
    sac_resume.add_argument("--checkpoint", type=Path, required=True)
    sac_resume.add_argument("--output", type=Path, required=True)
    sac_resume.add_argument("--steps", type=int, default=100)
    sac_resume.add_argument(
        "--raw-cache-bytes",
        type=int,
        help="Raw uint8 retention budget; 0 disables, omitted inherits",
    )
    sac_resume.add_argument("--checkpoint-sha256", help="Require this exact parent manifest digest")
    sac_resume.add_argument(
        "--imitation-comparison",
        type=Path,
        help="Recompute a frozen development comparison before guidance changes",
    )
    sac_resume.add_argument(
        "--imitation-registry",
        type=Path,
        help="Previously reserved development evidence usage registry",
    )
    sac_resume.add_argument(
        "--demonstration-fraction",
        type=float,
        help="Explicitly select source quotas in [0,1]; omitted inherits",
    )
    sac_resume.add_argument(
        "--add-replay",
        nargs=2,
        action="append",
        default=[],
        metavar=("PATH", "SHA256"),
        help="Append compatible sealed experience with explicit digest; repeat up to 10 times",
    )
    sac_policy = commands.add_parser("sac-policy-replay", help="Replay a frozen learned SAC policy")
    for name in ("checkpoint", "replay", "report"):
        sac_policy.add_argument("--" + name, type=Path, required=True)
    sac_policy.add_argument(
        "--raw-cache-bytes",
        type=int,
        help="Raw uint8 retention budget; 0 disables, omitted inherits",
    )
    sac_prepare = commands.add_parser(
        "sac-prepare", help="Prepare synthetic numerical learning transitions; no devices"
    )
    for name in ("recording", "trace", "task", "reward", "output"):
        sac_prepare.add_argument("--" + name, type=Path, required=True)
    sac_prepare.add_argument("--evidence", type=Path)
    sac_warmup = commands.add_parser(
        "sac-warmup", help="Warm two value heads with the entire BC frozen; no devices"
    )
    for name in ("model", "replay", "output"):
        sac_warmup.add_argument("--" + name, type=Path, required=True)
    sac_warmup.add_argument("--replay-sha256", required=True)
    sac_warmup.add_argument("--steps", type=int, default=100)
    sac_warmup.add_argument("--batch-size", type=int, default=32)
    sac_warmup.add_argument("--learning-rate", type=float, default=0.0001)
    sac_warmup.add_argument("--seed", type=int, default=7)
    sac_warmup.add_argument("--bounds", type=Path)
    warm_resume = commands.add_parser(
        "sac-warmup-resume", help="Resume remaining finite Q warm-up; no devices"
    )
    warm_resume.add_argument("--checkpoint", type=Path, required=True)
    warm_resume.add_argument("--output", type=Path, required=True)
    warm_resume.add_argument(
        "--steps", type=int, help="Cap this segment's updates; omitted finishes remaining budget"
    )
    warm_resume.add_argument(
        "--checkpoint-sha256", help="Require this exact parent manifest digest"
    )
    sac_replay = commands.add_parser(
        "sac-critic-replay", help="Reload frozen BC and warmed critics without updating"
    )
    for name in ("checkpoint", "replay", "report"):
        sac_replay.add_argument("--" + name, type=Path, required=True)
    prepare = commands.add_parser(
        "collection-prepare", help="Freeze a separate passive collector; no devices"
    )
    prepare.add_argument("--repository", type=Path, default=Path.cwd())
    prepare.add_argument("--capture-config", type=Path, required=True)
    prepare.add_argument("--input-profile", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seconds", type=float, default=14400)
    prepare.add_argument("--source", choices=("native", "synthetic"), default="native")
    prepare.add_argument("--port", type=int, default=5300)
    prepare.add_argument("--uv", type=Path)
    prepare.add_argument("--offline", action="store_true")
    start = commands.add_parser("collection-start", help="Start the frozen independent collector")
    start.add_argument("bundle", type=Path)
    start.add_argument("--live", action="store_true")
    dataset = commands.add_parser(
        "collection-dataset", help="Retired: use collection-bc-prepare with a v2 configuration"
    )
    dataset.add_argument("--config", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    dataset_review = commands.add_parser(
        "collection-dataset-review", help="Revalidate a fixed source selection"
    )
    dataset_review.add_argument("dataset", type=Path)
    dataset_review.add_argument("--report", type=Path, required=True)
    collection_bc = commands.add_parser(
        "collection-bc-prepare", help="Select sealed sources and prepare numeric BC inputs"
    )
    collection_bc.add_argument("--config", type=Path, required=True)
    collection_bc.add_argument("--output", type=Path, required=True)
    assess = commands.add_parser(
        "collection-bc-assess", help="Compare frozen models on development or final data"
    )
    assess.add_argument("--config", type=Path, required=True)
    assess.add_argument("--output", type=Path, required=True)
    scheduled = commands.add_parser(
        "collection-bc-train", help="Train a frozen snapshot within collection resource budgets"
    )
    scheduled.add_argument("--config", type=Path, required=True)
    scheduled.add_argument("--output", type=Path, required=True)
    scheduled_resume = commands.add_parser(
        "collection-bc-resume", help="Resume remaining CPU BC updates from a sealed learner"
    )
    scheduled_resume.add_argument("--run", type=Path, required=True)
    scheduled_resume.add_argument("--output", type=Path, required=True)
    scheduled_resume.add_argument("--checkpoint-sha256", required=True)
    for name in ("collection-status", "collection-stop", "collection-review"):
        collection = commands.add_parser(
            name, help="Inspect, stop or verify a passive collection session"
        )
        collection.add_argument("recording", type=Path)
        if name == "collection-review":
            collection.add_argument("--report", type=Path, required=True)
    realtime_replay = commands.add_parser(
        "realtime-replay", help="Independently replay recorded numerical predictions; no devices"
    )
    realtime_replay.add_argument("recording", type=Path)
    realtime_replay.add_argument("--model", type=Path, required=True)
    realtime_replay.add_argument("--report", type=Path, required=True)
    realtime_replay.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    realtime_replay.add_argument("--tolerance", type=float, default=1e-6)
    realtime_replay.add_argument("--allow-legacy-source-diagnostic", action="store_true")
    shadow = commands.add_parser(
        "realtime-shadow", help="Validate numerical shadow; --live is read-only"
    )
    shadow.add_argument("--config", type=Path, required=True)
    shadow.add_argument("--output", type=Path, required=True)
    shadow.add_argument("--seconds", type=float, default=30)
    shadow.add_argument("--hz", type=int, choices=(10, 20))
    shadow.add_argument("--allow-legacy-source-diagnostic", action="store_true")
    shadow.add_argument(
        "--live", action="store_true", help="Read FH5 and predict; never send game input"
    )
    drive = commands.add_parser(
        "realtime-drive",
        help="Validate numerical driving; --live explicitly enables bounded control",
    )
    drive.add_argument("--config", type=Path, required=True)
    drive.add_argument("--output", type=Path, required=True)
    drive.add_argument("--seconds", type=float, default=15)
    drive.add_argument("--live", action="store_true")
    trace = commands.add_parser("capture-frame-times", help="Attach independent PresentMon QPC CSV")
    trace.add_argument("recording", type=Path)
    trace.add_argument("--csv", type=Path, required=True)
    trace.add_argument("--pid", type=int, required=True)
    trace.add_argument("--swap-chain", required=True)
    trace.add_argument("--report", type=Path, required=True)
    capture = commands.add_parser(
        "capture-dxgi", help="Validate DXGI probe; --live passively captures FH5"
    )
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--seconds", type=float, default=30)
    capture.add_argument("--port", type=int, default=5300)
    capture.add_argument("--raw-samples", type=int, default=0, help="Bounded source samples, 0–8")
    capture.add_argument(
        "--diagnostic-mss",
        choices=("numeric", "jpeg"),
        help="Explicit MSS comparison, not a production fallback",
    )
    capture.add_argument(
        "--live", action="store_true", help="Read-only physical client capture; F8 stops"
    )
    policy = commands.add_parser("policy", help="Retired encoded-image driver; use realtime-drive")
    policy.add_argument("--config", type=Path)
    policy.add_argument("--output", type=Path)
    policy.add_argument("--port", type=int, default=5300)
    policy.add_argument("--live", action="store_true", help="Retired; no game input is sent")
    attempt = commands.add_parser(
        "attempt-review",
        help="Review complete local attempts with independent evidence; no game input",
    )
    attempt.add_argument("recording", type=Path)
    attempt.add_argument("--task", type=Path, required=True)
    attempt.add_argument("--evidence", type=Path)
    attempt.add_argument("--output", type=Path, required=True)
    reward = commands.add_parser("reward-replay", help="Settle local physical-time rewards offline")
    reward.add_argument("recording", type=Path)
    reward.add_argument("--task", type=Path, required=True)
    reward.add_argument("--reward", type=Path, required=True)
    reward.add_argument("--evidence", type=Path)
    reward.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser(
        "reward-audit", help="Replay synthetic complete reward counterexamples"
    )
    audit.add_argument("--reward", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    recovery = commands.add_parser(
        "recovery-replay", help="Replay synthetic recovery signals; never sends game input"
    )
    recovery.add_argument("--config", type=Path, required=True)
    recovery.add_argument("--trace", type=Path, required=True)
    recovery.add_argument("--output", type=Path, required=True)
    bc = commands.add_parser("bc-train", help="Train a bounded offline multimodal BC actor")
    bc.add_argument("--config", type=Path, required=True)
    bc.add_argument("--output", type=Path, required=True)
    bc_replay = commands.add_parser("bc-replay", help="Replay a frozen BC actor; no game input")
    bc_replay.add_argument("--model", type=Path, required=True)
    bc_replay.add_argument("--dataset", type=Path, required=True)
    bc_replay.add_argument("--report", type=Path, required=True)
    bc_replay.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for name in ("temporal-prepare", "temporal-train"):
        temporal = commands.add_parser(name, help="Prepare/train frozen numerical Δt BC offline")
        temporal.add_argument("--config", type=Path, required=True)
        temporal.add_argument("--output", type=Path, required=True)
    temporal_replay = commands.add_parser("temporal-replay", help="Reload numerical Δt BC offline")
    temporal_replay.add_argument("--model", type=Path, required=True)
    temporal_replay.add_argument("--dataset", type=Path, required=True)
    temporal_replay.add_argument("--report", type=Path, required=True)
    temporal_replay.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    numeric_prepare = commands.add_parser(
        "numeric-prepare", help="Decode legacy BC images once into a numerical offline source"
    )
    numeric_prepare.add_argument("--model", type=Path, required=True)
    numeric_prepare.add_argument("--dataset", type=Path, required=True)
    numeric_prepare.add_argument("--output", type=Path, required=True)
    numeric_prepare.add_argument("--max-decisions", type=int, default=200)
    numeric_prepare.add_argument(
        "--view", choices=("no_reference", "reference_assisted"), default="no_reference"
    )
    for name, description in (
        ("numeric-infer", "Run numerical prepared inputs through a frozen actor; no game input"),
        ("numeric-replay", "Verify exact numerical inputs and replay predictions; no game input"),
    ):
        numeric = commands.add_parser(name, help=description)
        numeric.add_argument("recording", type=Path)
        numeric.add_argument("--model", type=Path, required=True)
        numeric.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        numeric.add_argument("--legacy-diagnostic", action="store_true")
        if name == "numeric-infer":
            numeric.add_argument("--output", type=Path, required=True)
            numeric.add_argument("--max-decisions", type=int, default=1000)
            numeric.add_argument("--archive-capacity", type=int, default=8)
            numeric.add_argument("--archive-mib", type=int, default=32)
        else:
            numeric.add_argument("--report", type=Path, required=True)
            numeric.add_argument("--tolerance", type=float, default=1e-6)
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
    route_check = commands.add_parser(
        "route-check", help="Check a declared short recording against a frozen route; no control"
    )
    route_check.add_argument("recording", type=Path)
    route_check.add_argument("--route", type=Path, required=True)
    route_check.add_argument("--report", type=Path, required=True)
    route_check.add_argument("--first-packet", type=int, required=True)
    route_check.add_argument("--last-packet", type=int, required=True)
    route_check.add_argument("--max-speed-kmh", type=float, default=20.0)
    control = commands.add_parser(
        "control", help="Validate a bounded calibration; --live sends game input"
    )
    control.add_argument("--config", type=Path, required=True)
    control.add_argument("--output", type=Path, required=True)
    control.add_argument("--port", type=int, default=5300)
    control.add_argument(
        "--live", action="store_true", help="Send actual input; F8 or Ctrl+C releases it"
    )
    track = commands.add_parser(
        "track", help="Validate local route feedback; --live sends bounded input"
    )
    track.add_argument("--config", type=Path, required=True)
    track.add_argument("--output", type=Path, required=True)
    track.add_argument("--port", type=int, default=5300)
    track.add_argument("--live", action="store_true", help="Run with FH5 foreground; F8 stops")
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
        if args.mode == "collection-prepare":
            from fh5.capture_config import parse_capture_config
            from fh5.collection import CollectionConfig
            from fh5.collection_process import CollectionPrepare
            from fh5.collection_store import read_bounded

            pipeline, _ = parse_capture_config(
                json.loads(read_bounded(args.capture_config, 1024**2))
            )
            prepared = run_experiment(
                CollectionPrepare(
                    args.repository,
                    args.output,
                    args.capture_config,
                    args.input_profile,
                    CollectionConfig(
                        pixels=pipeline.pixels,
                        seconds=args.seconds,
                        observation_hz=pipeline.observation_hz,
                        max_age_ms=pipeline.max_age_ms,
                    ),
                    source=args.source,
                    port=args.port,
                    uv=args.uv,
                    offline=args.offline,
                )
            )
            print(
                json.dumps(
                    {
                        "state": prepared.summary["collection"]["state"],
                        "manifest": str(prepared.report_path),
                        "commands_sent": False,
                    }
                )
            )
            return 0
        if args.mode == "collection-start":
            from fh5.collection_process import CollectionStart

            started = run_experiment(CollectionStart(args.bundle, live=args.live))
            print(json.dumps(started.summary["collection"]))
            return 0
        if args.mode == "collection-dataset":
            raise ValueError(
                "collection-dataset is retired; use collection-bc-prepare with a v2 "
                "configuration containing sources, selection rules and observation settings. "
                "Existing selections remain readable with collection-dataset-review."
            )
        if args.mode == "collection-dataset-review":
            from fh5.collection_dataset import CollectionDatasetReview

            selected = run_experiment(CollectionDatasetReview(args.dataset, args.report))
            print(json.dumps(selected.summary["collection_dataset"], ensure_ascii=False))
            return 0
        if args.mode in ("candidate-archive", "candidate-restore"):
            from fh5.candidate_archive import CandidateArchive, CandidateRestore

            retained = run_experiment(
                CandidateArchive(args.checkpoint, args.output, args.checkpoint_sha256, args.reason)
                if args.mode == "candidate-archive"
                else CandidateRestore(args.archive, args.output, args.archive_sha256, args.reason)
            )
            result = retained.summary[args.mode.replace("-", "_")]
            print(json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=False))
            return 0
        if args.mode == "learning-storage-plan":
            from fh5.storage import LearningStoragePlan

            storage = run_experiment(LearningStoragePlan(args.config, args.output)).summary[
                "storage"
            ]
            print(
                json.dumps({k: v for k, v in storage.items() if k != "files"}, ensure_ascii=False)
            )
            return 0 if storage["status"] == "within_budget" else 4
        if args.mode in ("candidate-record", "candidate-rollback", "candidate-history"):
            from fh5.candidate_store import CandidateHistory, CandidateRecord, CandidateRollback

            operation: CandidateRecord | CandidateRollback | CandidateHistory
            if args.mode == "candidate-record":
                operation = CandidateRecord(
                    args.config, args.store, args.expected_revision, args.registry
                )
            elif args.mode == "candidate-rollback":
                operation = CandidateRollback(
                    args.store,
                    args.expected_revision,
                    args.target_revision,
                    args.reason,
                    args.registry,
                )
            else:
                operation = CandidateHistory(args.store, args.after_sequence, args.limit)
            print(
                json.dumps(run_experiment(operation).summary["candidate_store"], ensure_ascii=False)
            )
            return 0
        if args.mode == "candidate-compare":
            from fh5.candidate_selection import CandidateCompare

            compared = run_experiment(CandidateCompare(args.config, args.output, args.registry))
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in compared.summary["candidate_selection"].items()
                        if k != "reviews"
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.mode == "evaluation-run":
            from fh5.evaluation_cli import evaluation_command

            return evaluation_command(args)
        if args.mode in ("evaluation-prepare", "evaluation-review"):
            from fh5.evaluation import EvaluationPrepare, EvaluationReview

            evaluation_request = (
                EvaluationPrepare(args.config, args.output, args.registry)
                if args.mode == "evaluation-prepare"
                else EvaluationReview(args.batch, args.ledger, args.output, args.registry)
            )
            summary = run_experiment(evaluation_request).summary["evaluation"]
            print(
                json.dumps(
                    {k: v for k, v in summary.items() if k != "attempts"}, ensure_ascii=False
                )
            )
            independence = summary.get("independence", {})
            if (
                summary.get("unresolved_recordings")
                or summary.get("quarantined_executions")
                or independence.get("error")
            ):
                return 2
            if summary["purpose"] == "final" and independence.get("status") in (
                "known_overlap",
                "unknown",
            ):
                return 4
            return 0
        if args.mode in (
            "sac-prepare",
            "sac-warmup",
            "sac-warmup-resume",
            "sac-critic-replay",
            "sac-train",
            "sac-resume",
            "sac-policy-replay",
        ):
            return _sac(args)
        if args.mode == "evidence-use":
            from fh5.evidence_usage import RecordUsage

            usage = run_experiment(
                RecordUsage(
                    args.registry, tuple(args.recording), args.role, args.output, args.model_sha256
                )
            ).summary["evidence_usage"]
            print(json.dumps(usage, ensure_ascii=False))
            return 4 if usage["unidentified_recordings"] else 0
        if args.mode == "collection-bc-prepare":
            from fh5.collection_bc import CollectionBCPrepare

            prepared = run_experiment(CollectionBCPrepare(args.config, args.output))
            print(json.dumps(prepared.summary["collection_bc"], ensure_ascii=False))
            return 0
        if args.mode == "collection-bc-assess":
            from fh5.collection_assessment import CollectionBCAssess

            assessed = run_experiment(CollectionBCAssess(args.config, args.output))
            summary = assessed.summary["collection_assessment"]
            print(
                json.dumps(
                    {k: v for k, v in summary.items() if k != "decisions"}, ensure_ascii=False
                )
            )
            return 0
        if args.mode in ("collection-bc-train", "collection-bc-resume"):
            from fh5.learning_schedule import ScheduledBCResume, ScheduledBCTrain

            scheduled_result = run_experiment(
                ScheduledBCResume(args.run, args.output, args.checkpoint_sha256)
                if args.mode == "collection-bc-resume"
                else ScheduledBCTrain(args.config, args.output)
            )
            summary = scheduled_result.summary["learning_schedule"]
            print(
                json.dumps({k: v for k, v in summary.items() if k != "events"}, ensure_ascii=False)
            )
            return 0 if summary["state"] == "completed" else 2
        if args.mode.startswith("collection-"):
            from fh5.collection import CollectionControl, CollectionReview

            collection_request = (
                CollectionReview(args.recording, args.report)
                if args.mode == "collection-review"
                else CollectionControl(args.recording, stop=args.mode == "collection-stop")
            )
            result = run_experiment(collection_request)
            summary = result.summary["collection"]
            print(json.dumps(summary, ensure_ascii=False))
            return 4 if summary.get("errors") else 0
        if args.mode == "realtime-replay":
            from fh5.realtime_cli import replay_command

            return replay_command(args)
        if args.mode == "realtime-shadow":
            from fh5.realtime_cli import shadow_command

            return shadow_command(args)
        if args.mode == "realtime-drive":
            from fh5.numeric_drive_cli import drive_command

            return drive_command(args)
        if args.mode == "capture-frame-times":
            from fh5.capture_trace import CaptureTraceReview

            result = run_experiment(
                CaptureTraceReview(args.recording, args.csv, args.report, args.pid, args.swap_chain)
            )
            print(json.dumps(result.summary["capture"]["game_frame_time"], ensure_ascii=False))
            return int(result.summary["capture"]["game_frame_time"]["frame_count"] == 0)
        if args.mode == "capture-dxgi":
            from fh5.capture_cli import capture_command

            return capture_command(args)
        if args.mode in ("temporal-prepare", "temporal-train", "temporal-replay"):
            return _temporal_command(args)
        if args.mode in ("numeric-prepare", "numeric-infer", "numeric-replay"):
            return _numeric_command(args)
        if args.mode == "input-devices":
            from fh5.live_demonstration import input_devices

            print(json.dumps({"devices": input_devices(), "commands_sent": False}))
            return 0
        if args.mode == "policy":
            raise ValueError(
                "The encoded-image policy command is retired. Use realtime-drive with a numerical "
                "temporal BC model and configs/realtime-drive.example.json; old configurations "
                "are not interchangeable. Existing recordings remain readable with replay."
            )
        elif args.mode == "bc-train":
            result = run_experiment(BCTrain(args.config, args.output))
        elif args.mode == "bc-replay":
            result = run_experiment(BCReplay(args.model, args.dataset, args.report, args.device))
        elif args.mode == "demonstration-replay":
            result = run_experiment(DemonstrationReplay(args.recording, args.report))
        elif args.mode == "attempt-review":
            result = run_experiment(
                AttemptReplay(args.recording, args.output, args.task, args.evidence)
            )
        elif args.mode == "reward-replay":
            result = run_experiment(
                RewardReplay(args.recording, args.output, args.task, args.reward, args.evidence)
            )
        elif args.mode == "reward-audit":
            result = run_experiment(RewardAudit(args.reward, args.output))
        elif args.mode == "recovery-replay":
            result = run_experiment(RecoveryReplay(args.config, args.trace, args.output))
        elif args.mode == "route-check":
            result = run_experiment(
                RouteCheck(
                    args.recording,
                    args.report,
                    args.route,
                    args.first_packet,
                    args.last_packet,
                    args.max_speed_kmh,
                )
            )
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
        elif args.mode in ("control", "track"):
            config = (
                validate_tracking_file(args.config)[0]["tracking"]
                if args.mode == "track"
                else validate_control_file(args.config)
            )
            if not args.live:
                print(
                    json.dumps(
                        {
                            "status": "validated_only",
                            "duration_s": config["max_duration_s"] + config["braking_s"]
                            if args.mode == "track"
                            else sum(step["seconds"] for step in config["steps"]),
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
                control_request = (
                    TrackingDrive(args.config, args.output)
                    if args.mode == "track"
                    else Control(args.config, args.output)
                )
                result = run_experiment(control_request, environment=environment)
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
    if "recovery" in result.summary:
        print(
            json.dumps(
                {
                    "status": "synthetic_replay_complete",
                    "commands_sent": False,
                    "reason": result.summary["recovery"]["reason"],
                    "metrics": result.summary["recovery"]["metrics"],
                    "report": str(result.report_path),
                    "real_rewind_verified": False,
                }
            )
        )
        return 0
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
    if "rewards" in result.summary:
        rewards = result.summary["rewards"]
        segments = rewards["segments"]
        print(
            json.dumps(
                {
                    "rules_version": rewards["rules_version"],
                    "report": str(result.report_path),
                    "commands_sent": False,
                    "segments": [
                        {
                            k: s[k]
                            for k in (
                                "outcome",
                                "reward_usable",
                                "physical_duration_s",
                                "discounted_return",
                                "quarantine_reasons",
                            )
                        }
                        for s in segments
                    ],
                    "audit": result.summary.get("reward_audit"),
                }
            )
        )
        if "reward_audit" in result.summary:
            return 0 if result.summary["reward_audit"]["passed"] else 4
        return 0 if segments and all(s["reward_usable"] for s in segments) else 4
    if "attempt_review" in result.summary:
        review = result.summary["attempt_review"]
        print(
            json.dumps(
                {
                    "rules_version": review["rules_version"],
                    "recording_packet_count": review["recording_packet_count"],
                    "attempts": [
                        {k: a[k] for k in ("attempt_id", "outcome", "reasons", "record_eligible")}
                        for a in review["attempts"]
                    ],
                    "report": str(result.report_path),
                    "commands_sent": False,
                }
            )
        )
        return 0 if all(a["record_eligible"] for a in review["attempts"]) else 4
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
                **(
                    {"route_check": result.summary["route_check"]}
                    if "route_check" in result.summary
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
    if "route_check" in result.summary:
        return 0 if result.summary["route_check"]["passed"] else 4
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
    if args.mode == "track" and (
        result.summary["capture_status"] != "completed"
        or result.summary["control"]["stop_reason"] != "local_end"
        or not result.summary["control"]["release_sent"]
    ):
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


def _temporal_command(args: argparse.Namespace) -> int:
    from fh5.temporal_bc import TemporalBCReplay, TemporalBCTrain
    from fh5.temporal_import import TemporalBCPrepare

    if args.mode == "temporal-prepare":
        result = run_experiment(TemporalBCPrepare(args.config, args.output))
        summary = result.summary["temporal_import"]
    elif args.mode == "temporal-train":
        result = run_experiment(TemporalBCTrain(args.config, args.output))
        summary = result.summary["temporal_bc"]
    else:
        result = run_experiment(
            TemporalBCReplay(args.model, args.dataset, args.report, args.device)
        )
        summary = result.summary["temporal_bc"]
    print(
        json.dumps(
            {
                "commands_sent": False,
                "decisions": summary.get("selected_observations", len(summary["decisions"])),
                "dataset_sha256": summary["dataset_sha256"],
                "report": str(result.report_path),
                "training": {k: v for k, v in summary.get("training", {}).items() if k != "losses"},
            }
        )
    )
    return 0


def _numeric_command(args: argparse.Namespace) -> int:
    from fh5.numeric_actor import FrozenNumericActor
    from fh5.numeric_images import NumericInfer, NumericReplay, PixelContract
    from fh5.numeric_import import LegacyNumericImport, PreparedNumericSource

    if args.mode == "numeric-prepare":
        result = run_experiment(
            LegacyNumericImport(
                args.model,
                args.dataset,
                args.output,
                args.max_decisions,
                args.view,
            )
        )
        summary = result.summary["numeric_import"]
    elif args.mode == "numeric-infer":
        source = PreparedNumericSource(args.recording)
        actor = FrozenNumericActor(
            args.model, source.contract, args.device, legacy_diagnostic=args.legacy_diagnostic
        )
        result = run_experiment(
            NumericInfer(
                args.output,
                source.contract,
                args.max_decisions,
                args.archive_capacity,
                args.archive_mib * 1024**2,
            ),
            numeric_inputs=source,
            numeric_actor=actor,
        )
        summary = result.summary["numeric"]
    else:
        manifest = json.loads((args.recording / "numeric-run.json").read_text(encoding="utf-8"))
        contract = PixelContract.from_metadata(manifest["contract"])
        actor = FrozenNumericActor(
            args.model, contract, args.device, legacy_diagnostic=args.legacy_diagnostic
        )
        result = run_experiment(
            NumericReplay(args.recording, args.report, args.tolerance), numeric_actor=actor
        )
        summary = result.summary["numeric"]
    decisions = summary["decisions"]
    print(
        json.dumps(
            {
                "commands_sent": False,
                "decisions": len(decisions),
                "predicted": sum(d.get("status") == "predicted" for d in decisions),
                "exact_replay_available": sum(
                    d.get("exact_replay_available", False) for d in decisions
                ),
                "execution_error": summary.get("execution_error"),
                "replay_errors": summary.get("replay_errors", []),
                "report": str(result.report_path),
            }
        )
    )
    return (
        4
        if (
            summary.get("execution_error")
            or summary.get("replay_errors")
            or summary.get("archive", {}).get("error")
            or not summary.get("source_released", True)
            or (
                args.mode != "numeric-prepare"
                and not any(d.get("status") == "predicted" for d in decisions)
            )
        )
        else 0
    )
