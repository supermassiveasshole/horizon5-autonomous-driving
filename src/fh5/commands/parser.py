"""CLI argument definitions, grouped by workflow without loading runtime adapters."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the public command syntax in its established help order."""
    parser = argparse.ArgumentParser(description="Record and replay FH5 Data Out experiments")
    commands = parser.add_subparsers(dest="mode", required=True)
    _candidate_commands(commands)
    _evaluation_commands(commands)
    _sac_commands(commands)
    _collection_commands(commands)
    _realtime_commands(commands)
    _review_commands(commands)
    _offline_learning_commands(commands)
    _recording_and_route_commands(commands)
    _control_commands(commands)
    _vision_commands(commands)
    return parser


def _candidate_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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


def _evaluation_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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


def _sac_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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
    for name in ("checkpoint", "report"):
        sac_policy.add_argument("--" + name, type=Path, required=True)
    sac_policy.add_argument(
        "--replay", type=Path, help="Experience file; omitted uses the checkpoint's sealed replay"
    )
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
    sac_warmup.add_argument(
        "--replay-sha256", help="Require this exact replay digest; omitted binds the selected file"
    )
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
    for name in ("checkpoint", "report"):
        sac_replay.add_argument("--" + name, type=Path, required=True)
    sac_replay.add_argument(
        "--replay", type=Path, help="Experience file; omitted uses the checkpoint's sealed replay"
    )


def _collection_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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
    start.add_argument(
        "--output", type=Path, help="New recording directory; reuse the frozen install"
    )
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
    scheduled_resume.add_argument("--checkpoint-sha256", help="Optional expected learner digest")
    for name in ("collection-status", "collection-stop", "collection-review"):
        collection = commands.add_parser(
            name, help="Inspect, stop or verify a passive collection session"
        )
        collection.add_argument("recording", type=Path)
        if name == "collection-review":
            collection.add_argument("--report", type=Path, required=True)


def _realtime_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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


def _review_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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


def _offline_learning_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    bc = commands.add_parser("bc-train", help="Retired: import history and use temporal-train")
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
        "numeric-prepare", help="Retired: use temporal-prepare for historical training data"
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


def _recording_and_route_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
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


def _control_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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


def _vision_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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
