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

from fh5.capture.legacy import VisionRecord
from fh5.collection.demonstration_dataset import DemonstrationDataset
from fh5.collection.demonstrations import DemonstrationRecord, DemonstrationReplay
from fh5.commands.parser import build_parser
from fh5.driving.control import Control, validate_control_file
from fh5.driving.events import EventRun, validate_event_file
from fh5.driving.recovery import RecoveryReplay
from fh5.driving.tracking import TrackingDrive, validate_tracking_file
from fh5.evaluation.attempts import AttemptReplay
from fh5.evaluation.reward_audit import RewardAudit
from fh5.evaluation.rewards import RewardReplay
from fh5.experiment import run_experiment
from fh5.learning.bc.legacy import BCReplay
from fh5.observation.multimodal import ObservationReplay
from fh5.observation.perception import Perception, PerceptionReplay
from fh5.observation.routes import BuildRoute, RouteCheck
from fh5.telemetry.packet import Packet, Record, Replay


def _sac(args: argparse.Namespace) -> int:
    from fh5.artifacts.io import sha256_file
    from fh5.learning.sac.actions import ActionBounds
    from fh5.learning.sac.critic import SACCriticReplay, SACCriticResume, SACCriticWarmup
    from fh5.learning.sac.replay import SACReplayPrepare
    from fh5.learning.sac.training import SACPolicyReplay, SACResume, SACTrain

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
            warmup = args.config.parent / options.pop("warmup")
            replay = (
                args.config.parent / options.pop("replay")
                if "replay" in options
                else warmup / "experience/replay.json"
            )
            if options.get("imitation_protocol_batch") is not None:
                options["imitation_protocol_batch"] = (
                    args.config.parent / options["imitation_protocol_batch"]
                )
            request = SACTrain(warmup, replay, args.output, **options)
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid SAC training fields") from error
        summary = run_experiment(request).summary["sac_learning"]
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if summary["stop_reason"] == "budget_completed" else 4
    if args.mode == "sac-policy-replay":
        replay = (
            args.replay if args.replay is not None else args.checkpoint / "experience/replay.json"
        )
        replayed = run_experiment(
            SACPolicyReplay(
                args.checkpoint, replay, args.report, raw_cache_bytes=args.raw_cache_bytes
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
                sha256_file(args.replay) if args.replay_sha256 is None else args.replay_sha256,
                args.output,
                args.steps,
                args.batch_size,
                args.learning_rate,
                args.seed,
                bounds,
            )
        )
    else:
        replay = (
            args.replay if args.replay is not None else args.checkpoint / "experience/replay.json"
        )
        result = run_experiment(SACCriticReplay(args.checkpoint, replay, args.report))
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
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "collection-prepare":
            from fh5.artifacts.io import read_bounded
            from fh5.capture.config import parse_capture_config
            from fh5.collection.model import CollectionConfig
            from fh5.collection.process import CollectionPrepare

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
            from fh5.collection.process import CollectionStart

            started = run_experiment(
                CollectionStart(args.bundle, live=args.live, output_dir=args.output)
            )
            print(json.dumps(started.summary["collection"]))
            return 0
        if args.mode == "collection-dataset":
            raise ValueError(
                "collection-dataset is retired; use collection-bc-prepare with a v2 "
                "configuration containing sources, selection rules and observation settings. "
                "Existing selections remain readable with collection-dataset-review."
            )
        if args.mode == "collection-dataset-review":
            from fh5.collection.dataset import CollectionDatasetReview

            selected = run_experiment(CollectionDatasetReview(args.dataset, args.report))
            print(json.dumps(selected.summary["collection_dataset"], ensure_ascii=False))
            return 0
        if args.mode in ("candidate-archive", "candidate-restore"):
            from fh5.evaluation.candidate_archive import CandidateArchive, CandidateRestore

            retained = run_experiment(
                CandidateArchive(args.checkpoint, args.output, args.checkpoint_sha256, args.reason)
                if args.mode == "candidate-archive"
                else CandidateRestore(args.archive, args.output, args.archive_sha256, args.reason)
            )
            result = retained.summary[args.mode.replace("-", "_")]
            print(json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=False))
            return 0
        if args.mode == "learning-storage-plan":
            from fh5.learning.storage import LearningStoragePlan

            storage = run_experiment(LearningStoragePlan(args.config, args.output)).summary[
                "storage"
            ]
            print(
                json.dumps({k: v for k, v in storage.items() if k != "files"}, ensure_ascii=False)
            )
            return 0 if storage["status"] == "within_budget" else 4
        if args.mode in ("candidate-record", "candidate-rollback", "candidate-history"):
            from fh5.evaluation.candidate_store import (
                CandidateHistory,
                CandidateRecord,
                CandidateRollback,
            )

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
            from fh5.evaluation.candidate_selection import CandidateCompare

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
            from fh5.commands.evaluation import evaluation_command

            return evaluation_command(args)
        if args.mode in ("evaluation-prepare", "evaluation-review"):
            from fh5.evaluation.prepare import EvaluationPrepare, EvaluationReview

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
            from fh5.artifacts.usage import RecordUsage

            usage = run_experiment(
                RecordUsage(
                    args.registry, tuple(args.recording), args.role, args.output, args.model_sha256
                )
            ).summary["evidence_usage"]
            print(json.dumps(usage, ensure_ascii=False))
            return 4 if usage["unidentified_recordings"] else 0
        if args.mode == "collection-bc-prepare":
            from fh5.collection.bc import CollectionBCPrepare

            prepared = run_experiment(CollectionBCPrepare(args.config, args.output))
            print(json.dumps(prepared.summary["collection_bc"], ensure_ascii=False))
            return 0
        if args.mode == "collection-bc-assess":
            from fh5.collection.assessment import CollectionBCAssess

            assessed = run_experiment(CollectionBCAssess(args.config, args.output))
            summary = assessed.summary["collection_assessment"]
            print(
                json.dumps(
                    {k: v for k, v in summary.items() if k != "decisions"}, ensure_ascii=False
                )
            )
            return 0
        if args.mode in ("collection-bc-train", "collection-bc-resume"):
            from fh5.learning.bc.schedule import ScheduledBCResume, ScheduledBCTrain

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
            from fh5.collection.model import CollectionControl, CollectionReview

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
            from fh5.commands.realtime import replay_command

            return replay_command(args)
        if args.mode == "realtime-shadow":
            from fh5.commands.realtime import shadow_command

            return shadow_command(args)
        if args.mode == "realtime-drive":
            from fh5.commands.numeric_drive import drive_command

            return drive_command(args)
        if args.mode == "capture-frame-times":
            from fh5.capture.trace import CaptureTraceReview

            result = run_experiment(
                CaptureTraceReview(args.recording, args.csv, args.report, args.pid, args.swap_chain)
            )
            print(json.dumps(result.summary["capture"]["game_frame_time"], ensure_ascii=False))
            return int(result.summary["capture"]["game_frame_time"]["frame_count"] == 0)
        if args.mode == "capture-dxgi":
            from fh5.commands.capture import capture_command

            return capture_command(args)
        if args.mode in ("temporal-prepare", "temporal-train", "temporal-replay"):
            return _temporal_command(args)
        if args.mode == "numeric-prepare":
            raise ValueError(
                "numeric-prepare is retired. Use temporal-prepare to import historical "
                "demonstrations for new temporal training, or collection-bc-prepare for new "
                "numeric recordings. Existing prepared packages remain readable with "
                "numeric-infer; frozen v1 models remain readable with bc-replay. "
                "Single-frame data cannot provide temporal training history."
            )
        if args.mode in ("numeric-infer", "numeric-replay"):
            return _numeric_command(args)
        if args.mode == "input-devices":
            from fh5.collection.demonstration_windows import input_devices

            print(json.dumps({"devices": input_devices(), "commands_sent": False}))
            return 0
        if args.mode == "policy":
            raise ValueError(
                "The encoded-image policy command is retired. Use realtime-drive with a numerical "
                "temporal BC model and configs/realtime-drive.example.json; old configurations "
                "are not interchangeable. Existing recordings remain readable with replay."
            )
        elif args.mode == "bc-train":
            raise ValueError(
                "The legacy bc-train command is retired. Use temporal-prepare to import "
                "compatible historical demonstrations, then temporal-train; use "
                "collection-bc-prepare and collection-bc-train for new numeric collection. "
                "This requires retraining, not converting old weights. Existing v1 models "
                "remain readable with bc-replay. See docs/archive/legacy-models.md."
            )
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
            from fh5.observation.segformer import SegformerRoadModel

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
            from fh5.capture.legacy_windows import LiveVisionEnvironment, WindowsColorFrames
            from fh5.driving.windows import WindowsDesktop

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
                        from fh5.collection.demonstration_windows import (
                            HumanInputEnvironment,
                            XInputReader,
                        )
                        from fh5.collection.demonstrations import _profile

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
            from fh5.driving.windows import LiveEnvironment, WindowsDesktop, XboxController

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
            from fh5.driving.event_windows import BoundedFrames, LiveEventEnvironment, WindowsFrames
            from fh5.driving.windows import WindowsDesktop, XboxController

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
    from fh5.learning.bc.importer import TemporalBCPrepare
    from fh5.learning.bc.training import TemporalBCReplay, TemporalBCTrain

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
    from fh5.learning.bc.actor import FrozenNumericActor
    from fh5.learning.bc.legacy_import import PreparedNumericSource
    from fh5.observation.numeric import NumericInfer, NumericReplay, PixelContract

    if args.mode == "numeric-infer":
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
            or not any(d.get("status") == "predicted" for d in decisions)
        )
        else 0
    )
