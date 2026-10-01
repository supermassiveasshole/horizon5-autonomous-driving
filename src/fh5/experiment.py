"""The experiment-run interface for recording and replaying external telemetry."""

from __future__ import annotations

import json
import math
import struct
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fh5.attempts import AttemptReplay, review_attempts
from fh5.bc import BCReplay, BCTrain, run_bc
from fh5.candidate_archive import (
    CandidateArchive,
    CandidateRestore,
    archive_candidate,
    restore_candidate,
)
from fh5.candidate_selection import CandidateCompare, compare_candidates
from fh5.candidate_store import (
    CandidateHistory,
    CandidateRecord,
    CandidateRollback,
    read_candidate_history,
    record_candidate,
    rollback_candidate,
)
from fh5.capture import CaptureReplay, CaptureRun, replay_capture
from fh5.capture_runtime import CaptureSource, run_capture
from fh5.capture_trace import CaptureTraceReview, review_capture_trace
from fh5.collection import CollectionControl, CollectionEnvironment, CollectionReview, CollectionRun
from fh5.collection_assessment import CollectionBCAssess, assess_collection_bc
from fh5.collection_bc import CollectionBCPrepare, prepare_collection_bc
from fh5.collection_dataset import (
    CollectionDataset,
    CollectionDatasetReview,
    run_collection_dataset,
)
from fh5.collection_process import (
    CollectionInstaller,
    CollectionPrepare,
    CollectionStart,
    prepare_collection,
    start_collection,
)
from fh5.collection_review import review_collection
from fh5.collection_runtime import collect, control_collection
from fh5.collection_store import WriteFile
from fh5.control import Control, ControlEnvironment, read_control, run_control
from fh5.demonstration_dataset import DemonstrationDataset, export_demonstrations
from fh5.demonstrations import (
    DemonstrationRecord,
    DemonstrationReplay,
    record_demonstration,
    replay_demonstration,
)
from fh5.evaluation import (
    EvaluationPrepare,
    EvaluationReview,
    prepare_evaluation,
    review_evaluation,
)
from fh5.evaluation_run import EvaluationEnvironment, EvaluationRun, run_evaluation
from fh5.events import EventEnvironment, EventRun, read_event, run_event
from fh5.evidence_usage import RecordUsage, record_usage
from fh5.learning_loop import LearningContinue, LearningEnvironment, LearningLoop, run_learning_loop
from fh5.learning_schedule import LearningResources, ScheduledBCTrain, run_scheduled_bc
from fh5.numeric_images import (
    ContextualNumericActor,
    DecisionActor,
    NumericDecision,
    NumericInfer,
    NumericReplay,
    run_numeric,
)
from fh5.numeric_import import LegacyNumericImport, prepare_legacy
from fh5.observations import ObservationReplay, build_observations, read_settings
from fh5.perception import (
    Perception,
    PerceptionReplay,
    RoadModel,
    replay_perception,
    run_perception,
)
from fh5.policy import PolicyActor, PolicyDrive, PolicyEnvironment, read_policy, run_policy
from fh5.realtime import RealtimeEnvironment, RealtimeNumericReplay, RealtimeReplay, RealtimeRun
from fh5.realtime_numeric_replay import replay_realtime_numeric
from fh5.realtime_replay import replay_realtime
from fh5.realtime_runtime import run_realtime
from fh5.recovery import RecoveryReplay, replay_recovery
from fh5.report import write_report
from fh5.reward_audit import RewardAudit, audit_rewards
from fh5.rewards import RewardReplay, settle_rewards
from fh5.routes import (
    BuildRoute,
    RouteCheck,
    build_route,
    check_route_recording,
    load_route,
    locate_route,
)
from fh5.sac import SACCriticReplay, SACCriticResume, SACCriticWarmup, run_critic
from fh5.sac_cycle import SACCycle, SACEnvironment, run_sac_cycle
from fh5.sac_learning import (
    SACPolicyReplay,
    SACResume,
    SACTrain,
    run_sac_policy_replay,
    run_sac_training,
)
from fh5.sac_replay import SACReplayPrepare, prepare_sac_replay
from fh5.temporal_bc import TemporalBCReplay, TemporalBCTrain, run_temporal_bc
from fh5.temporal_import import TemporalBCPrepare, prepare_temporal
from fh5.tracking import TrackingDrive, read_tracking_route, run_tracking
from fh5.vision import VisionEnvironment, VisionRecord, read_vision, run_vision

FORMAT_VERSION = 1
DECODER_VERSION = "fh5-dash-324-v2"
DIAGNOSTICS = {
    "version": 1,
    "receive_gap_seconds": 0.5,
    "jump_slack_metres": 20.0,
    "jump_speed_metres_per_second": 200.0,
}
SNAPSHOT_FIELDS = ("vehicle", "variant", "tune", "assists", "event", "environment")


@dataclass(frozen=True)
class Packet:
    received_monotonic_ns: int
    received_utc: str
    payload: bytes


@dataclass(frozen=True)
class Record:
    config_file: Path
    output_dir: Path
    source_kind: Literal["udp", "synthetic"] = "synthetic"


@dataclass(frozen=True)
class Replay:
    recording_dir: Path
    report_path: Path
    route_file: Path | None = None


@dataclass(frozen=True)
class RunResult:
    metadata: dict[str, Any]
    samples: list[dict[str, Any]]
    events: list[dict[str, Any]]
    summary: dict[str, Any]
    report_path: Path


def _decode(packet: Packet) -> dict[str, Any]:
    if len(packet.payload) != 324:
        raise ValueError(f"Unsupported FH5 packet length: {len(packet.payload)} (expected 324)")
    data = packet.payload
    sample = {
        "received_monotonic_ns": packet.received_monotonic_ns,
        "received_utc": packet.received_utc,
        "game_timestamp_ms": struct.unpack_from("<I", data, 4)[0],
        "is_race_on": struct.unpack_from("<i", data, 0)[0],
        "position_m": list(struct.unpack_from("<fff", data, 244)),
        "speed_mps": struct.unpack_from("<f", data, 256)[0],
        "speed_kmh": struct.unpack_from("<f", data, 256)[0] * 3.6,
        "car_ordinal": struct.unpack_from("<i", data, 212)[0],
        "car_class": struct.unpack_from("<i", data, 216)[0],
        "car_performance_index": struct.unpack_from("<i", data, 220)[0],
        "telemetry_controls": {
            "accel": data[315],
            "brake": data[316],
            "steer": struct.unpack_from("<b", data, 320)[0],
        },
        "command": None,
    }
    values = [*sample["position_m"], sample["speed_mps"]]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite position or speed")
    if sample["is_race_on"] not in (0, 1):
        raise ValueError("IsRaceOn must be 0 or 1")
    velocity = list(struct.unpack_from("<fff", data, 32))
    angular = list(struct.unpack_from("<fff", data, 44))
    yaw = struct.unpack_from("<f", data, 56)[0]
    motion_valid = (
        all(math.isfinite(v) for v in [*velocity, *angular, yaw]) and abs(yaw) <= math.pi + 1e-5
    )
    sample["motion"] = (
        {"yaw_rad": yaw, "velocity_car_mps": velocity, "angular_velocity_car_radps": angular}
        if motion_valid
        else None
    )
    sample["motion_status"] = "decoded" if motion_valid else "invalid_motion"
    return sample


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _read_packet(line: bytes) -> Packet:
    row = json.loads(line)
    if not isinstance(row, dict):
        raise ValueError("Record must be a JSON object")
    if type(row.get("received_monotonic_ns")) is not int or row["received_monotonic_ns"] < 0:
        raise ValueError("Record requires a nonnegative integer receive time")
    if not isinstance(row.get("received_utc"), str) or not isinstance(row.get("payload_hex"), str):
        raise ValueError("Record requires received_utc and payload_hex strings")
    timestamp = datetime.fromisoformat(row["received_utc"])
    offset = timestamp.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("received_utc must include a UTC offset of zero")
    return Packet(
        row["received_monotonic_ns"], row["received_utc"], bytes.fromhex(row["payload_hex"])
    )


def _validate_config(config: object) -> dict[str, Any]:
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int:
        raise ValueError("Config must be an object with integer schema_version")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported config schema_version")
    if config.get("control_source") not in ("human", "unknown", "calibration", "policy"):
        raise ValueError("Unknown control_source")
    snapshot = config.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("Config requires a snapshot object")
    for name in set(SNAPSHOT_FIELDS) | set(snapshot):
        fact = snapshot.get(name)
        if not isinstance(fact, dict) or fact.get("status") not in (
            "unverified",
            "user_reported",
            "verified",
        ):
            raise ValueError(f"snapshot.{name} requires value and verification status")
        if "value" not in fact or (
            fact["value"] is not None and not isinstance(fact["value"], str)
        ):
            raise ValueError(f"snapshot.{name}.value must be text or null")
        if fact["status"] == "verified" and (
            not fact["value"]
            or not isinstance(fact.get("evidence"), str)
            or not fact["evidence"].strip()
        ):
            raise ValueError(f"snapshot.{name}: verified facts require a value and evidence")
    json.dumps(config, allow_nan=False)
    return config


def run_experiment(
    request: RealtimeReplay
    | CandidateArchive
    | CandidateRestore
    | CandidateCompare
    | CandidateHistory
    | CandidateRecord
    | CandidateRollback
    | EvaluationPrepare
    | EvaluationReview
    | EvaluationRun
    | RecordUsage
    | RealtimeRun
    | RealtimeNumericReplay
    | CollectionRun
    | CollectionPrepare
    | CollectionStart
    | CollectionReview
    | CollectionControl
    | CaptureReplay
    | CaptureRun
    | CaptureTraceReview
    | CollectionDataset
    | CollectionDatasetReview
    | CollectionBCPrepare
    | CollectionBCAssess
    | ScheduledBCTrain
    | TemporalBCPrepare
    | TemporalBCTrain
    | TemporalBCReplay
    | LegacyNumericImport
    | NumericInfer
    | NumericReplay
    | PolicyDrive
    | AttemptReplay
    | RewardReplay
    | SACReplayPrepare
    | SACCriticWarmup
    | SACCriticReplay
    | SACCriticResume
    | SACTrain
    | SACCycle
    | LearningLoop
    | LearningContinue
    | SACResume
    | SACPolicyReplay
    | RewardAudit
    | RecoveryReplay
    | TrackingDrive
    | BCTrain
    | BCReplay
    | DemonstrationRecord
    | DemonstrationDataset
    | DemonstrationReplay
    | Record
    | Replay
    | Control
    | EventRun
    | BuildRoute
    | RouteCheck
    | VisionRecord
    | ObservationReplay
    | Perception
    | PerceptionReplay,
    *,
    packets: Iterable[Packet] | None = None,
    environment: ControlEnvironment | None = None,
    event_environment: EventEnvironment | None = None,
    vision_environment: VisionEnvironment | None = None,
    road_model: RoadModel | None = None,
    policy_environment: PolicyEnvironment | None = None,
    policy_actor: PolicyActor | None = None,
    numeric_inputs: Iterable[NumericDecision] | None = None,
    numeric_actor: DecisionActor | None = None,
    capture_source_factory: Callable[[], CaptureSource] | None = None,
    capture_activity: Callable[[], dict[str, Any] | None] | None = None,
    capture_resources: Callable[[], dict[str, Any]] | None = None,
    realtime_environment: RealtimeEnvironment | None = None,
    numeric_actor_factory: Callable[[], DecisionActor] | None = None,
    realtime_journal_sink: Callable[[bytes], None] | None = None,
    collection_environment: CollectionEnvironment | None = None,
    learning_resources: LearningResources | None = None,
    sac_stop_requested: Callable[[int], bool] | None = None,
    sac_environment: SACEnvironment | None = None,
    evaluation_environment: EvaluationEnvironment | None = None,
    learning_environment: LearningEnvironment | None = None,
    collection_write: WriteFile | None = None,
    collection_installer: CollectionInstaller | None = None,
) -> RunResult:
    """Run one record/replay operation; injected packets are the environment seam."""
    if isinstance(request, (LearningLoop, LearningContinue)):
        if learning_environment is None:
            raise ValueError("Learning loop requires an explicit synthetic environment")
        return run_learning_loop(request, learning_environment)
    if isinstance(request, CandidateArchive):
        return archive_candidate(request)
    if isinstance(request, CandidateRestore):
        return restore_candidate(request)
    if isinstance(request, CandidateCompare):
        return compare_candidates(request)
    if isinstance(request, CandidateRecord):
        return record_candidate(request)
    if isinstance(request, CandidateRollback):
        return rollback_candidate(request)
    if isinstance(request, CandidateHistory):
        return read_candidate_history(request)
    if isinstance(request, EvaluationPrepare):
        return prepare_evaluation(request)
    if isinstance(request, EvaluationReview):
        return review_evaluation(request)
    if isinstance(request, RecordUsage):
        return record_usage(request)
    if isinstance(request, EvaluationRun):
        if evaluation_environment is None:
            raise ValueError("Repeated evaluation requires an explicit environment")
        return run_evaluation(request, evaluation_environment)
    if isinstance(request, CollectionPrepare):
        return prepare_collection(request, collection_installer)
    if isinstance(request, CollectionStart):
        return start_collection(request)
    if isinstance(request, CollectionControl):
        return control_collection(request)
    if isinstance(request, CollectionReview):
        return review_collection(request)
    if isinstance(request, CollectionRun):
        if collection_environment is None:
            raise ValueError("Collection requires an explicit passive input environment")
        return collect(request, collection_environment, collection_write)
    if isinstance(request, RealtimeReplay):
        return replay_realtime(request)
    if isinstance(request, RealtimeNumericReplay):
        if numeric_actor is None:
            raise ValueError("Numerical replay requires an explicit frozen actor")
        return replay_realtime_numeric(request, numeric_actor)
    if isinstance(request, RealtimeRun):
        if realtime_environment is None or numeric_actor_factory is None:
            raise ValueError(
                "Real-time experiment requires an environment and frozen actor factory"
            )
        return run_realtime(
            request, realtime_environment, numeric_actor_factory, realtime_journal_sink
        )
    if isinstance(request, CaptureReplay):
        return replay_capture(request)
    if isinstance(request, CaptureTraceReview):
        return review_capture_trace(request)
    if isinstance(request, CaptureRun):
        if capture_source_factory is None:
            raise ValueError("Capture requires an explicit passive source factory")
        return run_capture(request, capture_source_factory, capture_activity, capture_resources)
    if isinstance(request, TemporalBCPrepare):
        return prepare_temporal(request)
    if isinstance(request, CollectionBCPrepare):
        return prepare_collection_bc(request)
    if isinstance(request, CollectionBCAssess):
        return assess_collection_bc(request)
    if isinstance(request, ScheduledBCTrain):
        return run_scheduled_bc(request, learning_resources)
    if isinstance(request, (CollectionDataset, CollectionDatasetReview)):
        return run_collection_dataset(request)
    if isinstance(request, (TemporalBCTrain, TemporalBCReplay)):
        return run_temporal_bc(request)
    if isinstance(request, LegacyNumericImport):
        return prepare_legacy(request)
    if isinstance(request, (NumericInfer, NumericReplay)):
        if numeric_actor is None:
            raise ValueError("Numerical inference requires an explicit frozen actor")
        if isinstance(numeric_actor, ContextualNumericActor):
            raise ValueError("Contextual actors require the real-time decision interface")
        return run_numeric(request, numeric_actor, numeric_inputs)
    if isinstance(request, PolicyDrive):
        if policy_environment is None:
            raise ValueError("Policy execution requires an explicit game environment")
        return run_policy(request, policy_environment, policy_actor)
    if isinstance(request, AttemptReplay):
        return review_attempts(request)
    if isinstance(request, RewardReplay):
        return settle_rewards(request)
    if isinstance(request, SACReplayPrepare):
        return prepare_sac_replay(request)
    if isinstance(request, SACCycle):
        if sac_environment is None:
            raise ValueError("SAC cycle requires an explicit synthetic environment")
        return run_sac_cycle(request, sac_environment, sac_stop_requested)
    if isinstance(request, (SACCriticWarmup, SACCriticReplay, SACCriticResume)):
        return run_critic(request, sac_stop_requested)
    if isinstance(request, (SACTrain, SACResume)):
        return run_sac_training(request, sac_stop_requested)
    if isinstance(request, SACPolicyReplay):
        return run_sac_policy_replay(request)
    if isinstance(request, RewardAudit):
        return audit_rewards(request)
    if isinstance(request, RecoveryReplay):
        return replay_recovery(request)
    if isinstance(request, TrackingDrive):
        if environment is None:
            raise ValueError("TrackingDrive requires an external game environment")
        return run_tracking(request, environment)
    if isinstance(request, (BCTrain, BCReplay)):
        return run_bc(request)
    if isinstance(request, DemonstrationDataset):
        return export_demonstrations(request)
    if isinstance(request, DemonstrationReplay):
        return replay_demonstration(request)
    if isinstance(request, DemonstrationRecord):
        if vision_environment is None:
            raise ValueError("DemonstrationRecord requires a passive input environment")
        return record_demonstration(request, vision_environment)
    if isinstance(request, PerceptionReplay):
        return replay_perception(request)
    if isinstance(request, Perception):
        if road_model is None:
            raise ValueError("Perception requires a frozen external road model")
        return run_perception(request, road_model)
    if isinstance(request, VisionRecord):
        if vision_environment is None:
            raise ValueError("VisionRecord requires a passive observation environment")
        return run_vision(request, vision_environment)
    if isinstance(request, EventRun):
        if event_environment is None:
            raise ValueError("EventRun requires an external game environment")
        return run_event(request, event_environment)
    if isinstance(request, Control):
        if environment is None:
            raise ValueError("Control requires an external game environment")
        return run_control(request, environment)
    if isinstance(request, Record):
        if packets is None:
            raise ValueError("A record run requires an external packet source")
        config = _validate_config(json.loads(request.config_file.read_text(encoding="utf-8-sig")))
        if request.source_kind not in ("udp", "synthetic"):
            raise ValueError("Unknown packet source_kind")
        metadata = {
            "format_version": FORMAT_VERSION,
            "decoder_version": DECODER_VERSION,
            "diagnostics": DIAGNOSTICS.copy(),
            "created_utc": datetime.now(UTC).isoformat(),
            "source_kind": request.source_kind,
            "game_validation": "unverified",
            "control_source": config["control_source"],
            "snapshot": config["snapshot"],
            "capture_status": "recording",
        }
        request.output_dir.mkdir(parents=True, exist_ok=False)
        if request.source_kind == "udp":
            metadata["capture_start_monotonic_ns"] = time.perf_counter_ns()
        _write_json(request.output_dir / "session.json", metadata)
        try:
            with (request.output_dir / "packets.jsonl").open("w", encoding="utf-8") as target:
                for packet in packets:
                    target.write(
                        json.dumps(
                            {
                                "received_monotonic_ns": packet.received_monotonic_ns,
                                "received_utc": packet.received_utc,
                                "payload_hex": packet.payload.hex(),
                            }
                        )
                        + "\n"
                    )
                    target.flush()
        except KeyboardInterrupt:
            metadata["capture_status"] = "interrupted"
        except OSError as error:
            metadata["capture_status"] = "source_error"
            metadata["capture_error"] = str(error)
        else:
            metadata["capture_status"] = "completed"
        finally:
            close = getattr(packets, "close", None)
            if close is not None:
                close()
        metadata["ended_utc"] = datetime.now(UTC).isoformat()
        if request.source_kind == "udp":
            metadata["capture_end_monotonic_ns"] = time.perf_counter_ns()
        _write_json(request.output_dir / "session.json", metadata)
        directory = request.output_dir
        report_path = directory / "report.html"
    else:
        directory = request.recording_dir
        report_path = request.report_path
        metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("session.json must be an object")
        if (
            type(metadata.get("format_version")) is not int
            or metadata["format_version"] != FORMAT_VERSION
        ):
            raise ValueError("Unsupported session format_version")
        if metadata.get("decoder_version") not in ("fh5-dash-324-v1", DECODER_VERSION):
            raise ValueError("Unsupported session decoder_version")
        if metadata.get("diagnostics") != DIAGNOSTICS:
            raise ValueError("Unsupported session diagnostics")
        _validate_config(
            {
                "schema_version": 1,
                "control_source": metadata.get("control_source"),
                "snapshot": metadata.get("snapshot"),
            }
        )
        if metadata.get("source_kind") not in ("udp", "synthetic"):
            raise ValueError("Unknown session source_kind")
        if metadata.get("capture_status") not in (
            "recording",
            "completed",
            "interrupted",
            "source_error",
        ):
            raise ValueError("Unknown session capture_status")
        if metadata.get("game_validation") != "unverified" or not isinstance(
            metadata.get("created_utc"), str
        ):
            raise ValueError("Session requires created_utc and unverified game_validation")

    metadata["analysis_decoder_version"] = DECODER_VERSION
    samples: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    capture_status = metadata.get("capture_status", "recording")
    if capture_status != "completed":
        events.append(
            {
                "kind": "capture_" + capture_status,
                "detail": metadata.get("capture_error", "Capture did not reach its time limit"),
                "packet_index": None,
            }
        )
    segment = 0
    break_pending = False
    previous_receive: int | None = None
    first_receive: int | None = None
    packet_count = 0
    lines = (directory / "packets.jsonl").read_bytes().splitlines(keepends=True)
    for packet_index, line in enumerate(lines):
        packet_count += 1
        try:
            packet = _read_packet(line)
        except (ValueError, UnicodeError) as error:
            tail = packet_index == len(lines) - 1 and not line.endswith(b"\n")
            events.append(
                {
                    "kind": "incomplete_tail" if tail else "corrupt_record",
                    "packet_index": packet_index,
                    "detail": str(error),
                }
            )
            break_pending = True
            continue
        reasons: list[tuple[str, str]] = []
        if first_receive is None:
            first_receive = packet.received_monotonic_ns
        if previous_receive is not None:
            receive_delta = (packet.received_monotonic_ns - previous_receive) / 1e9
            if receive_delta > DIAGNOSTICS["receive_gap_seconds"]:
                reasons.append(("receive_gap", f"No datagram for {receive_delta:.3f} seconds"))
            elif receive_delta <= 0:
                reasons.append(("receive_clock_discontinuity", "Receive clock did not advance"))
        previous_receive = packet.received_monotonic_ns
        try:
            sample = _decode(packet)
        except ValueError as error:
            kind = "unsupported_packet" if len(packet.payload) != 324 else "invalid_value"
            reasons.append((kind, str(error)))
            sample = None
        if sample is not None and samples:
            previous = samples[-1]
            game_delta = sample["game_timestamp_ms"] - previous["game_timestamp_ms"]
            if game_delta < 0:
                wrapped = (
                    previous["game_timestamp_ms"] > 0xFFFF0000
                    and sample["game_timestamp_ms"] < 0x10000
                )
                if wrapped:
                    reasons.append(("game_clock_wrap", "Game uint32 timer crossed its boundary"))
                else:
                    reasons.append(("game_time_discontinuity", "Game time moved backwards"))
            if sample["is_race_on"] != previous["is_race_on"]:
                kind = "resumed" if sample["is_race_on"] else "paused"
                reasons.append((kind, "IsRaceOn changed; this is not a finish or rewind signal"))
            dt = (packet.received_monotonic_ns - previous["received_monotonic_ns"]) / 1e9
            displacement = math.dist(sample["position_m"], previous["position_m"])
            if (
                sample["is_race_on"]
                and previous["is_race_on"]
                and displacement
                > DIAGNOSTICS["jump_slack_metres"]
                + DIAGNOSTICS["jump_speed_metres_per_second"] * max(0.0, dt)
            ):
                reasons.append(("position_jump", f"Position changed by {displacement:.1f} metres"))
        for kind, detail in reasons:
            events.append(
                {
                    "kind": kind,
                    "detail": detail,
                    "packet_index": packet_index,
                    "received_monotonic_ns": packet.received_monotonic_ns,
                }
            )
        if sample is None:
            break_pending = True
            continue
        if samples and (break_pending or reasons):
            segment += 1
        sample["segment"] = segment
        sample["packet_index"] = packet_index
        samples.append(sample)
        break_pending = False
    for boundary, clock, received in (
        ("start", metadata.get("capture_start_monotonic_ns"), first_receive),
        ("end", metadata.get("capture_end_monotonic_ns"), previous_receive),
    ):
        if type(clock) is not int or received is None:
            continue
        duration = (received - clock if boundary == "start" else clock - received) / 1e9
        if duration > DIAGNOSTICS["receive_gap_seconds"]:
            events.append(
                {
                    "kind": "receive_gap",
                    "packet_index": None,
                    "boundary": boundary,
                    "duration_seconds": duration,
                    "detail": f"No datagram for {duration:.3f} seconds at capture {boundary}",
                }
            )
    if not samples:
        events.append(
            {
                "kind": "no_telemetry",
                "packet_index": None,
                "detail": "No supported, valid telemetry packets were received",
            }
        )
    receive_span = sum(
        max(0, b["received_monotonic_ns"] - a["received_monotonic_ns"]) / 1e9
        for a, b in zip(samples, samples[1:])
    )
    summary = {
        "capture_status": capture_status,
        "receive_span_seconds": receive_span,
        "packet_count": packet_count,
        "valid_packets": len(samples),
        "active_packets": sum(s["is_race_on"] for s in samples),
        "invalid_packets": packet_count - len(samples),
        "segments": segment + bool(samples),
    }
    if metadata.get("control_source") == "calibration" or (directory / "control.json").exists():
        try:
            summary["control"] = read_control(directory, samples)
        except ValueError as error:
            events.append(
                {"kind": "control_evidence_incomplete", "packet_index": None, "detail": str(error)}
            )
    if (directory / "policy.json").exists():
        summary["policy"] = read_policy(directory)
        summary["route"] = summary["policy"]["evaluation_route"]
    if summary.get("control", {}).get("controller_kind") == "route-feedback-v1":
        try:
            summary["route"] = read_tracking_route(directory, summary["control"])
            events.extend(locate_route(samples, summary["route"]))
        except (OSError, ValueError, KeyError) as error:
            events.append(
                {"kind": "tracking_route_incomplete", "packet_index": None, "detail": str(error)}
            )
    if any(
        (directory / name).exists()
        for name in ("event-run.json", "event-config.json", "event-journal.jsonl")
    ):
        event_run = read_event(directory)
        summary["event_run"] = event_run["summary"]
        events.extend(event_run["events"])
    if isinstance(request, (BuildRoute, RouteCheck)):
        route = (
            build_route(request, samples)
            if isinstance(request, BuildRoute)
            else load_route(request.route_file)
        )
        recording_packet_count = packet_count
        samples = [
            s for s in samples if request.first_packet <= s["packet_index"] <= request.last_packet
        ]
        if not samples or request.last_packet >= packet_count:
            raise ValueError("Local analysis packet range is outside the recording")
        events = [
            e
            for e in events
            if e.get("packet_index") is not None
            and request.first_packet <= e["packet_index"] <= request.last_packet
        ]
        metadata["analysis_packet_range"] = [request.first_packet, request.last_packet]
        selected_count = request.last_packet - request.first_packet + 1
        summary.update(
            packet_count=selected_count,
            valid_packets=len(samples),
            active_packets=sum(s["is_race_on"] for s in samples),
            invalid_packets=selected_count - len(samples),
            segments=len({s["segment"] for s in samples}),
            receive_span_seconds=(
                samples[-1]["received_monotonic_ns"] - samples[0]["received_monotonic_ns"]
            )
            / 1e9,
        )
        events.extend(locate_route(samples, route))
        summary["route"] = route
        if isinstance(request, RouteCheck):
            summary["route_check"] = check_route_recording(
                request, samples, route, metadata, recording_packet_count
            )
    elif (
        isinstance(request, (Replay, ObservationReplay))
        and request.route_file is not None
        and (
            not isinstance(request, ObservationReplay)
            or read_settings(request.config_file)["version"] == 1
        )
    ):
        route = load_route(request.route_file)
        events.extend(locate_route(samples, route))
        summary["route"] = route
    if (directory / "vision-session.json").exists():
        summary["vision"] = read_vision(directory, report_path, samples, events)
    if isinstance(request, ObservationReplay):
        summary["observations"] = build_observations(
            request, samples, events, summary.get("vision"), summary.get("route")
        )
    write_report(
        report_path,
        {"metadata": metadata, "samples": samples, "events": events, "summary": summary},
    )
    return RunResult(metadata, samples, events, summary, report_path)
