"""Public experiment interface: dispatch workflows and compose recording evidence."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from fh5.artifacts.io import WriteFile
from fh5.artifacts.usage import RecordUsage, record_usage
from fh5.capture.legacy import VisionEnvironment, VisionRecord, read_vision, run_vision
from fh5.capture.pipeline import CaptureReplay, CaptureRun, replay_capture
from fh5.capture.runtime import CaptureSource, run_capture
from fh5.capture.trace import CaptureTraceReview, review_capture_trace
from fh5.collection.assessment import CollectionBCAssess, assess_collection_bc
from fh5.collection.bc import CollectionBCPrepare, prepare_collection_bc
from fh5.collection.dataset import (
    CollectionDatasetReview,
    review_collection_dataset,
)
from fh5.collection.demonstration_dataset import DemonstrationDataset, export_demonstrations
from fh5.collection.demonstrations import (
    DemonstrationRecord,
    DemonstrationReplay,
    record_demonstration,
    replay_demonstration,
)
from fh5.collection.model import (
    CollectionControl,
    CollectionEnvironment,
    CollectionReview,
    CollectionRun,
)
from fh5.collection.process import (
    CollectionInstaller,
    CollectionPrepare,
    CollectionStart,
    control_collection,
    prepare_collection,
    start_collection,
)
from fh5.collection.review import review_collection
from fh5.collection.runtime import collect
from fh5.driving.control import Control, ControlEnvironment, read_control, run_control
from fh5.driving.events import EventEnvironment, EventRun, read_event, run_event
from fh5.driving.policy_recording import read_policy
from fh5.driving.realtime.model import (
    RealtimeEnvironment,
    RealtimeNumericReplay,
    RealtimeReplay,
    RealtimeRun,
)
from fh5.driving.realtime.numeric_replay import replay_realtime_numeric
from fh5.driving.realtime.replay import replay_realtime
from fh5.driving.realtime.runtime import run_realtime
from fh5.driving.recovery import RecoveryReplay, replay_recovery
from fh5.driving.tracking import TrackingDrive, read_tracking_route, run_tracking
from fh5.evaluation.attempts import AttemptReplay, review_attempts
from fh5.evaluation.candidate_archive import (
    CandidateArchive,
    CandidateRestore,
    archive_candidate,
    restore_candidate,
)
from fh5.evaluation.candidate_selection import CandidateCompare, compare_candidates
from fh5.evaluation.candidate_store import (
    CandidateHistory,
    CandidateRecord,
    CandidateRollback,
    read_candidate_history,
    record_candidate,
    rollback_candidate,
)
from fh5.evaluation.prepare import (
    EvaluationPrepare,
    EvaluationReview,
    prepare_evaluation,
    review_evaluation,
)
from fh5.evaluation.reward_audit import RewardAudit, audit_rewards
from fh5.evaluation.rewards import RewardReplay, settle_rewards
from fh5.evaluation.run import EvaluationEnvironment, EvaluationRun, run_evaluation
from fh5.learning.bc.importer import TemporalBCPrepare, prepare_temporal
from fh5.learning.bc.legacy import BCReplay
from fh5.learning.bc.legacy_training import run_offline
from fh5.learning.bc.schedule import (
    LearningResources,
    ScheduledBCResume,
    ScheduledBCTrain,
    run_scheduled_bc,
)
from fh5.learning.bc.training import TemporalBCReplay, TemporalBCTrain, run_temporal_bc
from fh5.learning.loop.runner import (
    LearningContinue,
    LearningEnvironment,
    LearningLoop,
    run_learning_loop,
)
from fh5.learning.sac.critic import SACCriticReplay, SACCriticResume, SACCriticWarmup, run_critic
from fh5.learning.sac.cycle import SACCycle, SACEnvironment, SACRealtimeCycle, run_sac_cycle
from fh5.learning.sac.realtime_experience import SACRealtimePrepare, prepare_realtime_experience
from fh5.learning.sac.realtime_sampler import SACRealtimeEnvironment
from fh5.learning.sac.replay import SACReplayPrepare, prepare_sac_replay
from fh5.learning.sac.training import (
    SACPolicyReplay,
    SACResume,
    SACTrain,
    run_sac_policy_replay,
    run_sac_training,
)
from fh5.learning.storage import LearningStoragePlan, plan_learning_storage
from fh5.observation.multimodal import ObservationReplay, build_observations, read_settings
from fh5.observation.numeric import (
    ContextualNumericActor,
    DecisionActor,
    NumericDecision,
    NumericInfer,
    NumericReplay,
)
from fh5.observation.perception import (
    Perception,
    PerceptionReplay,
    RoadModel,
    replay_perception,
    run_perception,
)
from fh5.observation.recording import infer_numeric, replay_numeric
from fh5.observation.routes import (
    BuildRoute,
    RouteCheck,
    build_route,
    check_route_recording,
    load_route,
    locate_route,
)
from fh5.reporting.telemetry import write_report
from fh5.result import RunResult as RunResult
from fh5.telemetry.packet import DIAGNOSTICS as DIAGNOSTICS
from fh5.telemetry.packet import Packet as Packet
from fh5.telemetry.packet import Record as Record
from fh5.telemetry.packet import Replay as Replay
from fh5.telemetry.recording import run_recording


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
    | CollectionDatasetReview
    | CollectionBCPrepare
    | CollectionBCAssess
    | ScheduledBCTrain
    | ScheduledBCResume
    | TemporalBCPrepare
    | TemporalBCTrain
    | TemporalBCReplay
    | NumericInfer
    | NumericReplay
    | AttemptReplay
    | RewardReplay
    | SACReplayPrepare
    | SACRealtimePrepare
    | SACRealtimeCycle
    | SACCriticWarmup
    | SACCriticReplay
    | SACCriticResume
    | SACTrain
    | SACCycle
    | LearningLoop
    | LearningContinue
    | LearningStoragePlan
    | SACResume
    | SACPolicyReplay
    | RewardAudit
    | RecoveryReplay
    | TrackingDrive
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
    sac_realtime_environment: SACRealtimeEnvironment | None = None,
    evaluation_environment: EvaluationEnvironment | None = None,
    learning_environment: LearningEnvironment | None = None,
    collection_write: WriteFile | None = None,
    collection_installer: CollectionInstaller | None = None,
) -> RunResult:
    """Run one record/replay operation; injected packets are the environment seam."""
    if isinstance(request, LearningStoragePlan):
        return plan_learning_storage(request)
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
    if isinstance(request, (ScheduledBCTrain, ScheduledBCResume)):
        return run_scheduled_bc(request, learning_resources)
    if isinstance(request, CollectionDatasetReview):
        return review_collection_dataset(request)
    if isinstance(request, (TemporalBCTrain, TemporalBCReplay)):
        return run_temporal_bc(request)
    if isinstance(request, (NumericInfer, NumericReplay)):
        if numeric_actor is None:
            raise ValueError("Numerical inference requires an explicit frozen actor")
        if isinstance(numeric_actor, ContextualNumericActor):
            raise ValueError("Contextual actors require the real-time decision interface")
        if isinstance(request, NumericReplay):
            return replay_numeric(request, numeric_actor)
        if numeric_inputs is None:
            raise ValueError("Numerical inference requires an explicit prepared input source")
        return infer_numeric(request, numeric_actor, numeric_inputs)
    if isinstance(request, AttemptReplay):
        return review_attempts(request)
    if isinstance(request, RewardReplay):
        return settle_rewards(request)
    if isinstance(request, SACReplayPrepare):
        return prepare_sac_replay(request)
    if isinstance(request, SACRealtimePrepare):
        if numeric_actor is None:
            raise ValueError("Asynchronous SAC preparation requires the frozen sampling actor")
        return prepare_realtime_experience(request, numeric_actor)
    if isinstance(request, SACRealtimeCycle):
        if sac_realtime_environment is None:
            raise ValueError("SAC realtime cycle requires an explicit external environment")
        return run_sac_cycle(request, sac_realtime_environment, sac_stop_requested)
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
    if isinstance(request, BCReplay):
        return run_offline(request)
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
        result = run_recording(request, packets=packets)
        directory = request.output_dir
    else:
        result = run_recording(Replay(request.recording_dir, request.report_path))
        directory = request.recording_dir
    metadata = result.metadata
    samples = result.samples
    events = result.events
    summary = result.summary
    report_path = result.report_path
    packet_count = summary["packet_count"]
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
