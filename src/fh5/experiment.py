"""Public experiment interface: dispatch workflows and compose recording evidence."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from fh5.artifacts.io import WriteFile
from fh5.artifacts.usage import RecordUsage, record_usage
from fh5.capture.legacy import VisionEnvironment, VisionRecord, run_vision
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
from fh5.driving.control import Control, ControlEnvironment, run_control
from fh5.driving.events import EventEnvironment, EventRun, run_event
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
from fh5.driving.tracking import TrackingDrive, run_tracking
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
from fh5.observation.multimodal import ObservationReplay
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
)
from fh5.reporting.recording import run_recording_report
from fh5.result import RunResult as RunResult
from fh5.telemetry.packet import DIAGNOSTICS as DIAGNOSTICS
from fh5.telemetry.packet import Packet as Packet
from fh5.telemetry.packet import Record as Record
from fh5.telemetry.packet import Replay as Replay


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
    return run_recording_report(request, packets=packets)
