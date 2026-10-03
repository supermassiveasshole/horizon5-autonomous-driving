"""Serial native learning leases with fresh qualification for each frozen candidate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from fh5.artifact_io import sha256_file
from fh5.collection_store import encode, write_file
from fh5.evaluation_native import NativeEvaluationEnvironment
from fh5.evaluation_native_io import native_event_environment
from fh5.evaluation_run import EvaluationRun, evaluation_inputs
from fh5.evaluation_start import event_payloads
from fh5.events import EventEnvironment
from fh5.learning_io import LearningUnavailable
from fh5.numeric_drive_cli import native_driving_environment
from fh5.numeric_drive_config import NumericDriveConfiguration
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig, RealtimeEnvironment, RealtimeRun
from fh5.realtime_shadow import ShadowEnvironment
from fh5.sac_native import NativeSACSamplingEnvironment
from fh5.sac_realtime_sampler import SACRealtimeStart, _AttemptDrive


class _NativeCandidateEvaluation:
    source_kind: Literal["native"] = "native"

    def __init__(self, identity: str, preparation: NativeSACSamplingEnvironment) -> None:
        self.identity, self.preparation = identity, preparation
        self.configuration: Path | None = None
        self.environment: NativeEvaluationEnvironment | None = None
        self.released: dict[str, Any] | None = None

    def prepare(self, request: EvaluationRun, batch: dict[str, Any]) -> dict[str, Any]:
        return self.prepare_stopped(request, batch, lambda: False)

    def prepare_stopped(
        self,
        request: EvaluationRun,
        batch: dict[str, Any],
        stopped: Callable[[], bool],
    ) -> dict[str, Any]:
        if self.environment is not None or self.released is not None:
            raise ValueError("Native candidate evaluation already prepared or closed")
        if stopped():
            raise InterruptedError("Native evaluation stopped before qualification")
        frozen, _, _ = evaluation_inputs(request)
        config = frozen["config"]
        task = json.loads((request.batch_dir / "task.json").read_bytes())
        if (
            not request.live
            or request.seconds > 30
            or frozen != batch
            or frozen["version"] != 3
            or config["model"]["kind"] != "sac"
            or config["model"]["device"] != "cpu"
            or task["version"] != 2
            or event_payloads(request.event_config_file) != self.preparation.event_assets
        ):
            raise ValueError("Native learning evaluation requires its frozen CPU SAC protocol")
        settings = dict(config["runtime"])
        settings["pixels"] = PixelContract.from_metadata(settings["pixels"])
        settings["action_offsets_ms"] = tuple(settings["action_offsets_ms"])
        runtime = RealtimeConfig(**settings)
        # EvaluationRun creates its output only after this preparation returns.
        base = request.output_dir.with_name(request.output_dir.name + "-preparation")
        root, attempt = base, 0
        while True:
            try:
                root.mkdir(parents=True)
                break
            except FileExistsError:
                # A stopped check is evidence to preserve, not an evaluation
                # attempt to repeat or an output directory to overwrite.
                attempt += 1
                root = base.with_name(f"{base.name}-{attempt:03d}")
        recording = root / "record.json"
        write_file(
            recording,
            encode(
                {
                    "schema_version": 1,
                    "control_source": "policy",
                    "snapshot": config["conditions"]["snapshot"],
                }
            ),
        )
        start = SACRealtimeStart(
            self.identity,
            RealtimeRun(root / "execution", runtime, request.seconds, live=True),
            request.batch_dir / "model",
            config["model"]["manifest_sha256"],
            0,
            recording,
            request.batch_dir / "task.json",
            stopped,
        )
        try:
            self.preparation.prepare(
                start,
                exploratory=False,
                expected_conditions=config["conditions"]["numeric_input_conditions"],
            )
            if stopped():
                raise InterruptedError("Native evaluation stopped after qualification")
            self.configuration = root / "drive.json"

            def driving(plan: NumericDriveConfiguration) -> RealtimeEnvironment:
                return _AttemptDrive(self.preparation.driving_factory(plan), stopped)

            self.environment = NativeEvaluationEnvironment(
                self.configuration,
                menu_factory=self.preparation.menu_factory,
                driving_factory=driving,
            )
            return self.environment.prepare(request, batch)
        except (Exception, KeyboardInterrupt) as error:
            released = self.close()
            if isinstance(error, LearningUnavailable):
                error.resources_released &= released["resources_released"]
                raise
            raise LearningUnavailable(
                str(error), resources_released=released["resources_released"]
            ) from error

    def event(self, slot_id: str) -> EventEnvironment:
        if self.environment is None or self.released is not None:
            raise ValueError("Native candidate evaluation is not prepared")
        return self.environment.event(slot_id)

    def driving(self, slot_id: str, ready_state: dict[str, Any]) -> RealtimeEnvironment:
        if self.environment is None or self.released is not None:
            raise ValueError("Native candidate evaluation is not prepared")
        return self.environment.driving(slot_id, ready_state)

    def close(self) -> dict[str, Any]:
        if self.released is not None:
            return self.released
        result = self.preparation.close()
        errors = list(result["errors"])
        released = result["resources_released"]
        sent = result["commands_sent_to_game"]
        menu_sends = result["menu_sends"]
        if self.environment is not None:
            try:
                outcome = self.environment.close()
                released &= outcome["resources_released"]
                menu_sends += outcome["menu_sends"]
                errors.extend(outcome["errors"])
                sent |= any(
                    isinstance(drive, _AttemptDrive)
                    and (drive.released or {}).get("controller_sends", 0) > 0
                    for drive in self.environment.drives
                )
            except (Exception, KeyboardInterrupt) as error:
                released = False
                errors.append(str(error))
            finally:
                self.environment.drives.clear()
                self.environment.menus.clear()
        self.released = {
            "resources_released": released and not errors,
            "commands_sent_to_game": sent or menu_sends > 0,
            "menu_sends": menu_sends,
            "errors": errors,
        }
        return self.released


class NativeLearningEnvironment:
    source_kind: Literal["native"] = "native"

    def __init__(
        self,
        configuration: Path,
        event_config: Path,
        *,
        shadow_seconds: float,
        handoff_timeout_s: float,
        review: Callable[[Path], Path | None] | None = None,
        shadow_factory: Callable[[NumericDriveConfiguration], RealtimeEnvironment] = (
            ShadowEnvironment.from_native
        ),
        menu_factory: Callable[[Path, NumericDriveConfiguration], EventEnvironment] = (
            native_event_environment
        ),
        driving_factory: Callable[[NumericDriveConfiguration], RealtimeEnvironment] = (
            native_driving_environment
        ),
    ) -> None:
        self.configuration = configuration.resolve()
        self.event_config = event_config.resolve()
        self.shadow_seconds, self.handoff_timeout_s = shadow_seconds, handoff_timeout_s
        self.reviewer = review
        self.shadow_factory, self.menu_factory, self.driving_factory = (
            shadow_factory,
            menu_factory,
            driving_factory,
        )
        self.expected_protocol = self.protocol
        self.active: NativeSACSamplingEnvironment | _NativeCandidateEvaluation | None = None
        self.resources_released = True
        self.commands_sent = False
        self.errors: list[str] = []

    @property
    def protocol(self) -> dict[str, Any]:
        raw = self.configuration.read_bytes()
        config = json.loads(raw)
        return {
            "configuration_sha256": hashlib.sha256(raw).hexdigest(),
            "capture_config_sha256": sha256_file(
                self.configuration.parent / config["capture_config"]
            ),
            "route_sha256": sha256_file(self.configuration.parent / config["task"]["route_file"]),
            "event_files": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in event_payloads(self.event_config).items()
            },
            "shadow_seconds": self.shadow_seconds,
            "handoff_timeout_s": self.handoff_timeout_s,
        }

    def _preparation(self) -> NativeSACSamplingEnvironment:
        if not self.close()["resources_released"]:
            raise LearningUnavailable(
                "Previous native learning lease remains open", resources_released=False
            )
        if self.protocol != self.expected_protocol:
            raise ValueError("Native learning deployment configuration changed")
        return NativeSACSamplingEnvironment(
            self.configuration,
            self.event_config,
            shadow_seconds=self.shadow_seconds,
            handoff_timeout_s=self.handoff_timeout_s,
            review=self.reviewer,
            shadow_factory=self.shadow_factory,
            menu_factory=self.menu_factory,
            driving_factory=self.driving_factory,
        )

    def sampling(self, identity: str) -> NativeSACSamplingEnvironment:
        source = self._preparation()
        self.active = source
        return source

    def evaluation(self, identity: str) -> _NativeCandidateEvaluation:
        source = _NativeCandidateEvaluation(identity, self._preparation())
        self.active = source
        return source

    def review(self, recording_dir: Path) -> Path | None:
        if not self.close()["resources_released"]:
            raise ValueError("Native learning resources remain open before independent review")
        return self.reviewer(recording_dir) if self.reviewer else None

    def close(self) -> dict[str, Any]:
        if self.active is not None:
            source, self.active = self.active, None
            try:
                result = source.close()
                self.resources_released &= result.get("resources_released") is True
                self.commands_sent |= result.get("commands_sent_to_game", False)
                self.errors.extend(result.get("errors", []))
            except (Exception, KeyboardInterrupt) as error:
                self.resources_released = False
                self.errors.append(str(error))
        return {
            "resources_released": self.resources_released and not self.errors,
            "commands_sent_to_game": self.commands_sent,
            "errors": list(self.errors),
        }
