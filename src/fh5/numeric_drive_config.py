"""Shared numerical shadow/driving configuration and pre-device qualification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from fh5.artifact_io import sha256_file
from fh5.capture_config import parse_capture_config
from fh5.numeric_actor import FrozenNumericActor
from fh5.numeric_images import DecisionActor
from fh5.realtime import RealtimeConfig, RealtimeRun
from fh5.realtime_model import ShadowNumericActor, sac_shadow_contract, shadow_model_contract
from fh5.realtime_numeric_replay import (
    read_realtime_decision,
    read_realtime_journal,
    read_realtime_recording,
)
from fh5.realtime_shadow import LocalTask
from fh5.realtime_udp import UDPTelemetry


class NumericDriveConfiguration:
    """Validate immutable inputs before constructing any native resource."""

    def __init__(
        self,
        config_file: Path,
        output_dir: Path,
        seconds: float,
        live: bool,
        *,
        mode: Literal["drive", "shadow", "legacy-shadow"] = "drive",
        hz: int | None = None,
    ) -> None:
        self.mode = mode
        raw = config_file.read_bytes()
        root = json.loads(raw)
        if (
            set(root)
            != {"version", "capture_config", "model", "task", "decision", "port", "shadow"}
            or root["version"] != 1
        ):
            raise ValueError(
                "Use the shared configs/realtime-drive.example.json configuration; "
                "old shadow-only configurations must add shadow: null"
            )
        base = config_file.parent
        capture_bytes = (base / root["capture_config"]).read_bytes()
        document = json.loads(capture_bytes)
        self.capture, self.target = parse_capture_config(document)
        if self.capture.pixels.origin != "direct_numeric":
            raise ValueError("Numerical driving requires direct numeric RGB")
        settings = dict(root["decision"])
        if "action_offsets_ms" in settings:
            settings["action_offsets_ms"] = tuple(settings["action_offsets_ms"])
        if hz is not None:
            settings["decision_hz"] = hz
        self.request = RealtimeRun(
            output_dir,
            RealtimeConfig(pixels=self.capture.pixels, **settings),
            seconds=seconds,
            live=live and mode == "drive",
        )
        if mode == "drive" and seconds > 30:
            raise ValueError("Numerical driving is limited to 30 seconds")
        task = dict(root["task"])
        task["route_file"] = base / task["route_file"]
        if "expected_route_sha256" not in task:
            task["expected_route_sha256"] = sha256_file(task["route_file"])
        self.task = LocalTask(**task)
        self.task.load()
        model = root["model"]
        self.model_kind = model.get("kind", "bc")
        allowed = {"directory", "device", "expected_sha256", "manifest_sha256"}
        if self.model_kind == "sac":
            allowed.update(("kind", "exploration_seed"))
        if (
            not {"directory", "device"} <= set(model) <= allowed
            or not all(
                isinstance(value, str) for key, value in model.items() if key != "exploration_seed"
            )
            or model["device"] not in ("cpu", "cuda")
        ):
            raise ValueError("Driving requires model directory/device and optional string hashes")
        self.model_dir = base / model["directory"]
        self.exploration_seed = model.get("exploration_seed")
        if self.model_kind == "sac":
            if mode != "shadow" or model["device"] != "cpu":
                raise ValueError(
                    "SAC requires read-only shadow on CPU with its exact source contract"
                )
            if "exploration_seed" in model and (
                type(self.exploration_seed) is not int or not 0 <= self.exploration_seed < 2**64
            ):
                raise ValueError("SAC exploration seed must be an unsigned 64-bit integer")
            self.metadata, self.model_hash = sac_shadow_contract(
                self.model_dir, self.capture.pixels, model.get("expected_sha256")
            )
            if "manifest_sha256" in model and model["manifest_sha256"] != self.model_hash:
                raise ValueError("Frozen SAC policy manifest changed")
            self.training_pixels = self.capture.pixels
        else:
            self.metadata, self.training_pixels, self.model_hash = shadow_model_contract(
                self.model_dir,
                self.capture.pixels,
                model.get("expected_sha256"),
                allow_legacy_source_diagnostic=mode == "legacy-shadow",
                expected_manifest_sha256=model.get("manifest_sha256"),
            )
        cfg = self.request.config
        if self.metadata["contract"]["actor_shape"] != {
            "action_count": len(cfg.action_offsets_ms),
            "reference_count": cfg.reference_count,
        }:
            raise ValueError("Driving actor history dimensions differ")
        self.device = model["device"]
        self.telemetry = UDPTelemetry(root["port"])
        self.bindings = {
            "drive_config_sha256" if mode == "drive" else "shadow_config_sha256": hashlib.sha256(
                raw
            ).hexdigest(),
            "capture_config_sha256": hashlib.sha256(capture_bytes).hexdigest(),
            "capture_target": asdict(self.target),
            "conditions": document["input_conditions"],
            "inference_device": self.device,
            "model_manifest_sha256": self.model_hash,
        }
        self.bindings = json.loads(json.dumps(self.bindings))
        self.qualification: dict[str, Any] = {
            "eligible": False,
            "reasons": ["read_only_mode"],
            "shadow": None,
        }
        if mode != "drive":
            return
        provenance = self.metadata.get("provenance", {})
        reasons = []
        if provenance.get("diagnostic_only") is not False:
            reasons.append("diagnostic_model")
        if provenance.get("kind") != "continuous_numeric_collection":
            reasons.append("unsupported_model_source")
        if provenance.get("input_conditions") != document["input_conditions"]:
            reasons.append("input_conditions_mismatch")
        if document["input_conditions"].get("status") != "confirmed":
            reasons.append("unconfirmed_input_conditions")
        if provenance.get("vehicle") != {
            "expected_car_ordinal": cfg.expected_car_ordinal,
            "expected_pi": cfg.expected_pi,
        }:
            reasons.append("vehicle_mismatch")
        if (
            provenance.get("config", {}).get("action_history_offsets_ms")
            != list(cfg.action_offsets_ms)
            or provenance.get("config", {}).get("max_action_age_ms") != 200
        ):
            reasons.append("action_history_mismatch")
        if self.metadata["contract"].get("action") != "xinput-lx-rt-lt-v1":
            reasons.append("action_contract_mismatch")
        if self.metadata["contract"].get("temporal", {}).get("mode") != "actual":
            reasons.append("actual_delta_time_model_required")
        if not self.metadata.get("training", {}).get("train_by_view", {}).get("no_reference", 0):
            reasons.append("no_reference_training_missing")
        shadow = None
        if root["shadow"] is None:
            reasons.append("missing_shadow_evidence")
        else:
            shadow, shadow_reasons = self._shadow(base, root["shadow"])
            reasons.extend(shadow_reasons)
        self.qualification = {
            "eligible": not reasons,
            "reasons": reasons,
            "shadow": shadow,
        }

    def _shadow(self, base: Path, entry: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        if (
            not isinstance(entry, dict)
            or not {"directory"} <= set(entry) <= {"directory", "manifest_sha256"}
            or not all(isinstance(value, str) for value in entry.values())
        ):
            raise ValueError("Driving requires a bound shadow recording manifest")
        root = base / entry["directory"]
        raw = (root / "realtime-manifest.json").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if "manifest_sha256" in entry and digest != entry["manifest_sha256"]:
            raise ValueError("Shadow evidence manifest changed")
        report = read_realtime_recording(root, expected_manifest_sha256=digest)
        read_realtime_journal(root, report)
        for row in report["decisions"]:
            if "actor" in row:
                read_realtime_decision(root, row, self.capture.pixels)
        reasons = []
        env = report["environment"]
        if (
            report["evidence_kind"] != "shadow"
            or env.get("capture", {}).get("source_kind") != "dxgi"
            or report["commands_sent_to_game"]
        ):
            reasons.append("shadow_not_native")
        if report["actor_kind"] != ShadowNumericActor.kind or any(
            report["model"].get(key) != value for key, value in self._candidate_identity().items()
        ):
            reasons.append("shadow_candidate_mismatch")
        expected = {**asdict(self.request.config), "pixels": self.capture.pixels.metadata()}
        if report["configuration"] != json.loads(json.dumps(expected)):
            reasons.append("shadow_decision_configuration_mismatch")
        bindings = env.get("input_conditions", {})
        for key in (
            "capture_config_sha256",
            "capture_target",
            "conditions",
            "inference_device",
            "model_manifest_sha256",
        ):
            if bindings.get(key) != json.loads(json.dumps(self.bindings[key])):
                reasons.append("shadow_" + key + "_mismatch")
        if report["inference"].get("inference_device") != self.device:
            reasons.append("shadow_worker_device_mismatch")
        task = env.get("task", {})
        if any(
            task.get(key) != value
            for key, value in {
                "route_sha256": self.task.expected_route_sha256,
                "start_station_m": self.task.start_station_m,
                "start_tolerance_m": self.task.start_tolerance_m,
                "end_margin_m": self.task.end_margin_m,
            }.items()
        ):
            reasons.append("shadow_task_mismatch")
        if (
            not report["resources_released"]
            or not report["evidence"]["recording_complete"]
            or not report["evidence"]["exact_replay_eligible"]
        ):
            reasons.append("shadow_incomplete")
        if report["stop_reason"] not in ("time_limit", "local_end"):
            reasons.append("shadow_stopped_on_fault")
        accepted = [d for d in report["decisions"] if d["status"] == "accepted"]
        # Require an observed sustained interval, not a single favorable call.
        span = (
            (accepted[-1]["decision_ns"] - accepted[0]["decision_ns"]) / 1e9
            if len(accepted) >= 2
            else 0
        )
        if span < 1:
            reasons.append("shadow_insufficient_active_interval")
        cfg = self.request.config
        if any(
            d["inference_returned_ns"] > d["deadline_ns"]
            or not 0
            <= d["inference_returned_ns"] - d["frames"][-1]["source_time_ns"]
            <= cfg.max_image_age_ms * 1_000_000
            for d in accepted
        ) or any(
            b["decision_ns"] - a["decision_ns"] >= cfg.watchdog_ms * 1_000_000
            for a, b in zip(accepted, accepted[1:])
        ):
            reasons.append("shadow_timing_outside_runtime_budget")
        return {
            "directory": str(root),
            "manifest_sha256": digest,
            "report_sha256": json.loads(raw)["report_sha256"],
            "accepted_decisions": len(accepted),
            "active_span_s": span,
            "reasons": reasons,
        }, reasons

    def actor(self) -> DecisionActor:
        if self.model_kind == "sac":
            from fh5.sac_evaluation_actor import SACEvaluationActor
            from fh5.sac_sampling_actor import SACSamplingActor

            if self.exploration_seed is not None:
                return SACSamplingActor(
                    self.model_dir,
                    self.capture.pixels,
                    self.model_hash,
                    exploration_seed=self.exploration_seed,
                    counterfactual=True,
                )
            return SACEvaluationActor(
                self.model_dir, self.capture.pixels, self.model_hash, counterfactual=True
            )
        if self.mode != "drive":
            return ShadowNumericActor(
                self.model_dir,
                self.capture.pixels,
                self.metadata["weights_sha256"],
                self.device,
                allow_legacy_source_diagnostic=self.mode == "legacy-shadow",
                expected_manifest_sha256=self.model_hash,
            )
        return FrozenNumericActor(
            self.model_dir,
            self.capture.pixels,
            self.device,
            expected_manifest_sha256=self.model_hash,
        )

    def require_eligible(self) -> None:
        if not self.qualification["eligible"]:
            raise ValueError(
                "Numerical driving not qualified: " + ", ".join(self.qualification["reasons"])
            )

    def _candidate_identity(self) -> dict[str, Any]:
        return {
            "weights_sha256": self.metadata["weights_sha256"],
            "numeric_contract": self.capture.pixels.metadata(),
            "model_contract": self.metadata["contract"],
            "provenance": self.metadata["provenance"],
        }

    def authorize(
        self, request: RealtimeRun, manifest: dict[str, Any], inference_device: str | None
    ) -> None:
        self.require_eligible()
        if request != self.request or not request.live:
            raise ValueError("Driving request differs from qualified configuration")
        if inference_device != self.device:
            raise ValueError("Loaded inference device differs from qualified timing evidence")
        if any(manifest.get(key) != value for key, value in self._candidate_identity().items()):
            raise ValueError("Loaded driving candidate differs from qualified model")
