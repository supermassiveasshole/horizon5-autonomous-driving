"""Bind numerical execution evidence to one frozen batch and recorded packet stream."""

from __future__ import annotations

import hashlib
import importlib
import json
from bisect import bisect_left
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.collection_store import read_bounded
from fh5.evaluation_metrics import execution_metrics
from fh5.learning_runtime import preserve_torch_state
from fh5.numeric_images import NumericActor, PixelContract
from fh5.realtime import RealtimeNumericReplay
from fh5.realtime_numeric_replay import read_realtime_journal, read_realtime_recording

if TYPE_CHECKING:
    from fh5.experiment import RunResult


def _bind_telemetry(root: Path, report: dict[str, Any], source: Path, recording: RunResult) -> int:
    packets = read_realtime_journal(root, report)
    original = [
        json.loads(line)
        for line in read_bounded(source / "packets.jsonl", 128 * 1024**2).splitlines()
    ]
    if not packets or packets != original:
        raise ValueError("Execution packet stream differs from complete attempt recording")
    expected_source = "synthetic" if report["evidence_kind"] == "synthetic" else "udp"
    if recording.metadata["source_kind"] != expected_source:
        raise ValueError("Execution and packet source kinds differ")
    samples = {s["received_monotonic_ns"]: s for s in recording.samples}
    if len(samples) != len(recording.samples):
        raise ValueError("Ambiguous telemetry timestamps")
    for row in report["decisions"]:
        if "actor" not in row:
            continue
        sample = samples.get(row["telemetry_received_ns"])
        if sample is None or row["actor"]["ego"] != {
            "speed_mps": sample["speed_mps"],
            "velocity_car_mps": sample["motion"]["velocity_car_mps"],
            "angular_velocity_car_radps": sample["motion"]["angular_velocity_car_radps"],
        }:
            raise ValueError("Execution observation differs from linked telemetry")
        if (
            row["telemetry_received_ns"] > row["decision_ns"]
            or row["actor"]["ego_age_ms"]
            != (row["decision_ns"] - row["telemetry_received_ns"]) / 1e6
        ):
            raise ValueError("Execution telemetry age is not causal")
        safety = row["safety_at_decision"]
        current = samples.get(safety["received_ns"])
        if (
            current is None
            or safety["received_ns"] > row["decision_ns"]
            or safety["game_timestamp_ms"] != current["game_timestamp_ms"]
            or safety["car_ordinal"] != current["car_ordinal"]
            or safety["pi"] != current["car_performance_index"]
            or safety["speed_kmh"] != current["speed_kmh"]
            or safety["active"] != bool(current["is_race_on"])
            or (
                safety["telemetry_packet_index"] is not None
                and safety["telemetry_packet_index"] != current["packet_index"]
            )
        ):
            raise ValueError("Execution safety state differs from linked telemetry")
    return len(packets)


def _verify_commands(report: dict[str, Any]) -> None:
    accepted = {d["decision_id"]: d for d in report["decisions"] if d["status"] == "accepted"}
    seen = set()
    config = report["configuration"]
    neutral = {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0}
    if (
        not report["commands"]
        or report["commands"][-1]["owner"] != "hard_stop"
        or report["commands"][-1]["sent"] != neutral
    ):
        raise ValueError("Execution lacks a recorded final input release")
    for decision in report["decisions"]:
        if (
            decision["deadline_ns"]
            != decision["decision_ns"] + config["inference_deadline_ms"] * 1_000_000
            or decision["valid_until_ns"]
            != decision["decision_ns"] + config["action_lease_ms"] * 1_000_000
        ):
            raise ValueError("Execution command timing differs from frozen bounds")
    for command in report["commands"]:
        if command["status"] != "sent" or command["target"] != command["sent"]:
            raise ValueError("Execution command target and successful send differ")
        if command["owner"] != "policy":
            if (
                command["owner"] not in ("hard_stop", "lease_expiry")
                or command["decision_id"] is not None
                or command["sent"] != neutral
            ):
                raise ValueError("Unrecognized non-policy execution command")
            continue
        decision_id = command["decision_id"]
        if decision_id not in accepted or decision_id in seen:
            raise ValueError("Execution policy command lacks a unique accepted decision")
        seen.add(decision_id)
        decision = accepted[decision_id]
        clock = [
            decision["decision_ns"],
            decision["worker_started_ns"],
            decision["worker_returned_ns"],
            decision["inference_returned_ns"],
            command["issued_ns"],
        ]
        if any(type(at) is not int for at in clock) or any(a > b for a, b in zip(clock, clock[1:])):
            raise ValueError("Execution inference and command timing is not causal")
        steer, longitudinal = decision["prediction"]
        expected = {
            "steer_i16": round(max(-config["max_steer"], min(config["max_steer"], steer)) * 32767),
            "throttle_u8": round(max(0, min(config["max_throttle"], longitudinal)) * 255),
            "brake_u8": round(max(0, min(config["max_brake"], -longitudinal)) * 255),
        }
        if (
            command["sent"] != expected
            or command["valid_until_ns"] != decision["valid_until_ns"]
            or not decision["decision_ns"] <= command["issued_ns"] < decision["deadline_ns"]
            or command["returned_ns"] > decision["valid_until_ns"]
        ):
            raise ValueError(
                "Execution command contradicts frozen prediction, envelope or deadline"
            )
    if seen != accepted.keys():
        raise ValueError("Accepted decision has no successful execution command")


def _verify_history(report: dict[str, Any]) -> None:
    commands = sorted(report["commands"], key=lambda command: command["returned_ns"])
    returned = [command["returned_ns"] for command in commands]
    for row in report["decisions"]:
        if "actor" not in row:
            continue
        actions: list[list[float] | None] = []
        ages: list[float | None] = []
        for offset in report["configuration"]["action_offsets_ms"]:
            target = row["decision_ns"] - offset * 1_000_000
            index = bisect_left(returned, target) - 1
            if (
                report["evidence_kind"] == "synthetic"
                and index >= 0
                and target - returned[index] <= 200_000_000
            ):
                sent = commands[index]["sent"]
                actions.append(
                    [sent["steer_i16"] / 32767, (sent["throttle_u8"] - sent["brake_u8"]) / 255]
                )
                ages.append((row["decision_ns"] - returned[index]) / 1e6)
            else:
                actions.append(None)
                ages.append(None)
        if (
            row["actor"]["actions"] != actions
            or row["actor"]["action_mask"] != [action is not None for action in actions]
            or row["actor"]["action_age_ms"] != ages
        ):
            raise ValueError("Execution action history differs from recorded successful sends")


def review_execution(
    binding: dict[str, Any] | None,
    *,
    ledger_dir: Path,
    batch_dir: Path,
    batch: dict[str, Any],
    source_dir: Path,
    recording: RunResult,
    reference_mode: str,
    output: Path,
) -> dict[str, Any]:
    from fh5.experiment import run_experiment
    from fh5.numeric_actor import FrozenNumericActor
    from fh5.realtime_model import ShadowNumericActor

    result: dict[str, Any] = {
        "status": "missing",
        "reasons": ["execution_not_supplied"],
        "verified_predictions": 0,
        "observed_reference_modes": [],
        "metrics": None,
        "recorded_gaps": None,
        "game_control_verified": False,
        "replay": None,
    }
    if binding is None:
        return result
    result.update(status="quarantined", reasons=[])
    try:
        if set(binding) != {"directory", "manifest_sha256"}:
            raise ValueError("Incomplete execution binding")
        root = ledger_dir / binding["directory"]
        manifest = root / "realtime-manifest.json"
        if hashlib.sha256(read_bounded(manifest, 4096)).hexdigest() != binding["manifest_sha256"]:
            raise ValueError("Execution manifest changed")
        report = read_realtime_recording(root)
        result["recorded_gaps"] = {
            "basis": "hash-bound recording declarations; successful validation reported separately",
            "journal_dropped": report["journal"]["dropped"],
            "journal_missing_sequences": len(report["journal"]["missing_sequences"]),
            "archive_error": report["archive"]["error"],
            "missing_input_archives": sum(
                "actor" in d and (not d.get("archive") or d.get("archive_reason") is not None)
                for d in report["decisions"]
            ),
        }
        if report["configuration"] != batch["config"]["runtime"]:
            raise ValueError("Execution runtime differs from frozen configuration")
        result["linked_packets"] = _bind_telemetry(root, report, source_dir, recording)
        contract = PixelContract.from_metadata(batch["config"]["runtime"]["pixels"])
        with preserve_torch_state(importlib.import_module("torch")):
            actor: NumericActor
            if report["actor_kind"] == ShadowNumericActor.kind:
                model = json.loads(read_bounded(batch_dir / "model/model.json", 128 * 1024**2))
                actor = ShadowNumericActor(batch_dir / "model", contract, model["weights_sha256"])
            else:
                actor = FrozenNumericActor(batch_dir / "model", contract)
            replay = run_experiment(
                RealtimeNumericReplay(root, output), numeric_actor=actor
            ).summary["realtime_numeric_replay"]
        result.update(
            replay=output.name,
            verified_predictions=replay["verified_predictions"],
            source_manifest_sha256=binding["manifest_sha256"],
        )
        if not replay["verified"]:
            raise ValueError("Execution numerical replay failed: " + str(replay["errors"]))
        _verify_commands(report)
        _verify_history(report)
        modes = {
            "reference_assisted" if any(d["actor"]["reference"]["mask"]) else "no_reference"
            for d in report["decisions"]
            if "actor" in d
        }
        result["observed_reference_modes"] = sorted(modes)
        if modes != {reference_mode}:
            raise ValueError("Observed execution reference modes differ from frozen plan")
        if hashlib.sha256(read_bounded(manifest, 4096)).hexdigest() != binding["manifest_sha256"]:
            raise ValueError("Execution manifest changed during review")
        result.update(
            status="bound_diagnostic",
            reasons=["synthetic_or_shadow_only", "game_action_timing_unverified"],
            metrics=execution_metrics(report),
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result["reasons"].append(str(error))
    return result
