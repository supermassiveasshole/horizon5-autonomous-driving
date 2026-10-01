"""Frozen event assets and independent menu-to-policy start verification."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.collection_store import encode, read_bounded, write_file
from fh5.events import ScreenFrame, _pgm, _recognize, read_event, validate_event_file
from fh5.realtime_numeric_replay import read_realtime_recording

if TYPE_CHECKING:
    from fh5.experiment import RunResult


def event_payloads(path: Path) -> dict[str, bytes]:
    config = validate_event_file(path)
    payloads = {}
    signatures = config["event_run"]["signatures"]
    for i, patch in enumerate(p for name in sorted(signatures) for p in signatures[name]):
        name = f"assets/template-{i}.pgm"
        payloads["start/" + name] = read_bounded(path.parent / patch["template"], 16 * 1024**2)
        patch["template"] = name
    for i, original in enumerate(config["event_run"]["verification_evidence"]):
        name = f"assets/evidence-{i}"
        payloads["start/" + name] = read_bounded(path.parent / original, 16 * 1024**2)
        config["event_run"]["verification_evidence"][i] = name
    payloads["start/event.json"] = encode(config)
    return payloads


def _inventory(directory: Path) -> dict[str, str]:
    paths = sorted(
        p
        for p in directory.rglob("*")
        if p.suffix in (".json", ".jsonl", ".pgm") and p.name != "start-manifest.json"
    )
    if not 1 <= len(paths) <= 5000:
        raise ValueError("Preparation inventory exceeds its bounded size")
    result = {}
    for path in paths:
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Preparation asset is outside its recording")
        result[path.relative_to(directory).as_posix()] = hashlib.sha256(
            read_bounded(path, 128 * 1024**2)
        ).hexdigest()
    return result


def seal_start(
    directory: Path, batch_sha256: str, slot: str, recording_sha256: str, execution_sha256: str
) -> str:
    raw = encode(
        {
            "version": 1,
            "batch_sha256": batch_sha256,
            "slot_id": slot,
            "recording_sha256": recording_sha256,
            "execution_sha256": execution_sha256,
            "files": _inventory(directory),
        }
    )
    write_file(directory / "start-manifest.json", raw)
    return hashlib.sha256(raw).hexdigest()


def _stationary(sample: dict[str, Any], config: dict[str, Any], speed: float) -> bool:
    return bool(
        sample["is_race_on"]
        and sample["car_ordinal"] == config["expected_car_ordinal"]
        and sample["car_performance_index"] == config["expected_pi"]
        and math.isfinite(sample["speed_kmh"])
        and 0 <= sample["speed_kmh"] <= speed
        and math.dist(sample["position_m"], config["start_position_m"]) <= config["start_radius_m"]
    )


def _menu_ready(
    directory: Path, config: dict[str, Any], operation: str, samples: list[dict[str, Any]]
) -> tuple[dict[str, Any], int]:
    recorded = read_event(directory)
    summary, events = recorded["summary"], recorded["events"]
    if (
        summary.get("evidence_status") != "complete"
        or not summary.get("release_sent")
        or summary.get("operation") != operation
        or summary.get("stop_reason") != "ready"
    ):
        raise ValueError("Preparation did not finish a complete ready operation")
    kinds = [event["kind"] for event in events]
    allowed = {
        "asset",
        "prepare",
        "screen",
        "menu_action",
        "recovery_verified",
        "ready_verified",
        "released",
        "adapter_event",
        "run_ended",
    }
    if (
        any(kind not in allowed for kind in kinds)
        or kinds.count("released") != 1
        or kinds.count("ready_verified") != 1
        or not kinds.index("ready_verified") < kinds.index("released") < len(kinds) - 1
        or any(
            kind not in {"adapter_event", "asset", "run_ended"}
            for kind in kinds[kinds.index("released") + 1 :]
        )
        or events[kinds.index("released")]["received_monotonic_ns"]
        < events[kinds.index("ready_verified")]["received_monotonic_ns"]
    ):
        raise ValueError("Preparation lacks consistent final release evidence")
    templates = {
        p["template"]: _pgm(read_bounded(directory / p["template"], 16 * 1024**2))[2]
        for ps in config["signatures"].values()
        for p in ps
    }
    steps = config["start_steps" if operation == "start_ready" else "restart_steps"]
    screens: list[tuple[str, int, int]] = []
    step = 0
    after = next(e["received_monotonic_ns"] for e in events if e["kind"] == "prepare")
    last_capture = -1
    for event in events:
        if event["kind"] == "screen":
            captured = event.get("captured_ns")
            observed = event["received_monotonic_ns"]
            if (
                type(captured) is not int
                or not last_capture < captured <= observed
                or not 0 <= observed - captured <= 500_000_000
            ):
                raise ValueError("Preparation frame timing is missing, stale or repeated")
            width, height, pixels = _pgm(read_bounded(directory / event["path"], 16 * 1024**2))
            screen = _recognize(ScreenFrame(captured, width, height, pixels), config, templates)
            screens = (screens + [(screen, captured, observed)])[-2:]
            last_capture = captured
        elif event["kind"] == "menu_action":
            issued, returned = event.get("issued_ns"), event.get("returned_ns")
            if (
                step >= len(steps)
                or event["status"] != "sent"
                or event["button"] != steps[step]["button"]
                or len(screens) != 2
                or type(issued) is not int
                or type(returned) is not int
                or returned < issued
                or any(
                    s[0] != steps[step]["screen"]
                    or not after < s[1] <= s[2] <= issued
                    or issued - s[1] > 500_000_000
                    for s in screens
                )
            ):
                raise ValueError("Menu action lacks matching fresh visual evidence")
            after, step, screens = returned, step + 1, []
    checked = summary.get("ready_observed_ns")
    if (
        step != len(steps)
        or len(screens) != 2
        or type(checked) is not int
        or any(
            s[0] != "driving" or not after < s[1] <= s[2] <= checked or checked - s[1] > 500_000_000
            for s in screens
        )
        or not samples
    ):
        raise ValueError("No two fresh driving frames after the complete menu recipe")
    ready = samples[-1]
    if (
        not after < ready["received_monotonic_ns"] <= checked
        or checked - ready["received_monotonic_ns"] > 500_000_000
        or not _stationary(ready, config, 3)
    ):
        raise ValueError("No fresh stationary telemetry after the menu actions")
    return ready, checked


def review_start(
    binding: Any,
    *,
    ledger_dir: Path,
    batch_dir: Path,
    batch: dict[str, Any],
    slot: str,
    execution: dict[str, Any],
    execution_binding: Any,
    recording: RunResult,
    output: Path,
) -> dict[str, Any]:
    from fh5.experiment import Replay, run_experiment

    task = json.loads(read_bounded(batch_dir / "task.json", 1024**2))
    result: dict[str, Any] = {
        "slot_id": slot,
        "status": "not_required",
        "reasons": [],
        "scope": "automatic local start only; not whole-task validity",
    }
    if task["version"] != 2:
        return result
    result.update(status="quarantined")
    try:
        if not isinstance(binding, dict) or set(binding) != {"directory", "manifest_sha256"}:
            raise ValueError("Automatic task requires a preparation binding")
        directory = ledger_dir / binding["directory"]
        raw = read_bounded(directory / "start-manifest.json", 4 * 1024**2)
        if hashlib.sha256(raw).hexdigest() != binding["manifest_sha256"]:
            raise ValueError("Preparation manifest changed")
        manifest = json.loads(raw)
        if (
            manifest["version"] != 1
            or manifest["slot_id"] != slot
            or manifest["batch_sha256"]
            != hashlib.sha256(read_bounded(batch_dir / "batch.json", 4 * 1024**2)).hexdigest()
            or manifest["recording_sha256"]
            != recording.summary["attempt_review"]["source_hashes"]["packets"]
            or execution["status"] != "bound_diagnostic"
            or manifest["execution_sha256"] != execution_binding["manifest_sha256"]
            or manifest["files"] != _inventory(directory)
        ):
            raise ValueError("Preparation is not bound to this complete attempt and execution")
        expected = {
            name: read_bounded(batch_dir / name, 16 * 1024**2)
            for name in batch["files"]
            if name.startswith("start/")
        }
        if event_payloads(directory / "event-config.json") != expected:
            raise ValueError("Preparation event protocol differs from the frozen task")
        operation = read_event(directory)["summary"].get("operation")
        allowed = (
            ("start_ready", "restart_ready")
            if slot == batch["config"]["plan"][0]["id"]
            else ("restart_ready",)
        )
        if operation not in allowed:
            raise ValueError("Preparation used an unsupported ready operation")
        observed_config = validate_event_file(directory / "event-config.json")["event_run"]
        output.parent.mkdir(parents=True, exist_ok=True)
        prepared = run_experiment(Replay(directory, output))
        ready, checked = _menu_ready(directory, observed_config, operation, prepared.samples)
        if (
            prepared.metadata["source_kind"] != recording.metadata["source_kind"]
            or prepared.metadata["snapshot"] != batch["config"]["conditions"]["snapshot"]
        ):
            raise ValueError("Preparation source or conditions differ from the attempt")
        first = recording.samples[0]
        bound = task["automatic_start"]["handoff_timeout_s"] * 1e9
        if (
            not 0 < first["received_monotonic_ns"] - checked <= bound
            or not 0
            < (first["game_timestamp_ms"] - ready["game_timestamp_ms"]) % 2**32
            <= bound / 1e6
            or not _stationary(
                first, observed_config, batch["config"]["runtime"]["start_speed_kmh"]
            )
            or any(first["telemetry_controls"].values())
        ):
            raise ValueError("First driving telemetry does not confirm a fresh stationary handoff")
        report = read_realtime_recording(ledger_dir / execution_binding["directory"])
        commands = [
            c
            for c in report["commands"]
            if c["owner"] == "policy" and c["status"] == "sent" and any(c["sent"].values())
        ]
        if (
            not commands
            or not first["received_monotonic_ns"] <= commands[0]["issued_ns"] <= checked + bound
        ):
            raise ValueError("No acknowledged nonzero policy command within the handoff deadline")
        before_command = [
            s for s in recording.samples if s["received_monotonic_ns"] <= commands[0]["issued_ns"]
        ]
        if commands[0]["issued_ns"] - before_command[-1]["received_monotonic_ns"] > batch["config"][
            "runtime"
        ]["max_telemetry_age_ms"] * 1e6 or any(
            not _stationary(s, observed_config, batch["config"]["runtime"]["start_speed_kmh"])
            or any(s["telemetry_controls"].values())
            for s in before_command
        ):
            raise ValueError("Handoff state changed or expired before first policy command")
        if manifest["files"] != _inventory(directory):
            raise ValueError("Preparation changed during verification")
        result.update(
            status="verified",
            operation=operation,
            source_kind=prepared.metadata["source_kind"],
            ready_packet_ns=ready["received_monotonic_ns"],
            ready_observed_ns=checked,
            first_driving_packet_ns=first["received_monotonic_ns"],
            first_policy_command_ns=commands[0]["issued_ns"],
            manifest_sha256=binding["manifest_sha256"],
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as error:
        result["reasons"].append(str(error))
    return result
