"""Retain complete learning candidates without changing their evaluated identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from fh5.collection_store import encode, read_bounded, write_file
from fh5.numeric_images import asset

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class CandidateArchive:
    checkpoint_dir: Path
    output_dir: Path
    expected_checkpoint_sha256: str
    reason: str


@dataclass(frozen=True)
class CandidateRestore:
    archive_dir: Path
    output_dir: Path
    expected_archive_sha256: str
    reason: str


def _paths(source: Path, expected: str) -> tuple[dict[str, Any], set[str]]:
    raw = read_bounded(source / "policy.json", 1024**2)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Candidate differs from its expected checkpoint identity")
    manifest = json.loads(raw)
    if manifest.get("version") not in (2, 3, 4) or manifest.get("stage") != "sac_updates":
        raise ValueError("Archive requires a complete SAC continuation checkpoint")
    names = {
        "policy.json",
        "policy.pt",
        "training-report.json",
        "bc/model.json",
        "bc/actor.pt",
        "experience/replay.json",
    }
    for entry in manifest["history"]:
        names.update((entry["checkpoint"], entry["report"]))
    for entry in manifest.get("imitation", {}).get("transitions", []):
        names.add(entry["review"])
    replay = json.loads(read_bounded(source / "experience/replay.json", 128 * 1024**2))
    names.update("experience/" + entry["path"] for entry in replay.get("source_inventory", []))
    for row in replay["transitions"]:
        for observation in (row["current"], row["next"]):
            if observation is not None:
                names.update("experience/" + frame["path"] for frame in observation["frames"])
    return manifest, names


def archive_candidate(request: CandidateArchive) -> RunResult:
    from fh5.experiment import RunResult, run_experiment
    from fh5.sac_learning import SACResume

    source, output = request.checkpoint_dir.resolve(), request.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Candidate archive must be outside its source")
    if not isinstance(request.reason, str) or not 1 <= len(request.reason.strip()) <= 2000:
        raise ValueError("Candidate archive requires a reason of 1..2000 characters")
    manifest, names = _paths(source, request.expected_checkpoint_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="fh5-candidate-", dir=output.parent) as temporary:
        root = Path(temporary)
        frozen = root / "checkpoint"
        inventory: dict[str, Any] = {}
        total = 0
        for name in sorted(names):
            raw = read_bounded(asset(source, name), 256 * 1024**2)
            total += len(raw)
            if total > 1024**3 or len(inventory) >= 50000:
                raise ValueError("Candidate archive exceeds its 1 GiB / 50000 file capacity")
            target = asset(frozen, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_file(target, raw)
            inventory[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        verified = run_experiment(
            SACResume(
                frozen,
                root / "probe",
                steps=0,
                expected_checkpoint_sha256=request.expected_checkpoint_sha256,
            )
        ).summary["sac_learning"]
        summary = {
            "version": 1,
            "checkpoint_sha256": request.expected_checkpoint_sha256,
            "learner_state_sha256": verified["learner_state_sha256"],
            "checkpoint_version": manifest["version"],
            "total_steps": verified["total_steps"],
            "reason": request.reason.strip(),
            "files": inventory,
            "bytes": total,
            "default_changed": False,
            "driving_qualification": "not_established_by_archive",
        }
        output.mkdir()
        for name in sorted(names, key=lambda value: (value == "policy.json", value)):
            target = asset(output / "checkpoint", name)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_file(target, read_bounded(asset(frozen, name), 256 * 1024**2))
        payload = encode(summary)
        if len(payload) > 16 * 1024**2:
            raise ValueError("Candidate archive manifest exceeds capacity")
        write_file(output / "archive.json", payload)
    result = {**summary, "archive_sha256": hashlib.sha256(payload).hexdigest()}
    return RunResult({}, [], [], {"candidate_archive": result}, output / "archive.json")


def restore_candidate(request: CandidateRestore) -> RunResult:
    from fh5.experiment import RunResult, run_experiment
    from fh5.sac_learning import SACResume

    source, output = request.archive_dir.resolve(), request.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Candidate restore must be outside its archive")
    if not isinstance(request.reason, str) or not 1 <= len(request.reason.strip()) <= 2000:
        raise ValueError("Candidate restore requires a reason of 1..2000 characters")
    payload = read_bounded(source / "archive.json", 16 * 1024**2)
    if hashlib.sha256(payload).hexdigest() != request.expected_archive_sha256:
        raise ValueError("Candidate archive manifest changed")
    manifest = json.loads(payload)
    if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported candidate archive")
    _, names = _paths(source / "checkpoint", manifest["checkpoint_sha256"])
    if names != manifest["files"].keys():
        raise ValueError("Candidate archive dependency inventory differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="fh5-restore-", dir=output.parent) as temporary:
        root = Path(temporary)
        frozen = root / "checkpoint"
        total = 0
        for name in sorted(names):
            raw = read_bounded(asset(source / "checkpoint", name), 256 * 1024**2)
            total += len(raw)
            if total > 1024**3 or len(names) > 50000:
                raise ValueError("Candidate archive exceeds capacity")
            if manifest["files"][name] != {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }:
                raise ValueError("Candidate archive dependency changed: " + name)
            target = asset(frozen, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_file(target, raw)
        verified = run_experiment(
            SACResume(
                frozen,
                root / "probe",
                steps=0,
                expected_checkpoint_sha256=manifest["checkpoint_sha256"],
            )
        ).summary["sac_learning"]
        if any(manifest[key] != verified[key] for key in ("learner_state_sha256", "total_steps")):
            raise ValueError("Candidate archive learner summary differs")
        receipt = {
            "version": 1,
            "archive_sha256": request.expected_archive_sha256,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "learner_state_sha256": verified["learner_state_sha256"],
            "total_steps": verified["total_steps"],
            "reason": request.reason.strip(),
            "default_changed": False,
            "driving_qualification": "not_established_by_archive",
        }
        output.mkdir()
        for name in sorted(names, key=lambda value: (value == "policy.json", value)):
            target = asset(output, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_file(target, read_bounded(asset(frozen, name), 256 * 1024**2))
        write_file(output / "restored-from.json", encode(receipt))
    return RunResult({}, [], [], {"candidate_restore": receipt}, output / "restored-from.json")
