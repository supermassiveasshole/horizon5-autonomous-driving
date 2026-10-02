"""Retain complete learning candidates without changing their evaluated identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from fh5.artifact_io import VerifiedFile, read_json, sha256_file
from fh5.checkpoint_history import NodeReader, history_assets
from fh5.collection_store import encode, write_file
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
    raw = (source / "policy.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Candidate differs from its expected checkpoint identity")
    manifest = json.loads(raw)
    replay = read_json(source / "experience/replay.json")
    return manifest, checkpoint_asset_names(source, manifest, replay)


def checkpoint_asset_names(
    root: Path,
    manifest: dict[str, Any],
    replay: dict[str, Any],
    *,
    read_history_node: NodeReader | None = None,
) -> set[str]:
    """Enumerate continuation assets, including independently indexed ancestry."""
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
    history = (
        history_assets(root, manifest["history"])
        if read_history_node is None
        else history_assets(root, manifest["history"], read_node=read_history_node)
    )
    names.update(name for name, _ in history)
    for entry in manifest.get("imitation", {}).get("transitions", []):
        names.add(entry["review"])
    names.update("experience/" + entry["path"] for entry in replay.get("source_inventory", []))
    for row in replay["transitions"]:
        for observation in (row["current"], row["next"]):
            if observation is not None:
                names.update("experience/" + frame["path"] for frame in observation["frames"])
    return names


def _copy_file(source: Path, target: Path, digest: str) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    VerifiedFile(source, digest).copy_to(target)
    return {"sha256": digest, "bytes": target.stat().st_size}


def _publish_checkpoint(source: Path, output: Path, inventory: dict[str, Any]) -> None:
    for name in sorted(inventory, key=lambda value: (value == "policy.json", value)):
        if name == "policy.json":
            for prior, entry in inventory.items():
                if prior != "policy.json":
                    VerifiedFile(asset(output, prior), entry["sha256"]).verify()
        target = asset(output, name)
        if _copy_file(asset(source, name), target, inventory[name]["sha256"]) != inventory[name]:
            raise ValueError("Candidate archive dependency changed: " + name)


def archive_candidate(request: CandidateArchive) -> RunResult:
    from fh5.experiment import RunResult
    from fh5.sac_learning import validate_sac_candidate

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
            original = asset(source, name)
            target = asset(frozen, name)
            inventory[name] = _copy_file(original, target, sha256_file(original))
            total += inventory[name]["bytes"]
        verified = validate_sac_candidate(frozen, request.expected_checkpoint_sha256)
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
        payload = encode(summary)
        output.mkdir()
        _publish_checkpoint(frozen, output / "checkpoint", inventory)
        write_file(output / "archive.json", payload)
    result = {**summary, "archive_sha256": hashlib.sha256(payload).hexdigest()}
    return RunResult({}, [], [], {"candidate_archive": result}, output / "archive.json")


def restore_candidate(request: CandidateRestore) -> RunResult:
    from fh5.experiment import RunResult
    from fh5.sac_learning import validate_sac_candidate

    source, output = request.archive_dir.resolve(), request.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Candidate restore must be outside its archive")
    if not isinstance(request.reason, str) or not 1 <= len(request.reason.strip()) <= 2000:
        raise ValueError("Candidate restore requires a reason of 1..2000 characters")
    payload = (source / "archive.json").read_bytes()
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
        for name in sorted(names):
            target = asset(frozen, name)
            actual = _copy_file(
                asset(source / "checkpoint", name), target, manifest["files"][name]["sha256"]
            )
            if manifest["files"][name] != actual:
                raise ValueError("Candidate archive dependency changed: " + name)
        verified = validate_sac_candidate(frozen, manifest["checkpoint_sha256"])
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
        _publish_checkpoint(frozen, output, manifest["files"])
        write_file(output / "restored-from.json", encode(receipt))
    return RunResult({}, [], [], {"candidate_restore": receipt}, output / "restored-from.json")
