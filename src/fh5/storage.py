"""Read-only capacity accounting for a learning session and its retained dependencies."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.artifact_io import read_json, sha256_file
from fh5.candidate_archive import checkpoint_asset_names
from fh5.candidate_store import _read_events
from fh5.collection_store import encode, write_file
from fh5.learning_recovery import sampling_bindings
from fh5.numeric_images import asset
from fh5.sampling_evidence import sampling_sources
from fh5.storage_inventory import StorageInventory, storage_inventory

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class LearningStoragePlan:
    config_file: Path
    output_dir: Path


class _Inventory:
    def __init__(self, root: Path, output: Path | None, index: StorageInventory) -> None:
        self.root, self.output = root, output
        self.index = index
        self.metadata_bytes = 0

    def path(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Storage dependency leaves the declared root: " + str(path))
        for part in (path, *path.parents):
            if part.is_symlink() or part.is_junction():
                raise ValueError("Storage dependencies cannot traverse links: " + str(part))
            if part == self.root:
                break
        return resolved

    def file(self, path: Path, role: str, expected_bytes: int | None = None) -> Path:
        path = self.path(path)
        status = path.stat()
        if not path.is_file():
            raise ValueError("Storage dependency is not a regular file: " + str(path))
        if expected_bytes is not None and status.st_size != expected_bytes:
            raise ValueError("Storage dependency size changed: " + str(path))
        self.index.file(path, status, role)
        return path

    def asset(self, root: Path, name: str) -> Path:
        self.path(root / name)
        return asset(root, name)

    def document(self, path: Path, role: str, expected: str | None = None) -> dict[str, Any]:
        path = self.file(path, role)
        raw = self.read_metadata(path)
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("Storage dependency manifest changed: " + str(path))
        self.index.document(path, digest)
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Storage dependency manifest must be an object")
        return value

    def read_metadata(self, path: Path) -> bytes:
        path = self.file(path, "metadata")
        raw = path.read_bytes()
        self.metadata_bytes += len(raw)
        return raw

    def tree(self, root: Path, role: str) -> None:
        root = self.path(root)
        if self.output is not None and self.output.is_relative_to(root):
            raise ValueError("Storage report must be outside retained source directories")
        if not root.is_dir():
            raise ValueError("Missing retained storage directory: " + str(root))
        if self.index.visited("tree", str(root), role):
            return

        self.index.queue_directory(root, role)
        while (pending := self.index.next_directory()) is not None:
            directory, role = pending
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        self.index.queue_directory(self.path(path), role)
                    else:
                        self.file(path, role)

    def checkpoint(self, root: Path, expected: str, role: str) -> None:
        root = self.path(root)
        manifest = self.document(root / "policy.json", role, expected)
        self.tree(root, role)
        replay = self.document(root / "experience/replay.json", role, manifest["replay_sha256"])
        for original in replay.get("source_inventory", []):
            self.document(
                self.asset(root / "experience", original["path"]), role, original["replay_sha256"]
            )
        names = checkpoint_asset_names(
            root,
            manifest,
            replay,
            read_history_node=lambda path, sha: self.document(path, role, sha),
        )
        for name in names:
            self.file(self.asset(root, name), role)

    def evidence(self, base: Path, binding: dict[str, Any], role: str) -> None:
        path = self.path(base / binding["file"])
        proof = self.document(path, role, binding["sha256"])
        for item in proof.get("items", []):
            self.file(self.asset(path.parent, item["path"]), role)

    def execution(self, root: Path, expected: str) -> None:
        root = self.path(root)
        role = "evaluation_original"
        self.tree(root, role)
        manifest = self.document(root / "realtime-manifest.json", role, expected)
        report = self.document(root / "report.json", role, manifest["report_sha256"])
        self.file(self.asset(root, report["journal"]["path"]), role)
        for decision in report["decisions"]:
            archive = decision.get("archive")
            if archive is None:
                continue
            source = self.document(self.asset(root, archive["path"]), role, archive["sha256"])
            for frame in source["frames"]:
                self.file(
                    self.asset(root, frame["path"]), role, 3 * frame["size"][0] * frame["size"][1]
                )

    def evaluation(self, base: Path, binding: dict[str, Any]) -> None:
        batch = self.path(base / binding["batch"])
        ledger = self.path(base / binding["ledger"])
        if self.index.visited(
            "evaluation", str(batch), binding["batch_sha256"], str(ledger), binding["ledger_sha256"]
        ):
            return
        role = "evaluation_original"
        frozen = self.document(batch / "batch.json", role, binding["batch_sha256"])
        entries = self.document(ledger, role, binding["ledger_sha256"])
        if entries["batch_sha256"] != binding["batch_sha256"]:
            raise ValueError("Storage evaluation ledger belongs to another batch")
        self.tree(batch, role)
        for name in frozen["files"]:
            self.file(self.asset(batch, name), role)
        for entry in entries["entries"]:
            recording = self.path(ledger.parent / entry["recording"])
            self.tree(recording, role)
            for name in entry["files"]:
                self.file(self.asset(recording, name), role)
            if entry.get("evidence") is not None:
                self.evidence(ledger.parent, entry["evidence"], role)
            if entry.get("execution") is not None:
                execution = entry["execution"]
                self.execution(ledger.parent / execution["directory"], execution["manifest_sha256"])
            if entry.get("preparation") is not None:
                prepared = entry["preparation"]
                root = self.path(ledger.parent / prepared["directory"])
                self.tree(root, role)
                source = self.document(
                    root / "start-manifest.json", role, prepared["manifest_sha256"]
                )
                for name in source["files"]:
                    self.file(self.asset(root, name), role)

    def store(self, root: Path, expected: str) -> None:
        root = self.path(root)
        self.tree(root, "candidate_history")
        self.file(root / "state.sqlite", "candidate_history")

        def digest_file(path: Path) -> str:
            return sha256_file(self.file(path, "metadata"))

        latest = None
        for event in _read_events(root, digest_file=digest_file):
            latest = event["revision"]
            for role in (
                event["default"],
                event["explorer"],
                *event["aggressive_by_reference"].values(),
            ):
                archive = self.asset(root, role["archive"])
                saved = self.document(
                    archive / "archive.json", "candidate_history", role["archive_sha256"]
                )
                self.checkpoint(
                    archive / "checkpoint", saved["checkpoint_sha256"], "candidate_history"
                )
                for name, entry in saved["files"].items():
                    self.file(
                        self.asset(archive / "checkpoint", name),
                        "candidate_history",
                        entry["bytes"],
                    )
            proof = event["qualification"]
            path = self.path(Path(proof["comparison_file"]))
            comparison = self.document(path, "candidate_history", proof["comparison_sha256"])
            for side in ("incumbent", "candidate"):
                self.evaluation(path.parent, comparison[side])
        if latest != expected:
            raise ValueError("Retained candidate store revision changed")
        for event in _read_events(root, digest_file=digest_file):
            latest = event["revision"]
        if latest != expected:
            raise ValueError("Candidate store changed during storage accounting")

    def session(self, root: Path, expected: str) -> None:
        root = self.path(root)
        state = self.document(root / "state.json", "session", expected)
        if state["version"] != 1 or state["scope"] != "synthetic_development_only":
            raise ValueError("Unsupported learning storage state")
        config = self.document(root / "config.json", "session", state["config_sha256"])
        self.tree(root, "session")
        for name, digest in state["source_files"].items():
            self.document(Path(name), "configuration", digest)
        task_path = Path(config["task"])
        task = self.document(task_path, "configuration", state["source_files"][str(task_path)])
        route_path = self.path(task_path.parent / task["route_file"])
        route = self.document(route_path, "task_route", task["route_sha256"])
        for item in [*route["assets"].values(), *route["evidence"]]:
            self.file(self.asset(route_path.parent, item["path"]), "task_route")
        if task.get("automatic_start") is not None:
            start = task["automatic_start"]
            event_path = self.path(task_path.parent / start["event_file"])
            event = self.document(event_path, "automatic_start", start["event_sha256"])["event_run"]
            for patches in event["signatures"].values():
                for patch in patches:
                    self.file(self.asset(event_path.parent, patch["template"]), "automatic_start")
            for name in event["verification_evidence"]:
                self.file(self.asset(event_path.parent, name), "automatic_start")
        self.file(Path(config["registry"]), "evidence_registry")
        roles = ("default", "explorer", "latest_learner")
        if state.get("initialized", True) is False:
            if state["rounds"] or any(role in state for role in roles):
                raise ValueError("Uninitialized storage state cannot contain learners or rounds")
        else:
            for role in roles:
                self.checkpoint(Path(state[role]["directory"]), state[role]["sha256"], role)
        self.evaluation(root, state["incumbent"])
        for row in state["rounds"]:
            # Parent review input may be frozen before its derived ledger exists.
            # Recordings are in the session tree; independent proof can live outside it.
            if "review_input" in row:
                for entry in row["review_input"]["entries"]:
                    if entry.get("evidence") is not None:
                        self.evidence(root, entry["evidence"], "parent_review")
            for learning in sampling_bindings(row):
                sampled = self.document(
                    Path(learning["directory"]) / "summary.json",
                    "sampling_original",
                    learning["summary_sha256"],
                )
                for attempt in sampled["attempts"]:
                    binding = attempt["source_assets"]
                    if binding.get("kind") == "sampling-source-index-v1":
                        self.file(Path(binding["path"]), "sampling_original_index")
                    with sampling_sources(binding) as sources:
                        for name in sources:
                            self.file(Path(name), "sampling_original")
            if "candidate_evaluation" in row:
                self.evaluation(root, row["candidate_evaluation"])
        self.store(Path(config["store"]["directory"]), state["store_revision"])

    def stable(self) -> None:
        for entry in self.index.files():
            path = self.path(Path(entry["path"]))
            status = path.stat()
            if not path.is_file() or (status.st_size, status.st_mtime_ns) != (
                entry["bytes"],
                entry["mtime_ns"],
            ):
                raise ValueError("Storage dependency changed during accounting: " + str(path))
        for path, digest in self.index.documents():
            if sha256_file(path) != digest:
                raise ValueError("Storage manifest changed during accounting: " + str(path))


def measure_learning_storage(namespace: Path, run_dir: Path, state_sha256: str) -> dict[str, int]:
    """Account at a quiescent learning boundary without writing an external report."""
    with storage_inventory() as index:
        inventory = _Inventory(namespace.resolve(), None, index)
        inventory.session(run_dir, state_sha256)
        inventory.stable()
        return {**index.totals(), "metadata_read_bytes": inventory.metadata_bytes}


def plan_learning_storage(request: LearningStoragePlan) -> RunResult:
    from fh5.experiment import RunResult

    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    config = read_json(request.config_file)
    if set(config) != {"version", "root", "learning", "budget_bytes"} or config["version"] != 1:
        raise ValueError("Unsupported learning storage configuration")
    if type(config["budget_bytes"]) is not int or config["budget_bytes"] <= 0:
        raise ValueError("Storage budget must be a positive integer number of bytes")
    if set(config["learning"]) != {"directory", "state_sha256"}:
        raise ValueError("Storage planning requires an immutable learning state reference")
    root = (request.config_file.parent / config["root"]).resolve()
    with storage_inventory() as index:
        inventory = _Inventory(root, request.output_dir.resolve(), index)
        inventory.session(
            root / config["learning"]["directory"], config["learning"]["state_sha256"]
        )
        inventory.stable()
        totals = index.totals()
        request.output_dir.mkdir(parents=True)
        files = index.publish(request.output_dir / "files.jsonl")
    summary = {
        "version": 2,
        "scope": "learning_session_and_retained_dependencies",
        "root": str(root),
        "state_sha256": config["learning"]["state_sha256"],
        "budget_bytes": config["budget_bytes"],
        "metadata_read_bytes": inventory.metadata_bytes,
        **totals,
        "status": "within_budget"
        if totals["protected_bytes"] <= config["budget_bytes"]
        else "over_budget",
        "files": files,
        "files_deleted": 0,
        "cleanup_authorized": False,
        "integrity_scope": "manifest_bindings_and_file_presence; not model_or_driving_qualification",
        "byte_basis": "unique_resolved_paths; logical_size_not_physical_disk_allocation",
    }
    path = request.output_dir / "storage-plan.json"
    write_file(path, encode(summary))
    return RunResult({}, [], [], {"storage": summary}, path)
