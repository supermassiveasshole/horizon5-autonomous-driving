"""Frozen SAC state and the numerical experience needed to continue learning."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fh5.artifact_io import VerifiedFile, sha256_file
from fh5.checkpoint_history import HistorySource, verify_history
from fh5.collection_store import encode, read_bounded, write_file
from fh5.numeric_images import NumericFrame, asset
from fh5.numeric_recording import read_numeric_frame
from fh5.sac_imitation import checkpoint_imitation, imitation_evidence


def _read_state(
    torch: Any,
    root: Path,
    manifest: dict[str, Any],
    weights_file: str,
    *,
    require_metadata: bool = True,
) -> dict[str, Any]:
    weights = VerifiedFile(root / weights_file, manifest["weights_sha256"])
    with weights.snapshot() as stream:
        saved: dict[str, Any] = torch.load(stream, map_location="cpu", weights_only=True)
    if require_metadata and saved.get("metadata") != {
        k: v for k, v in manifest.items() if k != "weights_sha256"
    }:
        raise ValueError("Checkpoint metadata mismatch")
    return saved


def _verify_training_state(
    torch: Any, root: Path, manifest: dict[str, Any], saved: dict[str, Any]
) -> None:
    if sha256_file(root / "training-report.json") != manifest["training_report_sha256"]:
        raise ValueError("Checkpoint training report changed")
    if (
        state_digest(torch, {k: v for k, v in saved.items() if k != "metadata"})
        != manifest["learner_state_sha256"]
    ):
        raise ValueError("Checkpoint learner state mismatch")


def resume_contract(torch: Any, version: int = 2) -> dict[str, Any]:
    contract = {
        "stage": "sac_updates",
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "threads": 2,
        "deterministic_algorithms": True,
        "optimizer_owners": {
            "critic": ["encoder", "critic"],
            "actor": ["policy"],
            "temperature": ["log_alpha"],
        },
        "state_digest": "canonical-cpu-tensors-v1",
        "imitation_schedule": "not_implemented",
        "auxiliary_components": [],
        "environment_state": "not_restored; new attempt required",
    }
    if version == 3:
        contract["replay_sampling"] = "source-quotas-v1; shrink without replacement or backfill"
    if version == 4:
        contract["replay_sampling"] = "uniform or explicit source-quotas-v1"
        contract["imitation_schedule"] = (
            "frozen-bc-interval-distance-v1; explicit phases ending at zero"
        )
    return contract


def read_checkpoint(torch: Any, root: Path) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    raw = (root / "policy.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("version") not in (1, 2, 3, 4) or (
        manifest.get("architecture"),
        manifest.get("stage"),
    ) != ("conditional-temporal-sac-v1", "sac_updates"):
        raise ValueError("Unsupported SAC policy checkpoint")
    saved = _read_state(torch, root, manifest, "policy.pt")
    imitation_evidence(root, checkpoint_imitation(manifest))
    if manifest["version"] in (2, 3, 4):
        if manifest["resume_contract"] != resume_contract(torch, manifest["version"]):
            raise ValueError("Unsupported SAC continuation contract or Torch runtime")
        _verify_training_state(torch, root, manifest, saved)
    return manifest, saved, raw


def critic_resume_contract(torch: Any) -> dict[str, Any]:
    return {
        "stage": "critic_warmup",
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "threads": 2,
        "deterministic_algorithms": True,
        "optimizer_owners": {"critic": ["critic"]},
        "actor": "entire BC frozen, including encoder and normalization",
        "state_digest": "canonical-cpu-tensors-v1",
        "phase_budget": "fixed total updates; resume consumes remaining budget",
        "sampling": "uniform without replacement within each batch",
        "imitation_schedule": "not_implemented",
        "auxiliary_components": [],
        "environment_state": "not_restored; new attempt required",
    }


def read_critic_checkpoint(torch: Any, root: Path) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    raw = (root / "critic.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("version") not in (1, 2) or manifest.get("stage") != "critic_warmup":
        raise ValueError("Unsupported critic checkpoint version or stage")
    saved = _read_state(
        torch, root, manifest, "critic.pt", require_metadata=manifest["version"] == 2
    )
    if manifest["version"] == 2:
        if manifest["resume_contract"] != critic_resume_contract(torch):
            raise ValueError("Unsupported critic continuation contract or Torch runtime")
        config = manifest["configuration"]
        if (
            set(config) != {"steps", "batch_size", "learning_rate", "seed", "target_tau", "device"}
            or config["target_tau"] != 0.005
            or config["device"] != "cpu"
        ):
            raise ValueError("Unsupported critic configuration")
        _verify_training_state(torch, root, manifest, saved)
        budget = manifest["configuration"]["steps"]
        if (
            type(budget) is not int
            or budget < 1
            or (type(saved["step"]) is not int or not 0 <= saved["step"] <= budget)
        ):
            raise ValueError("Invalid finite critic warm-up progress")
    return manifest, saved, raw


def publish_checkpoint(
    torch: Any, root: Path, metadata: dict[str, Any], state: dict[str, Any], report: bytes
) -> None:
    """Publish the manifest last, after state and required evidence are durable."""
    name = {"critic_warmup": "critic", "sac_updates": "policy"}[metadata["stage"]]
    weights = root / (name + ".pt")
    with weights.open("xb") as stream:
        torch.save({"metadata": metadata, **state}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {**metadata, "weights_sha256": sha256_file(weights)}
    raw = encode(manifest)
    write_file(root / "training-report.json", report)
    checkpoint_history(root, metadata)
    write_file(root / (name + ".json"), raw)


def checkpoint_history(root: Path, manifest: dict[str, Any]) -> None:
    """Validate retained history and the current stage without adding a successor."""
    if sha256_file(root / "training-report.json") != manifest["training_report_sha256"]:
        raise ValueError("SAC training report changed")
    verify_history(root, manifest["history"])


def continuation_history(root: Path, manifest: dict[str, Any], raw: bytes) -> HistorySource:
    checkpoint_history(root, manifest)
    return HistorySource(root, manifest["history"], raw, manifest["training_report_sha256"])


def state_digest(torch: Any, state: dict[str, Any]) -> str:
    """Canonical fingerprint independent of file paths and Torch ZIP serialization."""
    digest = hashlib.sha256()

    def add(value: Any) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            add(["tensor", str(tensor.dtype), list(tensor.shape)])
            raw = bytes(tensor.reshape(-1).view(torch.uint8).tolist())
            digest.update(len(raw).to_bytes(8, "little"))
            digest.update(raw)
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value, key=lambda k: (type(k).__name__, str(k))):
                add(key)
                add(value[key])
            digest.update(b"end")
        elif isinstance(value, (list, tuple)):
            digest.update(b"list")
            for item in value:
                add(item)
            digest.update(b"end")
        else:
            raw = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
            digest.update(len(raw).to_bytes(8, "little"))
            digest.update(raw)

    add(state)
    return digest.hexdigest()


def experience_frames(root: Path, replay: dict[str, Any]) -> Iterator[tuple[str, NumericFrame]]:
    """Check every referenced path, including terminal observations not used by the learner."""
    seen: dict[str, str] = {}
    total = 0
    for row in replay["transitions"]:
        for observation in (row["current"], row["next"]):
            if observation is None:
                continue
            for entry in observation["frames"]:
                name, sha = entry["path"], entry["sha256"]
                if name in seen:
                    if seen[name] != sha:
                        raise ValueError("SAC frame path has conflicting contents")
                    continue
                frame = read_numeric_frame(root, entry)
                total += frame.pixels.nbytes
                if total > 512 * 1024**2:
                    raise ValueError("Sealed SAC experience exceeds 512 MiB")
                seen[name] = sha
                yield name, frame


def seal_experience(raw: bytes, source: Path, output: Path) -> dict[str, Any]:
    replay = json.loads(raw)
    sources = source_replays(source.parent, replay)
    output.mkdir(parents=True)
    count, total = 0, 0
    for name, frame in experience_frames(source.parent, replay):
        target = asset(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_file(target, bytes(frame.pixels))
        count += 1
        total += frame.pixels.nbytes
    write_file(output / "replay.json", raw)
    for name, payload in sources.items():
        target = asset(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_file(target, payload)
    return {"replay": "experience/replay.json", "frame_files": count, "frame_bytes": total}


def source_replays(root: Path, replay: dict[str, Any]) -> dict[str, bytes]:
    """Retain source manifests, including excluded/failed experience diagnostics."""
    sources: dict[str, bytes] = {}
    total = 0
    for item in replay.get("source_inventory", []):
        name = item["path"]
        raw = read_bounded(asset(root, name), 128 * 1024**2)
        if (
            hashlib.sha256(raw).hexdigest() != item["replay_sha256"]
            or json.loads(raw)["source_hashes"] != item["source_hashes"]
        ):
            raise ValueError("SAC experience source manifest changed")
        total += len(raw)
        if total > 128 * 1024**2 or len(sources) >= 1000:
            raise ValueError("SAC experience source manifests exceed capacity")
        if name in sources:
            raise ValueError("Duplicate SAC experience source manifest")
        sources[name] = raw
    return sources
