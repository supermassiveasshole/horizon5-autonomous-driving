"""CPU temporal BC state sealed at a known complete optimizer boundary."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fh5.artifact_io import VerifiedFile, sha256_file
from fh5.bc_learning import _checked_config
from fh5.collection_store import atomic_json, encode
from fh5.replay_document import read_document_fields
from fh5.sac_checkpoint import state_digest
from fh5.temporal_features import TEMPORAL_ARCHITECTURE

# Growing groups/provenance belong to the retained dataset. Rebuild them from
# that verified input when sealing the final candidate, not in each learner.
BC_RESUME_METADATA_KEYS = {
    "version",
    "architecture",
    "contract",
    "config",
    "preprocessing",
    "future_supervision",
    "numeric_contract",
    "dataset_sha256",
}

_MANIFEST_FIELDS = {
    "version",
    "stage",
    "resume_contract",
    "model_metadata",
    "dataset",
    "configuration_sha256",
    "steps_completed",
    "total_steps",
    "learner_state_sha256",
    "weights_sha256",
    "statistics",
    "parent_checkpoint_sha256",
}


def _contract(torch: Any, threads: int) -> dict[str, Any]:
    if type(threads) is not int or threads < 1:
        raise ValueError("BC continuation needs a positive CPU thread count")
    return {
        "version": 1,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "cpu_threads": threads,
        "deterministic_algorithms": True,
        "optimizer": "torch.optim.Adam; all actor parameters in registration order",
        "sampling": "torch.randperm paired observations; source order; no_reference then reference_assisted",
        "phase_budget": "remaining original total updates",
        "state_digest": "canonical-cpu-tensors-v1",
        "recovery": "cooperative complete-update boundaries; no abrupt-kill or partial-Adam guarantee",
        "inputs": "original immutable dataset and pixel files must remain available",
        "loss_history": "optional; never required for continuation",
    }


def _hash_value(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _configuration(metadata: dict[str, Any]) -> dict[str, Any]:
    if set(metadata) != BC_RESUME_METADATA_KEYS or (
        metadata["version"],
        metadata["architecture"],
    ) != (2, TEMPORAL_ARCHITECTURE):
        raise ValueError("Unsupported temporal BC continuation metadata")
    config = metadata["config"]
    if not isinstance(config, dict) or set(config) != {
        "version",
        "dataset",
        "dataset_sha256",
        "seed",
        "steps",
        "batch_size",
        "learning_rate",
        "device",
        "time_mode",
    }:
        raise ValueError("Invalid temporal BC continuation configuration")
    _checked_config(
        {
            **{
                key: value
                for key, value in config.items()
                if key not in ("dataset_sha256", "time_mode")
            },
            "image_size": metadata["contract"]["image_size"],
        }
    )
    if config["device"] != "cpu" or config["time_mode"] not in ("actual", "fixed"):
        raise ValueError("BC continuation supports the frozen CPU temporal contract")
    if (
        not _hash_value(config["dataset_sha256"])
        or config["dataset_sha256"] != metadata["dataset_sha256"]
    ):
        raise ValueError("BC continuation dataset binding mismatch")
    return config


def _validate_state(torch: Any, state: dict[str, Any], config: dict[str, Any]) -> None:
    if set(state) != {"actor", "optimizer", "rng", "step"}:
        raise ValueError("Incomplete BC learner state")
    if type(state["step"]) is not int or not 0 <= state["step"] <= config["steps"]:
        raise ValueError("BC learner progress exceeds its original budget")
    if not isinstance(state["actor"], dict) or not isinstance(state["optimizer"], dict):
        raise ValueError("Invalid BC actor or optimizer state")
    optimizer = state["optimizer"]
    groups, history = optimizer.get("param_groups"), optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(history, dict):
        raise ValueError("Invalid BC Adam optimizer ownership")
    parameters = groups[0].get("params") if isinstance(groups[0], dict) else None
    # This actor has parameters only (no running-statistic buffers), and its
    # single Adam group owns them in the same registration order as state_dict.
    if parameters != list(range(len(state["actor"]))):
        raise ValueError("Invalid BC Adam optimizer parameter ownership")
    # Use the pinned runtime's defaults, with the frozen experiment's learning
    # rate. Construction over the saved tensors does not update them or draw RNG.
    expected_options = torch.optim.Adam(
        state["actor"].values(), lr=config["learning_rate"]
    ).defaults
    options = {key: value for key, value in groups[0].items() if key != "params"}

    def same_option(value: Any, expected: Any) -> bool:
        if type(value) is not type(expected):
            return False
        if isinstance(expected, tuple):
            return len(value) == len(expected) and all(
                same_option(item, target) for item, target in zip(value, expected, strict=True)
            )
        return bool(value == expected)

    if set(options) != set(expected_options) or any(
        not same_option(options[key], value) for key, value in expected_options.items()
    ):
        raise ValueError("BC Adam optimizer options differ from the frozen update rule")
    expected_history = set(parameters) if state["step"] else set()
    if set(history) != expected_history or any(
        not isinstance(entry, dict) or set(entry) != {"step", "exp_avg", "exp_avg_sq"}
        for entry in history.values()
    ):
        raise ValueError("Incomplete BC Adam optimizer history for completed updates")
    for identifier, parameter in enumerate(state["actor"].values()):
        if identifier not in history:
            continue
        for name in ("exp_avg", "exp_avg_sq"):
            moment = history[identifier][name]
            if (
                not torch.is_tensor(moment)
                or moment.shape != parameter.shape
                or moment.dtype != parameter.dtype
            ):
                raise ValueError("BC Adam optimizer moment differs from its actor parameter")
    if history:
        # Ask the pinned Adam runtime for its counter representation without
        # touching the actor or RNG. Repeated float additions of one saturate at
        # the mantissa boundary; completed remains the unrestricted Python count.
        scalar = torch.zeros((), device="cpu", requires_grad=True)
        scalar.grad = torch.zeros_like(scalar)
        probe = torch.optim.Adam([scalar])
        probe.step()
        native_counter = probe.state[scalar]["step"]
        saturation = int(2 / torch.finfo(native_counter.dtype).eps)
        expected_count = min(state["step"], saturation)
        for entry in history.values():
            counter = entry["step"]
            if (
                not torch.is_tensor(counter)
                or counter.shape != native_counter.shape
                or counter.dtype != native_counter.dtype
                or counter.item() != expected_count
            ):
                raise ValueError("BC Adam optimizer counter differs from completed updates")
    rng = state["rng"]
    if not torch.is_tensor(rng) or rng.dtype != torch.uint8 or rng.ndim != 1:
        raise ValueError("Invalid BC sampling RNG state")

    def finite_cpu(value: Any) -> None:
        if torch.is_tensor(value):
            if value.device.type != "cpu" or not torch.isfinite(value).all():
                raise ValueError("BC continuation needs finite CPU tensors")
        elif isinstance(value, dict):
            for item in value.values():
                finite_cpu(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                finite_cpu(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite BC optimizer setting")

    finite_cpu(state)
    # Validate without advancing or replacing the caller's global generator.
    torch.Generator(device="cpu").set_state(rng)


def _statistics(values: Any) -> dict[str, Any]:
    if not isinstance(values, dict):
        return {"status": "partial", "error": "Prior optional training statistics unavailable"}
    result: dict[str, Any] = {
        "status": "complete" if values.get("status", "complete") == "complete" else "partial"
    }
    for name in ("duration_s", "time_gradient_l1"):
        value: Any = values.get(name)
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if valid:
            result[name] = value
        else:
            result.update(
                status="partial", error="Missing or invalid optional training statistic: " + name
            )
    return result


@dataclass(frozen=True)
class BCCheckpoint:
    manifest: dict[str, Any]
    state: dict[str, Any]
    file: VerifiedFile

    @property
    def statistics(self) -> dict[str, Any]:
        return _statistics(self.manifest.get("statistics"))

    @property
    def dataset(self) -> VerifiedFile:
        dependency = self.manifest["dataset"]
        return VerifiedFile(Path(dependency["path"]), dependency["sha256"])

    @property
    def descriptor(self) -> dict[str, Any]:
        return _descriptor(self.file, self.manifest)


@dataclass
class BCRecovery:
    """Carry durable progress out even when a scheduled run stops or fails."""

    directory: Path
    parent: BCCheckpoint | None = None
    published: dict[str, Any] | None = None
    completed: int = 0

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.completed = self.parent.state["step"]

    @property
    def latest(self) -> dict[str, Any] | None:
        return self.published or (self.parent.descriptor if self.parent is not None else None)


def _descriptor(source: VerifiedFile, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "directory": str(source.path.parent.resolve()),
        "manifest_sha256": source.sha256,
        "learner_state_sha256": manifest["learner_state_sha256"],
        "steps_completed": manifest["steps_completed"],
    }


def publish_bc_checkpoint(
    torch: Any,
    root: Path,
    *,
    model_metadata: dict[str, Any],
    actor: Any,
    optimizer: Any,
    completed: int,
    dataset: VerifiedFile,
    cpu_threads: int,
    statistics: dict[str, Any] | None = None,
    parent_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Seal a fresh immutable directory; failed writes never replace its parent.

    The caller must supply a known complete update boundary and CPU state. This
    does not make optimizer.step transactional or provide periodic persistence.
    Dataset availability is deliberately not needed to save already-learned state.
    """
    if not BC_RESUME_METADATA_KEYS <= model_metadata.keys():
        raise ValueError("Incomplete temporal BC continuation metadata")
    model_metadata = {key: model_metadata[key] for key in BC_RESUME_METADATA_KEYS}
    config = _configuration(model_metadata)
    if dataset.sha256 != config["dataset_sha256"]:
        raise ValueError("BC checkpoint dataset differs from the frozen configuration")
    if parent_checkpoint_sha256 is not None and not _hash_value(parent_checkpoint_sha256):
        raise ValueError("Invalid BC parent checkpoint identity")
    state = {
        "actor": actor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": torch.get_rng_state(),
        "step": completed,
    }
    _validate_state(torch, state, config)
    metadata = {
        "version": 1,
        "stage": "temporal_bc_updates",
        "resume_contract": _contract(torch, cpu_threads),
        "model_metadata": model_metadata,
        "dataset": {"path": str(dataset.path.resolve()), "sha256": dataset.sha256},
        "configuration_sha256": hashlib.sha256(encode(config)).hexdigest(),
        "steps_completed": completed,
        "total_steps": config["steps"],
        "learner_state_sha256": state_digest(torch, state),
        "statistics": _statistics(statistics),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }
    root.mkdir(parents=True)
    with (root / "learner.pt").open("xb") as stream:
        torch.save({"metadata": metadata, **state}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {**metadata, "weights_sha256": sha256_file(root / "learner.pt")}
    path = root / "learner.json"
    atomic_json(path, manifest)
    # Returning state_dict here would retain aliases to live model/Adam tensors;
    # only the committed artifact descriptor escapes publication.
    return _descriptor(VerifiedFile(path, sha256_file(path)), manifest)


def read_bc_checkpoint(
    torch: Any, root: Path, *, expected_sha256: str, cpu_threads: int
) -> BCCheckpoint:
    """Validate the immutable learner; inputs are revalidated by the resume caller."""
    if not _hash_value(expected_sha256):
        raise ValueError("BC continuation requires an expected checkpoint SHA-256")
    source = VerifiedFile(root / "learner.json", expected_sha256)
    manifest = read_document_fields(source, _MANIFEST_FIELDS, reject_unknown=True)
    if not _MANIFEST_FIELDS - {"statistics"} <= manifest.keys() or (
        manifest["version"],
        manifest["stage"],
    ) != (
        1,
        "temporal_bc_updates",
    ):
        raise ValueError("Unsupported BC learner checkpoint")
    if manifest["resume_contract"] != _contract(torch, cpu_threads):
        raise ValueError("Unsupported BC continuation runtime or sampling contract")
    config = _configuration(manifest["model_metadata"])
    dependency = manifest["dataset"]
    if (
        not isinstance(dependency, dict)
        or set(dependency) != {"path", "sha256"}
        or not isinstance(dependency["path"], str)
        or not Path(dependency["path"]).is_absolute()
        or dependency["sha256"] != config["dataset_sha256"]
        or manifest["configuration_sha256"] != hashlib.sha256(encode(config)).hexdigest()
        or manifest["total_steps"] != config["steps"]
    ):
        raise ValueError("BC checkpoint frozen configuration or dataset mismatch")
    if not _hash_value(manifest["weights_sha256"]) or not _hash_value(
        manifest["learner_state_sha256"]
    ):
        raise ValueError("Invalid BC learner artifact identity")
    with VerifiedFile(root / "learner.pt", manifest["weights_sha256"]).snapshot() as stream:
        saved: dict[str, Any] = torch.load(stream, map_location="cpu", weights_only=True)
    metadata = saved.pop("metadata", None)
    if metadata != {key: value for key, value in manifest.items() if key != "weights_sha256"}:
        raise ValueError("BC checkpoint metadata mismatch")
    _validate_state(torch, saved, config)
    if (
        saved["step"] != manifest["steps_completed"]
        or state_digest(torch, saved) != manifest["learner_state_sha256"]
    ):
        raise ValueError("BC checkpoint learner state mismatch")
    return BCCheckpoint(manifest, saved, source)
