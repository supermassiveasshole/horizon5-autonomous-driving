"""Export fixed continuous selections as causal numeric BC inputs, without codecs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fh5.artifact_io import VerifiedFile, sha256_file
from fh5.collection_dataset import (
    _config,
    _evidence,
    _freeze_source,
    _report_destination,
    _summary,
    build_snapshot,
)
from fh5.collection_selection import sealed_rows
from fh5.collection_store import encode, read_bounded, write_file
from fh5.numeric_images import NumericDecision, PixelContract, asset, validate_decision
from fh5.numeric_recording import read_numeric_frame
from fh5.presentation import optional_report
from fh5.replay_document import read_document_fields, write_replay_document
from fh5.temporal_features import actor_shape

if TYPE_CHECKING:
    from fh5.collection_dataset import _CollectionIndex
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class CollectionBCPrepare:
    config_file: Path
    output_dir: Path


def _settings(config: dict[str, Any]) -> None:
    if (
        set(config) - {"reference"}
        != {
            "version",
            "seed",
            "sources",
            "rules",
            "action_history_offsets_ms",
            "max_action_age_ms",
            "waypoint_distances_m",
        }
        or config["version"] != 2
    ):
        raise ValueError(
            "Collection BC preparation requires v2 sources, seed, rules and observation settings; "
            "replace the old selection path/hash with its source configuration"
        )
    for name, maximum, descending in (
        ("action_history_offsets_ms", 2000, True),
        ("waypoint_distances_m", 500, False),
    ):
        values = config[name]
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 16
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= maximum
                for v in values
            )
            or values != sorted(set(values), reverse=descending)
            or (descending and values[-1] != 0)
        ):
            raise ValueError("Invalid causal history or waypoint layout")
    age = config["max_action_age_ms"]
    if type(age) not in (int, float) or not math.isfinite(age) or not 10 <= age <= 1000:
        raise ValueError("Invalid action history age budget")


def _reference(
    config: dict[str, Any],
    base: Path,
    sources: list[dict[str, Any]],
) -> tuple[dict[str, bytes], dict[str, Any]]:

    setting = config.get("reference")
    if setting is None:
        return {}, {"status": "absent", "paired_views_identical": True}
    if (
        not isinstance(setting, dict)
        or set(setting) != {"route_file", "independence_evidence"}
        or not _evidence(setting["independence_evidence"])
    ):
        raise ValueError("Optional reference requires explicit independent-source evidence")
    path = base / setting["route_file"]
    raw = read_bounded(path, 4 * 1024**2)
    manifest = json.loads(raw)
    if manifest["source"]["session_sha256"] in {s["session_sha256"] for s in sources}:
        raise ValueError("Collection cannot provide its own future navigation reference")
    files = {"route.json": raw}
    for entry in [*manifest["assets"].values(), *manifest["evidence"]]:
        payload = read_bounded(asset(path.parent, entry["path"]), 32 * 1024**2)
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("Reference asset differs from frozen hash")
        if entry["path"] in files and files[entry["path"]] != payload:
            raise ValueError("Conflicting reference asset path")
        files[entry["path"]] = payload
        if sum(map(len, files.values())) > 64 * 1024**2:
            raise ValueError("Reference exceeds 64 MiB preparation budget")
    # Load the copied bundle below, so concurrent changes to the source cannot
    # alter the waypoint inputs after these hashes were recorded.
    return (
        files,
        {
            "status": "loaded",
            "paired_views_identical": False,
            "route_sha256": hashlib.sha256(raw).hexdigest(),
            "independence_evidence": setting["independence_evidence"],
            "source": manifest["source"],
        },
    )


def _actor(
    row: dict[str, Any],
    sample: dict[str, Any],
    telemetry: dict[str, Any],
    history: deque[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    tick, floor = row["at_ns"], sample["history_floor_ns"]
    actions = []
    ages = []
    for offset in config["action_history_offsets_ms"]:
        cutoff = tick - int(offset * 1e6)
        candidates = [
            a for a in history if floor <= a["poll_ns"] < cutoff and a["available_ns"] < cutoff
        ]
        chosen = max(candidates, key=lambda a: a["poll_ns"], default=None)
        age = (tick - chosen["poll_ns"]) / 1e6 if chosen else None
        valid = (
            chosen is not None and age is not None and age <= offset + config["max_action_age_ms"]
        )
        actions.append(chosen["mapped"] if valid and chosen else None)
        ages.append(age if valid else None)
    motion = telemetry["motion"]
    count = len(row["frames"])
    return {
        "ego": {
            "speed_mps": telemetry["speed_mps"],
            "velocity_car_mps": motion["velocity_car_mps"],
            "angular_velocity_car_radps": motion["angular_velocity_car_radps"],
        },
        "ego_mask": True,
        "ego_age_ms": (tick - telemetry["received_monotonic_ns"]) / 1e6,
        "images": [None] * count,
        "image_mask": [True] * count,
        "image_age_ms": [(tick - f["source_time_ns"]) / 1e6 for f in row["frames"]],
        "actions": actions,
        "action_mask": [a is not None for a in actions],
        "action_age_ms": ages,
        "reference": {
            "waypoints_m": [None] * len(config["waypoint_distances_m"]),
            "mask": [False] * len(config["waypoint_distances_m"]),
        },
    }


def prepare_collection_bc(request: CollectionBCPrepare) -> RunResult:
    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    config = read_document_fields(
        VerifiedFile(request.config_file, sha256_file(request.config_file)),
        {
            "version",
            "dataset",
            "dataset_sha256",
            "seed",
            "sources",
            "rules",
            "action_history_offsets_ms",
            "max_action_age_ms",
            "waypoint_distances_m",
            "reference",
        },
        reject_unknown=True,
    )
    _settings(config)
    selection_config = {"version": 1, **{k: config[k] for k in ("seed", "sources", "rules")}}
    _config(selection_config)
    sources = [_freeze_source(s, request.config_file.parent) for s in config["sources"]]
    _report_destination(request.output_dir / "report.html", sources)
    with build_snapshot(selection_config, sources) as (data, index):
        return _export(request, config, data, index)


def _export(
    request: CollectionBCPrepare,
    config: dict[str, Any],
    data: dict[str, Any],
    index: _CollectionIndex,
) -> RunResult:
    from fh5.experiment import RunResult
    from fh5.observations import _preview
    from fh5.routes import load_route, locate_route

    contract = PixelContract.from_metadata(data["pixel_contract"])
    reference_files, reference_info = _reference(
        config, request.config_file.parent, data["sources"]
    )
    reference = None
    request.output_dir.mkdir(parents=True)
    (request.output_dir / "pixels").mkdir()
    selection = request.output_dir / "selection.json"
    write_replay_document(selection, data)
    digest = sha256_file(selection)
    for name, payload in reference_files.items():
        destination = asset(request.output_dir / "reference", name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_file(destination, payload)
    if reference_files:
        reference = load_route(request.output_dir / "reference/route.json")
    partitions = {
        g["id"]: "evaluation" if g["split"] == "evaluation" else "development"
        for g in data["groups"]
    }
    total_bytes = 0
    for source in data["sources"]:
        binding, root = source["session_sha256"], Path(source["recording"])
        session_bytes = read_bounded(root / "session.json", 1024**2)
        if hashlib.sha256(session_bytes).hexdigest() != binding:
            raise ValueError("Collection session changed during preparation")
        session = json.loads(session_bytes)
        history: deque[dict[str, Any]] = deque(maxlen=8192)
        latest = None
        previous = None
        route_state: dict[str, Any] = {}
        for block, row in sealed_rows(source, contract):
            if (
                previous is None
                or row["segment"] != previous["segment"]
                or row["sequence"] != previous["sequence"] + 1
            ):
                history.clear()
                latest = None
                route_state = {}
            previous = row
            if row["telemetry"]:
                if reference is not None:
                    for packet in row["telemetry"]:
                        packet.update(segment=row["segment"], packet_index=row["sequence"])
                    locate_route(row["telemetry"], reference, state=route_state)
                latest = row["telemetry"][-1]
            mapped = row["mapped_input"]
            if row["input_usable"] and mapped is not None and mapped["mapping_valid"]:
                history.append(mapped)
            earliest = row["at_ns"] - int(
                (max(config["action_history_offsets_ms"]) + config["max_action_age_ms"]) * 1e6
            )
            while history and history[0]["available_ns"] < earliest:
                history.popleft()
            sample = index.sample(binding, row["sequence"])
            if sample is None:
                continue
            partition = partitions[sample["group"]]
            if (
                latest is None
                or latest["motion"] is None
                or not latest["is_race_on"]
                or not (
                    sample["history_floor_ns"] <= latest["received_monotonic_ns"] <= row["at_ns"]
                    and row["at_ns"] - latest["received_monotonic_ns"]
                    <= session["configuration"]["max_age_ms"] * 1e6
                )
            ):
                index.append(
                    "excluded:" + partition,
                    {"id": sample["id"], "group": sample["group"], "reason": "unavailable_ego"},
                )
                continue
            actor = _actor(row, sample, latest, history, config)
            assisted = deepcopy(actor)
            if reference is not None:
                preview = _preview(latest, reference, config["waypoint_distances_m"])
                assisted["reference"] = {
                    "waypoints_m": preview["waypoints_m"],
                    "mask": preview["waypoint_mask"],
                }
            actor_shape(actor, len(row["frames"]))
            frames = tuple(read_numeric_frame(root / block, f) for f in row["frames"])
            reason = validate_decision(
                NumericDecision(sample["id"], row["capture_epoch"], row["at_ns"], frames, actor),
                contract,
            )
            if reason:
                raise ValueError("Invalid prepared observation: " + reason)
            metadata = []
            for frame, saved in zip(frames, row["frames"]):
                pixel_hash = saved["sha256"]
                item = {
                    **frame.metadata(),
                    "epoch": binding + ":" + frame.epoch,
                    "frame_id": binding + ":" + frame.frame_id,
                    "path": f"pixels/{pixel_hash}.rgb",
                    "sha256": pixel_hash,
                }
                if index.frame_is_new(item):
                    total_bytes += frame.pixels.nbytes
                if index.pixel_is_new(pixel_hash):
                    write_file(request.output_dir / item["path"], bytes(frame.pixels))
                metadata.append(item)
            index.append(
                "decisions:" + partition,
                {
                    "decision_id": sample["id"],
                    "group": sample["group"],
                    "epoch": binding + ":" + row["capture_epoch"],
                    "decision_ns": row["at_ns"],
                    "frames": metadata,
                    "views": {"no_reference": actor, "reference_assisted": assisted},
                    "bc_eligible": sample["bc_eligible"],
                    "variant": 0,
                    "source_sequence": row["sequence"],
                    "source_block": block,
                    "supervision": {
                        "action": sample["target_action"],
                        "action_mask": True,
                        "quality": "trusted" if sample["bc_eligible"] else "excluded",
                        "reasons": sample["reasons"],
                        "label_poll_ns": sample["label_poll_ns"],
                        "label_available_ns": sample["label_available_ns"],
                        "label_delay_ms": (sample["label_poll_ns"] - row["at_ns"]) / 1e6,
                    },
                },
            )
    coverage = {entry["attempt"]: entry for entry in data["coverage"]}
    groups = [
        {
            **g,
            "source_ranges": [
                {
                    "session_sha256": source["session_sha256"],
                    "start_sequence": attempt["start_sequence"],
                    "end_sequence": attempt["end_sequence"],
                    "first_ns": coverage[attempt["id"]]["first_ns"],
                    "last_ns": coverage[attempt["id"]]["last_ns"],
                }
                for source in data["sources"]
                for attempt in source["review"]["attempts"]
                if attempt["group"] == g["id"]
            ],
            "evidence_id": hashlib.sha256(
                encode({"selection": digest, "attempts": g["attempts"]})
            ).hexdigest(),
        }
        for g in data["groups"]
    ]
    provenance = {
        "kind": "continuous_numeric_collection",
        "selection_sha256": digest,
        "selection_path": "selection.json",
        "config": config,
        "sources": [{k: v for k, v in s.items() if k != "review"} for s in data["sources"]],
        "diagnostic_only": data["diagnostic_only"],
        "envelope": data["config"]["rules"],
        "reference": reference_info,
        "input_conditions": session["input_conditions"],
        "vehicle": {
            k: session["configuration"][k] for k in ("expected_car_ordinal", "expected_pi")
        },
        "input_mapping": session["profile"]["mapping"],
        "final_evaluation_available": any(g["split"] == "evaluation" for g in groups),
        "evaluation_policy": "separate sealed file; not training or development feedback",
    }
    hashes: dict[str, str] = {}
    # Bind the final file in training provenance before any candidate is trained.
    # This prevents rehashing a different holdout after seeing the model's results.
    for filename, evaluation in (("evaluation.json", True), ("dataset.json", False)):
        partition = "evaluation" if evaluation else "development"
        group_ids = {g["id"] for g in groups if (g["split"] == "evaluation") == evaluation}
        snapshot: dict[str, Any] = {
            "version": 1,
            "kind": "numeric-bc-snapshot-v1",
            "pixel_contract": contract.metadata(),
            "action_contract": "xinput-lx-rt-lt-v1",
            "groups": [g for g in groups if g["id"] in group_ids],
            "decisions": index.array("decisions:" + partition),
            "provenance": dict(provenance, partition=partition),
            "excluded": index.array("excluded:" + partition),
        }
        if not evaluation:
            snapshot["provenance"]["final_dataset_sha256"] = hashes["evaluation.json"]
        temporary = request.output_dir / (filename + ".tmp")
        write_replay_document(temporary, snapshot)
        temporary.rename(request.output_dir / filename)
        hashes[filename] = sha256_file(request.output_dir / filename)
    selection_summary = _summary(data, digest)
    summary = {
        "version": 1,
        "selection_sha256": digest,
        "selection": selection_summary,
        "snapshot_sha256": hashes,
        "commands_sent": False,
        "diagnostic_only": data["diagnostic_only"],
        "unique_frames": index.unique_frames,
        "decoded_frame_budget_bytes": total_bytes,
        "reference": provenance["reference"],
        "closed_loop_validated": False,
        "evaluation": {"metrics": "withheld", "path": "evaluation.json"},
    }
    report = request.output_dir / "report.html"
    write_file(report.with_suffix(".json"), encode(summary))
    report = optional_report(
        report,
        "数值 BC 数据准备（完整封存不等于优质示范）",
        summary,
        fallback=report.with_suffix(".json"),
        exclusive=True,
    )
    return RunResult(
        {}, [], [], {"collection_bc": summary, "collection_dataset": selection_summary}, report
    )
