"""Authenticate a completed sampling child before its parent acknowledges it."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fh5.collection_store import atomic_json, encode, read_bounded
from fh5.numeric_images import asset
from fh5.realtime_numeric_replay import read_realtime_journal, read_realtime_recording
from fh5.sac_cycle import SACCycle, SACRealtimeCycle
from fh5.sac_learning import validate_sac_candidate
from fh5.sampling_evidence import seal_sampling_sources, verify_sampling_sources

_INVENTORY_LIMIT = 64 * 1024**2


def sampling_bindings(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [*row.get("sampling_history", []), *([row["learning"]] if "learning" in row else [])]


def archive_failed_sampling(binding: dict[str, Any]) -> dict[str, Any]:
    """Keep large original inventories outside the bounded parent state document."""
    root = Path(binding["directory"])
    path = root.with_name(root.name + "-originals.json")
    inventory = seal_sampling_sources(root, None)
    payload = encode(inventory)
    if len(payload) > _INVENTORY_LIMIT:
        raise ValueError("Sampling originals inventory exceeds 64 MiB")
    if path.exists():
        if read_bounded(path, _INVENTORY_LIMIT) != payload:
            raise ValueError("Retained sampling originals inventory changed")
    else:
        atomic_json(path, inventory)
    return {
        **binding,
        "originals_file": str(path),
        "originals_sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_archived_sampling(binding: dict[str, Any]) -> None:
    root = Path(binding["directory"])
    path = root.with_name(root.name + "-originals.json")
    raw = read_bounded(path, _INVENTORY_LIMIT)
    if str(path) != binding.get("originals_file") or hashlib.sha256(raw).hexdigest() != binding.get(
        "originals_sha256"
    ):
        raise ValueError("Retained sampling originals inventory changed")
    verify_sampling_sources(json.loads(raw))


def _expected_protocol(request: SACCycle | SACRealtimeCycle) -> dict[str, Any]:
    files = (request.recording_config_file, request.task_file, request.reward_file)
    result = {
        "source_kind": "synthetic",
        "cycles": 1,
        **(
            {
                "runtime": asdict(request.runtime),
                "seconds_per_attempt": request.seconds_per_attempt,
                "max_updates_per_attempt": request.max_updates_per_attempt,
            }
            if isinstance(request, SACRealtimeCycle)
            else {"steps_per_attempt": request.steps_per_attempt}
        ),
        "seed": request.seed,
        "update_ratio": "at most one critic update per newly accepted transition",
        "protocol_files": {str(path): _sha(path, 1024**2) for path in files},
    }
    canonical: dict[str, Any] = json.loads(encode(result))
    return canonical  # Match the saved JSON tuple/list representation.


def retryable_sampling(
    request: SACCycle | SACRealtimeCycle,
    parent: dict[str, Any],
    expected_summary: str,
    *,
    pending: bool = False,
) -> bool:
    """Only released, sealed failures without any learner output may be resampled."""
    root = request.output_dir
    if _sha(root / "summary.json") != expected_summary:
        raise ValueError("Retained sampling result changed")
    summary = json.loads(read_bounded(root / "summary.json", 4 * 1024**2))
    reason = summary.get("stop_reason")
    attempts = summary.get("attempts", [])
    if (
        reason not in ("sampling_fault", "no_eligible_experience", "stop_requested")
        or summary.get("source_kind") != "synthetic"
        or summary.get("resources_released") is not True
        or summary.get("commands_sent_to_game") is not False
        or summary.get("latest_candidate")
        or summary.get("error")
        or summary.get("release_error")
        or len(attempts) > 1
        or pending
        and len(attempts) != 1
        or not attempts
        and reason != "stop_requested"
        or any(root.glob("candidate-*"))
    ):
        return False
    protocol = json.loads(read_bounded(root / "protocol.json", 1024**2))
    if protocol != _expected_protocol(request):
        raise ValueError("Failed sampling differs from its frozen retry protocol")
    attempt_names = {path.name for path in root.glob("attempt-*")}
    if attempt_names != ({"attempt-000"} if attempts else set()):
        raise ValueError("Failed sampling summary omits or invents an original attempt")
    for attempt in attempts:
        if (
            attempt.get("sampling_checkpoint_sha256") != parent["sha256"]
            or attempt.get("sampler_seed") != request.seed
            or attempt.get("learner_updates", 0) != 0
            or attempt.get("eligible_transitions", 0) != 0
            or attempt.get("candidate")
            or attempt.get("archive_error")
            or attempt.get("diagnostic_write_error")
        ):
            return False
        inventory = attempt.get("source_assets", {})
        verify_sampling_sources(inventory)
        count = attempt.get("received_packets")
        if type(count) is not int or count < 0:
            return False
        attempt_dir = root / "attempt-000"
        diagnostic = json.loads(read_bounded(attempt_dir / "sampling.json", 4 * 1024**2))
        if diagnostic.get("received_packets") != count:
            raise ValueError("Failed sampling count differs from its original diagnostic")
        _require_originals(
            attempt_dir,
            inventory,
            None,
            trace_required=count > 0 or (attempt_dir / "trace.json").exists(),
            review_binding_required=pending,
            asynchronous=request if isinstance(request, SACRealtimeCycle) else None,
        )
    return True


def _sha(path: Path, limit: int = 4 * 1024**2) -> str:
    return hashlib.sha256(read_bounded(path, limit)).hexdigest()


def _require_originals(
    root: Path,
    inventory: dict[str, str],
    sources: dict[str, Any] | None,
    *,
    trace_required: bool = True,
    review_binding_required: bool = False,
    asynchronous: SACRealtimeCycle | None = None,
) -> None:
    def require(path: Path, digest: str | None = None) -> None:
        stored = inventory.get(str(path.resolve()))
        if stored is None or (digest is not None and stored != digest):
            raise ValueError("Pending sampling original inventory is incomplete or inconsistent")

    for name in ("sampling.json", "recording/report.json", "recording/report.html"):
        require(root / name)
    for name, key in (
        ("trace.json", "trace"),
        ("recording/packets.jsonl", "packets"),
        ("recording/session.json", "session"),
    ):
        if name == "trace.json" and (not trace_required or asynchronous is not None):
            continue
        require(root / name, sources[key] if sources is not None else None)
    if asynchronous is not None:
        execution = root / "execution"
        require(execution / "realtime-manifest.json", sources["execution"] if sources else None)
        require(execution / "report.json")
        report = read_realtime_recording(execution)
        runtime = json.loads(
            encode(
                {**asdict(asynchronous.runtime), "pixels": asynchronous.runtime.pixels.metadata()}
            )
        )
        if (
            report["evidence_kind"] != "synthetic"
            or report["actor_kind"] != "frozen-numeric-sac-sampling-v1"
            or report["configuration"] != runtime
            or report["model"].get("sac_manifest_sha256") != asynchronous.expected_checkpoint_sha256
            or report["model"].get("noise", {}).get("seed") != asynchronous.seed
            or report["resources_released"] is not True
        ):
            raise ValueError("Pending async execution differs from its sampling contract")
        require(asset(execution, report["journal"]["path"]), report["journal"]["sha256"])
        read_realtime_journal(execution, report, time_limit_s=asynchronous.seconds_per_attempt)
        for decision in report["decisions"]:
            archived = decision.get("archive")
            if archived is None:
                continue
            path = asset(execution, archived["path"])
            require(path, archived["sha256"])
            for frame in json.loads(read_bounded(path, 4 * 1024**2))["frames"]:
                require(asset(execution, frame["path"]), frame["sha256"])
    elif trace_required:
        trace = json.loads(read_bounded(root / "trace.json", 32 * 1024**2))
        for observation in trace["observations"]:
            for frame in observation["frames"]:
                require(asset(root, frame["path"]), frame["sha256"])
    declaration = root / "sampling-sources.json"
    if review_binding_required or declaration.exists():
        require(declaration)
        bindings = json.loads(read_bounded(declaration, 4 * 1024**2))
        if bindings.get("version") != 1 or "review" not in bindings:
            raise ValueError("Pending sampling lacks its original review declaration")
        if bindings["review"] is not None:
            review = bindings["review"]
            path = Path(review["path"])
            if not path.is_absolute():
                raise ValueError("Pending sampling review path is not absolute")
            require(path, review["sha256"])
            proof = json.loads(read_bounded(path, 4 * 1024**2))
            for item in proof["items"]:
                require(asset(path.parent, item["path"]), item["sha256"])
    if sources is not None and sources["review"] is not None:
        proofs = [Path(name) for name, digest in inventory.items() if digest == sources["review"]]
        if not proofs:
            raise ValueError("Pending sampling inventory omits its independent review")
        # Same bytes at different paths can refer to different relative attachments.
        # Require the attachments of every retained matching original review.
        for path in proofs:
            proof = json.loads(read_bounded(path, 4 * 1024**2))
            for item in proof["items"]:
                require(asset(path.parent, item["path"]), item["sha256"])


def completed_sampling(
    request: SACCycle | SACRealtimeCycle,
    parent: dict[str, Any],
    *,
    allow_stopped_updates: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate a sealed single attempt and its complete learner snapshot."""
    root = request.output_dir
    protocol = json.loads(read_bounded(root / "protocol.json", 1024**2))
    summary: dict[str, Any] = json.loads(read_bounded(root / "summary.json", 4 * 1024**2))
    if protocol != _expected_protocol(request) or not (
        summary.get("source_kind") == "synthetic"
        and summary.get("stop_reason")
        in (
            ("budget_completed", "stop_requested")
            if allow_stopped_updates
            else ("budget_completed",)
        )
        and summary.get("resources_released") is True
        and summary.get("commands_sent_to_game") is False
        and summary.get("latest_candidate") == "candidate-000"
        and not summary.get("error")
        and not summary.get("release_error")
        and len(summary.get("attempts", [])) == 1
    ):
        raise ValueError("Pending sampling did not seal a matching successful child")
    attempt = summary["attempts"][0]
    if (
        attempt != json.loads(read_bounded(root / "attempt-000/cycle-result.json", 4 * 1024**2))
        or attempt.get("sampling_checkpoint_sha256") != parent["sha256"]
        or attempt.get("sampler_seed") != request.seed
        or attempt.get("candidate") != "candidate-000"
        or attempt.get("replay") != "attempt-000/prepared/replay.json"
        or attempt.get("error") is not None
        or attempt.get("inference_reload_max_error") != 0
    ):
        raise ValueError("Pending sampling result differs from its parent or sealed attempt")
    verify_sampling_sources(attempt.get("source_assets", {}))
    replay_path = root / attempt["replay"]
    if _sha(replay_path, 128 * 1024**2) != attempt["replay_sha256"]:
        raise ValueError("Pending sampling replay changed")
    replay = json.loads(read_bounded(replay_path, 128 * 1024**2))
    source_hashes = replay["source_hashes"]
    originals = {
        "packets": root / "attempt-000/recording/packets.jsonl",
        "session": root / "attempt-000/recording/session.json",
        **(
            {"execution": root / "attempt-000/execution/realtime-manifest.json"}
            if isinstance(request, SACRealtimeCycle)
            else {"trace": root / "attempt-000/trace.json"}
        ),
        "task": request.task_file,
        "reward": request.reward_file,
    }
    if any(source_hashes[key] != _sha(path, 256 * 1024**2) for key, path in originals.items()):
        raise ValueError("Pending sampling replay differs from its original evidence")
    _require_originals(
        root / "attempt-000",
        attempt["source_assets"],
        source_hashes,
        asynchronous=request if isinstance(request, SACRealtimeCycle) else None,
    )
    candidate = root / "candidate-000"
    manifest_sha = _sha(candidate / "policy.json")
    if attempt["candidate_sha256"] != manifest_sha:
        raise ValueError("Pending sampling candidate changed")
    learner = {
        "directory": str(candidate.resolve()),
        "sha256": manifest_sha,
        **validate_sac_candidate(candidate, manifest_sha),
    }
    manifest = json.loads(read_bounded(candidate / "policy.json", 4 * 1024**2))
    report = json.loads(read_bounded(candidate / "training-report.json", 128 * 1024**2))
    count, updates = attempt["eligible_transitions"], attempt["learner_updates"]
    maximum = (
        int(request.seconds_per_attempt * request.runtime.decision_hz) + 1
        if isinstance(request, SACRealtimeCycle)
        else request.steps_per_attempt
    )
    budget = (
        min(count, request.max_updates_per_attempt)
        if isinstance(request, SACRealtimeCycle)
        else count
    )
    if (
        type(count) is not int
        or not 1 <= count <= maximum
        or count != len(replay["transitions"])
        or type(updates) is not int
        or not 0 <= updates <= budget
        or (
            (updates != budget or report["stop_reason"] != "budget_completed")
            if summary["stop_reason"] == "budget_completed"
            else (updates >= budget or report["stop_reason"] != "stop_requested")
        )
        or report["steps_completed"] != updates
        or report["steps_requested"] != budget
        or learner["total_steps"] != parent["total_steps"] + updates
        or attempt["total_steps"] != learner["total_steps"]
        or manifest["continuation"]
        != {
            "parent_checkpoint_sha256": parent["sha256"],
            "parent_step": parent["total_steps"],
            "experience_additions": [attempt["replay_sha256"]],
            "new_transition_credit": count,
        }
    ):
        raise ValueError("Pending sampling has inconsistent learning progress or ancestry")
    return summary, learner
