"""Transactional candidate roles backed by complete learner and evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from fh5.artifact_io import VerifiedFile, read_json, sha256_file
from fh5.candidate_archive import CandidateArchive, CandidateRestore
from fh5.candidate_selection import CandidateCompare, _eligibility
from fh5.collection_store import encode, read_bounded, write_file
from fh5.numeric_images import asset
from fh5.replay_document import read_document_fields, replay_document
from fh5.sac_source_files import recording_origins

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class CandidateRecord:
    config_file: Path
    store_dir: Path
    expected_revision: str | None
    registry_file: Path


@dataclass(frozen=True)
class CandidateHistory:
    store_dir: Path
    after_sequence: int = 0
    limit: int | None = None


@dataclass(frozen=True)
class CandidateRollback:
    store_dir: Path
    expected_revision: str
    target_revision: str
    reason: str
    registry_file: Path


@contextmanager
def _database(root: Path, *, writable: bool = False) -> Iterator[sqlite3.Connection]:
    database = root / "state.sqlite"
    try:
        db = sqlite3.connect(
            str(database) if writable else database.as_uri() + "?mode=ro",
            uri=not writable,
            timeout=5,
        )
    except sqlite3.Error as error:
        raise ValueError("Candidate store unavailable: " + str(error)) from error
    try:
        if writable:
            db.execute("BEGIN IMMEDIATE")
        yield db
        if writable:
            db.commit()
    except sqlite3.Error as error:
        db.rollback()
        raise ValueError("Candidate store unavailable: " + str(error)) from error
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def _events(db: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    if db.execute("PRAGMA user_version").fetchone()[0] != 1:
        raise ValueError("Unsupported candidate store")
    rows = db.execute("SELECT sequence, revision, payload FROM events ORDER BY sequence")
    count = 0
    parent = None
    scope = None
    for number, revision, raw in rows:
        if hashlib.sha256(raw).hexdigest() != revision:
            raise ValueError("Candidate history changed")
        event = json.loads(raw)
        if number != count + 1 or event["parent"] != parent:
            raise ValueError("Candidate history is not continuous")
        if event["scope"] not in ("synthetic_development_only", "native_development_only"):
            raise ValueError("Unsupported candidate qualification scope")
        if scope is not None and event["scope"] != scope:
            raise ValueError("Candidate history qualification scope changed")
        scope = event["scope"]
        yield {**event, "revision": revision}
        count += 1
        parent = revision
    if not count:
        raise ValueError("Candidate store requires committed events")


def _backup_progress(status: int, _remaining: int, _total: int) -> None:
    if status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        raise ValueError(
            "Candidate history source is busy or locked; retry after the writer releases it"
        )


def _read_events(
    root: Path, *, digest_file: Callable[[Path], str] = sha256_file
) -> Iterator[dict[str, Any]]:
    database = root / "state.sqlite"
    if not database.is_file():
        raise ValueError("Missing candidate store")
    try:
        with TemporaryDirectory(prefix="fh5-candidate-history-") as temporary:
            with closing(sqlite3.connect(Path(temporary) / "history.sqlite")) as snapshot:
                with _database(root) as source:
                    # SQLite's smallest copy unit permits writes between backup steps.
                    # Release the source before attachment I/O or caller traversal.
                    source.backup(snapshot, pages=1, progress=_backup_progress)
                for event in _events(snapshot):
                    for name, digest in event["evidence"].items():
                        if digest_file(asset(root, name)) != digest:
                            raise ValueError("Candidate history evidence changed: " + name)
                    yield event
    except sqlite3.Error as error:
        raise ValueError("Candidate store unavailable: " + str(error)) from error


def _latest(events: Iterator[dict[str, Any]]) -> dict[str, Any]:
    latest = None
    for latest in events:
        pass
    if latest is None:
        raise ValueError("Candidate store requires committed events")
    return latest


def _commit(root: Path, event: dict[str, Any], expected: str | None) -> str:
    with _database(root, writable=True) as db:
        if expected is None:
            if db.execute("PRAGMA user_version").fetchone()[0] != 0:
                raise ValueError("Candidate store was initialized concurrently")
            db.execute(
                "CREATE TABLE events (sequence INTEGER PRIMARY KEY, revision TEXT UNIQUE NOT NULL, payload BLOB NOT NULL)"
            )
            db.execute("PRAGMA user_version=1")
            number = 1
        else:
            number = 1
            latest = None
            for latest in _events(db):
                number += 1
            if latest is None or latest["revision"] != expected:
                raise ValueError("Candidate store revision changed")
            if latest["scope"] != event["scope"]:
                raise ValueError("Candidate store qualification scope changed")
        raw = encode(event)
        revision = hashlib.sha256(raw).hexdigest()
        db.execute("INSERT INTO events VALUES (?, ?, ?)", (number, revision, raw))
        return revision


def _publish(root: Path, work: Path, event: dict[str, Any]) -> RunResult:
    from fh5.experiment import RunResult

    event = {
        **event,
        "comparison": (work / "comparison/selection.json").relative_to(root).as_posix(),
        "request": (work / "request.json").relative_to(root).as_posix(),
    }
    event["evidence"] = {
        event[key]: sha256_file(asset(root, event[key])) for key in ("comparison", "request")
    }
    revision = _commit(root, event, event["parent"])
    return RunResult(
        {}, [], [], {"candidate_store": {**event, "revision": revision}}, root / "state.sqlite"
    )


def _qualification_gate(
    side: str,
    reviews: dict[str, Any],
    conditions: dict[str, Any],
    checkpoint: Path,
    scope: str,
) -> list[str]:
    native = scope == "native_development_only"
    source_kind = "native" if native else "synthetic"
    review = reviews[side]
    reasons = _eligibility(side, review, conditions["criteria"])
    if any(
        not group["valid_duration_s"]["count"] or group["valid_duration_s"]["min"] <= 0
        for view in conditions["plan_by_reference"]
        for group in (review["by_reference"][view],)
    ):
        reasons.append(side + ":insufficient_valid_reference_group")
    actual = sum(
        item["status"] == "bound_diagnostic"
        and item["verified_predictions"] > 0
        and item.get("actor_kind") == "frozen-numeric-sac-v1"
        and (item.get("metrics") or {}).get("evidence_kind") == source_kind
        for item in review["executions"]
    )
    if actual != review["planned_runs"]:
        reasons.append(f"{side}:actual_{source_kind}_policy_execution_incomplete")
    if review["verified_starts"] != review["planned_runs"] or any(
        item.get("source_kind") != ("udp" if native else "synthetic") for item in review["starts"]
    ):
        reasons.append(f"{side}:verified_{source_kind}_starts_incomplete")
    if review["independence"]["status"] != "no_known_overlap":
        reasons.append(side + ":development_origins_not_separate_in_registry")
    policy = read_json(checkpoint / "policy.json")
    bc = read_document_fields(
        VerifiedFile(checkpoint / "bc/model.json", policy["bc_manifest_sha256"]),
        {"provenance"},
    )
    provenance = bc.get("provenance", {})
    if native:
        if (
            policy.get("source_kind") not in ("native", "mixed")
            or provenance.get("kind") != "continuous_numeric_collection"
            or provenance.get("diagnostic_only") is not False
        ):
            reasons.append(side + ":unsupported_training_lineage")
        frozen_conditions = conditions["conditions"].get("numeric_input_conditions")
        if (
            conditions.get("inference_device") != "cpu"
            or not isinstance(frozen_conditions, dict)
            or frozen_conditions.get("status") != "confirmed"
            or provenance.get("input_conditions") != frozen_conditions
        ):
            reasons.append(side + ":native_training_conditions_mismatch")
    elif provenance.get("kind") != "synthetic":
        reasons.append(side + ":unsupported_training_lineage")
    source = VerifiedFile(checkpoint / "experience/replay.json", policy["replay_sha256"])
    with replay_document(source) as replay:
        training = recording_origins(checkpoint / "experience", replay)
    origins = {
        row["source_hashes"]["packets"]
        for evaluation in reviews.values()
        for row in evaluation["attempts"]
        if row.get("source_hashes")
    }
    if origins & training:
        reasons.append(side + ":evaluation_reuses_learning_recording")
    return reasons


def record_candidate(request: CandidateRecord) -> RunResult:
    from fh5.experiment import run_experiment

    root = request.store_dir.resolve()
    previous = None
    if request.expected_revision is None:
        if root.exists():
            raise FileExistsError(root)
    else:
        previous = _latest(_read_events(root))
        if previous["revision"] != request.expected_revision:
            raise ValueError("Candidate store revision changed")
    raw = read_bounded(request.config_file, 1024**2)
    config = json.loads(raw)
    if set(config) != {"version", "comparison", "checkpoints"} or config["version"] not in (1, 2):
        raise ValueError("Unsupported candidate record configuration")
    source_kind = "native" if config["version"] == 2 else "synthetic"
    scope = source_kind + "_development_only"
    if previous is not None and previous["scope"] != scope:
        raise ValueError("Candidate store qualification scope changed")
    if set(config["checkpoints"]) != {"incumbent", "candidate"}:
        raise ValueError("Candidate record requires both complete checkpoints")
    checkpoints = {
        side: (request.config_file.parent / path).resolve()
        for side, path in config["checkpoints"].items()
    }
    if any(root.is_relative_to(path) or path.is_relative_to(root) for path in checkpoints.values()):
        raise ValueError("Candidate store must be separate from source checkpoints")
    comparison_file = (request.config_file.parent / config["comparison"]["file"]).resolve()
    comparison_bytes = read_bounded(comparison_file, 1024**2)
    if hashlib.sha256(comparison_bytes).hexdigest() != config["comparison"]["sha256"]:
        raise ValueError("Candidate comparison changed")
    root.mkdir(parents=True, exist_ok=previous is not None)
    folder = "events/" + uuid.uuid4().hex
    work = root / folder
    work.mkdir(parents=True)
    write_file(work / "request.json", raw)
    reviewed = run_experiment(
        CandidateCompare(comparison_file, work / "comparison", request.registry_file)
    ).summary["candidate_selection"]
    if reviewed["comparison_sha256"] != config["comparison"]["sha256"]:
        raise ValueError("Candidate comparison changed during review")
    protocol = hashlib.sha256(encode(reviewed["conditions"])).hexdigest()
    if previous is not None and (
        previous["protocol_sha256"] != protocol
        or previous["default"]["model_sha256"] != reviewed["models"]["incumbent"]
    ):
        raise ValueError("Comparison differs from current default or frozen conditions")
    roles, gates = {}, {}
    for side, checkpoint in checkpoints.items():
        archive = work / side
        saved = run_experiment(
            CandidateArchive(
                checkpoint, archive, reviewed["models"][side], "Retain candidate selection state"
            )
        ).summary["candidate_archive"]
        gates[side] = _qualification_gate(
            side,
            reviewed["reviews"],
            reviewed["conditions"],
            archive / "checkpoint",
            scope,
        )
        roles[side] = {
            "model_sha256": saved["checkpoint_sha256"],
            "archive_sha256": saved["archive_sha256"],
            "archive": folder + "/" + side,
        }
    if gates["incumbent"]:
        raise ValueError(
            f"Incumbent lacks {source_kind} qualification: " + ", ".join(gates["incumbent"])
        )
    select = (
        not gates["candidate"] and reviewed["local_recommendation"] == "prefer_candidate_locally"
    )
    reasons = list(dict.fromkeys(reviewed["reasons"] + gates["candidate"]))
    aggressive = dict(previous["aggressive_by_reference"]) if previous is not None else {}
    if not gates["candidate"]:
        aggressive.update(
            {view: roles["candidate"] for view in reviewed["aggressive_by_reference"]}
        )
    event = {
        "version": 1,
        "scope": scope,
        "parent": request.expected_revision,
        "operation": "selection",
        "protocol_sha256": protocol,
        "default": roles["candidate" if select else "incumbent"],
        "explorer": roles["candidate"],
        "aggressive_by_reference": aggressive,
        "selection": "prefer_candidate_locally" if select else "retain_incumbent",
        "reasons": reasons,
        "qualification": {
            "comparison_file": str(comparison_file),
            "comparison_sha256": config["comparison"]["sha256"],
            "side": "candidate" if select else "incumbent",
        },
        "default_changed": False,
        "real_driving_validated": False,
        source_kind + "_default_changed": previous is not None
        and roles["candidate" if select else "incumbent"]["model_sha256"]
        != previous["default"]["model_sha256"],
        "independence_limits": (
            "Registered raw origins and independently reviewed native development evidence; "
            "no driving improvement or controller activation established"
            if source_kind == "native"
            else "Registered raw origins and explicitly synthetic training only; no native qualification"
        ),
    }
    return _publish(root, work, event)


def read_candidate_history(request: CandidateHistory) -> RunResult:
    from fh5.experiment import RunResult

    root = request.store_dir.resolve()
    if (
        type(request.after_sequence) is not int
        or request.after_sequence < 0
        or (request.limit is not None and (type(request.limit) is not int or request.limit < 0))
    ):
        raise ValueError("Candidate history page requires nonnegative integers")
    events: list[dict[str, Any]] = []
    latest = None
    count = 0
    for count, latest in enumerate(_read_events(root), start=1):
        if count > request.after_sequence and (
            request.limit is None or len(events) < request.limit
        ):
            events.append(latest)
    if latest is None:
        raise ValueError("Candidate store requires committed events")
    next_sequence = min(count, request.after_sequence + len(events))
    summary = {
        **latest,
        "history": events,
        "history_count": count,
        "next_sequence": next_sequence,
        "history_complete": next_sequence == count,
    }
    return RunResult({}, [], [], {"candidate_store": summary}, root / "state.sqlite")


def rollback_candidate(request: CandidateRollback) -> RunResult:
    from fh5.experiment import run_experiment

    root = request.store_dir.resolve()
    target = previous = None
    for previous in _read_events(root):
        if previous["revision"] == request.target_revision:
            target = previous
    if previous is None:
        raise ValueError("Candidate store requires committed events")
    if previous["revision"] != request.expected_revision:
        raise ValueError("Candidate store revision changed")
    if not isinstance(request.reason, str) or not 1 <= len(request.reason.strip()) <= 2000:
        raise ValueError("Candidate rollback requires a reason of 1..2000 characters")
    if (
        target is None
        or target["protocol_sha256"] != previous["protocol_sha256"]
        or target["scope"] != previous["scope"]
    ):
        raise ValueError("Rollback target is not a compatible retained default")
    folder = "events/" + uuid.uuid4().hex
    work = root / folder
    work.mkdir(parents=True)
    payload = encode(
        {
            "expected_revision": request.expected_revision,
            "target_revision": request.target_revision,
            "reason": request.reason.strip(),
        }
    )
    write_file(work / "request.json", payload)
    role = target["default"]
    restored = work / "verified-default"
    verified = run_experiment(
        CandidateRestore(
            asset(root, role["archive"]), restored, role["archive_sha256"], request.reason
        )
    ).summary["candidate_restore"]
    if verified["checkpoint_sha256"] != role["model_sha256"]:
        raise ValueError("Rollback model differs from retained default")
    qualification = target["qualification"]
    comparison = Path(qualification["comparison_file"])
    if (
        hashlib.sha256(read_bounded(comparison, 1024**2)).hexdigest()
        != qualification["comparison_sha256"]
    ):
        raise ValueError("Rollback comparison changed")
    reviewed = run_experiment(
        CandidateCompare(comparison, work / "comparison", request.registry_file)
    ).summary["candidate_selection"]
    side = qualification["side"]
    if (
        reviewed["comparison_sha256"] != qualification["comparison_sha256"]
        or hashlib.sha256(encode(reviewed["conditions"])).hexdigest() != target["protocol_sha256"]
        or reviewed["models"][side] != role["model_sha256"]
    ):
        raise ValueError("Rollback evaluation differs from retained default")
    reasons = _qualification_gate(
        side, reviewed["reviews"], reviewed["conditions"], restored, target["scope"]
    )
    if reasons or (
        side == "candidate" and reviewed["local_recommendation"] != "prefer_candidate_locally"
    ):
        raise ValueError(
            "Rollback target no longer qualifies: " + ", ".join(reasons + reviewed["reasons"])
        )
    source_kind = "native" if target["scope"] == "native_development_only" else "synthetic"
    event = {
        **{key: value for key, value in previous.items() if key != "revision"},
        "parent": request.expected_revision,
        "operation": "rollback",
        "target_revision": request.target_revision,
        "default": role,
        "qualification": qualification,
        "selection": "restore_retained_default",
        "reasons": [request.reason.strip()],
        source_kind + "_default_changed": role["model_sha256"]
        != previous["default"]["model_sha256"],
    }
    return _publish(root, work, event)
