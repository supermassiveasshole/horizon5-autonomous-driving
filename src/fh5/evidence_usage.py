"""Transactional local evidence-use history; absence of a match is not independence proof."""

from __future__ import annotations

import hashlib
import html
import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fh5.collection_store import encode, read_bounded, write_file

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class RecordUsage:
    registry_file: Path
    recordings: tuple[Path, ...]
    role: Literal["training", "selection"]
    output_dir: Path
    model_sha256: str | None = None


class RegisteredSlotChanged(ValueError):
    """A previously reviewed attempt cannot disappear or become another recording."""


@contextmanager
def _registry(path: Path, *, create: bool = False) -> Iterator[sqlite3.Connection]:
    existed = path.exists()
    if not existed and not create:
        raise ValueError("Evidence usage registry does not exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        db = sqlite3.connect(path, timeout=5)
    except sqlite3.Error as error:
        raise ValueError("Evidence usage registry unavailable: " + str(error)) from error
    db.row_factory = sqlite3.Row
    try:
        db.execute("BEGIN IMMEDIATE")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if not existed:
            db.execute("CREATE TABLE identity (value TEXT NOT NULL)")
            db.execute("INSERT INTO identity VALUES (?)", (str(uuid.uuid4()),))
            db.execute(
                "CREATE TABLE uses (source_key TEXT NOT NULL, role TEXT NOT NULL, "
                "batch TEXT NOT NULL, model TEXT NOT NULL, origin TEXT NOT NULL, "
                "UNIQUE(source_key, role, batch, model, origin))"
            )
            db.execute(
                "CREATE TABLE reservations (batch TEXT PRIMARY KEY, model TEXT NOT NULL, "
                "purpose TEXT NOT NULL, reserved_utc TEXT NOT NULL)"
            )
            version = 1
        elif version not in (1, 2):
            raise ValueError("Unsupported evidence usage registry")
        if version == 1:
            had_slots = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='slots'"
            ).fetchone()
            db.execute(
                "CREATE TABLE IF NOT EXISTS slots (batch TEXT NOT NULL, slot TEXT NOT NULL, "
                "binding TEXT NOT NULL, PRIMARY KEY(batch, slot))"
            )
            db.execute("CREATE TABLE legacy_reviews (batch TEXT PRIMARY KEY)")
            if not had_slots:
                # Empty/unreadable old reviews may have no uses at all. Reservations
                # cannot distinguish those reviews from batches never started.
                db.execute(
                    "INSERT INTO legacy_reviews SELECT batch FROM reservations "
                    "UNION SELECT batch FROM uses WHERE batch<>''"
                )
            db.execute("PRAGMA user_version=2")
        if db.execute("SELECT count(*) FROM uses").fetchone()[0] > 20_000:
            raise ValueError("Evidence usage registry exceeds 20000-key bound")
        if db.execute("SELECT count(*) FROM slots").fetchone()[0] > 20_000:
            raise ValueError("Evidence usage registry exceeds 20000-slot bound")
        yield db
        if db.execute("SELECT count(*) FROM uses").fetchone()[0] > 20_000:
            raise ValueError("Evidence usage registry exceeds 20000-key bound")
        if db.execute("SELECT count(*) FROM slots").fetchone()[0] > 20_000:
            raise ValueError("Evidence usage registry exceeds 20000-slot bound")
        db.commit()
    except sqlite3.Error as error:
        db.rollback()
        raise ValueError("Evidence usage registry unavailable: " + str(error)) from error
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def source_keys(files: dict[str, str]) -> list[str]:
    if files.get("packets.jsonl") in (None, hashlib.sha256(b"").hexdigest()):
        return []
    return [f"{name}:{files[name]}" for name in ("session.json", "packets.jsonl")]


def _snapshot(db: sqlite3.Connection) -> dict[str, Any]:
    return {
        "version": 2,
        "registry_id": db.execute("SELECT value FROM identity").fetchone()[0],
        "uses": [
            dict(row)
            for row in db.execute(
                "SELECT * FROM uses ORDER BY source_key, role, batch, model, origin"
            )
        ],
        "reservations": [
            dict(row) for row in db.execute("SELECT * FROM reservations ORDER BY batch")
        ],
        "slots": [dict(row) for row in db.execute("SELECT * FROM slots ORDER BY batch, slot")],
        "legacy_reviews": [
            row[0] for row in db.execute("SELECT batch FROM legacy_reviews ORDER BY batch")
        ],
    }


def _insert(
    db: sqlite3.Connection, sources: list[dict[str, Any]], role: str, batch: str, model: str
) -> None:
    for source in sources:
        for key in source["keys"]:
            db.execute(
                "INSERT OR IGNORE INTO uses VALUES (?, ?, ?, ?, ?)",
                (key, role, batch, model, source["origin"]),
            )


def reserve_batch(registry: Path, batch: dict[str, Any]) -> None:
    with _registry(registry, create=True) as db:
        if db.execute("SELECT count(*) FROM reservations").fetchone()[0] >= 10_000:
            raise ValueError("Evidence registry exceeds 10000 reservations")
        batch["usage_registry_id"] = db.execute("SELECT value FROM identity").fetchone()[0]
        digest = hashlib.sha256(encode(batch)).hexdigest()
        db.execute(
            "INSERT INTO reservations VALUES (?, ?, ?, ?)",
            (
                digest,
                batch["config"]["model"]["manifest_sha256"],
                batch["config"]["purpose"],
                datetime.now(UTC).isoformat(),
            ),
        )


def bind_evaluation_slots(
    registry: Path | None, batch_sha256: str, entries: list[dict[str, Any]]
) -> str | None:
    if registry is None:
        return None
    bindings = {row["slot_id"]: hashlib.sha256(encode(row["files"])).hexdigest() for row in entries}
    try:
        with _registry(registry) as db:
            previous = {
                row["slot"]: row["binding"]
                for row in db.execute(
                    "SELECT slot, binding FROM slots WHERE batch=?", (batch_sha256,)
                )
            }
            if any(bindings.get(slot) != binding for slot, binding in previous.items()):
                raise RegisteredSlotChanged(
                    "A registered evaluation slot was omitted or its recording changed; retain the original attempts"
                )
            db.executemany(
                "INSERT OR IGNORE INTO slots VALUES (?, ?, ?)",
                [(batch_sha256, slot, binding) for slot, binding in bindings.items()],
            )
    except RegisteredSlotChanged:
        raise
    except (OSError, ValueError, sqlite3.Error) as error:
        # The review still reports all supplied attempts and the registry error.
        # A missing/broken registry never creates positive independence evidence.
        return str(error)
    return None


def _recorded_after(created: Any, reserved: str) -> bool:
    try:
        recording, reservation = datetime.fromisoformat(created), datetime.fromisoformat(reserved)
        return (
            recording.tzinfo is not None
            and reservation.tzinfo is not None
            and reservation <= recording <= datetime.now(UTC)
        )
    except (TypeError, ValueError):
        return False


def record_usage(request: RecordUsage) -> RunResult:
    from fh5.experiment import RunResult

    if request.role not in ("training", "selection") or not 1 <= len(request.recordings) <= 1000:
        raise ValueError("Evidence use needs training/selection and 1..1000 recordings")
    if request.model_sha256 is not None and (
        len(request.model_sha256) != 64
        or any(c not in "0123456789abcdef" for c in request.model_sha256)
    ):
        raise ValueError("Evidence use model binding must be SHA-256")
    if request.output_dir.exists():
        raise FileExistsError(request.output_dir)
    sources = []
    for directory in request.recordings:
        files = {
            name: hashlib.sha256(read_bounded(directory / name, 128 * 1024**2)).hexdigest()
            for name in ("session.json", "packets.jsonl")
        }
        sources.append(
            {"keys": source_keys(files), "origin": files["packets.jsonl"], "files": files}
        )
    with _registry(request.registry_file, create=True) as db:
        _insert(db, sources, request.role, "", request.model_sha256 or "")
        snapshot = _snapshot(db)
    summary = {
        "version": 1,
        "role": request.role,
        "model_sha256": request.model_sha256,
        "recordings": len(sources),
        "unidentified_recordings": sum(not s["keys"] for s in sources),
        "sources": sources,
        "registry_id": snapshot["registry_id"],
        "snapshot_sha256": hashlib.sha256(encode(snapshot)).hexdigest(),
        "registered_utc": datetime.now(UTC).isoformat(),
        "declaration_only": True,
        "commands_sent": False,
    }
    request.output_dir.mkdir(parents=True)
    write_file(request.output_dir / "usage.json", encode(summary))
    write_file(request.output_dir / "usage-snapshot.json", encode(snapshot))
    path = request.output_dir / "report.html"
    path.write_text(
        '<!doctype html><meta charset="utf-8"><h1>证据用途登记</h1>'
        "<p>登记声明的用途与文件指纹；不证明未登记历史完整。</p><pre>"
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>",
        encoding="utf-8",
    )
    return RunResult({}, [], [], {"evidence_usage": summary}, path)


def review_usage(
    registry: Path | None,
    batch: dict[str, Any],
    batch_sha256: str,
    sources: list[dict[str, Any]],
    output: Path,
    binding_error: str | None = None,
) -> dict[str, Any]:
    if registry is None:
        return {"status": "untracked", "independence_proven": False, "conflicts": []}
    if binding_error is not None:
        return {
            "status": "unknown",
            "independence_proven": False,
            "conflicts": [],
            "error": binding_error,
        }
    try:
        return _review_usage(registry, batch, batch_sha256, sources, output)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        return {
            "status": "unknown",
            "independence_proven": False,
            "conflicts": [],
            "error": str(error),
        }


def _review_usage(
    registry: Path,
    batch: dict[str, Any],
    batch_sha256: str,
    sources: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    with _registry(registry) as db:
        before = _snapshot(db)
        reservation = next((r for r in before["reservations"] if r["batch"] == batch_sha256), None)
        reserved = bool(
            reservation
            and batch.get("usage_registry_id") == before["registry_id"]
            and reservation["model"] == batch["config"]["model"]["manifest_sha256"]
            and reservation["purpose"] == batch["config"]["purpose"]
        )
        unknown = []
        if batch_sha256 in before["legacy_reviews"]:
            unknown.append("legacy_slot_history_unavailable")
        if not reserved:
            unknown.append("batch_not_reserved_in_this_registry")
        if len(sources) != len(batch["config"]["plan"]):
            unknown.append("unstarted_slots")
        for source in sources:
            if not source["keys"]:
                unknown.append("unidentified_source:" + source["slot_id"])
            if reservation and not _recorded_after(
                source.get("created_utc"), reservation["reserved_utc"]
            ):
                unknown.append("recording_not_known_after_reservation:" + source["slot_id"])
        conflicts = []
        for source in sources:
            for prior in before["uses"]:
                if prior["source_key"] in source["keys"] and prior["batch"] != batch_sha256:
                    conflicts.append(
                        {"slot_id": source["slot_id"], "prior_role": prior["role"], **prior}
                    )
        _insert(
            db,
            sources,
            "final" if batch["config"]["purpose"] == "final" else "selection",
            batch_sha256,
            batch["config"]["model"]["manifest_sha256"],
        )
        snapshot = _snapshot(db)
    write_file(output / "usage-snapshot.json", encode(snapshot))
    return {
        "status": "known_overlap" if conflicts else "unknown" if unknown else "no_known_overlap",
        "reservation_verified": reserved,
        "unknown_reasons": unknown,
        "independence_proven": False,
        "conflicts": conflicts,
        "registry_id": snapshot["registry_id"],
        "snapshot_sha256": hashlib.sha256(encode(snapshot)).hexdigest(),
        "unidentified_slots": [s["slot_id"] for s in sources if not s["keys"]],
        "limits": "Exact file fingerprints and registered uses only; unknown or transformed history is not proven independent",
    }
