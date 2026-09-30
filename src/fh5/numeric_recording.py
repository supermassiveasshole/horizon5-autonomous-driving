"""Numerical inference evidence; asynchronous exact storage and offline reconstruction."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any

from fh5.bc_learning import _numeric
from fh5.numeric_images import (
    NumericActor,
    NumericDecision,
    NumericFrame,
    NumericInfer,
    NumericReplay,
    PixelContract,
    asset,
    validate_decision,
)
from fh5.numeric_report import preview_png, write_numeric_report

if TYPE_CHECKING:
    from fh5.experiment import RunResult


def _encode(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


class NumericArchive:
    """Only this bounded consumer hashes or writes pixels, never the decision caller."""

    def __init__(self, directory: Path, capacity: int, byte_limit: int) -> None:
        self.directory = directory
        self.capacity, self.byte_limit = capacity, byte_limit
        self.queue: Queue[tuple[dict[str, Any], tuple[NumericFrame, ...], int]] = Queue(capacity)
        self.lock = threading.Lock()
        self.pending = self.pending_bytes = self.peak_bytes = 0
        self.finished, self.aborted = threading.Event(), threading.Event()
        self.error: str | None = None
        self.preview_errors = 0
        self.records: dict[str, dict[str, Any]] = {}
        self.worker = threading.Thread(target=self._work, name="fh5-numeric-archive", daemon=True)
        self.worker.start()

    def submit(self, row: dict[str, Any], frames: tuple[NumericFrame, ...]) -> str | None:
        size = sum(f.pixels.nbytes for f in frames)
        with self.lock:
            if self.error or self.finished.is_set():
                return "archive_unavailable"
            if self.pending >= self.capacity:
                return "archive_capacity"
            if self.pending_bytes + size > self.byte_limit:
                return "archive_byte_limit"
            self.pending += 1
            self.pending_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.pending_bytes)
        try:
            self.queue.put_nowait((row, frames, size))
        except Full:
            with self.lock:
                self.pending -= 1
                self.pending_bytes -= size
            return "archive_capacity"
        return None

    def _work(self) -> None:
        try:
            while not self.aborted.is_set():
                try:
                    row, frames, size = self.queue.get(timeout=0.01)
                except Empty:
                    if self.finished.is_set():
                        break
                    continue
                try:
                    stored = []
                    previews: list[str | None] = []
                    for frame in frames:
                        payload = bytes(frame.pixels)
                        digest = hashlib.sha256(payload).hexdigest()
                        path = f"pixels/{digest}.rgb"
                        target = self.directory / path
                        if not target.exists():
                            target.write_bytes(payload)
                        stored.append(dict(frame.metadata(), path=path, sha256=digest))
                        preview = f"previews/{digest}.png"
                        try:
                            if not (self.directory / preview).exists():
                                (self.directory / preview).write_bytes(
                                    preview_png(payload, frame.size)
                                )
                            previews.append(preview)
                        except OSError:
                            self.preview_errors += 1
                            previews.append(None)
                    document = _encode(dict(row, frames=stored))
                    path = f"inputs/{row['index']:06d}.json"
                    if self.aborted.is_set():
                        break
                    (self.directory / path).write_bytes(document)
                    self.records[row["decision_id"]] = {
                        "path": path,
                        "sha256": hashlib.sha256(document).hexdigest(),
                        "previews": previews,
                    }
                finally:
                    with self.lock:
                        self.pending -= 1
                        self.pending_bytes -= size
                del frames, row
        except Exception as error:
            self.error = str(error)

    def close(self) -> dict[str, Any]:
        self.finished.set()
        self.worker.join(timeout=2)
        released = not self.worker.is_alive()
        if not released:
            self.aborted.set()
            self.error = "Numerical archive worker did not finish within its shutdown budget"
        while True:
            try:
                _, _, size = self.queue.get_nowait()
            except Empty:
                break
            with self.lock:
                self.pending -= 1
                self.pending_bytes -= size
        return {
            "resources_released": released,
            "error": self.error,
            "peak_pending_bytes": self.peak_bytes,
            "capacity": self.capacity,
            "byte_limit": self.byte_limit,
            "pending_after_close": self.pending,
            "preview_errors": self.preview_errors,
        }


def _prediction(actor: NumericActor, decision: NumericDecision) -> list[float]:
    result = actor.predict(json.loads(json.dumps(decision.actor)), decision.frames)
    if len(result) != 2 or any(not math.isfinite(v) or abs(v) > 1 for v in result):
        raise ValueError("Numerical actor must return two finite bounded actions")
    return result


def _result(
    path: Path, summary: dict[str, Any], *, section: str = "numeric", root: Path | None = None
) -> RunResult:
    from fh5.experiment import RunResult

    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_bytes(_encode(summary))
    write_numeric_report(path, summary, root or path.parent)
    return RunResult({"source_kind": "numeric_diagnostic"}, [], [], {section: summary}, path)


def infer_numeric(
    request: NumericInfer, actor: NumericActor, inputs: Iterable[NumericDecision]
) -> RunResult:
    expected = request.contract.metadata()
    if actor.manifest.get("numeric_contract", expected) != expected:
        raise ValueError("Inference pixel contract differs from frozen model")
    if getattr(inputs, "contract", request.contract) != request.contract:
        raise ValueError("Prepared input pixel contract differs from inference")
    directory = request.output_dir
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "pixels").mkdir()
    (directory / "inputs").mkdir()
    (directory / "previews").mkdir()
    archive = NumericArchive(directory, request.archive_capacity, request.archive_bytes)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "version": 1,
        "contract": request.contract.metadata(),
        "actor_kind": actor.kind,
        "model": json.loads(json.dumps(actor.manifest)),
        "commands_sent": False,
        "decisions": rows,
        "stop_reason": "source_end",
    }
    seen = set()
    try:
        for index, decision in enumerate(islice(inputs, request.max_decisions)):
            if decision.decision_id in seen:
                raise ValueError("Duplicate numerical decision identity")
            seen.add(decision.decision_id)
            reason = validate_decision(decision, request.contract)
            row = {
                "index": index,
                "decision_id": decision.decision_id,
                "epoch": decision.epoch,
                "decision_ns": decision.decision_ns,
                "actor": json.loads(json.dumps(decision.actor)),
                "frames": [f.metadata() for f in decision.frames],
                "features": None,
                "prediction": None,
                "status": "predicted" if reason is None else "skipped",
                "reason": reason,
            }
            if reason is None:
                try:
                    row["features"] = _numeric(decision.actor)
                    row["prediction"] = _prediction(actor, decision)
                except Exception as error:
                    rows.append(
                        dict(
                            row,
                            status="error",
                            reason=str(error),
                            archive_accepted=False,
                            archive_reason="execution_error",
                        )
                    )
                    raise
            archive_reason = archive.submit(row, decision.frames) if reason is None else "skipped"
            rows.append(
                dict(row, archive_accepted=archive_reason is None, archive_reason=archive_reason)
            )
        if len(rows) == request.max_decisions:
            summary["stop_reason"] = "decision_limit"
    except Exception as error:
        summary.update(
            stop_reason="execution_error", execution_error=f"{type(error).__name__}: {error}"
        )
    finally:
        closer = getattr(inputs, "close", None)
        if closer is not None:
            try:
                summary["source_released"] = closer() is not False
            except Exception as error:
                summary.update(source_released=False, source_close_error=str(error))
        else:
            summary["source_released"] = True
        clear = getattr(actor, "clear_input_cache", None)
        if clear is not None:
            clear()
        summary["archive"] = archive.close()
    for row in rows:
        reference = archive.records.get(row["decision_id"])
        row["archive"] = reference
        row["exact_replay_available"] = reference is not None
    (directory / "numeric-run.json").write_bytes(_encode(summary))
    return _result(directory / "report.html", summary)


def replay_numeric(request: NumericReplay, actor: NumericActor) -> RunResult:
    directory = request.recording_dir
    summary = json.loads((directory / "numeric-run.json").read_text(encoding="utf-8"))
    if summary.get("version") != 1 or summary.get("commands_sent") is not False:
        raise ValueError("Unsupported numerical recording")
    contract = PixelContract.from_metadata(summary["contract"])
    if actor.manifest.get("numeric_contract", contract.metadata()) != contract.metadata():
        raise ValueError("Replay pixel contract differs from frozen model")
    if summary["model"] != actor.manifest or summary["actor_kind"] != actor.kind:
        raise ValueError("Numerical replay model differs from the recorded model")
    summary["replay_errors"] = []
    for row in summary["decisions"]:
        if row["status"] != "predicted":
            continue
        try:
            reference = row["archive"]
            if not reference:
                raise ValueError("Numerical input was not archived")
            payload = asset(directory, reference["path"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
                raise ValueError("Numerical input metadata hash mismatch")
            recorded = json.loads(payload)
            for key in (
                "index",
                "decision_id",
                "epoch",
                "decision_ns",
                "actor",
                "features",
                "prediction",
                "status",
                "reason",
            ):
                if recorded[key] != row[key]:
                    raise ValueError("Numerical summary differs from archived decision")
            frames = []
            for frame in recorded["frames"]:
                pixels = asset(directory, frame["path"]).read_bytes()
                if hashlib.sha256(pixels).hexdigest() != frame["sha256"]:
                    raise ValueError("Numerical pixel hash mismatch")
                metadata = {k: v for k, v in frame.items() if k not in ("path", "sha256")}
                metadata["size"] = tuple(metadata["size"])
                frames.append(NumericFrame(pixels=memoryview(pixels), **metadata))
            features = _numeric(recorded["actor"])
            row.update(pixels_match=True, features_match=features == recorded["features"])
            decision = NumericDecision(
                recorded["decision_id"],
                recorded["epoch"],
                recorded["decision_ns"],
                tuple(frames),
                recorded["actor"],
            )
            if [f.metadata() for f in frames] != row["frames"]:
                raise ValueError("Numerical frame metadata differs from summary")
            reason = validate_decision(decision, contract)
            if reason is not None:
                raise ValueError(f"Invalid replay observation: {reason}")
            prediction = _prediction(actor, decision)
            error = max(abs(a - b) for a, b in zip(prediction, recorded["prediction"]))
            row.update(replayed_prediction=prediction, prediction_max_abs_error=error)
            if not row["features_match"] or error > request.tolerance:
                raise ValueError("Numerical features or prediction failed exact replay tolerance")
        except (OSError, ValueError, KeyError, TypeError) as error:
            row.update(exact_replay_available=False, replay_error=str(error))
            summary["replay_errors"].append(
                {"decision_id": row["decision_id"], "error": str(error)}
            )
    clear = getattr(actor, "clear_input_cache", None)
    if clear is not None:
        clear()
    summary["replay_tolerance"] = request.tolerance
    return _result(request.report_path, summary, root=directory)
