"""Independent input, inference and action supervision, with bounded shutdown."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from queue import Empty
from typing import TYPE_CHECKING, Any

from fh5.numeric_images import NumericActor
from fh5.numeric_recording import NumericArchive
from fh5.realtime import RealtimeEnvironment, RealtimeObservation, RealtimeRun
from fh5.realtime_journal import RealtimeJournal
from fh5.realtime_report import write_realtime_result
from fh5.realtime_state import DecisionState
from fh5.realtime_worker import InferenceWorker

if TYPE_CHECKING:
    from fh5.experiment import RunResult


def run_realtime(
    request: RealtimeRun,
    environment: RealtimeEnvironment,
    factory: Callable[[], NumericActor],
    journal_sink: Callable[[bytes], None] | None = None,
) -> RunResult:
    if environment.source_kind not in ("synthetic", "shadow"):
        raise ValueError(
            "Real-time foundation only accepts simulated actuators or read-only shadow"
        )
    request.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("pixels", "inputs", "previews"):
        (request.output_dir / name).mkdir()
    journal = RealtimeJournal(request.output_dir, request.journal_capacity, journal_sink)
    archive = NumericArchive(request.output_dir, 8, 32 * 1024**2)
    archive_bytes = 0
    done = threading.Event()
    lock = threading.Lock()
    state = DecisionState(
        request.config,
        environment.send,
        clock=time.perf_counter_ns,
        simulated_history=environment.source_kind == "synthetic",
        notify=journal.submit,
    )
    worker = InferenceWorker(factory, request.config)
    latest: RealtimeObservation | None = None

    def receive() -> None:
        nonlocal latest
        try:
            while not done.is_set():
                value = environment.read(0.005)
                for packet in value.raw_packets:
                    journal.submit("packet", packet)
                with lock:
                    state.update(value.safety)
                    if value.safety.fault:
                        state.stop(
                            time.perf_counter_ns(),
                            "user_stop" if value.safety.stop_requested else value.safety.fault,
                        )
                    latest = value.observation
                    if value.capture_epoch is not None:
                        state.capture_epoch = value.capture_epoch
                    elif latest:
                        state.capture_epoch = latest.epoch
        except Exception as error:
            with lock:
                state.stop(time.perf_counter_ns(), f"input_error: {error}")
            done.set()

    def supervise() -> None:
        try:
            while not done.is_set():
                focused, stop = environment.signals()
                with lock:
                    if state.safety:
                        state.update(replace(state.safety, focused=focused, stop_requested=stop))
                    elif stop:
                        state.stop(time.perf_counter_ns(), "user_stop")
                    now = time.perf_counter_ns()
                    state.supervise(now)
                    try:
                        returned, work, prediction, error = worker.results.get_nowait()
                    except Empty:
                        pass
                    else:
                        work.row["worker_returned_ns"] = returned
                        state.complete(time.perf_counter_ns(), work, prediction, error)
                    if worker.error:
                        state.stop(time.perf_counter_ns(), "inference_worker_error")
                    if state.stop_reason:
                        done.set()
                done.wait(0.005)
        except Exception as error:
            with lock:
                state.stop(time.perf_counter_ns(), f"supervisor_error: {error}")
            done.set()

    threads: list[threading.Thread] = []
    started = time.perf_counter_ns()
    try:
        if not worker.ready.wait(request.startup_timeout_s) or not worker.warmup_completed:
            state.stop(time.perf_counter_ns(), "model_startup_failed")
        else:
            threads = [
                threading.Thread(target=receive, name="fh5-runtime-input", daemon=True),
                threading.Thread(target=supervise, name="fh5-action-supervisor", daemon=True),
            ]
            for thread in threads:
                thread.start()
            started = time.perf_counter_ns()
            end = started + int(request.seconds * 1e9)
            next_tick = started
            while not done.is_set() and time.perf_counter_ns() < end:
                with lock:
                    work = state.begin(time.perf_counter_ns(), latest)
                    if work:
                        worker.requests.put_nowait(work)
                        size = sum(f.pixels.nbytes for f in work.observation.frames)
                        reason: str | None = "archive_disk_budget"
                        if archive_bytes + size <= request.archive_limit_bytes:
                            reason = archive.submit(dict(work.row), work.observation.frames)
                            if reason is None:
                                archive_bytes += size
                        work.row["archive_reason"] = reason
                next_tick = max(
                    next_tick + 1_000_000_000 // request.config.decision_hz, time.perf_counter_ns()
                )
                done.wait(max(0, (min(next_tick, end) - time.perf_counter_ns()) / 1e9))
    except KeyboardInterrupt:
        with lock:
            state.stop(time.perf_counter_ns(), "interrupted")
    except Exception as error:
        with lock:
            state.stop(time.perf_counter_ns(), f"runtime_error: {error}")
    finally:
        with lock:
            state.stop(time.perf_counter_ns(), "time_limit")
        ended = time.perf_counter_ns()
        done.set()
        for thread in threads:
            thread.join(timeout=0.5)
        inference = worker.close()
        if state.pending:
            state.pending.row["status"] = "abandoned_inference"
            journal.submit("decision_abandoned", dict(state.pending.row))
        try:
            environment_result = environment.close()
        except Exception as error:
            environment_result = {"resources_released": False, "close_error": str(error)}
        archive_result = archive.close()
        journal_result = journal.close()
    image_complete = all(
        d.get("archive_reason") is None and d["decision_id"] in archive.records
        for d in state.decisions
        if "actor" in d
    )
    complete = (
        image_complete
        and not journal_result["dropped"]
        and not journal_result["error"]
        and journal_result["resources_released"]
        and not journal_result["external_sink"]
        and not archive_result["error"]
        and state.pending is None
        and all(c["status"] == "sent" for c in state.commands)
        and inference["warmup_completed"]
        and not inference["error"]
        and any(d["prediction"] is not None for d in state.decisions)
    )
    for row in state.decisions:
        row["archive"] = archive.records.get(row["decision_id"])
    result: dict[str, Any] = {
        "version": 1,
        "evidence_kind": environment.source_kind,
        "started_ns": started,
        "ended_ns": ended,
        "finalized_ns": time.perf_counter_ns(),
        "commands_sent_to_game": False,
        "decisions": state.decisions,
        "commands": state.commands,
        "stop_reason": state.stop_reason,
        "inference": inference,
        "model": worker.manifest,
        "actor_kind": worker.kind,
        "environment": environment_result,
        "journal": journal_result,
        "archive": archive_result,
        "evidence": {
            "recording_complete": complete,
            "training_eligible": False,
            "exact_replay_eligible": complete,
            "promotion_eligible": False,
            "reason": "synthetic_or_shadow_only; independent replay and validity review required"
            if complete
            else "recording_gap_or_writer_failure",
        },
        "resources_released": inference["resources_released"]
        and environment_result.get("resources_released", False)
        and all(not t.is_alive() for t in threads)
        and journal_result["resources_released"]
        and archive_result["resources_released"],
        "real_game_validation": False,
    }
    return write_realtime_result(request.output_dir, result, request.config)
