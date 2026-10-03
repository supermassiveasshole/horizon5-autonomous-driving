"""Attach a bounded independent PresentMon QPC trace to a capture report."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fh5.capture.metrics import percentiles
from fh5.capture.pipeline import QpcMapping
from fh5.reporting.numeric import write_numeric_report

if TYPE_CHECKING:
    from fh5.result import RunResult


@dataclass(frozen=True)
class CaptureTraceReview:
    recording_dir: Path
    trace_csv: Path
    report_path: Path
    process_id: int
    swap_chain: str

    def __post_init__(self) -> None:
        if type(self.process_id) is not int or self.process_id <= 0 or not self.swap_chain:
            raise ValueError("Select an explicit FH5 process and swap chain")


def review_capture_trace(request: CaptureTraceReview) -> RunResult:
    from fh5.result import RunResult
    from fh5.telemetry.recording import write_json as _write_json

    if request.report_path.exists() or request.report_path.with_suffix(".json").exists():
        raise FileExistsError(request.report_path)
    source = (request.recording_dir / "capture.json").read_bytes()
    capture = json.loads(source)
    if capture.get("version") != 1 or capture.get("commands_sent") is not False:
        raise ValueError("Expected a passive capture report")
    mappings = [
        frame["source_layout"]["qpc"]
        for row in capture["decisions"]
        for frame in row["frames"]
        if frame["time_quality"] == "dxgi_qpc"
    ]
    if not mappings:
        raise ValueError(
            "Independent trace requires recorded native QPC mapping, not a proxy clock"
        )
    mapping = QpcMapping(**mappings[0])
    start, end = capture["started_ns"], capture["ended_ns"]
    with request.trace_csv.open("rb") as stream:
        payload = stream.read(32 * 1024**2 + 1)
    if len(payload) > 32 * 1024**2:
        raise ValueError("Frame trace exceeds diagnostic budget")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    required = {"Application", "ProcessID", "SwapChainAddress", "CPUStartQPC", "MsBetweenPresents"}
    if not required <= set(reader.fieldnames or ()):
        raise ValueError("PresentMon trace requires QPC ticks and MsBetweenPresents columns")
    intervals: list[float] = []
    displayed: list[float] = []
    selected = rejected = 0
    for index, row in enumerate(reader):
        if index >= 100_000:
            raise ValueError("Frame trace exceeds row budget")
        if (
            row["Application"].casefold() != "forzahorizon5.exe"
            or row["ProcessID"] != str(request.process_id)
            or row["SwapChainAddress"].casefold() != request.swap_chain.casefold()
        ):
            rejected += 1
            continue
        stamp = mapping.convert(int(row["CPUStartQPC"]))
        if not start <= stamp <= end:
            rejected += 1
            continue
        selected += 1
        for name, target in (("MsBetweenPresents", intervals), ("DisplayedTime", displayed)):
            value = row.get(name)
            if value is None or value in ("", "NA", "N/A"):
                continue
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError("Invalid frame-time value")
            target.append(number)
    capture["game_frame_time"] = {
        "status": "imported" if selected else "no_matching_frames",
        "frame_count": selected,
        "excluded_rows": rejected,
        "process_id": request.process_id,
        "swap_chain": request.swap_chain,
        "ms_between_presents": percentiles(intervals),
        "displayed_time_ms": percentiles(displayed),
        "trace_sha256": hashlib.sha256(payload).hexdigest(),
        "capture_sha256": hashlib.sha256(source).hexdigest(),
        "time_filter": "CPUStartQPC mapped to capture monotonic interval",
        "source": "external PresentMon CSV; collection conditions require independent verification",
        "interpretation": "MsBetweenPresents is application Present-call spacing; DisplayedTime is screen residency, not GPU render duration",
    }
    capture["dynamic_game_validation"] = False
    request.report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(request.report_path.with_suffix(".json"), capture)
    write_numeric_report(request.report_path, capture, request.recording_dir)
    return RunResult({}, [], [], {"capture": capture}, request.report_path)
