"""Compose telemetry, control, route and observation evidence for a recording."""

from __future__ import annotations

from collections.abc import Iterable

from fh5.capture.legacy import read_vision
from fh5.driving.control import read_control
from fh5.driving.events import read_event
from fh5.driving.policy_recording import read_policy
from fh5.driving.tracking import read_tracking_route
from fh5.observation.multimodal import ObservationReplay, build_observations, read_settings
from fh5.observation.routes import (
    BuildRoute,
    RouteCheck,
    build_route,
    check_route_recording,
    load_route,
    locate_route,
)
from fh5.reporting.telemetry import write_report
from fh5.result import RunResult
from fh5.telemetry.packet import Packet, Record, Replay
from fh5.telemetry.recording import run_recording


def run_recording_report(
    request: Record | Replay | BuildRoute | RouteCheck | ObservationReplay,
    *,
    packets: Iterable[Packet] | None = None,
) -> RunResult:
    """Record or replay telemetry, compose available evidence, and render its report."""
    if isinstance(request, Record):
        result = run_recording(request, packets=packets)
        directory = request.output_dir
    else:
        result = run_recording(Replay(request.recording_dir, request.report_path))
        directory = request.recording_dir
    metadata = result.metadata
    samples = result.samples
    events = result.events
    summary = result.summary
    report_path = result.report_path
    packet_count = summary["packet_count"]
    if metadata.get("control_source") == "calibration" or (directory / "control.json").exists():
        try:
            summary["control"] = read_control(directory, samples)
        except ValueError as error:
            events.append(
                {"kind": "control_evidence_incomplete", "packet_index": None, "detail": str(error)}
            )
    if (directory / "policy.json").exists():
        summary["policy"] = read_policy(directory)
        summary["route"] = summary["policy"]["evaluation_route"]
    if summary.get("control", {}).get("controller_kind") == "route-feedback-v1":
        try:
            summary["route"] = read_tracking_route(directory, summary["control"])
            events.extend(locate_route(samples, summary["route"]))
        except (OSError, ValueError, KeyError) as error:
            events.append(
                {"kind": "tracking_route_incomplete", "packet_index": None, "detail": str(error)}
            )
    if any(
        (directory / name).exists()
        for name in ("event-run.json", "event-config.json", "event-journal.jsonl")
    ):
        event_run = read_event(directory)
        summary["event_run"] = event_run["summary"]
        events.extend(event_run["events"])
    if isinstance(request, (BuildRoute, RouteCheck)):
        route = (
            build_route(request, samples)
            if isinstance(request, BuildRoute)
            else load_route(request.route_file)
        )
        recording_packet_count = packet_count
        samples = [
            s for s in samples if request.first_packet <= s["packet_index"] <= request.last_packet
        ]
        if not samples or request.last_packet >= packet_count:
            raise ValueError("Local analysis packet range is outside the recording")
        events = [
            e
            for e in events
            if e.get("packet_index") is not None
            and request.first_packet <= e["packet_index"] <= request.last_packet
        ]
        metadata["analysis_packet_range"] = [request.first_packet, request.last_packet]
        selected_count = request.last_packet - request.first_packet + 1
        summary.update(
            packet_count=selected_count,
            valid_packets=len(samples),
            active_packets=sum(s["is_race_on"] for s in samples),
            invalid_packets=selected_count - len(samples),
            segments=len({s["segment"] for s in samples}),
            receive_span_seconds=(
                samples[-1]["received_monotonic_ns"] - samples[0]["received_monotonic_ns"]
            )
            / 1e9,
        )
        events.extend(locate_route(samples, route))
        summary["route"] = route
        if isinstance(request, RouteCheck):
            summary["route_check"] = check_route_recording(
                request, samples, route, metadata, recording_packet_count
            )
    elif (
        isinstance(request, (Replay, ObservationReplay))
        and request.route_file is not None
        and (
            not isinstance(request, ObservationReplay)
            or read_settings(request.config_file)["version"] == 1
        )
    ):
        route = load_route(request.route_file)
        events.extend(locate_route(samples, route))
        summary["route"] = route
    if (directory / "vision-session.json").exists():
        summary["vision"] = read_vision(directory, report_path, samples, events)
    if isinstance(request, ObservationReplay):
        summary["observations"] = build_observations(
            request, samples, events, summary.get("vision"), summary.get("route")
        )
    write_report(
        report_path,
        {"metadata": metadata, "samples": samples, "events": events, "summary": summary},
    )
    return RunResult(metadata, samples, events, summary, report_path)
