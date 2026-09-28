"""The command-line experiment entry point receives real UDP with synthetic payloads."""

import json
import socket
import subprocess
import sys
from pathlib import Path

from test_control import control_config
from test_experiment import config_file, sample_packet


def test_cli_records_udp_then_replays_without_claiming_game_validation(tmp_path: Path) -> None:
    run_dir = tmp_path / "udp-run"
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fh5",
            "record",
            "--config",
            str(config_file(tmp_path)),
            "--output",
            str(run_dir),
            "--seconds",
            "0.8",
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ) as recorder:
        assert recorder.stdout is not None
        ready = json.loads(recorder.stdout.readline())
        assert ready["status"] == "listening"
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(sample_packet(), ("127.0.0.1", ready["port"]))
        output, error = recorder.communicate(timeout=10)
    assert recorder.returncode == 0, error
    assert json.loads(output.splitlines()[-1])["valid_packets"] == 1
    replay = subprocess.run(
        [
            sys.executable,
            "-m",
            "fh5",
            "replay",
            str(run_dir),
            "--report",
            str(tmp_path / "replayed.html"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    assert replay.returncode == 0, replay.stderr
    result = json.loads(replay.stdout)
    assert result["valid_packets"] == 1
    assert result["game_validation"] == "unverified"
    report = json.loads((tmp_path / "replayed.json").read_text(encoding="utf-8"))
    tail_gaps = [
        event
        for event in report["events"]
        if event["kind"] == "receive_gap" and event.get("boundary") == "end"
    ]
    assert len(tail_gaps) == 1
    assert 0.5 < tail_gaps[0]["duration_seconds"] < 2


def test_no_datagrams_is_reported_as_an_unsuccessful_capture(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty-run"
    capture = subprocess.run(
        [
            sys.executable,
            "-m",
            "fh5",
            "record",
            "--config",
            str(config_file(tmp_path)),
            "--output",
            str(run_dir),
            "--seconds",
            "0.05",
            "--port",
            "0",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    assert capture.returncode == 3
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["valid_packets"] == 0
    assert "no_telemetry" in {event["kind"] for event in report["events"]}


def test_control_cli_defaults_to_validation_without_a_driver(tmp_path: Path) -> None:
    run_dir = tmp_path / "never-created"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fh5",
            "control",
            "--config",
            str(control_config(tmp_path)),
            "--output",
            str(run_dir),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "validated_only"
    assert not run_dir.exists()


def test_event_cli_defaults_to_validation_without_capture_or_input(tmp_path):
    from test_event_run import verified_config

    output = tmp_path / "no-device"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fh5",
            "event",
            "--config",
            str(verified_config(tmp_path)),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "validated_only"
    assert not output.exists()
