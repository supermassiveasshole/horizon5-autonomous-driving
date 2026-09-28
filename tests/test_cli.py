"""The command-line experiment entry point receives real UDP with synthetic payloads."""

import json
import socket
import subprocess
import sys
from pathlib import Path

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
            "0.4",
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
