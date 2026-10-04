"""SAC shadow through the public CLI; only camera/desktop/UDP peers are synthetic."""

import hashlib
import json
import socket
import threading
import tracemalloc
from pathlib import Path

import pytest

import fh5.capture.resources as capture_resources
import fh5.capture.windows as dxgi_windows
import fh5.driving.config as numeric_drive_config
import fh5.driving.windows as live
from fh5.cli import main
from fh5.driving.realtime.udp import UDPTelemetry
from fh5.observation.numeric import PixelContract
from tests.driving.test_realtime_shadow import Capture, Desktop, Telemetry
from tests.learning.bc.test_bc_manifest_resources import add_loss_history
from tests.learning.sac.test_sac_evaluation import sac_policy as sac_policy
from tests.observation.test_route_check import route


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def shadow_config(root, checkpoint, port, exploration_seed=None):
    repository = Path(__file__).resolve().parents[3]
    capture = json.loads((repository / "configs/capture-dxgi.example.json").read_bytes())
    capture["pixels"] = PixelContract(size=(64, 36)).metadata()
    capture["input_conditions"] = {
        "version": 1,
        "id": "sac-shadow-simulated-ports",
        "status": "synthetic_test_fixture_not_game_qualification",
    }
    capture["target"]["condition_id"] = capture["input_conditions"]["id"]
    capture_path = root / "capture.json"
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    task_file = route(root)
    model = {"kind": "sac", "directory": str(checkpoint), "device": "cpu"}
    if exploration_seed is not None:
        model["exploration_seed"] = exploration_seed
    path = root / "shadow.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "capture_config": str(capture_path),
                "model": model,
                "task": {"route_file": str(task_file), "end_margin_m": 0.5},
                "decision": {"reference_count": 1},
                "port": port,
                "shadow": None,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("exploration_seed", [None, 37], ids=["deterministic", "seeded"])
def test_sac_shadow_predicts_without_commands_and_replays_exactly(
    tmp_path, sac_policy, monkeypatch, capsys, exploration_seed, record_property
):
    parent = {name: sha(sac_policy / name) for name in ("policy.json", "policy.pt")}
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    address = receiver.getsockname()
    telemetry = UDPTelemetry(address[1], receiver=receiver)
    capture = Capture()
    done = threading.Event()

    def publish():
        packets = Telemetry()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            while not done.is_set():
                for packet in packets.read(0.005).packets:
                    sender.sendto(packet.payload, address)

    publisher = threading.Thread(target=publish, daemon=True)

    def camera(target):
        publisher.start()  # No UDP traffic until model warmup finishes and capture opens.
        return capture

    def telemetry_port(port):
        assert port == address[1]
        return telemetry

    def forbidden_controller(*args, **kwargs):
        pytest.fail("Read-only shadow must never construct a controller")

    monkeypatch.setattr(dxgi_windows, "WindowsDXGIFrames", camera)
    monkeypatch.setattr(live, "WindowsDesktop", Desktop)
    monkeypatch.setattr(live, "XboxController", forbidden_controller)
    monkeypatch.setattr(numeric_drive_config, "UDPTelemetry", telemetry_port)
    monkeypatch.setattr(
        capture_resources, "WindowsResources", lambda: lambda: {"source": "fixture", "gpus": []}
    )
    config = shadow_config(tmp_path, sac_policy, address[1], exploration_seed)
    root = tmp_path / "shadow-run"
    try:
        code = main(
            [
                "realtime-shadow",
                "--config",
                str(config),
                "--output",
                str(root),
                "--seconds",
                "0.8",
                "--live",
            ]
        )
        output = capsys.readouterr()
        assert code == 0, output.out + output.err
        report = json.loads((root / "report.json").read_bytes())
        assert report["stop_reason"] == "time_limit", report
        assert report["evidence_kind"] == "shadow" and report["commands"] == []
        assert not report["commands_sent_to_game"] and not report["real_game_validation"]
        assert report["resources_released"] and capture.closed and telemetry.closed
        assert report["evidence"]["exact_replay_eligible"]
        assert not report["evidence"]["training_eligible"]
        assert not report["evidence"]["promotion_eligible"]
        assert not report["environment"]["controller_created"]
        assert report["environment"]["capture"]["source_kind"] == "synthetic-capture"
        assert report["model"]["sac_manifest_sha256"] == parent["policy.json"]
        assert report["model"]["command_context"] == "counterfactual-proposal-v1"
        assert report["model"]["diagnostic_only"]
        assert report["model"]["exploration"] is (exploration_seed is not None)
        assert report["inference"]["inference_device"] == "cpu"
        proposals = report["proposals"]
        assert proposals and proposals[0]["owner"] == "initial_neutral"
        assert proposals[0]["proposal"] == {"steer_i16": 0, "throttle_u8": 0, "brake_u8": 0}
        accepted = [row for row in report["decisions"] if row["status"] == "accepted"]
        assert accepted
        # Optional statistics are rendered after the replay evidence is sealed.
        # Check the public display, without requiring derived values in the raw file.
        html = (root / "report.html").read_text(encoding="utf-8")
        display, _ = json.JSONDecoder().raw_decode(html.split("const data=", 1)[1])
        assert display["metrics"]["source_to_send_return_ms"]["count"] == 0
        assert display["metrics"]["source_to_proposal_ms"]["count"] == len(accepted)
        assert display["commands"] == report["commands"] == []
        assert [row["decision_id"] for row in display["decisions"]] == [
            row["decision_id"] for row in report["decisions"]
        ]
        for row in accepted:
            assert row["actor"]["actions"] == [None, None, None]
            assert row["actor"]["action_mask"] == [False, False, False]
            context = row["command_context"]
            previous = proposals[context["proposal_index"]]
            assert context["proposal"] == previous["proposal"]
            assert context["proposed_ns"] == previous["proposed_ns"] < row["decision_ns"]
            archive = root / row["archive"]["path"]
            assert sha(archive) == row["archive"]["sha256"]
            saved = json.loads(archive.read_bytes())
            assert saved["command_context"] == context
            for frame in saved["frames"]:
                assert (root / frame["path"]).read_bytes() == bytes([17, 34, 51]) * (64 * 36)
                assert frame["source_time_ns"] <= row["decision_ns"]
        replay_path = tmp_path / "replayed.html"
        code = main(
            [
                "realtime-replay",
                str(root),
                "--model",
                str(sac_policy),
                "--report",
                str(replay_path),
                "--tolerance",
                "0",
            ]
        )
        output = capsys.readouterr()
        assert code == 0, output.out + output.err
        replay = json.loads(replay_path.with_suffix(".json").read_bytes())
        assert replay["verified"] and replay["errors"] == []
        assert replay["source_evidence_kind"] == "shadow"
        assert replay["model"] == report["model"]
        assert replay["verified_predictions"] >= len(accepted)
        assert all(check["prediction_max_abs_error"] == 0 for check in replay["checks"])
        assert parent == {name: sha(sac_policy / name) for name in parent}
        record_property("accepted_predictions", len(accepted))
        record_property("verified_predictions", replay["verified_predictions"])
        # Re-sign a forged proposal in both files: hashes alone cannot establish
        # that the reported hypothetical command came from the frozen prediction.
        proposal = next(item for item in proposals if item["owner"] == "policy")
        proposal["proposal"]["steer_i16"] += 1
        journal_path = root / "realtime-events.jsonl"
        journal = [json.loads(line) for line in journal_path.read_bytes().splitlines()]
        for event in journal:
            if (
                event["kind"] == "proposal"
                and event["data"]["proposal_index"] == proposal["proposal_index"]
            ):
                event["data"] = proposal
        journal_path.write_text("".join(json.dumps(event) + "\n" for event in journal))
        report["journal"]["sha256"] = sha(journal_path)
        (root / "report.json").write_text(json.dumps(report))
        manifest_path = root / "realtime-manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["report_sha256"] = sha(root / "report.json")
        manifest_path.write_text(json.dumps(manifest))
        assert (
            main(
                [
                    "realtime-replay",
                    str(root),
                    "--model",
                    str(sac_policy),
                    "--report",
                    str(tmp_path / "forged.html"),
                    "--tolerance",
                    "0",
                ]
            )
            == 4
        )
        forged = json.loads((tmp_path / "forged.json").read_bytes())
        assert not forged["verified"]
        assert any("proposal differs" in error["error"] for error in forged["errors"])
    finally:
        done.set()
        if publisher.ident is not None:
            publisher.join(2)
            assert not publisher.is_alive()
        telemetry.close()


@pytest.mark.parametrize("change", [None, "drive", "hash", "pixels", "cuda", "legacy"])
def test_sac_configuration_checks_before_opening_devices(
    tmp_path, sac_policy, monkeypatch, capsys, change
):
    def forbidden(*args, **kwargs):
        pytest.fail("Configuration validation must not open native devices")

    monkeypatch.setattr(dxgi_windows, "WindowsDXGIFrames", forbidden)
    monkeypatch.setattr(live, "WindowsDesktop", forbidden)
    monkeypatch.setattr(live, "XboxController", forbidden)
    config = shadow_config(tmp_path, sac_policy, 5300)
    document = json.loads(config.read_bytes())
    if change == "hash":
        document["model"]["expected_sha256"] = "0" * 64
    elif change == "cuda":
        document["model"]["device"] = "cuda"
    elif change == "pixels":
        capture_path = tmp_path / "capture.json"
        capture = json.loads(capture_path.read_bytes())
        capture["pixels"] = PixelContract(size=(128, 72)).metadata()
        capture_path.write_text(json.dumps(capture))
    config.write_text(json.dumps(document))
    root = tmp_path / "not-created"
    args = [
        "realtime-drive" if change == "drive" else "realtime-shadow",
        "--config",
        str(config),
        "--output",
        str(root),
    ]
    if change is not None:
        args.append("--live")
    if change == "legacy":
        args.append("--allow-legacy-source-diagnostic")
    assert main(args) == (0 if change is None else 2)
    output = capsys.readouterr()
    if change is None:
        report = json.loads(output.out)
        assert report["status"] == "validated_only" and not report["devices_opened"]
    else:
        assert json.loads(output.err)["status"] == "error"
    assert not root.exists()


def test_sac_shadow_checks_growing_bc_history_without_loading_it_all(
    tmp_path, sac_policy, capsys, record_property
):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "bc").mkdir(parents=True)
    bc_manifest = checkpoint / "bc/model.json"
    bc_manifest.write_bytes((sac_policy / "bc/model.json").read_bytes())
    history_bytes = add_loss_history(checkpoint / "bc")
    policy = json.loads((sac_policy / "policy.json").read_bytes())
    policy["bc_manifest_sha256"] = sha(bc_manifest)
    (checkpoint / "policy.json").write_text(json.dumps(policy))
    config = shadow_config(tmp_path, checkpoint, 5300)
    output = tmp_path / "not-created"
    tracemalloc.start()
    try:
        code = main(["realtime-shadow", "--config", str(config), "--output", str(output)])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    captured = capsys.readouterr()
    assert code == 0, captured.out + captured.err
    assert json.loads(captured.out)["status"] == "validated_only"
    assert peak < history_bytes, "Configuration materialized the archived BC diagnostic history"
    assert not output.exists()
    record_property("bc_history_bytes", history_bytes)
    record_property("peak_python_bytes", peak)
