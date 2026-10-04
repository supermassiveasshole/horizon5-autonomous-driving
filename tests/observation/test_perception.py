"""Pixel estimation and independent evaluation through the experiment entry point."""

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from fh5.capture.legacy import ColorFrame, VisionInput, VisionRecord
from fh5.cli import main
from fh5.experiment import run_experiment
from fh5.observation.perception import Perception, PerceptionReplay, PixelPrediction
from tests.capture.test_vision import Observations, packet
from tests.telemetry.test_experiment import config_file


class FrozenPixels:
    """Fixed external model output; no learned accuracy is asserted by these tests."""

    metadata = {"method": "fixture", "weights_sha256": "fixture", "device": "test"}

    def predict(self, image: Path) -> PixelPrediction:
        return PixelPrediction(
            (10, 10),
            bytes(0 if 3 <= x <= 6 else 2 for y in range(10) for x in range(10)),
            bytes([250] * 100),
        )


def recorded_dataset(tmp_path: Path) -> tuple[Path, Path]:
    encoded = io.BytesIO()
    Image.new("RGB", (10, 10), "gray").save(encoded, format="PNG")
    recording = tmp_path / "recording"
    frame = ColorFrame(
        1_000_000_000, 1_020_000_000, 1_030_000_000, encoded.getvalue(), "png", (10, 10), (10, 10)
    )
    run_experiment(
        VisionRecord(config_file(tmp_path), recording),
        vision_environment=Observations([VisionInput(packets=(packet(990),), frame=frame)]),
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "version": 1,
                "clips": [
                    {
                        "id": "one",
                        "split": "development",
                        "recording_dir": str(recording),
                        "vision_sha256": hashlib.sha256(
                            (recording / "vision.jsonl").read_bytes()
                        ).hexdigest(),
                        "first_frame": 0,
                        "last_frame": 0,
                        "annotation_frames": [0],
                        "tags": ["straight", "low_speed"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    protocol = Path(__file__).parents[2] / "configs/perception-t23.json"
    return dataset, protocol


def test_frozen_pixels_replay_as_candidates_without_claiming_road_truth(tmp_path: Path):
    dataset, protocol = recorded_dataset(tmp_path)
    result = run_experiment(
        Perception(dataset, protocol, tmp_path / "estimate"), road_model=FrozenPixels()
    )
    road = result.summary["perception"]
    frame = road["frames"][0]
    assert frame["status"] == "ok"
    assert frame["coverage"]["road"] == 0.4
    assert frame["coverage"]["obstacle_candidate"] == 0.6
    assert all(b["left"] == 3 and b["right"] == 6 for b in frame["boundaries"])
    assert frame["capture_start_ns"] == 1_000_000_000
    assert frame["online_usable"] is False
    assert road["evaluation"]["status"] == "awaiting_independent_labels"
    assert road["geometry_ready"] is False
    assert "verified" not in frame
    assert (tmp_path / "estimate" / "annotate.html").is_file()


def test_replay_rejects_changed_pixels_and_reports_late_results(tmp_path: Path):
    dataset, protocol = recorded_dataset(tmp_path)
    protocol_data = json.loads(protocol.read_text(encoding="utf-8"))
    protocol_data["max_result_age_ms"] = 1
    short_protocol = tmp_path / "short.json"
    short_protocol.write_text(json.dumps(protocol_data), encoding="utf-8")
    output = tmp_path / "estimate"
    original = run_experiment(
        Perception(dataset, short_protocol, output), road_model=FrozenPixels()
    )
    assert original.summary["perception"]["frames"][0]["stale"] is True
    replay = run_experiment(PerceptionReplay(output, tmp_path / "late.html"))
    assert replay.summary["perception"]["frames"][0]["stale"] is True
    (output / "pixels/000000.png").write_bytes(b"corrupt")
    damaged = run_experiment(PerceptionReplay(output, tmp_path / "damaged.html"))
    frame = damaged.summary["perception"]["frames"][0]
    assert frame["status"] == "invalid"
    assert frame["overlay_url"] is None
    assert "integrity" in frame["error"]
    assert damaged.summary["perception"]["geometry_ready"] is False


def test_independent_labels_measure_errors_instead_of_accepting_model_truth(tmp_path: Path):
    dataset, protocol = recorded_dataset(tmp_path)
    output = tmp_path / "estimate"
    predicted = run_experiment(Perception(dataset, protocol, output), road_model=FrozenPixels())
    road = predicted.summary["perception"]
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "version": 1,
                "annotator": {"kind": "human", "name": "test fixture"},
                "protocol_sha256": road["protocol_sha256"],
                "dataset_sha256": road["dataset_sha256"],
                "frames": [
                    {
                        "id": "one:0",
                        "image_sha256": road["frames"][0]["image_sha256"],
                        "reviewed": True,
                        "road_polygons": [[[2, 0], [7, 0], [7, 9], [2, 9]]],
                        "ignore_polygons": [],
                        "boundaries": [
                            {"y": 4, "left": 2, "right": 7},
                            {"y": 5, "left": 2, "right": 7},
                        ],
                        "notes": "Independent 6-column road; prediction is only 4 columns.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evaluated = run_experiment(PerceptionReplay(output, tmp_path / "evaluated.html", labels))
    metrics = evaluated.summary["perception"]["evaluation"]["splits"]["development"]
    assert metrics["road_iou"] == 2 / 3
    assert metrics["road_miss_fraction"] == 1 / 3
    assert metrics["left_mae_px"] == 1
    assert metrics["right_mae_px"] == 1
    assert metrics["high_confidence_error_fraction"] == 0.2
    assert metrics["high_confidence_pixels"] == 100
    assert metrics["annotated_frames"] == 1
    assert evaluated.summary["perception"]["geometry_ready"] is False


def test_pixel_replay_cli_finishes_without_requiring_new_udp_packets(tmp_path: Path, capsys):
    dataset, protocol = recorded_dataset(tmp_path)
    output = tmp_path / "estimate"
    run_experiment(Perception(dataset, protocol, output), road_model=FrozenPixels())
    assert main(["perception-replay", str(output), "--report", str(tmp_path / "cli.html")]) == 0
    assert json.loads(capsys.readouterr().out)["perception"]["geometry_ready"] is False


def test_malformed_manual_labels_fail_as_a_readable_experiment_error(tmp_path: Path):
    dataset, protocol = recorded_dataset(tmp_path)
    output = tmp_path / "estimate"
    initial = run_experiment(Perception(dataset, protocol, output), road_model=FrozenPixels())
    road = initial.summary["perception"]
    labels = tmp_path / "broken-labels.json"
    labels.write_text(
        json.dumps(
            {
                "version": 1,
                "annotator": {"kind": "human", "name": "test fixture"},
                "protocol_sha256": road["protocol_sha256"],
                "dataset_sha256": road["dataset_sha256"],
                "frames": [
                    {
                        "id": "one:0",
                        "image_sha256": road["frames"][0]["image_sha256"],
                        "reviewed": True,
                        "road_polygons": [],
                        "ignore_polygons": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="label"):
        run_experiment(PerceptionReplay(output, tmp_path / "broken.html", labels))


@pytest.mark.parametrize(
    "malformed",
    [
        [],
        {"version": 1, "annotator": None},
        {"version": 1, "annotator": {"kind": "human", "name": None}},
    ],
)
def test_invalid_label_containers_return_cli_error(tmp_path: Path, malformed, capsys):
    dataset, protocol = recorded_dataset(tmp_path)
    output = tmp_path / "estimate"
    initial = run_experiment(Perception(dataset, protocol, output), road_model=FrozenPixels())
    if isinstance(malformed, dict):
        road = initial.summary["perception"]
        malformed = {
            **malformed,
            "protocol_sha256": road["protocol_sha256"],
            "dataset_sha256": road["dataset_sha256"],
            "frames": [],
        }
    labels = tmp_path / "invalid.json"
    labels.write_text(json.dumps(malformed), encoding="utf-8")
    assert (
        main(
            [
                "perception-replay",
                str(output),
                "--report",
                str(tmp_path / "bad.html"),
                "--labels",
                str(labels),
            ]
        )
        == 2
    )
    assert "error" in capsys.readouterr().err


def test_development_and_holdout_cannot_reuse_the_same_recording(tmp_path: Path):
    dataset, protocol = recorded_dataset(tmp_path)
    data = json.loads(dataset.read_text(encoding="utf-8"))
    data["clips"].append({**data["clips"][0], "id": "held", "split": "holdout"})
    dataset.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="independent"):
        run_experiment(
            Perception(dataset, protocol, tmp_path / "estimate"), road_model=FrozenPixels()
        )


def test_replay_never_overwrites_a_frozen_source_json(tmp_path: Path):
    dataset, protocol = recorded_dataset(tmp_path)
    output = tmp_path / "estimate"
    run_experiment(Perception(dataset, protocol, output), road_model=FrozenPixels())
    original = (output / "protocol.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(PerceptionReplay(output, output / "protocol.html"))
    assert (output / "protocol.json").read_bytes() == original
    assert not (output / "protocol.html").exists()


def test_unknown_pixels_cannot_dilute_confident_false_road_errors(tmp_path: Path):
    class MostlyUnknown(FrozenPixels):
        def predict(self, image: Path) -> PixelPrediction:
            return PixelPrediction((10, 10), bytes([0] + [10] * 99), bytes([250] * 100))

    dataset, protocol = recorded_dataset(tmp_path)
    output = tmp_path / "estimate"
    result = run_experiment(Perception(dataset, protocol, output), road_model=MostlyUnknown())
    road = result.summary["perception"]
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "version": 1,
                "annotator": {"kind": "human", "name": "test fixture"},
                "protocol_sha256": road["protocol_sha256"],
                "dataset_sha256": road["dataset_sha256"],
                "frames": [
                    {
                        "id": "one:0",
                        "image_sha256": road["frames"][0]["image_sha256"],
                        "reviewed": True,
                        "road_polygons": [],
                        "ignore_polygons": [],
                        "boundaries": [],
                        "notes": "All pixels are visibly non-road.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    replay = run_experiment(PerceptionReplay(output, tmp_path / "false-road.html", labels))
    metrics = replay.summary["perception"]["evaluation"]["splits"]["development"]
    assert metrics["high_confidence_pixels"] == 1
    assert metrics["high_confidence_error_fraction"] == 1
    assert metrics["unknown_fraction"] == 0.99
