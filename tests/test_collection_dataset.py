"""Frozen continuous-source datasets through the public experiment interface."""

import hashlib
import json
import threading
import time
from dataclasses import replace
from itertools import count
from pathlib import Path

import pytest
from test_collection import Stream, input_at, request

from fh5.collection_bc import CollectionBCPrepare
from fh5.collection_dataset import CollectionDatasetReview
from fh5.experiment import run_experiment


def dataset_inputs(tmp_path, invalid_packet=False, vary_action=False, overlap=False, size=(2, 1)):
    from fh5.numeric_images import PixelContract

    sources = []
    for index, split in enumerate(("train", "development", "evaluation")):
        folder = tmp_path / f"source-{index}"
        folder.mkdir()
        req = request(folder)
        req = replace(req, input_conditions={"camera": "chase_far", "blueprint": "105657219"})
        req = replace(req, config=replace(req.config, pixels=PixelContract(size=size)))
        points = [input_at(ms + (0 if overlap else index * 10000)) for ms in range(250, 951, 50)]
        points = [
            replace(
                p,
                frames=tuple(
                    replace(
                        f, size=size, pixels=memoryview(bytes([51, 17, 34] * (size[0] * size[1])))
                    )
                    for f in p.frames
                ),
            )
            for p in points
        ]
        if invalid_packet and index == 0:
            points[6] = replace(points[6], packets=(replace(points[6].packets[0], payload=b"bad"),))
        if vary_action and index == 0:
            human = {
                **points[6].human_input,
                "raw": {**points[6].human_input["raw"], "thumb_lx": 16384},
            }
            points[6] = replace(points[6], human_input=human)
        run_experiment(req, collection_environment=Stream(points))
        session_hash = hashlib.sha256((req.output_dir / "session.json").read_bytes()).hexdigest()
        review = folder / "review.json"
        review.write_text(
            json.dumps(
                {
                    "version": 1,
                    "session_sha256": session_hash,
                    "conditions_verified": True,
                    "evidence": ["synthetic fixture only"],
                    "independence_evidence": [],
                    "attempts": [
                        {
                            "id": f"attempt-{index}",
                            "group": f"group-{index}",
                            "split": split,
                            "start_sequence": 0,
                            "end_sequence": 15,
                            "related_attempts": [],
                            "intervals": [
                                {
                                    "start_sequence": 0,
                                    "end_sequence": 15,
                                    "quality": "trusted",
                                    "reasons": [],
                                    "evidence": ["synthetic normal input"],
                                    "road_kind": "unknown",
                                }
                            ],
                        }
                    ],
                }
            )
        )
        sources.append({"recording": str(req.output_dir), "review": str(review)})
    config = tmp_path / "dataset-config.json"
    config.write_text(
        json.dumps(
            {
                "version": 2,
                "seed": 7,
                "sources": sources,
                "rules": {
                    "max_samples_per_attempt": 100,
                    "speed_range_mps": [0, 100],
                    "steering_limit": 1.0,
                    "longitudinal_limit": 1.0,
                    "max_label_delay_ms": 50,
                },
                "action_history_offsets_ms": [100, 50, 0],
                "max_action_age_ms": 100,
                "waypoint_distances_m": [5, 10, 20],
            }
        )
    )
    return config


def test_freeze_binds_sealed_sources_reviews_and_independent_groups(tmp_path):
    config = dataset_inputs(tmp_path)
    output = tmp_path / "snapshot"
    result = run_experiment(CollectionBCPrepare(config, output))
    data = json.loads((output / "selection.json").read_bytes())
    assert data["kind"] == "collection-dataset-snapshot-v1"
    assert data["version"] == data["config"]["version"] == 1
    assert {g["split"] for g in data["groups"]} == {"train", "development", "evaluation"}
    assert data["diagnostic_only"] and not data["closed_loop_validated"]
    assert data["samples"] and all(s["bc_eligible"] for s in data["samples"])
    assert all(not s["q_eligible"] for s in data["samples"])
    assert all(source["blocks"] for source in data["sources"])
    assert result.summary["collection_dataset"]["ready_for_software_training"]
    assert result.summary["collection_dataset"]["evaluation"]["coverage"] == "withheld"
    before = (output / "selection.json").read_bytes()
    for source in json.loads(config.read_bytes())["sources"]:
        Path(source["review"]).write_text("changed later review")
    verified = run_experiment(
        CollectionDatasetReview(output / "selection.json", tmp_path / "reviewed.html")
    )
    assert verified.summary["collection_dataset"]["verified"]
    assert (output / "selection.json").read_bytes() == before


@pytest.mark.parametrize(
    "fault", ["group_split", "related_split", "duplicate_source", "review_binding"]
)
def test_invalid_independence_or_review_never_publishes_snapshot(tmp_path, fault):
    config = dataset_inputs(tmp_path)
    value = json.loads(config.read_bytes())
    review_path = Path(value["sources"][1]["review"])
    review = json.loads(review_path.read_bytes())
    if fault == "group_split":
        review["attempts"][0]["group"] = "group-0"
    elif fault == "related_split":
        review["attempts"][0]["related_attempts"] = ["attempt-0"]
    elif fault == "duplicate_source":
        value["sources"][1] = value["sources"][0]
    else:
        review["session_sha256"] = "0" * 64
    review_path.write_text(json.dumps(review))
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        run_experiment(CollectionBCPrepare(config, tmp_path / "snapshot"))
    assert not (tmp_path / "snapshot").exists()


def test_failed_and_out_of_envelope_labels_are_retained_without_clipping(tmp_path):
    config = dataset_inputs(tmp_path)
    value = json.loads(config.read_bytes())
    value["rules"]["steering_limit"] = 0.5
    config.write_text(json.dumps(value))
    review_path = Path(value["sources"][0]["review"])
    review = json.loads(review_path.read_bytes())
    interval = review["attempts"][0]["intervals"][0]
    interval.update(quality="failed", reasons=["offroad"])
    review_path.write_text(json.dumps(review))
    run_experiment(CollectionBCPrepare(config, tmp_path / "snapshot"))
    data = json.loads((tmp_path / "snapshot/selection.json").read_bytes())
    assert data["samples"] and all(not s["bc_eligible"] for s in data["samples"])
    assert all(s["target_action"][0] == -1 for s in data["samples"])
    assert all("action_outside_envelope" in s["reasons"] for s in data["samples"])
    assert any("unreviewed_or_failed" in s["reasons"] for s in data["samples"])


@pytest.mark.parametrize("changed", ["pixel", "rows", "session", "selection"])
def test_changed_frozen_dependencies_or_selection_are_rejected(tmp_path, changed):
    config = dataset_inputs(tmp_path)
    output = tmp_path / "snapshot"
    run_experiment(CollectionBCPrepare(config, output))
    document = output / "selection.json"
    data = json.loads(document.read_bytes())
    root = Path(data["sources"][0]["recording"])
    if changed == "pixel":
        next((root / "blocks").rglob("*.rgb")).write_bytes(b"broken pixels")
    elif changed == "rows":
        next((root / "blocks").rglob("rows.jsonl")).write_bytes(b"broken rows")
    elif changed == "session":
        with (root / "session.json").open("a") as stream:
            stream.write(" ")
    else:
        data["samples"][0]["target_action"] = [0, 0]
        document.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        run_experiment(CollectionDatasetReview(document, tmp_path / "reviewed.html"))
    assert not (tmp_path / "reviewed.html").exists()


def test_event_coverage_counts_one_contiguous_turn_and_withholds_final_holdout(tmp_path):
    config = dataset_inputs(tmp_path)
    result = run_experiment(CollectionBCPrepare(config, tmp_path / "snapshot"))
    coverage = result.summary["collection_dataset"]["development_coverage"]
    assert {c["attempt"] for c in coverage} == {"attempt-0", "attempt-1"}
    assert all(c["trusted_events"]["left"] == 1 for c in coverage)
    assert all(c["observations"] == 15 for c in coverage)
    assert all(c["road_kind_unknown_polls"] == 15 for c in coverage)


@pytest.mark.parametrize("destination", ["snapshot", "source", "source_new", "existing", "json"])
def test_review_report_never_overwrites_frozen_or_existing_evidence(tmp_path, destination):
    config = dataset_inputs(tmp_path)
    output = tmp_path / "snapshot"
    run_experiment(CollectionBCPrepare(config, output))
    source = Path(json.loads(config.read_bytes())["sources"][0]["recording"])
    report = {
        "snapshot": output / "selection.html",
        "source": source / "session.html",
        "source_new": source / "new-report.html",
        "existing": output / "report.html",
        "json": tmp_path / "new-report.json",
    }[destination]
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises((ValueError, FileExistsError)):
        run_experiment(CollectionDatasetReview(output / "selection.json", report))
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("middle_quality, expected_events", [("trusted", 1), ("failed", 2)])
def test_review_subdivision_preserves_events_unless_trust_is_interrupted(
    tmp_path, middle_quality, expected_events
):
    config = dataset_inputs(tmp_path)
    review_path = Path(json.loads(config.read_bytes())["sources"][0]["review"])
    review = json.loads(review_path.read_bytes())
    attempt = review["attempts"][0]
    original = attempt["intervals"][0]
    attempt["intervals"] = [
        dict(original, end_sequence=7),
        dict(
            original,
            start_sequence=7,
            end_sequence=8,
            quality=middle_quality,
            reasons=[] if middle_quality == "trusted" else ["offroad"],
        ),
        dict(original, start_sequence=8),
    ]
    review_path.write_text(json.dumps(review))
    result = run_experiment(CollectionBCPrepare(config, tmp_path / "snapshot"))
    coverage = result.summary["collection_dataset"]["development_coverage"][0]
    assert coverage["trusted_events"]["left"] == expected_events
    assert coverage["trusted_events"]["throttle"] == expected_events
    assert coverage["trusted_events"]["medium_speed"] == expected_events
    data = json.loads((tmp_path / "snapshot/selection.json").read_bytes())
    tail = [s for s in data["samples"] if s["attempt"] == "attempt-0" and s["sequence"] >= 8]
    assert tail and all(s["history_floor_ns"] == 1_650_000_000 for s in tail)


def test_invalid_packet_is_excluded_without_discarding_valid_parts_of_attempt(tmp_path):
    config = dataset_inputs(tmp_path, invalid_packet=True)
    result = run_experiment(CollectionBCPrepare(config, tmp_path / "snapshot"))
    data = json.loads((tmp_path / "snapshot/selection.json").read_bytes())
    kept = [s for s in data["samples"] if s["attempt"] == "attempt-0"]
    assert kept and all(s["sequence"] != 6 for s in kept)
    assert result.summary["collection_dataset"]["ready_for_software_training"]


def test_label_is_first_next_poll_and_cannot_become_current_action_history(tmp_path):
    config = dataset_inputs(tmp_path, vary_action=True)
    run_experiment(CollectionBCPrepare(config, tmp_path / "snapshot"))
    data = json.loads((tmp_path / "snapshot/selection.json").read_bytes())
    sample = next(s for s in data["samples"] if s["attempt"] == "attempt-0" and s["sequence"] == 5)
    assert sample["target_action"][0] == pytest.approx(16384 / 32767)
    assert sample["label_sequence"] == 6
    assert sample["label_poll_ns"] >= sample["decision_ns"]


def test_overlapping_source_windows_cannot_claim_independent_attempts(tmp_path):
    config = dataset_inputs(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="Overlapping"):
        run_experiment(CollectionBCPrepare(config, tmp_path / "snapshot"))


def test_reservoir_is_reproducible_and_does_not_change_event_counts(tmp_path):
    config = dataset_inputs(tmp_path)
    value = json.loads(config.read_bytes())
    value["rules"]["max_samples_per_attempt"] = 2
    config.write_text(json.dumps(value))
    first = run_experiment(CollectionBCPrepare(config, tmp_path / "first"))
    run_experiment(CollectionBCPrepare(config, tmp_path / "second"))
    assert (tmp_path / "first/selection.json").read_bytes() == (
        tmp_path / "second/selection.json"
    ).read_bytes()
    data = json.loads((tmp_path / "first/selection.json").read_bytes())
    assert len(data["samples"]) == 6
    assert all(
        c["trusted_events"]["left"] == 1
        for c in first.summary["collection_dataset"]["development_coverage"]
    )


def test_snapshot_stays_fixed_while_collector_publishes_later_blocks(tmp_path):
    from fh5.collection import CollectionControl

    config = dataset_inputs(tmp_path)
    folder = tmp_path / "continuing"
    folder.mkdir()
    req = replace(
        request(folder, seconds=30, block_rows=10),
        input_conditions={"camera": "chase_far", "blueprint": "105657219"},
    )

    def points():
        for ms in count(100250, 50):
            time.sleep(0.01)
            yield input_at(ms)

    completed = []
    worker = threading.Thread(
        target=lambda: completed.append(
            run_experiment(req, collection_environment=Stream(points()))
        )
    )
    worker.start()
    try:
        deadline = time.monotonic() + 4
        while not (req.output_dir / "index.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        binding = hashlib.sha256((req.output_dir / "session.json").read_bytes()).hexdigest()
        original = json.loads(
            Path(json.loads(config.read_bytes())["sources"][0]["review"]).read_bytes()
        )
        original["session_sha256"] = binding
        attempt = original["attempts"][0]
        attempt.update(id="ongoing", group="ongoing", end_sequence=10000)
        attempt["intervals"][0]["end_sequence"] = 10000
        review = folder / "review.json"
        review.write_text(json.dumps(original))
        value = json.loads(config.read_bytes())
        value["sources"].append({"recording": str(req.output_dir), "review": str(review)})
        config.write_text(json.dumps(value))
        run_experiment(CollectionBCPrepare(config, tmp_path / "snapshot"))
        path = tmp_path / "snapshot/selection.json"
        before = path.read_bytes()
        frozen_count = len(json.loads(before)["sources"][-1]["blocks"])
        deadline = time.monotonic() + 4
        while (
            len(json.loads((req.output_dir / "index.json").read_bytes())["blocks"]) <= frozen_count
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert worker.is_alive()
        assert (
            len(json.loads((req.output_dir / "index.json").read_bytes())["blocks"]) > frozen_count
        )
        result = run_experiment(CollectionDatasetReview(path, tmp_path / "verified.html"))
        assert result.summary["collection_dataset"]["verified"] and path.read_bytes() == before
    finally:
        run_experiment(CollectionControl(req.output_dir, stop=True))
        worker.join(timeout=4)
    assert not worker.is_alive() and completed[0].summary["collection"]["complete"]
