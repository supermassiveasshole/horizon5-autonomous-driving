"""Source/review growth is accepted without weakening independent reconstruction."""

import hashlib
import json
import tracemalloc
from pathlib import Path

import pytest

from fh5.collection.bc import CollectionBCPrepare
from fh5.collection.dataset import CollectionDatasetReview
from fh5.experiment import run_experiment
from tests.collection.test_collection_growth import _growing_source, _json


def test_more_than_one_hundred_independent_sources_prepare_and_review(tmp_path):
    sources = []
    for number in range(101):
        root = tmp_path / f"source-{number}"
        root.mkdir()
        config = _growing_source(
            root,
            attempt_rows=8,
            attempt_count=1,
            clock_offset_ns=number * 10_000_000_000,
            identity_prefix=f"s{number}-",
        )
        options = json.loads(config.read_bytes())
        entry = options["sources"][0]
        review = Path(entry["review"])
        value = json.loads(review.read_bytes())
        value["attempts"][0]["split"] = ("train", "development", "evaluation")[number % 3]
        _json(review, value)
        sources.append(entry)
    options["sources"] = sources
    config = tmp_path / "config.json"
    _json(config, options)
    output = tmp_path / "prepared"
    result = run_experiment(CollectionBCPrepare(config, output))
    assert result.summary["collection_bc"]["selection"]["ready_for_software_training"]
    selection = json.loads((output / "selection.json").read_bytes())
    assert len(selection["sources"]) == len(selection["groups"]) == 101
    assert len({s["session_sha256"] for s in selection["sources"]}) == 101
    for source in selection["sources"]:
        canonical = (
            json.dumps(source["review"], sort_keys=True, allow_nan=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        assert source["review_sha256"] == hashlib.sha256(canonical).hexdigest()
    reviewed = run_experiment(
        CollectionDatasetReview(output / "selection.json", tmp_path / "review-result.html")
    )
    assert reviewed.summary["collection_dataset"]["verified"]
    train = json.loads((output / "dataset.json").read_bytes())
    holdout = json.loads((output / "evaluation.json").read_bytes())
    assert {d["decision_id"] for d in train["decisions"]}.isdisjoint(
        d["decision_id"] for d in holdout["decisions"]
    )
    assert train["decisions"] and holdout["decisions"]


@pytest.mark.parametrize("growth", ["attempts", "intervals"])
def test_more_than_one_thousand_review_items_prepare_and_reconstruct(tmp_path, growth):
    config = _growing_source(
        tmp_path,
        attempt_rows=8 if growth == "attempts" else 8008,
        attempt_count=1001 if growth == "attempts" else 1,
    )
    options = json.loads(config.read_bytes())
    review = Path(options["sources"][0]["review"])
    value = json.loads(review.read_bytes())
    if growth == "intervals":
        attempt = value["attempts"][0]
        template = attempt["intervals"][0]
        attempt["intervals"] = [
            {**template, "start_sequence": number * 8, "end_sequence": (number + 1) * 8}
            for number in range(1001)
        ]
        _json(review, value)
    output = tmp_path / "prepared"
    result = run_experiment(CollectionBCPrepare(config, output))
    assert sum(result.summary["collection_dataset"]["bc_samples_by_split"].values()) > 0
    selection = json.loads((output / "selection.json").read_bytes())
    attempts = selection["sources"][0]["review"]["attempts"]
    assert len(attempts if growth == "attempts" else attempts[0]["intervals"]) == 1001
    reviewed = run_experiment(
        CollectionDatasetReview(output / "selection.json", tmp_path / "review-result.html")
    )
    assert reviewed.summary["collection_dataset"]["verified"]
    damaged = json.loads((output / "selection.json").read_bytes())
    damaged["samples"][0]["target_action"][0] += 0.1
    _json(tmp_path / "damaged.json", damaged)
    with pytest.raises(ValueError, match="canonical frozen source reconstruction"):
        run_experiment(
            CollectionDatasetReview(tmp_path / "damaged.json", tmp_path / "damaged-result.html")
        )


def test_growing_review_evidence_is_frozen_without_full_document_residency(
    tmp_path, record_property
):
    config = _growing_source(tmp_path, attempt_rows=8, attempt_count=1)
    options = json.loads(config.read_bytes())
    review = Path(options["sources"][0]["review"])
    value = json.loads(review.read_bytes())
    attempt = value["attempts"][0]
    template = attempt["intervals"][0]
    # A review can describe the unsealed tail of an ongoing attempt. Only the
    # actual eight sealed rows are exported; evidence remains bound in full.
    attempt["end_sequence"] = 999 * 8
    attempt["intervals"] = [
        {**template, "start_sequence": number * 8, "end_sequence": (number + 1) * 8}
        for number in range(999)
    ]
    _json(review, value)

    def prepare(name):
        tracemalloc.start()
        try:
            result = run_experiment(CollectionBCPrepare(config, tmp_path / name))
            return result, tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    reference, small_peak = prepare("small")
    before = review.stat().st_size
    for interval in attempt["intervals"]:
        interval["evidence"] = [
            f"independent annotation {number}: " + "e" * 600 for number in range(10)
        ]
    _json(review, value)
    added = review.stat().st_size - before
    assert review.stat().st_size > 4 * 1024**2
    # Do not count fixture construction or retain its Python objects while measuring.
    del value, attempt, interval, template
    result, large_peak = prepare("large")
    for key in ("unique_frames", "decoded_frame_budget_bytes"):
        assert result.summary["collection_bc"][key] == reference.summary["collection_bc"][key]
    for name in ("dataset.json", "evaluation.json"):
        small = json.loads((tmp_path / "small" / name).read_bytes())
        large = json.loads((tmp_path / "large" / name).read_bytes())
        assert small["decisions"] == large["decisions"]
    record_property("added_review_bytes", added)
    record_property("small_peak", small_peak)
    record_property("large_peak", large_peak)
    assert large_peak - small_peak < added
    reviewed = run_experiment(
        CollectionDatasetReview(tmp_path / "large/selection.json", tmp_path / "reviewed.html")
    )
    assert reviewed.summary["collection_dataset"]["verified"]


def test_ongoing_review_can_extend_beyond_ten_million_sequences(tmp_path):
    config = _growing_source(tmp_path, attempt_rows=8, attempt_count=1)
    options = json.loads(config.read_bytes())
    path = Path(options["sources"][0]["review"])
    review = json.loads(path.read_bytes())
    review["attempts"][0]["end_sequence"] = 10_000_001
    review["attempts"][0]["intervals"][0]["end_sequence"] = 10_000_001
    _json(path, review)
    output = tmp_path / "prepared"
    result = run_experiment(CollectionBCPrepare(config, output))
    assert sum(result.summary["collection_dataset"]["bc_samples_by_split"].values()) > 0
    selection = json.loads((output / "selection.json").read_bytes())
    assert selection["sources"][0]["review"]["attempts"][0]["end_sequence"] == 10_000_001
    # Only sealed evidence becomes samples; declaring a future tail creates no data.
    assert all(sample["sequence"] < 8 for sample in selection["samples"])
    reviewed = run_experiment(
        CollectionDatasetReview(output / "selection.json", tmp_path / "reconstructed.html")
    )
    assert reviewed.summary["collection_dataset"]["verified"]
