"""Numerical model inputs and exact replay at the agreed experiment boundary."""

from dataclasses import replace

import pytest

from fh5.experiment import run_experiment
from fh5.numeric_images import (
    NumericDecision,
    NumericFrame,
    NumericInfer,
    NumericReplay,
    PixelContract,
)


def actor_state():
    return {
        "actions": [None, None, None],
        "action_mask": [False, False, False],
        "action_age_ms": [None, None, None],
        "images": [None, None, None],
        "image_mask": [True, True, True],
        "image_age_ms": [210, 110, 10],
        "ego_mask": True,
        "ego_age_ms": 5,
        "ego": {
            "speed_mps": 10,
            "velocity_car_mps": [0, 0, 10],
            "angular_velocity_car_radps": [0, 0, 0],
        },
        "reference": {"waypoints_m": [None], "mask": [False]},
    }


class NumericalProbe:
    kind = "synthetic_numeric_probe"
    manifest = {"weights_sha256": "synthetic", "diagnostic_only": True}

    def predict(self, actor, frames):
        assert all(f.pixels.readonly and f.pixels.format == "B" for f in frames)
        assert all(f.pixels.shape == (1, 2, 3) for f in frames)
        return [frames[-1].pixels[0, 0, 0] / 255, -actor["ego"]["speed_mps"] / 100]


def decisions():
    shared = bytearray([51, 17, 34] * 2)
    frames = tuple(
        NumericFrame(
            epoch="drive-1",
            frame_id=str(i),
            source_time_ns=1_000_000_000 + i * 100_000_000,
            capture_received_ns=1_002_000_000 + i * 100_000_000,
            preprocess_ready_ns=1_004_000_000 + i * 100_000_000,
            time_quality="synthetic",
            uncertainty_ns=0,
            size=(2, 1),
            pixels=memoryview(shared),
            source_layout={"size": [2, 1], "format": "RGB", "stride_bytes": 6},
        )
        for i in range(3)
    )
    # A producer immediately reuses its own capture buffer after publication.
    shared[:] = bytes([255] * 6)
    yield NumericDecision("d0", "drive-1", 1_210_000_000, frames, actor_state())
    yield NumericDecision(
        "d1",
        "drive-1",
        1_220_000_000,
        frames,
        dict(actor_state(), image_age_ms=[220, 120, 20]),
    )


def test_numeric_pixels_survive_producer_reuse_and_replay_exactly(tmp_path):
    result = run_experiment(
        NumericInfer(tmp_path / "numeric", PixelContract(size=(2, 1))),
        numeric_inputs=decisions(),
        numeric_actor=NumericalProbe(),
    )
    summary = result.summary["numeric"]
    assert summary["commands_sent"] is False
    assert [d["prediction"] for d in summary["decisions"]] == [[0.2, -0.1]] * 2
    assert all(d["exact_replay_available"] for d in summary["decisions"])
    assert summary["archive"]["resources_released"] is True
    replay = run_experiment(
        NumericReplay(tmp_path / "numeric", tmp_path / "replay.html"),
        numeric_actor=NumericalProbe(),
    )
    assert replay.summary["numeric"]["replay_errors"] == []
    assert all(
        d["pixels_match"] and d["features_match"] for d in replay.summary["numeric"]["decisions"]
    )
    assert [d["prediction_max_abs_error"] for d in replay.summary["numeric"]["decisions"]] == [0, 0]


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("future_ready", "image_not_available"),
        ("other_epoch", "history_crosses_epoch"),
        ("repeated_frame", "repeated_image"),
        ("missing_history", "incomplete_history"),
        ("age_changed", "image_age_mismatch"),
        ("wrong_dimensions", "pixel_contract_mismatch"),
        ("wrong_preprocess", "pixel_contract_mismatch"),
        ("layout_changed", "history_layout_changed"),
    ],
)
def test_invalid_numeric_history_is_visible_and_never_reaches_model(tmp_path, fault, reason):
    decision = next(decisions())
    frames = list(decision.frames)
    contract = PixelContract(size=(2, 1))
    if fault == "future_ready":
        frames[-1] = replace(frames[-1], preprocess_ready_ns=1_211_000_000)
    elif fault == "other_epoch":
        frames[0] = replace(frames[0], epoch="before-rewind")
    elif fault == "repeated_frame":
        frames[1] = frames[0]
    elif fault == "missing_history":
        frames.pop()
    elif fault == "age_changed":
        decision = replace(decision, actor=dict(decision.actor, image_age_ms=[200, 100, 0]))
    elif fault == "wrong_dimensions":
        contract = PixelContract(size=(1, 2))
    elif fault == "wrong_preprocess":
        frames[-1] = replace(frames[-1], preprocess_version="another-resize")
    elif fault == "layout_changed":
        frames[-1] = replace(
            frames[-1], source_layout=dict(frames[-1].source_layout, client_size=[3840, 2160])
        )

    class MustNotPredict(NumericalProbe):
        def predict(self, actor, frames):
            pytest.fail("Unusable history reached the model")

    result = run_experiment(
        NumericInfer(tmp_path / fault, contract),
        numeric_inputs=[replace(decision, frames=tuple(frames))],
        numeric_actor=MustNotPredict(),
    )
    row = result.summary["numeric"]["decisions"][0]
    assert row["status"] == "skipped"
    assert row["reason"] == reason
    assert row["prediction"] is None


def test_existing_frozen_bc_uses_prepared_numbers_without_image_io_or_codecs(tmp_path, monkeypatch):
    import hashlib
    import threading
    from pathlib import Path

    pytest.importorskip("torch")
    from PIL import Image
    from test_bc import setup_bc

    from fh5.numeric_actor import FrozenNumericActor
    from fh5.numeric_import import LegacyNumericImport, PreparedNumericSource

    request = setup_bc(tmp_path)
    baseline = run_experiment(request)
    prepared = run_experiment(
        LegacyNumericImport(
            request.output_dir,
            tmp_path / "data/dataset.json",
            tmp_path / "prepared",
            max_decisions=2,
        )
    )
    contract = PixelContract(size=(64, 36), origin="legacy_offline", history_offsets_ms=(0,))
    model = FrozenNumericActor(request.output_dir, contract, legacy_diagnostic=True)
    source = PreparedNumericSource(tmp_path / "prepared")
    read_bytes, digest = Path.read_bytes, hashlib.sha256

    def no_decode(*args, **kwargs):
        pytest.fail("Compressed image decoding reached the numerical execution phase")

    def prepared_read(path):
        if path.suffix in (".rgb", ".jpeg", ".png", ".jpg"):
            assert threading.current_thread().name == "fh5-numeric-source"
        return read_bytes(path)

    def background_digest(*args, **kwargs):
        assert threading.current_thread().name in ("fh5-numeric-archive", "fh5-numeric-source")
        return digest(*args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Image, "open", no_decode)
        guard.setattr(Image.Image, "save", no_decode)
        guard.setattr(Path, "read_bytes", prepared_read)
        guard.setattr(hashlib, "sha256", background_digest)
        result = run_experiment(
            NumericInfer(tmp_path / "numeric", contract),
            numeric_inputs=source,
            numeric_actor=model,
        )
    summary = result.summary["numeric"]
    assert prepared.summary["numeric_import"]["decoded_unique_frames"] > 0
    assert summary["model"]["diagnostic_only"] is True
    assert source.closed
    assert not model.cache
    for row in summary["decisions"]:
        original = next(
            r
            for r in baseline.summary["bc"]["predictions"]
            if r["example_index"] == int(row["decision_id"].split(":")[-1])
            and r["view"] == "no_reference"
        )
        assert row["prediction"] == pytest.approx(original["prediction"], abs=1e-6)
    replay = run_experiment(
        NumericReplay(tmp_path / "numeric", tmp_path / "reloaded.html"),
        numeric_actor=model,
    )
    assert replay.summary["numeric"]["replay_errors"] == []


def test_stalled_archive_drops_evidence_without_waiting_or_changing_predictions(
    tmp_path, monkeypatch
):
    import threading
    from pathlib import Path

    started, release = threading.Event(), threading.Event()
    write = Path.write_bytes
    source_released = []

    def stalled_write(path, data):
        if path.suffix == ".rgb":
            assert threading.current_thread().name == "fh5-numeric-archive"
            started.set()
            assert release.wait(2), "Decisions waited for the blocked archive"
        return write(path, data)

    def inputs():
        try:
            first, second = list(decisions())
            yield first
            assert started.wait(2)
            yield second
        finally:
            source_released.append(True)
            release.set()

    monkeypatch.setattr(Path, "write_bytes", stalled_write)
    result = run_experiment(
        NumericInfer(tmp_path / "numeric", PixelContract(size=(2, 1)), archive_capacity=1),
        numeric_inputs=inputs(),
        numeric_actor=NumericalProbe(),
    )
    rows = result.summary["numeric"]["decisions"]
    assert [d["prediction"] for d in rows] == [[0.2, -0.1]] * 2
    assert [d["exact_replay_available"] for d in rows] == [True, False]
    assert rows[1]["archive_reason"] == "archive_capacity"
    assert result.summary["numeric"]["archive"]["peak_pending_bytes"] == 18
    assert result.summary["numeric"]["archive"]["pending_after_close"] == 0
    assert source_released == [True]
    replay = run_experiment(
        NumericReplay(tmp_path / "numeric", tmp_path / "missing.html"),
        numeric_actor=NumericalProbe(),
    )
    assert len(replay.summary["numeric"]["replay_errors"]) == 1
    assert replay.summary["numeric"]["replay_errors"][0]["decision_id"] == "d1"


@pytest.mark.parametrize(
    "change", ["zero_capacity", "zero_bytes", "zero_limit", "bad_size", "history_order"]
)
def test_numeric_resource_and_pixel_contracts_are_validated_before_running(tmp_path, change):
    with pytest.raises(ValueError):
        contract = PixelContract(size=(2, 1))
        options = {}
        if change == "zero_capacity":
            options["archive_capacity"] = 0
        elif change == "zero_bytes":
            options["archive_bytes"] = 0
        elif change == "zero_limit":
            options["max_decisions"] = 0
        elif change == "bad_size":
            contract = PixelContract(size=(-1, 1))
        else:
            contract = PixelContract(size=(2, 1), history_offsets_ms=(0, 100, 200))
        run_experiment(
            NumericInfer(tmp_path / "invalid", contract, **options),
            numeric_inputs=decisions(),
            numeric_actor=NumericalProbe(),
        )
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    "fault", ["missing_pixels", "corrupt_pixels", "summary_tampered", "escaped_path"]
)
def test_numeric_replay_never_substitutes_damaged_or_mismatched_evidence(tmp_path, fault):
    import hashlib
    import json

    directory = tmp_path / "numeric"
    run_experiment(
        NumericInfer(directory, PixelContract(size=(2, 1))),
        numeric_inputs=[next(decisions())],
        numeric_actor=NumericalProbe(),
    )
    manifest = json.loads((directory / "numeric-run.json").read_text())
    row = manifest["decisions"][0]
    archived = directory / row["archive"]["path"]
    record = json.loads(archived.read_text())
    pixels = directory / record["frames"][0]["path"]
    if fault == "missing_pixels":
        pixels.unlink()
    elif fault == "corrupt_pixels":
        pixels.write_bytes(bytes([255] * 6))
    elif fault == "summary_tampered":
        row["actor"]["ego"]["speed_mps"] = 90
    else:
        external = tmp_path / "outside.rgb"
        external.write_bytes(pixels.read_bytes())
        record["frames"][0]["path"] = "../outside.rgb"
        archived.write_text(json.dumps(record))
        row["archive"]["sha256"] = hashlib.sha256(archived.read_bytes()).hexdigest()
    (directory / "numeric-run.json").write_text(json.dumps(manifest))
    result = run_experiment(
        NumericReplay(directory, tmp_path / "broken.html"), numeric_actor=NumericalProbe()
    )
    summary = result.summary["numeric"]
    assert len(summary["replay_errors"]) == 1
    assert summary["decisions"][0]["exact_replay_available"] is False


@pytest.mark.parametrize("fault", ["source", "model", "duplicate_decision"])
def test_numeric_failure_retains_completed_evidence_and_releases_sources(tmp_path, fault):
    released = []
    first, second = list(decisions())

    def source():
        try:
            yield first
            if fault == "source":
                raise OSError("capture source failed")
            yield replace(second, decision_id="d0") if fault == "duplicate_decision" else second
        finally:
            released.append(True)

    class FailingModel(NumericalProbe):
        def predict(self, actor, frames):
            if fault == "model" and actor["image_age_ms"][-1] == 20:
                raise RuntimeError("model failed")
            return super().predict(actor, frames)

    result = run_experiment(
        NumericInfer(tmp_path / fault, PixelContract(size=(2, 1))),
        numeric_inputs=source(),
        numeric_actor=FailingModel(),
    )
    summary = result.summary["numeric"]
    assert summary["stop_reason"] == "execution_error"
    assert summary["execution_error"]
    assert summary["decisions"][0]["prediction"] == [0.2, -0.1]
    assert summary["decisions"][0]["exact_replay_available"] is True
    assert summary["archive"]["resources_released"] is True
    assert released == [True]
    if fault == "model":
        assert summary["decisions"][1]["status"] == "error"
        assert summary["decisions"][1]["prediction"] is None


def test_stopping_at_decision_limit_does_not_pull_more_source_frames(tmp_path):
    released = []

    def source():
        try:
            yield next(decisions())
            pytest.fail("Consumer continued pulling after the configured limit")
        finally:
            released.append(True)

    result = run_experiment(
        NumericInfer(tmp_path / "limited", PixelContract(size=(2, 1)), max_decisions=1),
        numeric_inputs=source(),
        numeric_actor=NumericalProbe(),
    )
    assert result.summary["numeric"]["stop_reason"] == "decision_limit"
    assert released == [True]


@pytest.mark.parametrize(
    "fault",
    [
        "removed_preview",
        "changed_preview",
        "preview_write_failure",
        "archive_failure",
        "byte_limit",
    ],
)
def test_presentation_and_archive_failures_never_change_model_inputs(tmp_path, monkeypatch, fault):
    from pathlib import Path

    directory = tmp_path / fault
    write = Path.write_bytes

    def fail_write(path, data):
        if path.suffix == (".png" if fault == "preview_write_failure" else ".rgb"):
            raise OSError("Injected output failure")
        return write(path, data)

    with monkeypatch.context() as guard:
        if fault in ("preview_write_failure", "archive_failure"):
            guard.setattr(Path, "write_bytes", fail_write)
        result = run_experiment(
            NumericInfer(
                directory,
                PixelContract(size=(2, 1)),
                archive_bytes=1 if fault == "byte_limit" else 1000,
            ),
            numeric_inputs=list(decisions()),
            numeric_actor=NumericalProbe(),
        )
    assert [d["prediction"] for d in result.summary["numeric"]["decisions"]] == [[0.2, -0.1]] * 2
    for preview in (directory / "previews").glob("*.png"):
        if fault == "removed_preview":
            preview.unlink()
        elif fault == "changed_preview":
            preview.write_bytes(b"unrelated presentation")
    replay = run_experiment(
        NumericReplay(directory, tmp_path / f"{fault}.html"), numeric_actor=NumericalProbe()
    )
    rows = replay.summary["numeric"]["decisions"]
    if fault in ("archive_failure", "byte_limit"):
        assert len(replay.summary["numeric"]["replay_errors"]) == 2
        assert all(not d["exact_replay_available"] for d in rows)
    else:
        assert replay.summary["numeric"]["replay_errors"] == []
        assert all(d["prediction_max_abs_error"] == 0 for d in rows)
