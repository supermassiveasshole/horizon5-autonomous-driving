"""Independent numerical reproduction through the experiment-run interface."""

import hashlib
import json
import time
from dataclasses import replace

import pytest
from test_realtime import ThreadedGame

from fh5.experiment import run_experiment
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig, RealtimeNumericReplay, RealtimeRun


class PixelTimeActor:
    kind = "synthetic_pixel_time"
    manifest = {"fixture_version": 1}

    def input_features(self, actor, frames):
        return [actor["image_age_ms"][-1], 0.1, frames[-1].pixels[0, 0, 0] / 255]

    def predict(self, actor, frames):
        age, delta, red = self.input_features(actor, frames)
        return [red, delta + age / 1000]


def record(tmp_path, factory=PixelTimeActor, **options):
    return run_experiment(
        RealtimeRun(
            tmp_path / "run",
            RealtimeConfig(pixels=PixelContract(size=(2, 1))),
            seconds=0.3,
            **options,
        ),
        realtime_environment=ThreadedGame(),
        numeric_actor_factory=factory,
    ).summary["realtime"]


def test_independent_replay_restores_pixels_time_features_predictions_and_all_outcomes(tmp_path):
    original = record(tmp_path)
    result = run_experiment(
        RealtimeNumericReplay(tmp_path / "run", tmp_path / "replayed.html"),
        numeric_actor=PixelTimeActor(),
    ).summary["realtime_numeric_replay"]
    assert result["verified"] and result["errors"] == []
    assert result["verified_predictions"] >= 3
    assert result["decisions"] == original["decisions"]
    assert result["commands"] == original["commands"]
    assert result["timing_reexecuted"] is False
    assert result["training_eligible"] is result["promotion_eligible"] is False
    predicted = [d for d in original["decisions"] if d["prediction"] is not None]
    for row in predicted:
        assert row["features"][1:] == [0.1, 0.2]
        assert row["prediction"][0] == 0.2
        assert row["decision_ns"] <= row["worker_started_ns"] <= row["worker_returned_ns"]
        assert row["worker_returned_ns"] <= row["inference_returned_ns"]
    assert original["version"] == 2
    assert (tmp_path / "run/realtime-manifest.json").is_file()


@pytest.mark.parametrize(
    "target", ["report", "metadata", "pixels", "journal", "model", "prediction", "features"]
)
def test_replay_rejects_changed_assets_model_or_computation(tmp_path, target):
    original = record(tmp_path)
    row = next(d for d in original["decisions"] if d["prediction"] is not None)
    model = PixelTimeActor()
    root = tmp_path / "run"
    if target in ("report", "journal", "metadata", "pixels"):
        paths = {
            "report": root / "report.json",
            "journal": root / "realtime-events.jsonl",
            "metadata": root / row["archive"]["path"],
        }
        if target == "pixels":
            saved = json.loads(paths["metadata"].read_bytes())
            path = root / saved["frames"][0]["path"]
        else:
            path = paths[target]
        payload = path.read_bytes()
        path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    elif target == "model":
        model.manifest = {"fixture_version": 2}
    elif target == "prediction":
        model.predict = lambda actor, frames: [0.9, 0.9]
    else:
        model.input_features = lambda actor, frames: [100, 0.1, 0.2]
    result = run_experiment(
        RealtimeNumericReplay(root, tmp_path / "review.html"),
        numeric_actor=model,
    ).summary["realtime_numeric_replay"]
    assert not result["verified"] and result["errors"]


def test_archive_budget_gap_cannot_be_promoted_to_exact_replay(tmp_path):
    record(tmp_path, archive_limit_bytes=1)
    result = run_experiment(
        RealtimeNumericReplay(tmp_path / "run", tmp_path / "review.html"),
        numeric_actor=PixelTimeActor(),
    ).summary["realtime_numeric_replay"]
    assert not result["verified"]
    assert "incomplete" in result["errors"][0]["error"]
    assert len(result["decisions"]) >= 3


def test_deadline_discarded_prediction_is_also_reproduced(tmp_path):
    class SlowActor(PixelTimeActor):
        def __init__(self):
            self.calls = 0

        def predict(self, actor, frames):
            self.calls += 1
            if self.calls == 2:
                time.sleep(0.12)
            return super().predict(actor, frames)

    original = record(tmp_path, SlowActor)
    discarded = [d for d in original["decisions"] if d["status"] == "discard_deadline"]
    assert len(discarded) == 1
    result = run_experiment(
        RealtimeNumericReplay(tmp_path / "run", tmp_path / "review.html"),
        numeric_actor=PixelTimeActor(),
    ).summary["realtime_numeric_replay"]
    assert result["verified"]
    assert discarded[0]["decision_id"] in [c["decision_id"] for c in result["checks"]]
    assert result["decisions"] == original["decisions"]


def test_missing_journal_event_fails_even_when_file_hash_is_updated(tmp_path):
    record(tmp_path)
    root = tmp_path / "run"
    journal = root / "realtime-events.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    journal.write_bytes(b"".join(lines[1:]))
    report = json.loads((root / "report.json").read_bytes())
    report["journal"]["sha256"] = hashlib.sha256(journal.read_bytes()).hexdigest()
    payload = json.dumps(report).encode()
    (root / "report.json").write_bytes(payload)
    (root / "realtime-manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "report_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    )
    result = run_experiment(
        RealtimeNumericReplay(root, tmp_path / "review.html"),
        numeric_actor=PixelTimeActor(),
    ).summary["realtime_numeric_replay"]
    assert not result["verified"]
    assert "Missing journal" in result["errors"][0]["error"]


def test_cli_replays_real_frozen_time_model_without_an_environment(tmp_path, capsys):
    pytest.importorskip("torch")
    from test_temporal_bc import temporal_fixture

    from fh5.cli import main
    from fh5.realtime_model import ShadowNumericActor
    from fh5.temporal_bc import TemporalBCTrain

    config, _ = temporal_fixture(tmp_path)
    model_dir = tmp_path / "model"
    run_experiment(TemporalBCTrain(config, model_dir))
    digest = json.loads((model_dir / "model.json").read_text())["weights_sha256"]
    pixels = PixelContract(size=(64, 36))

    class BiggerGame(ThreadedGame):
        def read(self, period_s):
            point = super().read(period_s)
            return replace(
                point,
                observation=replace(
                    point.observation,
                    frames=tuple(
                        replace(f, size=(64, 36), pixels=memoryview(bytes([51, 17, 34] * 64 * 36)))
                        for f in point.observation.frames
                    ),
                ),
            )

    run_experiment(
        RealtimeRun(
            tmp_path / "run",
            RealtimeConfig(pixels=pixels, reference_count=1),
            seconds=0.5,
        ),
        realtime_environment=BiggerGame(),
        numeric_actor_factory=lambda: ShadowNumericActor(model_dir, pixels, digest),
    )
    assert (
        main(
            [
                "realtime-replay",
                str(tmp_path / "run"),
                "--model",
                str(model_dir),
                "--report",
                str(tmp_path / "review.html"),
            ]
        )
        == 0
    )
    summary = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert summary["verified"] and summary["verified_predictions"] > 0
    assert max(c["prediction_max_abs_error"] for c in summary["checks"]) <= 1e-6
    assert json.loads(capsys.readouterr().out)["verified"] is True


def test_journal_scheduling_snapshot_must_match_final_outcome_and_archived_input(tmp_path):
    record(tmp_path)
    root = tmp_path / "run"
    journal = root / "realtime-events.jsonl"
    events = [json.loads(line) for line in journal.read_bytes().splitlines()]
    start = next(e for e in events if e["kind"] == "decision_started")
    start["data"]["decision_ns"] += 50_000_000
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    report = json.loads((root / "report.json").read_bytes())
    report["journal"]["sha256"] = hashlib.sha256(journal.read_bytes()).hexdigest()
    payload = json.dumps(report).encode()
    (root / "report.json").write_bytes(payload)
    (root / "realtime-manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "report_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    )
    result = run_experiment(
        RealtimeNumericReplay(root, tmp_path / "review.html"),
        numeric_actor=PixelTimeActor(),
    ).summary["realtime_numeric_replay"]
    assert not result["verified"]
    assert "scheduling snapshot" in result["errors"][0]["error"]


def test_runtime_report_above_32_mib_is_still_replayable(tmp_path):
    class DiagnosticGame(ThreadedGame):
        def close(self):
            return {**super().close(), "diagnostics": "x" * (33 * 1024**2)}

    run_experiment(
        RealtimeRun(
            tmp_path / "run", RealtimeConfig(pixels=PixelContract(size=(2, 1))), seconds=0.2
        ),
        realtime_environment=DiagnosticGame(),
        numeric_actor_factory=PixelTimeActor,
    )
    result = run_experiment(
        RealtimeNumericReplay(tmp_path / "run", tmp_path / "review.html"),
        numeric_actor=PixelTimeActor(),
    ).summary["realtime_numeric_replay"]
    assert result["verified"]


def test_actor_reused_buffers_cannot_rewrite_previous_features_or_predictions(tmp_path):
    class ReusingActor(PixelTimeActor):
        def __init__(self):
            self.features = []
            self.prediction = []

        def input_features(self, actor, frames):
            self.features[:] = super().input_features(actor, frames)
            return self.features

        def predict(self, actor, frames):
            self.prediction[:] = super().predict(actor, frames)
            return self.prediction

    record(tmp_path, ReusingActor)
    result = run_experiment(
        RealtimeNumericReplay(tmp_path / "run", tmp_path / "review.html"),
        numeric_actor=PixelTimeActor(),
    ).summary["realtime_numeric_replay"]
    assert result["verified"]


def test_replay_preserves_runtime_isolation_between_model_hooks(tmp_path):
    class MutatingFeatures(PixelTimeActor):
        def input_features(self, actor, frames):
            actor["image_age_ms"][-1] += 1
            return super().input_features(actor, frames)

    original = record(tmp_path, MutatingFeatures)
    result = run_experiment(
        RealtimeNumericReplay(tmp_path / "run", tmp_path / "review.html"),
        numeric_actor=MutatingFeatures(),
    ).summary["realtime_numeric_replay"]
    assert result["verified"]
    assert result["decisions"] == original["decisions"]
