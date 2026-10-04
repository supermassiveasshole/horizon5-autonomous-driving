"""Collection-aware learning is exercised through complete experiment runs."""

import hashlib
import json
from pathlib import Path

import pytest

from fh5.collection.bc import CollectionBCPrepare
from fh5.experiment import run_experiment
from fh5.learning.bc.training import TemporalBCTrain
from tests.collection.test_collection_bc import prepare_inputs

pytest.importorskip("torch")


class Resources:
    source_kind = "synthetic"

    def __init__(self, pressures=(), *, permanent=None):
        self.time = 10.0
        self.reads = 0
        self.pressures = set(pressures)
        self.permanent = permanent
        self.waited = 0
        self.closed = False

    def now_ns(self):
        return int(self.time * 1e9)

    def sample(self):
        self.reads += 1
        self.time += 0.001
        now = self.now_ns()
        data = {
            "observed_ns": now,
            "process_private_bytes": 64 * 1024**2,
            "free_disk_bytes": 50 * 1024**3,
            "gpus": [{"memory_used_mib": 1000, "utilization_percent": 5}],
            "collector": {
                "process_liveness": "running",
                "software_snapshot_verified": True,
                "state": "recording",
                "heartbeat_ns": now,
                "last_poll_ns": now,
                "latest_image_source_ns": now - 50_000_000,
                "pending_bytes": 0,
                "dropped_rows": 0,
                "seen_rows": self.reads * 20,
                "archive_error": None,
            },
        }
        if self.reads in self.pressures or self.permanent == "backlog":
            data["collector"]["pending_bytes"] = 100 * 1024**2
        if self.permanent == "delayed_heartbeat":
            for field in ("heartbeat_ns", "last_poll_ns", "latest_image_source_ns"):
                data["collector"][field] -= 1_000_000_000
        if self.permanent == "stale":
            data["collector"]["heartbeat_ns"] -= 10_000_000_000
        if self.permanent == "missing":
            data.pop("process_private_bytes")
        if self.permanent == "private_memory":
            data["process_private_bytes"] = 16 * 1024**3
        if self.permanent == "archive_failed":
            data["collector"]["archive_error"] = "writer_failed"
        if self.permanent == "incomplete_exit":
            data["collector"].update(
                process_liveness="exited", final_status_present=True, complete=False
            )
        return data

    def wait(self, seconds):
        self.time += seconds
        self.waited += seconds

    def close(self):
        self.closed = True


def configuration(tmp_path, **budget_changes):
    prepare = prepare_inputs(tmp_path)
    numeric = tmp_path / "numeric"
    run_experiment(CollectionBCPrepare(prepare, numeric))
    dataset = numeric / "dataset.json"
    training = tmp_path / "train.json"
    training.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": str(dataset),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "seed": 19,
                "steps": 3,
                "batch_size": 4,
                "learning_rate": 0.001,
                "device": "cpu",
                "time_mode": "actual",
            }
        )
    )
    config = tmp_path / "scheduled.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "training_config": str(training),
                "training_config_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
                "collector_bundle": "synthetic-bundle",
                "collector_manifest_sha256": "0" * 64,
                "budget": {
                    "cpu_threads": 2,
                    "poll_interval_s": 0.1,
                    "max_wait_s": 1,
                    "max_total_s": 120,
                    "max_unit_s": 10,
                    "max_private_bytes": 4 * 1024**3,
                    "min_free_disk_bytes": 1024**3,
                    "max_status_age_ms": 3000,
                    "max_image_age_ms": 250,
                    "max_pending_bytes": 16 * 1024**2,
                    "max_gpu_memory_mib": 20000,
                    "max_gpu_utilization_percent": 80,
                    **budget_changes,
                },
            }
        )
    )
    return config, training, dataset


def test_pressure_pauses_actual_training_without_changing_the_frozen_candidate(tmp_path):
    from fh5.learning.bc.schedule import ScheduledBCTrain

    class WaitOnlyForPressure(Resources):
        def sample(self):
            sample = super().sample()
            self.pressure = sample["collector"]["pending_bytes"] > 0
            return sample

        def wait(self, seconds):
            assert self.pressure, "Resource budget is already satisfied; no extra wait is justified"
            super().wait(seconds)

    config, training, dataset = configuration(tmp_path)
    frozen = dataset.read_bytes()
    plain = run_experiment(TemporalBCTrain(training, tmp_path / "plain")).summary["temporal_bc"]
    resources = WaitOnlyForPressure(pressures=(5, 6))
    result = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
    ).summary["learning_schedule"]
    assert result["state"] == "completed"
    assert result["steps_completed"] == 3
    assert resources.closed and resources.waited > 0
    assert result["pauses"] >= 1
    with (tmp_path / "scheduled" / result["event_history"]["path"]).open(
        encoding="utf-8"
    ) as stream:
        assert any("collection_backlog" in json.loads(line)["reasons"] for line in stream)
    assert result["commands_sent"] is False
    assert result["diagnostic_only"] is True
    actual = json.loads((tmp_path / "scheduled/candidate/report.json").read_bytes())
    assert [r["prediction"] for r in actual["decisions"]] == [
        r["prediction"] for r in plain["decisions"]
    ]
    assert dataset.read_bytes() == frozen


def test_large_candidate_manifest_does_not_block_publication(tmp_path, monkeypatch):
    from fh5.learning.bc.schedule import ScheduledBCTrain
    from fh5.learning.bc.training import TemporalBCReplay

    config, _, dataset = configuration(tmp_path)
    output = tmp_path / "scheduled"
    manifest = output / ".candidate/model.json"
    write_text = Path.write_text
    expanded = []

    def write_large_manifest(path, text, *args, **kwargs):
        written = write_text(path, text, *args, **kwargs)
        if path == manifest and '"verification":' in text:
            # Real training/reload has finished. Grow only its final JSON file,
            # preserving every field and tensor; this is a filesystem substitution.
            with path.open("ab") as stream:
                block = b" " * 1024**2
                for _ in range(129):
                    stream.write(block)
            expanded.append(path.stat().st_size)
        return written

    resources = Resources()
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", write_large_manifest)
        result = run_experiment(
            ScheduledBCTrain(config, output), learning_resources=resources
        ).summary["learning_schedule"]

    assert len(expanded) == 1 and expanded[0] > 128 * 1024**2
    assert result["state"] == "completed"
    assert result["steps_completed"] == result["durable_steps_completed"] == 3
    assert resources.closed and not result["commands_sent"]
    assert (output / "learner/learner.json").is_file()
    assert not manifest.exists()
    with (output / "candidate/model.json").open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest() == result["candidate_manifest_sha256"]
        )
    replay = run_experiment(
        TemporalBCReplay(output / "candidate", dataset, tmp_path / "replayed.html")
    ).summary["temporal_bc"]
    assert replay["verification"]["max_abs_error"] <= 1e-6


def test_recent_collector_heartbeat_uses_image_age_at_its_actual_poll(tmp_path):
    from fh5.learning.bc.schedule import ScheduledBCTrain

    config, _, _ = configuration(tmp_path)
    resources = Resources(permanent="delayed_heartbeat")
    result = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
    ).summary["learning_schedule"]
    assert result["state"] == "completed"
    assert result["pauses"] == 0


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("backlog", "resource_wait_timeout"),
        ("stale", "resource_wait_timeout"),
        ("missing", "resource_wait_timeout"),
        ("private_memory", "resource_wait_timeout"),
        ("archive_failed", "collector_failed"),
        ("incomplete_exit", "collector_failed"),
    ],
)
def test_unusable_resources_exit_with_evidence_without_a_candidate(tmp_path, fault, reason):
    from fh5.learning.bc.schedule import ScheduledBCTrain

    config, _, dataset = configuration(tmp_path)
    before = dataset.read_bytes()
    resources = Resources(permanent=fault)
    result = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
    ).summary["learning_schedule"]
    assert result["state"] == "stopped" and result["stop_reason"] == reason
    assert result["steps_completed"] == 0
    assert result["candidate"] is None
    assert not (tmp_path / "scheduled/candidate").exists()
    assert (tmp_path / "scheduled/schedule.json").is_file()
    assert resources.closed and resources.waited <= 1.1
    assert dataset.read_bytes() == before


def test_pressure_after_an_update_preserves_partial_progress_without_publishing(tmp_path):
    from fh5.learning.bc.schedule import ScheduledBCTrain

    class MidRun(Resources):
        def sample(self):
            if self.reads >= 3:
                self.permanent = "backlog"
            return super().sample()

    config, _, _ = configuration(tmp_path)
    result = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=MidRun()
    ).summary["learning_schedule"]
    assert result["steps_completed"] == 1
    assert result["state"] == "stopped" and result["stop_reason"] == "resource_wait_timeout"
    assert result["candidate"] is None and not (tmp_path / "scheduled/candidate").exists()


def test_collector_on_gpu_host_is_not_interrupted_to_start_cuda_training(tmp_path):
    from fh5.learning.bc.schedule import ScheduledBCTrain

    config, training, _ = configuration(tmp_path)
    values = json.loads(training.read_bytes())
    values["device"] = "cuda"
    training.write_text(json.dumps(values))
    options = json.loads(config.read_bytes())
    options["training_config_sha256"] = hashlib.sha256(training.read_bytes()).hexdigest()
    config.write_text(json.dumps(options))
    resources = Resources()
    result = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
    ).summary["learning_schedule"]
    assert result["steps_completed"] == 0 and result["candidate"] is None
    assert result["pressure_counts"]["cuda_waits_for_collection_exit"] > 0
    assert result["stop_reason"] == "resource_wait_timeout"
    assert result["commands_sent"] is False


def test_stop_request_during_resource_wait_preserves_evidence(tmp_path):
    from fh5.learning.bc.schedule import ScheduledBCTrain

    class StopDuringWait(Resources):
        def wait(self, seconds):
            super().wait(seconds)
            (tmp_path / "scheduled/stop.request").touch()

    config, _, _ = configuration(tmp_path)
    resources = StopDuringWait(permanent="backlog")
    summary = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
    ).summary["learning_schedule"]
    assert summary["stop_reason"] == "requested_stop" and resources.closed
    assert summary["steps_completed"] == 0 and summary["candidate"] is None


def test_sampling_that_exhausts_total_budget_cannot_start_training(tmp_path):
    from fh5.learning.bc.schedule import ScheduledBCTrain

    class SlowSample(Resources):
        def sample(self):
            self.time += 2
            return super().sample()

    config, _, _ = configuration(tmp_path, max_total_s=1)
    resources = SlowSample()
    summary = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
    ).summary["learning_schedule"]
    assert summary["stop_reason"] == "total_time_limit" and resources.closed
    assert summary["steps_completed"] == 0 and summary["candidate"] is None


def test_modified_training_config_is_rejected_before_resource_queries(tmp_path):
    from fh5.learning.bc.schedule import ScheduledBCTrain

    config, training, _ = configuration(tmp_path)
    training.write_bytes(training.read_bytes() + b" ")
    resources = Resources()
    with pytest.raises(ValueError, match="configuration changed"):
        run_experiment(
            ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
        )
    assert resources.reads == 0 and not (tmp_path / "scheduled").exists()


def test_overlong_work_unit_stops_before_the_next_update(tmp_path):
    from fh5.learning.bc.schedule import ScheduledBCTrain

    class SlowClock(Resources):
        def now_ns(self):
            self.time += 1
            return super().now_ns()

    config, _, _ = configuration(tmp_path, max_unit_s=0.1)
    resources = SlowClock()
    summary = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
    ).summary["learning_schedule"]
    assert summary["stop_reason"] == "work_unit_overrun"
    assert summary["steps_completed"] == 0 and summary["max_work_unit_s"] >= 1
    assert resources.closed and not (tmp_path / "scheduled/candidate").exists()


def test_native_probe_rejects_unbound_collector_before_querying_host_resources(tmp_path):
    from fh5.learning.bc.schedule import ScheduledBCTrain

    config, _, _ = configuration(tmp_path)
    bundle = tmp_path / "synthetic-bundle"
    bundle.mkdir()
    (bundle / "frozen.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest differs"):
        run_experiment(ScheduledBCTrain(config, tmp_path / "scheduled"))
    assert not (tmp_path / "scheduled").exists()


def test_completed_collection_does_not_override_gpu_resource_pressure(tmp_path):
    from fh5.learning.bc.schedule import ScheduledBCTrain

    class BusyGPU(Resources):
        def sample(self):
            value = super().sample()
            value["collector"].update(
                process_liveness="exited", final_status_present=True, complete=True
            )
            value["gpus"][0]["memory_used_mib"] = 23000
            return value

    config, training, _ = configuration(tmp_path)
    values = json.loads(training.read_bytes())
    values["device"] = "cuda"
    training.write_text(json.dumps(values))
    options = json.loads(config.read_bytes())
    options["training_config_sha256"] = hashlib.sha256(training.read_bytes()).hexdigest()
    config.write_text(json.dumps(options))
    resources = BusyGPU()
    summary = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
    ).summary["learning_schedule"]
    assert summary["steps_completed"] == 0 and summary["candidate"] is None
    assert summary["pressure_counts"]["gpu_pressure"] > 0
    assert "cuda_waits_for_collection_exit" not in summary["pressure_counts"]
    assert summary["stop_reason"] == "resource_wait_timeout" and resources.closed


def test_frozen_schedule_remains_runnable_after_developer_configs_change(tmp_path):
    from fh5.learning.bc.schedule import ScheduledBCTrain

    config, training, _ = configuration(tmp_path)
    run_experiment(ScheduledBCTrain(config, tmp_path / "first"), learning_resources=Resources())
    config.write_text("developer schedule changed")
    training.write_text("developer training configuration changed")
    summary = run_experiment(
        ScheduledBCTrain(tmp_path / "first/schedule-config.json", tmp_path / "second"),
        learning_resources=Resources(),
    ).summary["learning_schedule"]
    assert summary["state"] == "completed"
    reports = [
        json.loads((tmp_path / name / "candidate/report.json").read_bytes())
        for name in ("first", "second")
    ]
    assert [r["prediction"] for r in reports[0]["decisions"]] == [
        r["prediction"] for r in reports[1]["decisions"]
    ]


def test_published_candidate_previews_resolve_after_staging_directory_is_moved(tmp_path):
    from pathlib import Path
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname

    from fh5.learning.bc.schedule import ScheduledBCTrain

    config, _, _ = configuration(tmp_path)
    output = tmp_path / "scheduled with spaces"
    result = run_experiment(
        ScheduledBCTrain(config, output), learning_resources=Resources()
    ).summary["learning_schedule"]
    assert result["state"] == "completed"
    candidate = output / "candidate"
    html = (candidate / "report.html").read_text(encoding="utf-8")
    display, _ = json.JSONDecoder().raw_decode(html.split("const data=", 1)[1])
    urls = [url for row in display["decisions"] for url in row["preview_urls"]]
    assert urls
    for url in urls:
        parsed = urlparse(url)
        preview = (
            Path(url2pathname(parsed.path))
            if parsed.scheme == "file"
            else candidate / unquote(parsed.path)
        )
        assert preview.is_file(), url
        assert preview.read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("total", "total_time_limit"),
        ("unit", "work_unit_overrun"),
        ("stop", "requested_stop"),
    ],
)
def test_device_transfer_cannot_dispatch_training_after_budget_or_stop(
    tmp_path, monkeypatch, fault, reason
):
    import torch

    from fh5.learning.bc.schedule import ScheduledBCTrain

    class ClearingPressure(Resources):
        saw_pressure = False
        pressure_cleared = False

        def sample(self):
            sample = super().sample()
            pressured = sample["collector"]["pending_bytes"] > 0
            self.pressure_cleared = self.saw_pressure and not pressured
            self.saw_pressure |= pressured
            return sample

    resources = ClearingPressure(pressures=(3,))
    config, _, _ = configuration(
        tmp_path,
        max_total_s=5 if fault == "total" else 120,
        max_unit_s=5 if fault == "unit" else 60,
    )
    original_to = torch.nn.Module.to
    delayed = False

    def device_transfer(model, *args, **kwargs):
        nonlocal delayed
        result = original_to(model, *args, **kwargs)
        # Model a slow external Torch device transfer after pressure clears.
        # Training, scheduling and optimizer logic remain real.
        if resources.pressure_cleared and not delayed:
            delayed = True
            resources.time += 10
            if fault == "stop":
                (tmp_path / "scheduled/stop.request").touch()
        return result

    monkeypatch.setattr(torch.nn.Module, "to", device_transfer)
    summary = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
    ).summary["learning_schedule"]
    assert delayed and resources.closed
    assert summary["stop_reason"] == reason
    assert summary["steps_completed"] == 0
    assert summary["max_work_unit_s"] >= 10
    assert summary["candidate"] is None


@pytest.mark.parametrize("mismatch", [None, "session", "worker"])
def test_scheduler_binds_actual_collector_session_and_worker_to_its_manifest(tmp_path, mismatch):
    import time

    from fh5.collection.model import CollectionControl
    from fh5.collection.process import CollectionStart
    from fh5.learning.bc.schedule import ScheduledBCTrain
    from tests.collection.test_collection_process import prepare

    config, training, _ = configuration(tmp_path)
    folder = tmp_path / "collector"
    folder.mkdir()
    installation, _, _ = prepare(folder, seconds=2)
    bundle = folder / "recording-run"
    run_experiment(CollectionStart(installation, output_dir=bundle))
    status = {}
    try:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            status = run_experiment(CollectionControl(bundle)).summary["collection"]
            if status.get("complete") and status["process_liveness"] == "exited":
                break
            time.sleep(0.05)
    finally:
        run_experiment(CollectionControl(bundle, stop=True))
    assert status.get("complete") and status["process_liveness"] == "exited"
    manifest = bundle / "frozen.json"
    options = json.loads(config.read_bytes())
    if mismatch is None:
        # The normal v2 request names the selected recording, not a copied digest
        # or the reusable installation (which has no process/recording status).
        options.pop("training_config")
        options.pop("training_config_sha256")
        options.pop("collector_manifest_sha256")
        options.update(
            version=2,
            training=json.loads(training.read_bytes()),
            collector_bundle=bundle.relative_to(config.parent).as_posix(),
        )
        config.write_text(json.dumps(options))
        original = config.read_bytes()
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        summary = run_experiment(ScheduledBCTrain(config, tmp_path / "scheduled")).summary[
            "learning_schedule"
        ]
        assert summary["state"] == "completed"
        assert summary["steps_completed"] == 3
        assert summary["config"]["collector_bundle"] == str(bundle.resolve())
        assert summary["config"]["collector_manifest_sha256"] == digest
        assert config.read_bytes() == original
        assert (tmp_path / "scheduled/requested-schedule.json").read_bytes() == original
        frozen = tmp_path / "scheduled/schedule-config.json"
        assert json.loads(frozen.read_bytes())["collector_manifest_sha256"] == digest
        plain = run_experiment(TemporalBCTrain(training, tmp_path / "plain")).summary["temporal_bc"]
        candidate = json.loads((tmp_path / "scheduled/candidate/report.json").read_bytes())
        assert [r["prediction"] for r in candidate["decisions"]] == [
            r["prediction"] for r in plain["decisions"]
        ]
        assert summary["diagnostic_only"] and not summary["commands_sent"]

        manifest.write_bytes(manifest.read_bytes() + b" ")
        from fh5.learning.bc.schedule import ScheduledBCResume

        with pytest.raises(ValueError, match="manifest differs"):
            run_experiment(
                ScheduledBCResume(
                    tmp_path / "scheduled",
                    tmp_path / "rejected-resume",
                    summary["learner_checkpoint"]["manifest_sha256"],
                )
            )
        assert not (tmp_path / "rejected-resume").exists()
        options["collector_manifest_sha256"] = digest
        config.write_text(json.dumps(options))
        with pytest.raises(ValueError, match="manifest differs"):
            run_experiment(ScheduledBCTrain(config, tmp_path / "rejected-new"))
        assert not (tmp_path / "rejected-new").exists()
        return
    if mismatch == "worker":
        manifest.write_bytes(manifest.read_bytes() + b" ")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    session_path = bundle / "recording/session.json"
    session = json.loads(session_path.read_bytes())
    session["software_snapshot"]["manifest_sha256"] = digest if mismatch == "worker" else "1" * 64
    session_path.write_text(json.dumps(session))
    final_path = bundle / "recording/final.json"
    final = json.loads(final_path.read_bytes())
    final["session_sha256"] = hashlib.sha256(session_path.read_bytes()).hexdigest()
    final_path.write_text(json.dumps(final))
    options.update(collector_bundle=str(bundle), collector_manifest_sha256=digest)
    config.write_text(json.dumps(options))
    with pytest.raises(ValueError, match="snapshot differs"):
        run_experiment(ScheduledBCTrain(config, tmp_path / "scheduled"))
    summary = json.loads((tmp_path / "scheduled/schedule.json").read_bytes())
    assert summary["steps_completed"] == 0 and summary["candidate"] is None
