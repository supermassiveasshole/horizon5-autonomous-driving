"""Collection-aware learning is exercised through complete experiment runs."""

import hashlib
import json

import pytest
from test_collection_bc import prepare_inputs

from fh5.collection_bc import CollectionBCPrepare
from fh5.experiment import run_experiment
from fh5.temporal_bc import TemporalBCTrain

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
    prepare, _ = prepare_inputs(tmp_path)
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
    from fh5.learning_schedule import ScheduledBCTrain

    config, training, dataset = configuration(tmp_path)
    frozen = dataset.read_bytes()
    plain = run_experiment(TemporalBCTrain(training, tmp_path / "plain")).summary["temporal_bc"]
    resources = Resources(pressures=(5, 6))
    result = run_experiment(
        ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
    ).summary["learning_schedule"]
    assert result["state"] == "completed"
    assert result["steps_completed"] == 3
    assert resources.closed and resources.waited > 0
    assert result["pauses"] >= 1
    assert any("collection_backlog" in e["reasons"] for e in result["events"])
    assert result["commands_sent"] is False
    assert result["diagnostic_only"] is True
    actual = json.loads((tmp_path / "scheduled/candidate/report.json").read_bytes())
    assert [r["prediction"] for r in actual["decisions"]] == [
        r["prediction"] for r in plain["decisions"]
    ]
    assert dataset.read_bytes() == frozen


def test_recent_collector_heartbeat_uses_image_age_at_its_actual_poll(tmp_path):
    from fh5.learning_schedule import ScheduledBCTrain

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
        ("private_memory", "resource_limit:process_private_bytes"),
        ("archive_failed", "collector_failed"),
        ("incomplete_exit", "collector_failed"),
    ],
)
def test_unusable_resources_exit_with_evidence_without_a_candidate(tmp_path, fault, reason):
    from fh5.learning_schedule import ScheduledBCTrain

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
    from fh5.learning_schedule import ScheduledBCTrain

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
    from fh5.learning_schedule import ScheduledBCTrain

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
    from fh5.learning_schedule import ScheduledBCTrain

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
    from fh5.learning_schedule import ScheduledBCTrain

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
    from fh5.learning_schedule import ScheduledBCTrain

    config, training, _ = configuration(tmp_path)
    training.write_bytes(training.read_bytes() + b" ")
    resources = Resources()
    with pytest.raises(ValueError, match="configuration changed"):
        run_experiment(
            ScheduledBCTrain(config, tmp_path / "scheduled"), learning_resources=resources
        )
    assert resources.reads == 0 and not (tmp_path / "scheduled").exists()


def test_overlong_work_unit_stops_before_the_next_update(tmp_path):
    from fh5.learning_schedule import ScheduledBCTrain

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
    from fh5.learning_schedule import ScheduledBCTrain

    config, _, _ = configuration(tmp_path)
    bundle = tmp_path / "synthetic-bundle"
    bundle.mkdir()
    (bundle / "frozen.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest differs"):
        run_experiment(ScheduledBCTrain(config, tmp_path / "scheduled"))
    assert not (tmp_path / "scheduled").exists()


def test_completed_collection_does_not_override_gpu_resource_pressure(tmp_path):
    from fh5.learning_schedule import ScheduledBCTrain

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
