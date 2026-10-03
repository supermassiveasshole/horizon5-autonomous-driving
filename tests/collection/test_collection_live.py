"""Passive numerical source composition at the agreed experiment-run seam."""

import json
import threading
import time
from dataclasses import replace

import pytest

from fh5.capture.pipeline import CaptureConfig
from fh5.collection.model import CollectionReview
from fh5.experiment import run_experiment
from tests.collection.test_collection import request, saved_rows
from tests.collection.test_demonstrations import raw_input
from tests.driving.test_realtime_shadow import Capture, Desktop, Telemetry


class Inputs:
    def __init__(self, identity):
        self.device = identity
        self.closed = False
        self.polls = 0

    def identity(self):
        return self.device

    def read(self):
        self.polls += 1
        now = time.perf_counter_ns()
        return raw_input(0, observed_ns=now, available_ns=time.perf_counter_ns())

    def close(self):
        self.closed = True


def setup(tmp_path, lease_path=None):
    from fh5.collection.live import PassiveCollectionEnvironment

    req = request(tmp_path, seconds=0.9, block_rows=20, expected_car_ordinal=2941, expected_pi=999)
    inputs = Inputs(json.loads(req.input_profile.read_bytes())["device"])
    telemetry, capture = Telemetry(), Capture()
    env = PassiveCollectionEnvironment(
        req,
        CaptureConfig(pixels=req.config.pixels),
        lambda: capture,
        telemetry,
        Desktop(),
        inputs,
        source_kind="synthetic",
        lease_path=lease_path or tmp_path / "resource.lock",
    )
    return req, env, inputs, telemetry, capture


def test_passive_adapter_records_raw_rgb_and_inputs_without_any_control_interface(tmp_path):
    req, env, inputs, telemetry, capture = setup(tmp_path)
    result = run_experiment(req, collection_environment=env).summary["collection"]
    assert result["complete"] and result["stop_reason"] == "time_limit"
    assert result["counts"]["synchronized_observations"] > 2
    rows = saved_rows(req.output_dir)
    assert any(row["frames"] for row in rows)
    row = next(row for row in rows if row["frames"])
    assert row["mapped_input"]["mapped"] == [-1, 128 / 255]
    assert result["environment"]["capture"]["source_kind"] == "synthetic-capture"
    assert inputs.closed and telemetry.closed and capture.closed
    assert result["commands_sent"] is False
    review = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html"))
    assert review.summary["collection"]["complete"]


@pytest.mark.parametrize("problem", ["disconnected", "different_device"])
def test_passive_adapter_isolates_input_loss_and_recovers_fresh_history(tmp_path, problem):
    req, env, inputs, telemetry, capture = setup(tmp_path)

    class InterruptedInputs(Inputs):
        def identity(self):
            return {} if problem == "different_device" and 30 <= self.polls < 40 else self.device

        def read(self):
            value = super().read()
            if problem == "disconnected" and 30 <= self.polls < 40:
                value["connected"] = False
            return value

    env.inputs = InterruptedInputs(inputs.device)
    run_experiment(req, collection_environment=env)
    rows = saved_rows(req.output_dir)
    reason = "disconnected" if problem == "disconnected" else "device_identity_mismatch"
    rejected = [row for row in rows if reason in row["reasons"]]
    assert rejected and all(not row["input_usable"] and not row["frames"] for row in rejected)
    assert any(row["frames"] for row in rows if row["sequence"] > rejected[-1]["sequence"])
    assert all(
        f["source_time_ns"] >= row["segment_start_ns"] for row in rows for f in row["frames"]
    )


def test_resource_owner_excludes_second_collector_and_releases_for_next_run(tmp_path):
    first, second, third = (tmp_path / name for name in ("first", "second", "third"))
    for folder in (first, second, third):
        folder.mkdir()
    lease = tmp_path / "resources.lock"
    req, env, inputs, _, _ = setup(first, lease)
    req = replace(req, config=replace(req.config, seconds=1.5))
    completed = []
    worker = threading.Thread(
        target=lambda: completed.append(run_experiment(req, collection_environment=env))
    )
    worker.start()
    deadline = time.monotonic() + 1
    while inputs.polls < 5 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert inputs.polls >= 5
    req2, env2, inputs2, _, capture2 = setup(second, lease)
    result = run_experiment(req2, collection_environment=env2).summary["collection"]
    assert result["stop_reason"] == "source_error" and "already owned" in result["error"]
    assert inputs2.polls == 0 and not capture2.closed
    worker.join(timeout=3)
    assert not worker.is_alive() and completed[0].summary["collection"]["complete"]
    req3, env3, _, _, _ = setup(third, lease)
    assert run_experiment(req3, collection_environment=env3).summary["collection"]["complete"]
