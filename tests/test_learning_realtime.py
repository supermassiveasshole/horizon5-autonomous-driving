"""Continuous asynchronous learning through the experiment entry point."""

import json
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_candidate_store import candidate_setup, record_config
from test_evaluation import sha
from test_learning_loop import SharedBackend, loop_request
from test_learning_recovery import interrupt_selection
from test_sac_realtime_cycle import AsyncEnvironment, CycleGame

from fh5.candidate_store import CandidateRecord
from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig


@pytest.fixture(scope="module")
def async_seed(tmp_path_factory):
    root = tmp_path_factory.mktemp("async-learning-seed")
    source = root / "source"
    source.mkdir()
    candidates = candidate_setup(source, candidate_steps=2)
    store = root / "versions"
    # This selection preserves the independently evaluated default.
    saved = run_experiment(
        CandidateRecord(
            record_config(root, candidates, missing_execution=True), store, None, candidates[3]
        )
    ).summary["candidate_store"]
    return candidates[0], store, saved, candidates[3]


def async_request(tmp_path, setup, *, rounds=2):
    operation = loop_request(tmp_path, setup, rounds=rounds, evaluation_seconds=0.6)
    config = json.loads(operation.config_file.read_bytes())
    config.pop("steps_per_attempt")
    runtime = RealtimeConfig(pixels=PixelContract(size=(64, 36)), reference_count=1)
    config.update(
        version=3,
        sampling={
            "runtime": {**asdict(runtime), "pixels": runtime.pixels.metadata()},
            "seconds": 0.6,
            "max_updates": 2,
        },
    )
    operation.config_file.write_text(json.dumps(config))
    return operation


class AsyncBackend(SharedBackend):
    def sampling(self, identity):
        assert self.active is None
        owner = self

        class Lease(AsyncEnvironment):
            def close(self):
                result = super().close()
                owner.active = None
                return result

        lease = Lease(self.source)
        self.active = lease
        self.leases.append(lease)
        return lease


class LargeMetadataBackend(AsyncBackend):
    def sampling(self, identity):
        class LargeMetadataGame(CycleGame):
            def read(self, period_s):
                point = super().read(period_s)
                return replace(
                    point,
                    observation=replace(
                        point.observation,
                        frames=tuple(
                            # Large source metadata exercises the original archive reader
                            # while stable image IDs keep actor-input copying inexpensive.
                            replace(frame, time_quality=frame.time_quality + "x" * 1500000)
                            for frame in point.observation.frames
                        ),
                    ),
                )

        lease = super().sampling(identity)
        lease.game_type = LargeMetadataGame
        return lease


def test_async_loop_samples_learns_evaluates_and_retains_each_round(
    tmp_path, async_seed, monkeypatch
):
    operation = async_request(tmp_path, async_seed)
    backend = AsyncBackend(async_seed[0])
    original_open = Path.open

    def unavailable_parent_html(path, mode="r", *args, **kwargs):
        if path == operation.output_dir / "report.html" and any(flag in mode for flag in "wx"):
            raise OSError("parent presentation unavailable")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unavailable_parent_html)
    result = run_experiment(operation, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "budget_completed", result
    assert result["rounds_completed"] == 2 and result["learner_updates"] == 4
    assert result["latest_learner"]["total_steps"] == 6
    assert result["eligible_transitions"] >= 4
    first, second = result["rounds"]
    assert first["candidate_sha256"] == second["sampling_checkpoint_sha256"]
    assert all(row["learner_updates"] == row["update_budget"] == 2 for row in result["rounds"])
    assert all(row["evaluation"]["metrics"]["all_attempts"] == 2 for row in result["rounds"])
    assert result["default"]["sha256"] == async_seed[2]["default"]["model_sha256"]
    assert result["resources_released"] and backend.closed
    assert len(backend.leases) == 4 and all(lease.closed for lease in backend.leases)
    assert result["presentation"]["status"] == "unavailable"


def test_async_stopped_child_recovers_only_its_update_budget_without_resampling(
    tmp_path, async_seed
):
    operation = async_request(tmp_path, async_seed, rounds=1)
    interrupt_selection(
        operation, async_seed[0], "before_learned", scenario="async_stopped_updates"
    )
    root = operation.output_dir
    child = root / "round-000/learning"
    stopped = json.loads((child / "summary.json").read_bytes())
    assert stopped["stop_reason"] == "stop_requested"
    assert stopped["attempts"][0]["eligible_transitions"] > 2
    assert stopped["attempts"][0]["learner_updates"] == 0
    original = {path: sha(path) for path in child.rglob("*") if path.is_file()}

    class NoSampling(AsyncBackend):
        def sampling(self, identity):
            raise AssertionError("Retained async experience must not be sampled again")

    backend = NoSampling(async_seed[0])
    held = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert held["stop_reason"] == "stop_requested", held
    assert held["learner_updates"] == 0 and held["rounds"][0]["update_budget"] == 2
    assert not backend.leases and held["resources_released"]
    (root / "stop.request").unlink()
    backend = NoSampling(async_seed[0])
    recovered = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert recovered["stop_reason"] == "budget_completed", recovered
    assert recovered["rounds_completed"] == 1
    assert recovered["learner_updates"] == 2 < recovered["eligible_transitions"]
    assert recovered["latest_learner"]["total_steps"] == 4
    assert recovered["rounds"][0]["evaluation"]["metrics"]["all_attempts"] == 2
    assert len(backend.leases) == 1 and recovered["resources_released"]
    assert all(sha(path) == digest for path, digest in original.items())


def test_async_failed_send_retries_once_and_preserves_original_attempt(tmp_path, async_seed):
    class UnavailableSink(CycleGame):
        def send(self, command):
            if command.throttle_u8:
                raise OSError("synthetic command sink disconnected")
            super().send(command)

    class OnceUnavailable(AsyncBackend):
        def sampling(self, identity):
            lease = super().sampling(identity)
            if len(self.leases) == 1:
                lease.game_type = UnavailableSink
            return lease

    operation = async_request(tmp_path, async_seed, rounds=1)
    config = json.loads(operation.config_file.read_bytes())
    config["sampling_retry"] = {"max_retries": 1}
    operation.config_file.write_text(json.dumps(config))
    backend = OnceUnavailable(async_seed[0])
    result = run_experiment(operation, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "budget_completed", result
    assert result["rounds_completed"] == 1 and result["learner_updates"] == 2
    row = result["rounds"][0]
    assert row["sampling_attempt"] == len(row["sampling_history"]) == 1
    failed = json.loads((operation.output_dir / "round-000/learning/summary.json").read_bytes())
    assert failed["stop_reason"] == "sampling_fault" and failed["resources_released"]
    assert not failed.get("latest_candidate")
    assert len(backend.leases) == 3 and result["resources_released"]


def test_async_pending_child_cannot_omit_original_pixels_from_its_inventory(tmp_path, async_seed):
    operation = async_request(tmp_path, async_seed, rounds=1)
    interrupt_selection(operation, async_seed[0], "before_learned", scenario="async_sampling")
    root = operation.output_dir
    child = root / "round-000/learning"
    summary_file = child / "summary.json"
    summary = json.loads(summary_file.read_bytes())
    attempt = summary["attempts"][0]
    binding = attempt["source_assets"]
    index = Path(binding["path"])
    connection = sqlite3.connect(index)
    try:
        omitted = connection.execute("SELECT path FROM assets WHERE path LIKE '%.rgb'").fetchone()[
            0
        ]
        connection.execute("DELETE FROM assets WHERE path = ?", (omitted,))
        connection.commit()
    finally:
        connection.close()
    binding["sha256"] = sha(index)
    summary_file.write_text(json.dumps(summary))
    (child / "attempt-000/cycle-result.json").write_text(json.dumps(attempt))
    before = {path: sha(path) for path in root.rglob("*") if path.is_file()}
    backend = AsyncBackend(async_seed[0])
    with pytest.raises(ValueError, match="original inventory is incomplete"):
        run_experiment(
            LearningContinue(root, sha(root / "state.json")), learning_environment=backend
        )
    assert not backend.leases and backend.closed
    assert all(sha(path) == digest for path, digest in before.items())


def test_existing_synchronous_loop_keeps_its_original_update_count(tmp_path, async_seed):
    operation = loop_request(tmp_path, async_seed, rounds=1, evaluation_seconds=0.6)
    backend = SharedBackend(async_seed[0])
    result = run_experiment(operation, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "budget_completed", result
    assert result["rounds_completed"] == 1
    assert result["eligible_transitions"] == result["learner_updates"] == 3
    assert result["latest_learner"]["total_steps"] == 5
    assert result["default"]["sha256"] == async_seed[2]["default"]["model_sha256"]
    assert len(backend.leases) == 2 and result["resources_released"]


@pytest.mark.parametrize("pending", [False, True])
def test_large_async_result_can_be_acknowledged_and_resumed(tmp_path, async_seed, pending):
    operation = async_request(tmp_path, async_seed, rounds=1)
    config = json.loads(operation.config_file.read_bytes())
    config["sampling"]["seconds"] = 1.0
    operation.config_file.write_text(json.dumps(config))
    root = operation.output_dir
    if pending:
        interrupt_selection(operation, async_seed[0], "before_learned", scenario="async_large")
        backend = AsyncBackend(async_seed[0])
        result = run_experiment(
            LearningContinue(root, sha(root / "state.json")), learning_environment=backend
        ).summary["learning_loop"]
        assert len(backend.leases) == 1  # Evaluation only; sealed sampling is reused.
    else:
        backend = LargeMetadataBackend(async_seed[0])
        result = run_experiment(operation, learning_environment=backend).summary["learning_loop"]
    child = root / "round-000/learning"
    execution = child / "attempt-000/execution/report.json"
    assert execution.stat().st_size > 4 * 1024**2  # Cross the old refusal threshold.
    recorded = json.loads(execution.read_bytes())
    inputs = [row["archive"] for row in recorded["decisions"] if row.get("archive")]
    assert inputs and all(
        (execution.parent / item["path"]).stat().st_size > 4 * 1024**2 for item in inputs
    )
    assert (child / "summary.json").stat().st_size < execution.stat().st_size
    attempt = json.loads((child / "summary.json").read_bytes())["attempts"][0]
    assert "decisions" not in attempt
    assert attempt["source_assets"]["kind"] == "sampling-source-index-v1"
    replay = json.loads((child / attempt["replay"]).read_bytes())
    eligible = len(replay["transitions"])
    # Sampling earns one update per eligible transition, up to the explicit budget.
    # Large metadata can reduce that count within the unchanged wall-clock budget.
    expected_updates = min(eligible, config["sampling"]["max_updates"])
    assert expected_updates > 0
    training = json.loads((child / attempt["candidate"] / "training-report.json").read_bytes())
    initial = json.loads((root / "initial/explorer/training-report.json").read_bytes())
    assert training["steps_requested"] == training["steps_completed"] == expected_updates
    archived_decisions = {row["decision_id"] for row in recorded["decisions"] if row.get("archive")}
    large_transition_ids = {
        f"{attempt['replay_sha256']}:{transition['id']}"
        for transition in replay["transitions"]
        if all(
            transition[side]["decision_id"] in archived_decisions for side in ("current", "next")
        )
    }
    updates_file = child / attempt["candidate"] / training["updates"]["path"]
    updates = [json.loads(line) for line in updates_file.read_text().splitlines()]
    sampled_ids = {identity for update in updates for identity in update["transition_ids"]}
    assert large_transition_ids & sampled_ids  # The learner consumed oversized originals.
    assert training["critic_change_max"] > 0 and training["encoder_change_max"] > 0
    assert training["learner_state_sha256"] != initial["learner_state_sha256"]
    assert attempt["eligible_transitions"] == result["eligible_transitions"] == eligible
    assert attempt["learner_updates"] == result["learner_updates"] == expected_updates
    row = result["rounds"][0]
    assert row["eligible_transitions"] == eligible
    assert row["update_budget"] == row["learner_updates"] == expected_updates
    assert result["latest_learner"]["learner_state_sha256"] == training["learner_state_sha256"]
    assert (
        result["latest_learner"]["total_steps"]
        == training["total_steps"]
        == (initial["total_steps"] + expected_updates)
    )
    assert result["stop_reason"] == "budget_completed", result.get("error")
    assert result["rounds_completed"] == 1
    assert result["resources_released"]
    backend = AsyncBackend(async_seed[0])
    resumed = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed"
    assert resumed["learner_updates"] == expected_updates and not backend.leases
    assert resumed["eligible_transitions"] == eligible and resumed["rounds_completed"] == 1
    assert resumed["latest_learner"] == result["latest_learner"]
    assert resumed["rounds"] == result["rounds"] and resumed["resources_released"]
