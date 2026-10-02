"""Learning state growth through public continuation and persisted evidence."""

import json
from pathlib import Path

import pytest
from learning_files import stage_records
from test_candidate_store import candidates as candidates
from test_evaluation import sha
from test_learning_loop import SharedBackend, loop_request
from test_learning_loop import seeded_loop as seeded_loop

from fh5.experiment import run_experiment
from fh5.learning_loop import LearningContinue


def test_long_legacy_stage_history_can_resume_into_compact_parent_state(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop)
    backend = SharedBackend(seeded_loop[0])
    backend.stop_sampling_file = request.output_dir / "stop.request"
    stopped = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    state_path = request.output_dir / "state.json"
    legacy = json.loads(state_path.read_bytes())
    # A valid legacy document spanning both old ceilings. These are retained
    # stage records, not a claim that this test executed 1001 learning stages.
    legacy["stages"] = [
        {"phase": "ready", "at_ns": index + 1, "round": -1} for index in range(999)
    ] + [
        {"phase": "driving", "at_ns": 1000, "round": 0},
        {"phase": "stopped", "at_ns": 1001, "round": 0},
    ]
    raw = json.dumps(legacy).encode() + b" " * (4 * 1024**2)
    state_path.write_bytes(raw)
    second_backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(state_path)), learning_environment=second_backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "stop_requested"
    assert resumed["latest_learner"] == stopped["latest_learner"]
    assert resumed["learner_updates"] == 0 and second_backend.leases == []
    descriptor = resumed["stages"]
    assert isinstance(descriptor, dict), "Stage history must not accumulate in parent snapshots"
    assert descriptor["format"] == "learning-stages-v1"
    assert descriptor["events"] == 1003
    assert [row["phase"] for row in descriptor["tail"]] == ["resuming", "stopped"]
    saved = json.loads(state_path.read_bytes())
    assert saved["stages"] == descriptor
    assert resumed["interruptions"][-1]["phase"] == "driving"
    records = stage_records(request.output_dir, resumed)
    assert records[:1001] == legacy["stages"]
    assert len(records) == 1003


def test_stage_segments_preserve_committed_history_and_ignore_uncommitted_tail(
    tmp_path, seeded_loop
):
    request = loop_request(tmp_path, seeded_loop)
    backend = SharedBackend(seeded_loop[0])
    backend.stop_sampling_file = request.output_dir / "stop.request"
    first = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    root = request.output_dir
    initial_records = stage_records(root, first)
    first_log = root / first["stages"]["diagnostic"]["path"]
    # Simulate diagnostics written after the last acknowledged parent state.
    with first_log.open("ab") as stream:
        stream.write(b'{"unacknowledged":"crash tail"}\n')
    original_bytes = first_log.read_bytes()
    previous = first
    for invocation in range(2):
        backend = SharedBackend(seeded_loop[0])
        current = run_experiment(
            LearningContinue(root, sha(root / "state.json")), learning_environment=backend
        ).summary["learning_loop"]
        assert current["stop_reason"] == "stop_requested" and not backend.leases
        assert current["stages"]["events"] == first["stages"]["events"] + 2 * (invocation + 1)
        assert first_log.read_bytes() == original_bytes
        current_log = root / current["stages"]["diagnostic"]["path"]
        with current_log.open("rb") as stream:
            header = json.loads(stream.readline())
        assert header["previous"] == previous["stages"]["diagnostic"]
        records = stage_records(root, current)
        assert records[: len(initial_records)] == initial_records
        assert len(records) == current["stages"]["events"]
        previous = current
    # History is optional; a missing segment cannot remove required stop evidence.
    current_log.unlink()
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "stop_requested" and not backend.leases
    assert resumed["latest_learner"] == first["latest_learner"]
    assert resumed["interruptions"][-1]["phase"] == "resuming"


@pytest.mark.parametrize("failure", [OSError, MemoryError])
@pytest.mark.parametrize("stage", ["open", "write", "flush", "close"])
def test_optional_stage_log_failure_preserves_finished_learner(
    tmp_path, seeded_loop, monkeypatch, failure, stage
):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    root = request.output_dir
    original_open = Path.open
    streams = []

    class FailingSink:
        def __init__(self, stream):
            self.stream = stream
            self.writes = 0

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, raw):
            self.writes += 1
            if stage == "write" and self.writes == 2:
                raise failure("optional stage diagnostic exhausted")
            return self.stream.write(raw)

        def flush(self):
            if stage == "flush":
                raise failure("optional stage diagnostic exhausted")
            return self.stream.flush()

        def close(self):
            self.stream.close()
            if stage == "close":
                raise failure("optional stage diagnostic exhausted")

    def failing_log(path, mode="r", *args, **kwargs):
        if path.parent == root / "diagnostics" and path.name.startswith("stages-") and mode == "xb":
            if stage == "open":
                raise failure("optional stage diagnostic exhausted")
            stream = original_open(path, mode, *args, **kwargs)
            streams.append(stream)
            return FailingSink(stream)
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", failing_log)
        trained = run_experiment(
            request, learning_environment=SharedBackend(seeded_loop[0])
        ).summary["learning_loop"]
    assert trained["stop_reason"] == "budget_completed", trained.get("error")
    assert trained["learner_updates"] == 3 and trained["latest_learner"]["total_steps"] == 33
    assert trained["stages"]["diagnostic"]["status"] == "unavailable"
    assert "optional stage diagnostic exhausted" in trained["stages"]["diagnostic"]["error"]
    assert trained["stages"]["tail"][-1]["phase"] == "stopped"
    assert all(stream.closed for stream in streams)
    assert json.loads((root / "state.json").read_bytes())["stages"] == trained["stages"]
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed" and not backend.leases
    assert (
        resumed["learner_updates"] == 3 and resumed["latest_learner"] == trained["latest_learner"]
    )
