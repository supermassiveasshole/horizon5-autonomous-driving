"""Real process exits at filesystem publication boundaries, through run_experiment."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fh5.evaluation.candidate_store import CandidateHistory, CandidateRecord
from fh5.experiment import run_experiment
from fh5.learning.loop.runner import LearningContinue
from tests.evaluation.test_candidate_store import candidates as candidates
from tests.evaluation.test_evaluation import sha
from tests.learning.loop.test_learning_loop import SharedBackend, loop_request
from tests.learning.loop.test_learning_loop import seeded_loop as seeded_loop


def interrupt_selection(
    request,
    source,
    boundary="after_commit",
    *,
    scenario="normal_trace",
):
    assert scenario in {
        "normal_trace",
        "large_trace",
        "failed_evaluation",
        "parent_evidence",
        "stopped_evaluation",
        "failed_sampling",
        "stopped_updates",
        "async_sampling",
        "async_stopped_updates",
        "async_large",
    }
    repository = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(repository / "src"), str(repository)))
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, os, sys
from contextlib import nullcontext
from dataclasses import replace as replaced
from pathlib import Path
from fh5.experiment import run_experiment
from fh5.learning.loop.runner import LearningLoop
from tests.learning.loop.test_learning_loop import SharedBackend
root = Path(sys.argv[2]).resolve()
original_replace = os.replace
def replace(source, destination, *args, **kwargs):
    if (sys.argv[4] == 'before_parent_ledger'
            and Path(destination).resolve() == root / 'round-000/parent-ledger.json'):
        os._exit(73)
    if Path(destination).resolve() == root / 'state.json':
        value = json.loads(Path(source).read_bytes())
        if sys.argv[4] == 'before_commit' and value['phase'] == 'saving_versions':
            original_replace(source, destination, *args, **kwargs)
            os._exit(73)
        if sys.argv[4] == 'after_commit' and value['phase'] == 'ready' and value['rounds_completed'] == 1:
            os._exit(73)
        if sys.argv[4] == 'before_learned' and value['phase'] == 'learned':
            os._exit(73)
        if (sys.argv[4] == 'before_retried_failure' and value['phase'] == 'learned'
                and value['rounds'][0].get('sampling_attempt') == 1):
            os._exit(73)
        if sys.argv[4] == 'before_evaluation_ack' and value['phase'] == 'reviewing_evaluation':
            os._exit(73)
        if sys.argv[4] == 'before_evaluation_run' and value['phase'] == 'evaluating':
            original_replace(source, destination, *args, **kwargs)
            os._exit(73)
        if sys.argv[4] == 'before_parent_review_ack' and value['phase'] == 'evaluated':
            os._exit(73)
    return original_replace(source, destination, *args, **kwargs)
os.replace = replace
original_open = Path.open
def open_file(path, *args, **kwargs):
    if (sys.argv[4] == 'before_review_report'
            and path.resolve() == root / 'round-000/reviewed/batch-report.json'):
        os._exit(73)
    return original_open(path, *args, **kwargs)
Path.open = open_file
if sys.argv[5] == 'async_large':
    from tests.learning.loop.test_learning_realtime import LargeMetadataBackend
    backend = LargeMetadataBackend(Path(sys.argv[3]))
elif sys.argv[5].startswith('async_'):
    from tests.learning.loop.test_learning_realtime import AsyncBackend
    backend = AsyncBackend(Path(sys.argv[3]))
else:
    backend = SharedBackend(Path(sys.argv[3]))
if sys.argv[5] == 'failed_sampling':
    original_sampling = backend.sampling
    def sampling(identity):
        lease = original_sampling(identity)
        lease.fail_at = 0
        return lease
    backend.sampling = sampling
if sys.argv[5] == 'stopped_evaluation':
    from fh5.driving.control import Command
    original_evaluation = backend.evaluation
    def evaluation(identity):
        lease = original_evaluation(identity)
        original_driving = lease.driving
        def driving(slot, ready):
            game = original_driving(slot, ready)
            original_send = game.send
            def send(command):
                original_send(command)
                if command != Command(0, 0, 0):
                    (root / 'stop.request').write_text('external stop')
            game.send = send
            return game
        lease.driving = driving
        return lease
    backend.evaluation = evaluation
if sys.argv[5] == 'parent_evidence':
    from tests.evaluation.test_attempts import evidence
    def review(recording):
        target = root.parent / 'independent-evidence' / recording.parent.name
        target.mkdir(parents=True, exist_ok=True)
        return evidence(target, recording)
    backend.review = review
if sys.argv[5] == 'failed_evaluation':
    original_evaluation = backend.evaluation
    def evaluation(identity):
        lease = original_evaluation(identity)
        original_driving = lease.driving
        def driving(slot, ready):
            game = original_driving(slot, ready)
            game.signals = lambda: (len(game.sent) < 3, False)
            return game
        lease.driving = driving
        return lease
    backend.evaluation = evaluation
if sys.argv[5] == 'large_trace':
    # Expand permitted external frame metadata without fabricating learned results.
    original_sampling = backend.sampling
    def sampling(identity):
        lease = original_sampling(identity)
        original_sample = lease.sample
        def sample():
            value = original_sample()
            frames = tuple(replaced(f, frame_id=f.frame_id + 'x' * 400000)
                           for f in value.decision.frames)
            return replaced(value, decision=replaced(value.decision, frames=frames))
        lease.sample = sample
        return lease
    backend.sampling = sampling
from tests.learning.loop.test_learning_update_resume import stop_on_creation
guard = (stop_on_creation(root / 'round-000/learning/candidate-000', root / 'stop.request')
         if sys.argv[5] in ('stopped_updates', 'async_stopped_updates') else nullcontext())
with guard:
    run_experiment(LearningLoop(Path(sys.argv[1]), root),
                   learning_environment=backend)
raise SystemExit('Expected filesystem exit was not reached')
""",
            str(request.config_file),
            str(request.output_dir),
            str(source),
            boundary,
            scenario,
        ],
        env=environment,
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert process.returncode == 73, process.stdout + process.stderr


def test_sealed_evaluation_survives_exit_before_parent_acknowledgement(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_evaluation_ack")
    state_file = request.output_dir / "state.json"
    interrupted = json.loads(state_file.read_bytes())
    assert interrupted["phase"] == "evaluating"
    assert interrupted["learner_updates"] == 3
    assert "evaluation_run" not in interrupted["rounds"][0]
    root = request.output_dir / "round-000/evaluation"
    execution = json.loads((root / "run.json").read_bytes())
    assert execution["stop_reason"] == "plan_complete"
    assert execution["resources_released"] is True
    originals = {path: sha(path) for path in root.rglob("*") if path.is_file()}
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(state_file)), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed", resumed.get("error")
    assert resumed["rounds_completed"] == 1
    assert resumed["learner_updates"] == resumed["eligible_transitions"] == 3
    assert resumed["latest_learner"]["sha256"] == interrupted["latest_learner"]["sha256"]
    assert resumed["latest_learner"]["total_steps"] == 33
    assert resumed["rounds"][0]["evaluation_run"] == execution
    assert resumed["rounds"][0]["evaluation"]["metrics"]["all_attempts"] == 2
    assert resumed["rounds"][0]["evaluation"]["execution_metrics"]["bound_runs"] == 2
    assert resumed["recoveries"][0]["kind"] == "sealed_evaluation"
    assert resumed["interruptions"][0] == {
        "phase": "evaluating",
        "stop_reason": "unclean_exit",
        "error": None,
    }
    assert not backend.leases and backend.closed and resumed["resources_released"]
    assert all(sha(path) == digest for path, digest in originals.items())
    history = run_experiment(CandidateHistory(tmp_path / "versions")).summary["candidate_store"]
    assert len(history["history"]) == 2
    assert history["explorer"]["model_sha256"] == resumed["latest_learner"]["sha256"]
    assert history["default"]["model_sha256"] == seeded_loop[2]["default"]["model_sha256"]


def test_sampling_recovery_accepts_the_same_trace_size_as_the_sampler(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_learned", scenario="large_trace")
    root = request.output_dir
    trace = root / "round-000/learning/attempt-000/trace.json"
    assert 4 * 1024**2 < trace.stat().st_size < 32 * 1024**2
    backend = SharedBackend(seeded_loop[0])
    result = run_experiment(
        LearningContinue(root, sha(root / "state.json")), learning_environment=backend
    ).summary["learning_loop"]
    assert result["stop_reason"] == "budget_completed", result.get("error")
    assert result["learner_updates"] == 3
    assert result["latest_learner"]["total_steps"] == 33
    assert len(backend.leases) == 1 and backend.closed


def test_completed_failed_evaluation_keeps_its_attempt_and_unstarted_slots_on_recovery(
    tmp_path, seeded_loop
):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(
        request, seeded_loop[0], "before_evaluation_ack", scenario="failed_evaluation"
    )
    state = request.output_dir / "state.json"
    interrupted = json.loads(state.read_bytes())
    root = request.output_dir / "round-000/evaluation"
    execution = json.loads((root / "run.json").read_bytes())
    assert execution["stop_reason"] == "execution_stopped"
    assert execution["resources_released"] is True
    assert execution["started_slots"] == ["run-0"]
    assert execution["unstarted_slots"] == ["run-1"]
    originals = {path: sha(path) for path in root.rglob("*") if path.is_file()}
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(state)), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "evaluation_execution_stopped"
    assert resumed["rounds_completed"] == 1
    assert resumed["learner_updates"] == 3
    assert resumed["latest_learner"] == interrupted["latest_learner"]
    row = resumed["rounds"][0]
    assert row["evaluation_run"] == execution
    assert row["evaluation"]["metrics"]["all_attempts"] == 1
    assert row["evaluation"]["unstarted_slots"] == ["run-1"]
    assert row["selection"] == "retain_incumbent"
    assert resumed["resources_released"] and backend.closed and not backend.leases
    assert all(sha(path) == digest for path, digest in originals.items())
    history = run_experiment(CandidateHistory(tmp_path / "versions")).summary["candidate_store"]
    assert len(history["history"]) == 2
    assert history["default"]["model_sha256"] == seeded_loop[2]["default"]["model_sha256"]


def test_evaluation_recovery_requires_the_complete_original_child(tmp_path, seeded_loop):
    import hashlib

    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_evaluation_ack")
    state = request.output_dir / "state.json"
    original_state = state.read_bytes()
    root = request.output_dir / "round-000/evaluation"
    completion = root / "completion.json"
    original_completion = completion.read_bytes()
    frame = next((root / "attempt-0000/execution/pixels").glob("*.rgb"))
    cases = [{completion: None}, {frame: None}, {root / "frozen/model/policy.pt": b"changed"}]
    for name, key, value in (
        ("run.json", "resources_released", False),
        ("run.json", "started_slots", ["run-0"]),
        ("run-protocol.json", "seconds_per_attempt", 99),
        ("review/batch-report.json", "verified_starts", 1),
    ):
        document = json.loads((root / name).read_bytes())
        document[key] = value
        payload = json.dumps(document).encode()
        seal = json.loads(original_completion)
        seal["files"][name] = hashlib.sha256(payload).hexdigest()
        cases.append({root / name: payload, completion: json.dumps(seal).encode()})
    for changes in cases:
        originals = {path: path.read_bytes() for path in changes}
        backend = SharedBackend(seeded_loop[0])
        try:
            for path, payload in changes.items():
                if payload is None:
                    path.unlink()
                else:
                    path.write_bytes(payload)
            with pytest.raises((OSError, ValueError)):
                run_experiment(
                    LearningContinue(request.output_dir, sha(state)), learning_environment=backend
                )
            assert state.read_bytes() == original_state
            assert not backend.leases and backend.closed
            history = run_experiment(CandidateHistory(tmp_path / "versions")).summary[
                "candidate_store"
            ]
            assert len(history["history"]) == 1
        finally:
            for path, payload in originals.items():
                path.write_bytes(payload)


def test_evaluation_recovery_rejects_start_in_place_of_the_requested_restart(tmp_path, seeded_loop):
    from fh5.evaluation.run import EvaluationRun
    from tests.learning.sac.test_sac_imitation_evaluation import SteeringDragBatch

    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_evaluation_run")
    state = request.output_dir / "state.json"
    original_state = state.read_bytes()
    root = request.output_dir / "round-000"
    batch = root / "batch"
    child = root / "evaluation"
    result = run_experiment(
        EvaluationRun(
            batch,
            sha(batch / "batch.json"),
            batch / "start/event.json",
            child,
            1.0,
            seeded_loop[3],
            initial_operation="start_ready",
        ),
        evaluation_environment=SteeringDragBatch(),
    ).summary
    assert result["evaluation_run"]["stop_reason"] == "plan_complete"
    assert result["evaluation"]["verified_starts"] == 2
    assert result["evaluation"]["starts"][0]["operation"] == "start_ready"
    protocol_path = child / "run-protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    protocol["initial_operation"] = "restart_ready"
    protocol_path.write_text(json.dumps(protocol))
    completion = child / "completion.json"
    seal = json.loads(completion.read_bytes())
    seal["files"]["run-protocol.json"] = sha(protocol_path)
    completion.write_text(json.dumps(seal))
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises(ValueError, match="operation"):
        run_experiment(
            LearningContinue(request.output_dir, sha(state)), learning_environment=backend
        )
    assert state.read_bytes() == original_state
    assert not backend.leases and backend.closed


def test_evaluation_recovery_rejects_a_shorter_execution_than_its_declared_time_limit(
    tmp_path, seeded_loop
):
    from fh5.evaluation.run import EvaluationRun

    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_evaluation_run")
    state = request.output_dir / "state.json"
    original_state = state.read_bytes()
    root = request.output_dir / "round-000"
    batch = root / "batch"
    child = root / "evaluation"
    producer = SharedBackend(seeded_loop[0])
    result = run_experiment(
        EvaluationRun(
            batch,
            sha(batch / "batch.json"),
            batch / "start/event.json",
            child,
            0.3,
            seeded_loop[3],
            initial_operation="restart_ready",
        ),
        evaluation_environment=producer.evaluation("shortened"),
    ).summary
    assert result["evaluation_run"]["stop_reason"] == "plan_complete"
    assert result["evaluation"]["verified_starts"] == 2
    assert producer.close()["resources_released"]
    for index in range(2):
        report = json.loads((child / f"attempt-{index:04d}/execution/report.json").read_bytes())
        assert report["stop_reason"] == "time_limit"
        assert report["ended_ns"] - report["started_ns"] < 1_000_000_000
    protocol_path = child / "run-protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    protocol["seconds_per_attempt"] = 1.0
    protocol_path.write_text(json.dumps(protocol))
    completion = child / "completion.json"
    seal = json.loads(completion.read_bytes())
    seal["files"]["run-protocol.json"] = sha(protocol_path)
    completion.write_text(json.dumps(seal))
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises(ValueError, match="duration"):
        run_experiment(
            LearningContinue(request.output_dir, sha(state)), learning_environment=backend
        )
    assert state.read_bytes() == original_state
    assert not backend.leases and backend.closed


def test_sealed_sampling_is_adopted_after_exit_before_parent_acknowledgement(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_learned")
    state_file = request.output_dir / "state.json"
    interrupted = json.loads(state_file.read_bytes())
    assert interrupted["phase"] == "updating"
    assert interrupted["learner_updates"] == 0
    assert "candidate_sha256" not in interrupted["rounds"][0]
    learning = request.output_dir / "round-000/learning"
    summary = json.loads((learning / "summary.json").read_bytes())
    candidate = learning / summary["latest_candidate"]
    candidate_sha = sha(candidate / "policy.json")
    originals = {path: sha(path) for path in learning.rglob("*") if path.is_file()}
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(state_file)), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed", resumed.get("error")
    assert resumed["rounds_completed"] == 1
    assert resumed["learner_updates"] == resumed["eligible_transitions"] == 3
    assert resumed["latest_learner"]["sha256"] == candidate_sha
    assert resumed["latest_learner"]["total_steps"] == 33
    assert resumed["recoveries"][0]["kind"] == "sealed_sampling"
    assert resumed["interruptions"][0]["phase"] == "updating"
    assert resumed["interruptions"][0]["stop_reason"] == "unclean_exit"
    assert len(backend.leases) == 1  # Evaluate the saved model; no new sampling lease.
    assert backend.closed and resumed["resources_released"]
    assert all(sha(path) == digest for path, digest in originals.items())
    history = run_experiment(CandidateHistory(tmp_path / "versions")).summary["candidate_store"]
    assert len(history["history"]) == 2
    assert history["explorer"]["model_sha256"] == candidate_sha
    assert history["default"]["model_sha256"] == seeded_loop[2]["default"]["model_sha256"]


def test_committed_candidate_is_reconciled_after_process_exit_without_duplicate_learning(
    tmp_path, seeded_loop
):
    request = loop_request(tmp_path, seeded_loop)
    interrupt_selection(request, seeded_loop[0])
    state_file = request.output_dir / "state.json"
    interrupted = json.loads(state_file.read_bytes())
    assert interrupted["phase"] == "saving_versions"
    assert interrupted["rounds_completed"] == 0
    assert interrupted["learner_updates"] == 3
    history = run_experiment(CandidateHistory(tmp_path / "versions")).summary["candidate_store"]
    assert len(history["history"]) == 2
    assert history["revision"] != interrupted["store_revision"]
    saved_candidate = history["explorer"]["model_sha256"]
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(state_file)), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed"
    assert resumed["rounds_completed"] == 2
    assert resumed["learner_updates"] == resumed["eligible_transitions"] == 6
    assert resumed["latest_learner"]["total_steps"] == 36
    assert resumed["rounds"][1]["sampling_checkpoint_sha256"] == saved_candidate
    assert len(backend.leases) == 2  # Only the unfinished second round samples/evaluates.
    final = run_experiment(CandidateHistory(tmp_path / "versions")).summary["candidate_store"]
    assert len(final["history"]) == 3
    assert final["default"]["model_sha256"] == history["default"]["model_sha256"]
    assert resumed["resources_released"]
    assert resumed["recoveries"][0]["kind"] == "candidate_commit"
    assert resumed["interruptions"][0] == {
        "phase": "saving_versions",
        "stop_reason": "unclean_exit",
        "error": None,
    }


def test_prepared_selection_survives_exit_before_the_store_commit(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_commit")
    state_file = request.output_dir / "state.json"
    interrupted = json.loads(state_file.read_bytes())
    assert interrupted["phase"] == "saving_versions"
    assert interrupted["learner_updates"] == 3
    store = CandidateHistory(tmp_path / "versions")
    before = run_experiment(store).summary["candidate_store"]
    assert before["revision"] == interrupted["store_revision"]
    root = request.output_dir / "round-000"
    originals = {
        path: path.read_bytes() for path in (root / "comparison.json", root / "retain.json")
    }
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(state_file)), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed"
    assert resumed["rounds_completed"] == 1
    assert resumed["learner_updates"] == 3
    assert not backend.leases and backend.closed
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    after = run_experiment(store).summary["candidate_store"]
    assert len(after["history"]) == 2
    assert after["explorer"]["model_sha256"] == interrupted["latest_learner"]["sha256"]
    assert resumed["interruptions"][0] == {
        "phase": "saving_versions",
        "stop_reason": "unclean_exit",
        "error": None,
    }


def test_unacknowledged_sampling_rejects_missing_or_inconsistent_child_evidence(
    tmp_path, seeded_loop
):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_learned")
    state = request.output_dir / "state.json"
    original_state = state.read_bytes()
    root = request.output_dir / "round-000/learning"
    summary_path = root / "summary.json"
    result_path = root / "attempt-000/cycle-result.json"
    summary = json.loads(summary_path.read_bytes())
    wrong_count = json.loads(summary_path.read_bytes())
    wrong_count["attempts"][0]["eligible_transitions"] = 2
    unreleased = dict(summary, resources_released=False)
    cases = (
        {summary_path: None},
        {summary_path: json.dumps(unreleased).encode()},
        {root / "protocol.json": b'{"source_kind": "synthetic", "seed": 192}'},
        {root / "attempt-000/trace.json": b"original action trace changed"},
        {root / "candidate-000/policy.pt": b"weights changed after child finished"},
        {
            summary_path: json.dumps(wrong_count).encode(),
            result_path: json.dumps(wrong_count["attempts"][0]).encode(),
        },
    )
    for changes in cases:
        originals = {path: path.read_bytes() for path in changes}
        backend = SharedBackend(seeded_loop[0])
        try:
            for path, raw in changes.items():
                if raw is None:
                    path.unlink()
                else:
                    path.write_bytes(raw)
            with pytest.raises((OSError, ValueError)):
                run_experiment(
                    LearningContinue(request.output_dir, sha(state)), learning_environment=backend
                )
            assert state.read_bytes() == original_state
            assert not backend.leases and backend.closed
            history = run_experiment(CandidateHistory(tmp_path / "versions")).summary[
                "candidate_store"
            ]
            assert len(history["history"]) == 1
            assert history["revision"] == seeded_loop[2]["revision"]
        finally:
            for path, raw in originals.items():
                path.write_bytes(raw)


def test_sampling_recovery_requires_raw_frame_and_review_attachment_inventory(
    tmp_path, seeded_loop
):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0], "before_learned")
    state = request.output_dir / "state.json"
    original_state = state.read_bytes()
    root = request.output_dir / "round-000/learning"
    summary_path = root / "summary.json"
    result_path = root / "attempt-000/cycle-result.json"
    original_summary, original_result = summary_path.read_bytes(), result_path.read_bytes()
    inventory = json.loads(original_summary)["attempts"][0]["source_assets"]
    frame = next(Path(name) for name in inventory if name.endswith(".rgb"))
    attachment = root / "attempt-000/independent-review.md"
    for omitted in (frame, attachment):
        original_asset = omitted.read_bytes()
        summary = json.loads(original_summary)
        del summary["attempts"][0]["source_assets"][str(omitted.resolve())]
        backend = SharedBackend(seeded_loop[0])
        try:
            summary_path.write_text(json.dumps(summary))
            result_path.write_text(json.dumps(summary["attempts"][0]))
            omitted.unlink()
            with pytest.raises(ValueError, match="inventory"):
                run_experiment(
                    LearningContinue(request.output_dir, sha(state)), learning_environment=backend
                )
            assert state.read_bytes() == original_state
            assert not backend.leases and backend.closed
        finally:
            omitted.write_bytes(original_asset)
            summary_path.write_bytes(original_summary)
            result_path.write_bytes(original_result)


def test_commit_recovery_rejects_changed_assets_and_another_store_successor(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)
    interrupt_selection(request, seeded_loop[0])
    state = request.output_dir / "state.json"
    original_state = state.read_bytes()
    store = tmp_path / "versions"
    history = run_experiment(CandidateHistory(store)).summary["candidate_store"]
    proposal = request.output_dir / "round-000/retain.json"
    archived_weights = store / history["explorer"]["archive"] / "checkpoint/policy.pt"
    for path in (proposal, archived_weights):
        original = path.read_bytes()
        path.write_bytes(original + b"changed externally")
        backend = SharedBackend(seeded_loop[0])
        try:
            with pytest.raises(ValueError):
                run_experiment(
                    LearningContinue(request.output_dir, sha(state)), learning_environment=backend
                )
            assert state.read_bytes() == original_state
            assert not backend.leases and backend.closed
        finally:
            path.write_bytes(original)
    # Even a further legitimate commit with the same candidate is not this pending write.
    other = run_experiment(
        CandidateRecord(proposal, store, history["revision"], seeded_loop[3])
    ).summary["candidate_store"]
    assert other["explorer"]["model_sha256"] == history["explorer"]["model_sha256"]
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises(ValueError, match="outside the pending candidate commit"):
        run_experiment(
            LearningContinue(request.output_dir, sha(state)), learning_environment=backend
        )
    assert state.read_bytes() == original_state
    assert not backend.leases and backend.closed
    assert (
        run_experiment(CandidateHistory(store)).summary["candidate_store"]["revision"]
        == other["revision"]
    )
