"""Unattended learning through the experiment-run seam and synthetic external I/O."""

import json
import shutil
from dataclasses import replace

import pytest
from learning_files import stage_records
from test_candidate_store import candidates as candidates
from test_candidate_store import record_config
from test_evaluation import sha
from test_sac_cycle import ResponsiveEnvironment
from test_sac_imitation_evaluation import SteeringDragBatch

from fh5.candidate_store import CandidateHistory, CandidateRecord
from fh5.experiment import run_experiment


@pytest.fixture(scope="module")
def seeded_loop(tmp_path_factory, candidates):
    root = tmp_path_factory.mktemp("loop-seed")
    store = root / "versions"
    saved = run_experiment(
        CandidateRecord(
            record_config(root, candidates, missing_execution=True), store, None, candidates[3]
        )
    ).summary["candidate_store"]
    return candidates[0], store, saved, candidates[3]


def loop_request(tmp_path, setup, **changes):
    from fh5.learning_loop import LearningLoop

    source, store, saved, registry = setup
    shutil.copytree(store, tmp_path / "versions")
    config = {
        "version": 1,
        "store": {"directory": str(tmp_path / "versions"), "revision": saved["revision"]},
        "registry": str(registry),
        "recording": str(source / "record.json"),
        "task": str(source / "task.json"),
        "reward": str(source / "reward.json"),
        "rounds": 2,
        "steps_per_attempt": 3,
        "evaluation_seconds": 1.0,
        "seed": 191,
        **changes,
    }
    path = tmp_path / "loop-config.json"
    path.write_text(json.dumps(config))
    return LearningLoop(path, tmp_path / "loop")


class SharedBackend:
    source_kind = "synthetic"

    def __init__(self, source):
        self.source = source
        self.active = None
        self.leases = []
        self.closed = False
        self.stop_sampling_file = None

    def sampling(self, identity):
        assert self.active is None
        owner = self

        class SamplingLease(ResponsiveEnvironment):
            def close(self):
                result = super().close()
                owner.active = None
                return result

            def sample(self):
                sample = super().sample()
                if self.index == 1 and owner.stop_sampling_file is not None:
                    owner.stop_sampling_file.write_text("operator stop")
                return replace(
                    sample,
                    packet=replace(
                        sample.packet,
                        received_utc=f"2026-10-01T00:00:{len(owner.leases):02d}+00:00",
                    ),
                )

        lease = SamplingLease(self.source)
        self.active = lease
        self.leases.append(lease)
        return lease

    def evaluation(self, identity):
        assert self.active is None
        owner = self

        class EvaluationLease(SteeringDragBatch):
            def event(self, slot_id):
                menu = super().event(slot_id)
                menu.screen = "driving"
                return menu

            def close(self):
                result = super().close()
                owner.active = None
                return result

        lease = EvaluationLease()
        self.active = lease
        self.leases.append(lease)
        return lease

    def review(self, recording):
        # Unknown road legality must retain the reliable default, not stop learning.
        return None

    def close(self):
        self.closed = True
        return {"resources_released": self.active is None and all(x.closed for x in self.leases)}


def test_two_rounds_keep_default_and_continue_rejected_explorer(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop)
    backend = SharedBackend(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "budget_completed"
    assert result["rounds_completed"] == 2
    assert result["eligible_transitions"] == 6
    assert result["learner_updates"] == 6
    assert result["latest_learner"]["total_steps"] == 36
    first, second = result["rounds"]
    assert second["sampling_checkpoint_sha256"] == first["candidate_sha256"]
    assert first["candidate_sha256"] != second["candidate_sha256"]
    assert all(r["selection"] == "retain_incumbent" for r in result["rounds"])
    assert all(r["evaluation"]["metrics"]["all_attempts"] == 2 for r in result["rounds"])
    assert all(r["evaluation"]["execution_metrics"]["bound_runs"] == 2 for r in result["rounds"])
    history = run_experiment(CandidateHistory(tmp_path / "versions")).summary["candidate_store"]
    assert len(history["history"]) == 3
    assert history["default"]["model_sha256"] == seeded_loop[2]["default"]["model_sha256"]
    assert history["explorer"]["model_sha256"] == second["candidate_sha256"]
    assert result["store_revision"] == history["revision"]
    assert all(lease.closed for lease in backend.leases) and backend.closed
    assert result["resources_released"] is True
    assert result["real_driving_validated"] is False
    assert result["driving_improvement_validated"] is False
    assert (
        sha(seeded_loop[0] / "candidate-model/policy.json")
        == seeded_loop[2]["explorer"]["model_sha256"]
    )


def test_stop_during_sampling_closes_the_lease_without_updating_or_evaluating(
    tmp_path, seeded_loop
):
    request = loop_request(tmp_path, seeded_loop)
    backend = SharedBackend(seeded_loop[0])
    backend.stop_sampling_file = request.output_dir / "stop.request"
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "stop_requested"
    assert result["learner_updates"] == 0
    assert result["latest_learner"]["total_steps"] == 30
    assert result["store_revision"] == seeded_loop[2]["revision"]
    assert len(backend.leases) == 1
    assert len(backend.leases[0].commands) == 1
    assert backend.leases[0].closed and result["resources_released"]
    assert "driving" in [item["phase"] for item in stage_records(request.output_dir, result)]


def test_resume_evaluates_the_saved_learner_without_sampling_or_updating_it_twice(
    tmp_path, seeded_loop, monkeypatch
):
    from pathlib import Path

    request = loop_request(tmp_path, seeded_loop)

    class UnavailableEvaluation(SharedBackend):
        def evaluation(self, identity):
            raise OSError("synthetic evaluation receiver unavailable")

    backend = UnavailableEvaluation(seeded_loop[0])
    original_open = Path.open

    def unavailable_display(path, mode="r", *args, **kwargs):
        if (
            path.parent == request.output_dir
            and path.name in ("summary.tmp", "summary.json", "report.html")
            and any(flag in mode for flag in "wax")
        ):
            assert backend.closed
            assert (request.output_dir / "state.json").is_file()
            raise OSError("loop display unavailable")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", unavailable_display)
        stopped = run_experiment(request, learning_environment=backend)
    failed = stopped.summary["learning_loop"]
    assert stopped.report_path == request.output_dir / "state.json"
    assert failed["presentation"]["status"] == "unavailable"
    assert json.loads(stopped.report_path.read_bytes()) == {
        key: value for key, value in failed.items() if key != "presentation"
    }
    assert not (request.output_dir / "summary.json").exists()
    assert failed["stop_reason"] == "interface_error"
    assert failed["latest_learner"]["total_steps"] == 33
    assert failed["explorer"]["total_steps"] == 30
    assert failed["store_revision"] == seeded_loop[2]["revision"]
    saved = failed["latest_learner"]["sha256"]
    from fh5.learning_loop import LearningContinue

    # Older runs may still contain a display copy. It is neither recovery input
    # nor a publication target; retain its original bytes during continuation.
    historical_summary = request.output_dir / "summary.json"
    historical_summary.write_bytes(b'{"historical_display": true}\n')
    backend = SharedBackend(seeded_loop[0])
    continued = run_experiment(
        LearningContinue(request.output_dir),
        learning_environment=backend,
    )
    result = continued.summary["learning_loop"]
    assert continued.report_path == request.output_dir / "report.html"
    assert continued.report_path.is_file()
    assert historical_summary.read_bytes() == b'{"historical_display": true}\n'
    assert json.loads((request.output_dir / "state.json").read_bytes()) == result
    assert result["stop_reason"] == "budget_completed"
    assert result["rounds_completed"] == 2
    assert result["learner_updates"] == 6
    assert result["latest_learner"]["total_steps"] == 36
    assert result["rounds"][0]["candidate_sha256"] == saved
    assert result["rounds"][1]["sampling_checkpoint_sha256"] == saved
    assert sum(isinstance(lease, ResponsiveEnvironment) for lease in backend.leases) == 1
    assert result["interruptions"][0]["stop_reason"] == "interface_error"
    assert result["resources_released"] and backend.closed


def test_stale_continuation_cannot_rewrite_the_saved_session(tmp_path, seeded_loop):
    from fh5.learning_loop import LearningContinue

    request = loop_request(tmp_path, seeded_loop, rounds=1)

    class UnavailableSampling(SharedBackend):
        def sampling(self, identity):
            raise OSError("synthetic sampling receiver unavailable")

    failed = run_experiment(
        request, learning_environment=UnavailableSampling(seeded_loop[0])
    ).summary["learning_loop"]
    assert failed["stop_reason"] == "interface_error"
    state_file = request.output_dir / "state.json"
    before = state_file.read_bytes()
    backend = SharedBackend(seeded_loop[0])
    with pytest.raises(ValueError, match="state changed"):
        run_experiment(LearningContinue(request.output_dir, "0" * 64), learning_environment=backend)
    assert state_file.read_bytes() == before
    assert not backend.leases
    assert backend.closed


def test_evaluation_restarts_from_the_state_left_by_sampling(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)

    class StillDriving(SharedBackend):
        def evaluation(self, identity):
            lease = super().evaluation(identity)
            original_event = lease.event

            def event(slot):
                menu = original_event(slot)
                menu.screen = "driving"
                return menu

            lease.event = event
            return lease

    backend = StillDriving(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["stop_reason"] == "budget_completed"
    assert result["rounds"][0]["evaluation"]["verified_starts"] == 2
    assert backend.leases[1].menus[0].pulses == ["START", "X", "A", "A"]
    assert result["rounds"][0]["evaluation"]["starts"][0]["operation"] == "restart_ready"


def test_stop_during_evaluation_releases_and_resume_keeps_the_interrupted_attempt(
    tmp_path, seeded_loop
):
    from fh5.control import Command
    from fh5.learning_loop import LearningContinue

    request = loop_request(tmp_path, seeded_loop)

    class StopAfterFirstCommand(SharedBackend):
        def evaluation(self, identity):
            lease = super().evaluation(identity)
            original_driving = lease.driving

            def driving(slot, ready):
                game = original_driving(slot, ready)
                original_send = game.send

                def send(command):
                    original_send(command)
                    if command != Command(0, 0, 0):
                        (request.output_dir / "stop.request").write_text("external stop")

                game.send = send
                return game

            lease.driving = driving
            return lease

    backend = StopAfterFirstCommand(seeded_loop[0])
    stopped = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert stopped["stop_reason"] == "stop_requested"
    assert stopped["latest_learner"]["total_steps"] == 33
    evaluation = backend.leases[1]
    assert (
        sum(command != Command(0, 0, 0) for game in evaluation.drives for _, command in game.sent)
        == 1
    )
    assert len(evaluation.menus) == 1
    assert evaluation.drives[0].sent[-1][1] == Command(0, 0, 0)
    assert stopped["resources_released"]
    ledger = request.output_dir / "round-000/evaluation/ledger.json"
    original = ledger.read_bytes()
    (request.output_dir / "stop.request").unlink()
    resumed = run_experiment(
        LearningContinue(request.output_dir, sha(request.output_dir / "state.json")),
        learning_environment=SharedBackend(seeded_loop[0]),
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed"
    assert resumed["rounds_completed"] == 2
    assert resumed["learner_updates"] == 6
    assert ledger.read_bytes() == original
    assert resumed["rounds"][0]["evaluation"]["metrics"]["all_attempts"] == 1
    assert resumed["rounds"][0]["evaluation"]["unstarted_slots"] == ["run-1"]
    assert resumed["rounds"][0]["selection"] == "retain_incumbent"


def test_model_replaced_while_opening_sampler_never_receives_a_command(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)

    class ReplacedCheckpoint(SharedBackend):
        def sampling(self, identity):
            lease = super().sampling(identity)
            shutil.copytree(
                seeded_loop[0] / "incumbent-model",
                request.output_dir / "initial/explorer",
                dirs_exist_ok=True,
            )
            return lease

    backend = ReplacedCheckpoint(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert not backend.leases[0].commands
    assert result["learner_updates"] == 0
    assert result["store_revision"] == seeded_loop[2]["revision"]
    assert result["stop_reason"] == "sampling_interface_error"
    assert result["resources_released"]


def test_unconfirmed_restart_retains_the_candidate_and_stops_further_sampling(
    tmp_path, seeded_loop
):
    request = loop_request(tmp_path, seeded_loop)

    class UnconfirmedRestart(SharedBackend):
        sample_requests = 0

        def sampling(self, identity):
            self.sample_requests += 1
            if self.sample_requests > 1:
                raise OSError("unexpected further sample request")
            return super().sampling(identity)

        def evaluation(self, identity):
            lease = super().evaluation(identity)
            original_event = lease.event

            def event(slot):
                menu = original_event(slot)
                menu.fail = True
                return menu

            lease.event = event
            return lease

    backend = UnconfirmedRestart(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert backend.sample_requests == 1
    assert result["stop_reason"] == "evaluation_ready_unconfirmed"
    assert result["latest_learner"]["total_steps"] == result["explorer"]["total_steps"] == 33
    assert result["rounds"][0]["evaluation"]["unstarted_slots"] == ["run-0", "run-1"]
    assert result["rounds"][0]["selection"] == "retain_incumbent"
    assert result["resources_released"]


def test_menu_release_failure_survives_successful_backend_close(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)

    class FailedMenuRelease(SharedBackend):
        def evaluation(self, identity):
            lease = super().evaluation(identity)
            original_event = lease.event

            def event(slot):
                menu = original_event(slot)

                def release():
                    raise OSError("synthetic menu release failed")

                menu.release = release
                return menu

            lease.event = event
            return lease

    backend = FailedMenuRelease(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["environment"]["resources_released"] is True
    assert result["resources_released"] is False
    assert result["child_resources_released"] is False
    assert result["stop_reason"] == "release_fault"
    assert result["learner_updates"] == 3
    assert result["store_revision"] == seeded_loop[2]["revision"]
    assert not backend.leases[1].drives


def test_menu_close_exception_survives_successful_backend_close(tmp_path, seeded_loop):
    request = loop_request(tmp_path, seeded_loop, rounds=1)

    class FailedMenuClose(SharedBackend):
        def evaluation(self, identity):
            lease = super().evaluation(identity)
            original_event = lease.event
            original_close = lease.close

            def close_backend():
                original_close()
                return {"resources_released": True}

            def event(slot):
                menu = original_event(slot)

                def close():
                    raise OSError("synthetic menu close failed")

                menu.close = close
                return menu

            lease.event = event
            lease.close = close_backend
            return lease

    backend = FailedMenuClose(seeded_loop[0])
    result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
    assert result["environment"]["resources_released"] is True
    assert result["resources_released"] is False
    assert result["child_resources_released"] is False
    assert result["stop_reason"] == "release_fault"
    assert result["store_revision"] == seeded_loop[2]["revision"]
    assert not backend.leases[1].drives


def test_rejected_child_cannot_hide_failed_cleanup(tmp_path, seeded_loop):
    for kind in ("sampling", "evaluation"):
        folder = tmp_path / kind
        folder.mkdir()
        request = loop_request(folder, seeded_loop, rounds=1)

        class RejectedChild(SharedBackend):
            def reject(self, lease):
                lease.source_kind = "unsupported"
                original_close = lease.close

                def close():
                    original_close()
                    return {"resources_released": False}

                lease.close = close
                return lease

            def sampling(self, identity):
                lease = super().sampling(identity)
                return self.reject(lease) if kind == "sampling" else lease

            def evaluation(self, identity):
                return self.reject(super().evaluation(identity))

        backend = RejectedChild(seeded_loop[0])
        result = run_experiment(request, learning_environment=backend).summary["learning_loop"]
        assert result["environment"]["resources_released"] is True
        assert result["resources_released"] is False
        assert result["child_resources_released"] is False
        assert result["stop_reason"] == "release_fault"
        assert result["store_revision"] == seeded_loop[2]["revision"]
        if kind == "sampling":
            assert not backend.leases[-1].commands
        else:
            assert not backend.leases[-1].menus


def test_continuation_verifies_sampling_originals_and_external_review(tmp_path, seeded_loop):
    from fh5.learning_loop import LearningContinue

    request = loop_request(tmp_path, seeded_loop, rounds=1)
    proof_dir = tmp_path / "observer"
    proof_dir.mkdir()

    class ExternalReview(SharedBackend):
        def sampling(self, identity):
            lease = super().sampling(identity)
            original_finish = lease.finish

            def finish(recording):
                proof = original_finish(recording)
                for name in (proof.name, "independent-review.md"):
                    shutil.move(proof.parent / name, proof_dir / name)
                return proof_dir / proof.name

            lease.finish = finish
            return lease

    result = run_experiment(request, learning_environment=ExternalReview(seeded_loop[0])).summary[
        "learning_loop"
    ]
    assert result["stop_reason"] == "budget_completed"
    attempt = request.output_dir / "round-000/learning/attempt-000"
    state = request.output_dir / "state.json"
    original_state = state.read_bytes()
    trace = json.loads((attempt / "trace.json").read_bytes())
    paths = [
        attempt / "trace.json",
        attempt / "recording/packets.jsonl",
        attempt / "recording/session.json",
        attempt / trace["observations"][0]["frames"][0]["path"],
        proof_dir / "evidence.json",
        proof_dir / "independent-review.md",
    ]
    for path in paths:
        raw = path.read_bytes()
        path.write_bytes(b'{"corrupted":true}')
        backend = SharedBackend(seeded_loop[0])
        try:
            with pytest.raises(ValueError, match="sampling.*changed"):
                run_experiment(LearningContinue(request.output_dir), learning_environment=backend)
            assert state.read_bytes() == original_state
            assert not backend.leases and backend.closed
        finally:
            path.write_bytes(raw)
    backend = SharedBackend(seeded_loop[0])
    resumed = run_experiment(
        LearningContinue(request.output_dir), learning_environment=backend
    ).summary["learning_loop"]
    assert resumed["stop_reason"] == "budget_completed"
    assert resumed["learner_updates"] == 3
    assert not backend.leases and backend.closed
