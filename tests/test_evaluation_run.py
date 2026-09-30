"""Repeated frozen execution through the experiment seam, without real devices."""

import json
import struct
import time
from dataclasses import replace

import pytest
from test_evaluation import policy as policy
from test_evaluation import prepare, sha
from test_evaluation_execution import PacketGame
from test_event_run import PATTERNS, verified_config

from fh5.evaluation import EvaluationPrepare
from fh5.events import EventInput, ScreenFrame
from fh5.experiment import run_experiment


def request(tmp_path, policy):
    from fh5.evaluation_run import EvaluationRun

    _, path = prepare(tmp_path, policy, ["no_reference"] * 2)
    options = json.loads(path.read_bytes())
    for key, value in options["conditions"]["snapshot"].items():
        value.update(status="verified", value=key, evidence="synthetic fixture")
    task_path = tmp_path / "task.json"
    task = json.loads(task_path.read_bytes())
    task["control_owner"] = "policy"
    task_path.write_text(json.dumps(task))
    options["task"]["sha256"] = sha(task_path)
    path.write_text(json.dumps(options))
    run_experiment(EvaluationPrepare(path, tmp_path / "batch"))
    menu = tmp_path / "menu"
    menu.mkdir()
    event = verified_config(menu)
    configuration = json.loads(event.read_bytes())
    configuration["snapshot"] = options["conditions"]["snapshot"]
    configuration["event_run"].update(
        expected_car_ordinal=2941, expected_pi=999, start_position_m=[0, 2, 0.2]
    )
    event.write_text(json.dumps(configuration))
    return EvaluationRun(
        tmp_path / "batch",
        sha(tmp_path / "batch/batch.json"),
        event,
        tmp_path / "run",
        seconds=0.35,
    )


class Menu:
    source_kind = "synthetic"

    def __init__(self, restart, fail=False):
        self.car = PacketGame()
        self.screen = "driving" if restart else "ready"
        self.pulses = []
        self.closed = False
        self.fail = fail

    def now_ns(self):
        return time.perf_counter_ns()

    def read(self, period_s):
        value = self.car.read(0.01)
        return EventInput(
            value.raw_packets,
            ScreenFrame(value.at_ns, 4, 2, bytes(8) if self.fail else PATTERNS[self.screen]),
            True,
        )

    def pulse(self, button):
        self.pulses.append(button)
        self.screen = {
            ("ready", "A"): "driving",
            ("driving", "START"): "pause",
            ("pause", "X"): "confirm",
            ("confirm", "A"): "ready",
        }[self.screen, button]

    def release(self):
        pass

    def close(self):
        self.closed = True


class Batch:
    source_kind = "synthetic"

    def __init__(self, fail_restart=False):
        self.menus = []
        self.drives = []
        self.fail_restart = fail_restart
        self.closed = False

    def event(self, slot_id):
        menu = Menu(bool(self.menus), bool(self.menus) and self.fail_restart)
        self.menus.append(menu)
        return menu

    def driving(self, slot_id, ready_state):
        assert self.menus[-1].closed
        game = PacketGame()
        self.drives.append(game)
        return game

    def close(self):
        self.closed = True
        return {"resources_released": all(g.closed for g in self.drives + self.menus)}


def test_frozen_policy_runs_twice_with_confirmed_restart_and_fresh_histories(tmp_path, policy):
    operation = request(tmp_path, policy)
    game = Batch()
    result = run_experiment(operation, evaluation_environment=game)
    summary = result.summary["evaluation_run"]
    assert summary["stop_reason"] == "plan_complete"
    assert summary["started_slots"] == ["run-0", "run-1"]
    assert summary["unstarted_slots"] == []
    assert game.menus[1].pulses == ["START", "X", "A", "A"]
    assert all(any(c.throttle_u8 for _, c in drive.sent) for drive in game.drives)
    for index in range(2):
        recorded = json.loads(
            (operation.output_dir / f"attempt-{index:04d}/execution/report.json").read_bytes()
        )
        first = next(d for d in recorded["decisions"] if "actor" in d)
        assert first["actor"]["actions"] == [None, None, None]
    assert result.summary["evaluation"]["metrics"]["all_attempts"] == 2
    assert result.summary["evaluation"]["execution_metrics"]["bound_runs"] == 2
    assert game.closed and summary["resources_released"]
    assert summary["commands_sent_to_game"] is False
    assert sha(operation.batch_dir / "model/actor.pt") == sha(
        operation.output_dir / "frozen/model/actor.pt"
    )


def test_unconfirmed_restart_keeps_first_attempt_and_does_not_open_second_driver(tmp_path, policy):
    operation = request(tmp_path, policy)
    game = Batch(fail_restart=True)
    result = run_experiment(operation, evaluation_environment=game)
    summary = result.summary["evaluation_run"]
    assert summary["stop_reason"] == "ready_unconfirmed"
    assert summary["started_slots"] == ["run-0"]
    assert summary["unstarted_slots"] == ["run-1"]
    assert len(game.drives) == 1
    assert result.summary["evaluation"]["metrics"]["all_attempts"] == 1
    assert game.closed


def test_parent_close_failure_is_reported_without_discarding_finished_attempts(tmp_path, policy):
    class FailedClose(Batch):
        def close(self):
            super().close()
            raise OSError("synthetic close failed")

    result = run_experiment(request(tmp_path, policy), evaluation_environment=FailedClose())
    assert result.report_path.is_file()
    assert result.summary["evaluation_run"]["resources_released"] is False
    assert result.summary["evaluation_run"]["stop_reason"] == "close_failed"
    assert result.summary["evaluation"]["metrics"]["all_attempts"] == 2


@pytest.mark.parametrize("change", ["position", "clock", "input"])
def test_changed_position_after_menu_confirmation_never_receives_a_policy_command(
    tmp_path, policy, change
):
    class WrongPosition(PacketGame):
        def read(self, period_s):
            value = super().read(period_s)
            raw = bytearray(value.raw_packets[0].payload)
            if change == "position":
                struct.pack_into("<f", raw, 244, 100)
            elif change == "clock":
                struct.pack_into("<I", raw, 4, self.ready_clock)
                value = replace(
                    value, safety=replace(value.safety, game_timestamp_ms=self.ready_clock)
                )
            else:
                raw[315] = 255
            return replace(value, raw_packets=(replace(value.raw_packets[0], payload=bytes(raw)),))

    class MovedBatch(Batch):
        def driving(self, slot_id, ready_state):
            drive = WrongPosition()
            drive.ready_clock = ready_state["game_timestamp_ms"]
            self.drives.append(drive)
            return drive

    game = MovedBatch()
    result = run_experiment(request(tmp_path, policy), evaluation_environment=game)
    assert not any(c.throttle_u8 or c.brake_u8 or c.steer_i16 for _, c in game.drives[0].sent)
    assert len(game.drives) == 1
    assert result.summary["evaluation_run"]["unstarted_slots"] == ["run-1"]
    assert result.summary["evaluation"]["metrics"]["all_attempts"] == 1


def test_changed_frozen_policy_stops_next_attempt_and_preserves_completed_run_record(
    tmp_path, policy
):
    operation = request(tmp_path, policy)

    class ChangedModel(PacketGame):
        def close(self):
            result = super().close()
            weights = operation.output_dir / "frozen/model/actor.pt"
            weights.write_bytes(b"changed after first attempt")
            return result

    class EditingBatch(Batch):
        def driving(self, slot_id, ready_state):
            drive = ChangedModel()
            self.drives.append(drive)
            return drive

    game = EditingBatch()
    result = run_experiment(operation, evaluation_environment=game)
    assert result.report_path.is_file()
    assert len(game.drives) == 1
    assert result.summary["evaluation_run"]["started_slots"] == ["run-0"]
    assert result.summary["evaluation_run"]["unstarted_slots"] == ["run-1"]
    assert result.summary["evaluation_run"]["review_error"]
    assert len(json.loads((operation.output_dir / "ledger.json").read_bytes())["entries"]) == 1


def test_event_recipe_cannot_change_between_frozen_attempts(tmp_path, policy):
    operation = request(tmp_path, policy)

    class ChangedRecipe(PacketGame):
        def close(self):
            result = super().close()
            path = operation.output_dir / "event.json"
            config = json.loads(path.read_bytes())
            config["event_run"]["restart_timeout_s"] = 1
            path.write_text(json.dumps(config))
            return result

    class EditingBatch(Batch):
        def driving(self, slot_id, ready_state):
            drive = ChangedRecipe()
            self.drives.append(drive)
            return drive

    game = EditingBatch()
    result = run_experiment(operation, evaluation_environment=game)
    assert len(game.drives) == 1
    assert result.summary["evaluation_run"]["stop_reason"] == "interface_error"
    assert result.summary["evaluation_run"]["unstarted_slots"] == ["run-1"]
    assert result.summary["evaluation"]["metrics"]["all_attempts"] == 1


def test_second_driver_open_failure_remains_an_interface_attempt_in_the_batch(tmp_path, policy):
    class FailedOpen(Batch):
        def driving(self, slot_id, ready_state):
            if self.drives:
                raise OSError("synthetic driver unavailable")
            return super().driving(slot_id, ready_state)

    result = run_experiment(request(tmp_path, policy), evaluation_environment=FailedOpen())
    summary = result.summary["evaluation_run"]
    assert summary["started_slots"] == ["run-0", "run-1"]
    assert len(summary["attempts"]) == 2
    assert summary["attempts"][1]["stop_reason"] == "interface_error"
    assert result.summary["evaluation"]["metrics"]["all_attempts"] == 2
    assert result.summary["evaluation"]["unresolved_recordings"] == 1


def test_valid_model_replacement_during_driver_open_is_rejected_before_commands(tmp_path, policy):
    import torch

    from fh5.numeric_actor import FrozenNumericActor
    from fh5.numeric_images import PixelContract

    operation = request(tmp_path, policy)

    class SwappingBatch(Batch):
        def driving(self, slot_id, ready_state):
            model = operation.output_dir / "frozen/model"
            saved = torch.load(model / "actor.pt", weights_only=True)
            next(iter(saved["actor"].values())).add_(0.01)
            torch.save(saved, model / "actor.pt")
            manifest = json.loads((model / "model.json").read_bytes())
            manifest["weights_sha256"] = sha(model / "actor.pt")
            (model / "model.json").write_text(json.dumps(manifest))
            # A valid alternative model must still be rejected by this frozen batch.
            FrozenNumericActor(model, PixelContract(size=(64, 36)))
            return super().driving(slot_id, ready_state)

    game = SwappingBatch()
    result = run_experiment(operation, evaluation_environment=game)
    assert len(game.drives) == 1
    assert not any(c.throttle_u8 or c.brake_u8 or c.steer_i16 for _, c in game.drives[0].sent)
    assert result.summary["evaluation_run"]["unstarted_slots"] == ["run-1"]
    assert game.closed and game.drives[0].closed


def test_driver_cannot_rewrite_ready_clock_to_accept_a_backward_handoff(tmp_path, policy):
    class BackwardClock(PacketGame):
        def read(self, period_s):
            value = super().read(period_s)
            stamp = (self.start_clock - 100 + len(self.packets)) % 2**32
            raw = bytearray(value.raw_packets[0].payload)
            struct.pack_into("<I", raw, 4, stamp)
            return replace(
                value,
                safety=replace(value.safety, game_timestamp_ms=stamp),
                raw_packets=(replace(value.raw_packets[0], payload=bytes(raw)),),
            )

    class RewritingBatch(Batch):
        def driving(self, slot_id, ready_state):
            drive = BackwardClock()
            drive.start_clock = ready_state["game_timestamp_ms"]
            ready_state["game_timestamp_ms"] = (drive.start_clock - 1000) % 2**32
            self.drives.append(drive)
            return drive

    operation = request(tmp_path, policy)
    game = RewritingBatch()
    result = run_experiment(operation, evaluation_environment=game)
    assert not any(c.throttle_u8 or c.brake_u8 or c.steer_i16 for _, c in game.drives[0].sent)
    summary = result.summary["evaluation_run"]
    assert (
        summary["preparations"][0]["ready_state"]["game_timestamp_ms"] == game.drives[0].start_clock
    )
    assert summary["unstarted_slots"] == ["run-1"]
    assert result.summary["evaluation"]["metrics"]["all_attempts"] == 1


def test_driver_cannot_move_frozen_start_position_to_accept_a_different_location(tmp_path, policy):
    operation = request(tmp_path, policy)

    class DifferentLocation(PacketGame):
        def read(self, period_s):
            value = super().read(period_s)
            raw = bytearray(value.raw_packets[0].payload)
            struct.pack_into("<f", raw, 244, 100)
            return replace(value, raw_packets=(replace(value.raw_packets[0], payload=bytes(raw)),))

    class ChangingStart(Batch):
        def driving(self, slot_id, ready_state):
            path = operation.output_dir / "event.json"
            config = json.loads(path.read_bytes())
            config["event_run"]["start_position_m"] = [100, 2, 0.2]
            path.write_text(json.dumps(config))
            drive = DifferentLocation()
            self.drives.append(drive)
            return drive

    game = ChangingStart()
    result = run_experiment(operation, evaluation_environment=game)
    assert not any(c.throttle_u8 or c.brake_u8 or c.steer_i16 for _, c in game.drives[0].sent)
    assert len(game.drives) == 1
    assert result.summary["evaluation_run"]["unstarted_slots"] == ["run-1"]


def test_menu_open_cannot_change_the_frozen_button_recipe(tmp_path, policy):
    operation = request(tmp_path, policy)

    class AcceptingMenu(Menu):
        def pulse(self, button):
            if button == "B":
                self.pulses.append(button)
                self.screen = "driving"
            else:
                super().pulse(button)

    class ChangingRecipe(Batch):
        def event(self, slot_id):
            path = operation.output_dir / "event.json"
            config = json.loads(path.read_bytes())
            config["event_run"]["start_steps"][0]["button"] = "B"
            path.write_text(json.dumps(config))
            menu = AcceptingMenu(False)
            self.menus.append(menu)
            return menu

    game = ChangingRecipe()
    result = run_experiment(operation, evaluation_environment=game)
    assert game.menus[0].pulses == []
    assert game.menus[0].closed and not game.drives
    assert result.summary["evaluation_run"]["started_slots"] == []
    assert result.summary["evaluation_run"]["unstarted_slots"] == ["run-0", "run-1"]
