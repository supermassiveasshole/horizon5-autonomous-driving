"""Trusted demonstration behavior at the experiment-run seam."""

import hashlib
import json
from pathlib import Path

import pytest
from test_experiment import config_file
from test_observations import motion_packet
from test_vision import Observations, frame

from fh5.experiment import run_experiment
from fh5.vision import VisionInput, VisionRecord


def profile_file(tmp_path):
    path = tmp_path / "input-profile.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "mapping": "xinput-lx-rt-lt-v1",
                "device": {
                    "api": "xinput1_4",
                    "index": 0,
                    "capabilities": {"type": 1, "subtype": 1, "flags": 0},
                },
                "calibration": {"status": "unverified", "evidence": []},
            }
        )
    )
    return path


def raw_input(ms=1050, lx=-32768, rt=128, lt=0, buttons=0, **extra):
    return {
        "kind": "human_input",
        "observed_ns": ms * 1_000_000,
        "available_ns": (ms + 1) * 1_000_000,
        "device_index": 0,
        "connected": True,
        "focused": True,
        "other_input": False,
        "raw": {
            "packet_number": 1,
            "buttons": buttons,
            "left_trigger": lt,
            "right_trigger": rt,
            "thumb_lx": lx,
            "thumb_ly": 0,
            "thumb_rx": 0,
            "thumb_ry": 0,
        },
        **extra,
    }


def test_demo_records_raw_and_normalized_inputs_without_claiming_game_adoption(tmp_path: Path):
    from fh5.demonstrations import DemonstrationRecord

    directory = tmp_path / "demo"
    profile = profile_file(tmp_path)
    environment = Observations(
        [
            VisionInput(
                packets=(motion_packet(990), motion_packet(1090)),
                frame=frame(),
                events=(raw_input(),),
            )
        ]
    )
    result = run_experiment(
        DemonstrationRecord(VisionRecord(config_file(tmp_path), directory), profile),
        vision_environment=environment,
    )
    demo = result.summary["demonstration"]
    row = demo["inputs"][0]
    assert row["raw"]["thumb_lx"] == -32768
    assert row["mapped"] == [-1.0, 128 / 255]
    assert row["poll_ns"] == 1_050_000_000
    assert row["available_ns"] == 1_051_000_000
    assert row["game_adoption"] == "unverified"
    assert row["mapping_valid"]
    assert demo["profile"]["device"]["index"] == 0
    assert demo["commands_sent"] is False
    assert not demo["calibrated"]
    assert environment.closed


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"lt": 20}, "simultaneous_pedals"),
        ({"buttons": 0x1000}, "button_action"),
        ({"connected": False}, "disconnected"),
        ({"focused": False}, "focus_lost"),
        ({"other_input": True}, "other_input"),
        ({"device_index": 1}, "device_mismatch"),
        ({"lx": 50000}, "invalid_raw_state"),
    ],
)
def test_unrepresentable_or_unowned_input_is_preserved_but_not_a_dual_axis_label(
    tmp_path, change, reason
):
    from fh5.demonstrations import DemonstrationRecord

    row = raw_input(**change)
    result = run_experiment(
        DemonstrationRecord(
            VisionRecord(config_file(tmp_path), tmp_path / "demo"), profile_file(tmp_path)
        ),
        vision_environment=Observations(
            [
                VisionInput(
                    packets=(motion_packet(990), motion_packet(1090)), frame=frame(), events=(row,)
                )
            ]
        ),
    )
    saved = result.summary["demonstration"]["inputs"][0]
    assert saved["raw"] == row["raw"]
    assert saved["mapped"] is None
    assert not saved["mapping_valid"]
    assert reason in saved["reasons"]


def test_passive_input_adapter_marks_control_changes_and_closes_reader(tmp_path):
    from fh5.demonstrations import DemonstrationRecord
    from fh5.live_demonstration import HumanInputEnvironment

    class Reader:
        closed = False

        def read(self):
            return raw_input(other_input=True)

        def close(self):
            self.closed = True

    reader = Reader()
    profile = profile_file(tmp_path)
    base = Observations(
        [VisionInput(packets=(motion_packet(990), motion_packet(1090)), frame=frame())]
    )
    result = run_experiment(
        DemonstrationRecord(VisionRecord(config_file(tmp_path), tmp_path / "demo"), profile),
        vision_environment=HumanInputEnvironment(base, reader, profile),
    )
    assert reader.closed and base.closed
    assert any(e["kind"] == "input_boundary" for e in result.summary["vision"]["events"])
    assert result.summary["demonstration"]["inputs"][0]["mapped"] is None


def test_demo_replay_binds_raw_input_and_profile_and_builds_only_prior_action_history(tmp_path):
    from dataclasses import replace

    from test_navigation_observations import navigation_fixture

    from fh5.demonstrations import DemonstrationRecord, DemonstrationReplay

    setup = navigation_fixture(tmp_path)
    directory = tmp_path / "demo"
    result = run_experiment(
        DemonstrationRecord(
            VisionRecord(config_file(tmp_path), directory, observation_config=setup.config_file),
            profile_file(tmp_path),
        ),
        vision_environment=Observations(
            [
                VisionInput(
                    packets=(motion_packet(990), motion_packet(1090)),
                    frame=frame(),
                    events=(raw_input(),),
                ),
            ]
        ),
    )
    replay = run_experiment(DemonstrationReplay(directory, directory / "again.html"))
    assert replay.summary["demonstration"] == result.summary["demonstration"]
    current = replay.summary["observations"]["decisions"][0]
    assert current["actor"]["actions"][-1] == [-1, 128 / 255]
    assert current["actor"]["action_age_ms"][-1] == 50
    assert not replay.summary["demonstration"]["integrity_errors"]
    (directory / "input-profile.json").write_text("{}")
    with pytest.raises(ValueError, match="Demonstration integrity"):
        run_experiment(replace(DemonstrationReplay(directory, directory / "bad.html")))


def dataset_fixture(tmp_path, boundary=False):
    from test_navigation_observations import navigation_fixture

    from fh5.demonstrations import DemonstrationRecord

    setup = navigation_fixture(tmp_path, max_telemetry_age_ms=200)
    profile = profile_file(tmp_path)
    value = json.loads(profile.read_text())
    value["calibration"] = {
        "status": "verified",
        "evidence": ["Synthetic mapping only; not live evidence"],
    }
    profile.write_text(json.dumps(value))
    entries = []
    for split, shift in [("train", 0), ("holdout", 10000)]:
        directory = tmp_path / split
        env = Observations(
            [
                VisionInput(
                    packets=(motion_packet(shift + 990 + i * 200, (0, 0, i * 2)),),
                    frame=frame(
                        shift + 1000 + i * 200, shift + 1020 + i * 200, shift + 1030 + i * 200
                    ),
                    events=(raw_input(shift + 1050 + i * 200, lx=-32768 if i % 2 == 0 else 32767),)
                    + (
                        (
                            {
                                "kind": "input_boundary",
                                "observed_ns": (shift + 1060 + i * 200) * 1_000_000,
                            },
                        )
                        if boundary and i == 0
                        else ()
                    ),
                )
                for i in range(8)
            ]
        )
        env.clock += shift * 1_000_000
        run_experiment(
            DemonstrationRecord(
                VisionRecord(
                    config_file(tmp_path),
                    directory,
                    observation_config=setup.config_file,
                    route_file=setup.route_file,
                ),
                profile,
            ),
            vision_environment=env,
        )
        review = tmp_path / f"{split}-review.json"
        review.write_text(
            json.dumps(
                {
                    "version": 1,
                    "packets_sha256": hashlib.sha256(
                        (directory / "packets.jsonl").read_bytes()
                    ).hexdigest(),
                    "vision_sha256": hashlib.sha256(
                        (directory / "vision.jsonl").read_bytes()
                    ).hexdigest(),
                    "intervals": [
                        {
                            "start_ns": (shift + 900) * 1_000_000,
                            "end_ns": (shift + 2600) * 1_000_000,
                            "quality": "trusted",
                            "evidence": "Synthetic expected action sequence",
                        }
                    ],
                    "intent_changes": [],
                    "notes": "Synthetic navigation conditions, not live validation",
                }
            )
        )
        entries.append({"directory": str(directory), "split": split, "review": str(review)})
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "max_label_delay_ms": 200,
                "future_offsets_ms": [200, 600],
                "runs": entries,
            }
        )
    )
    return manifest


def test_dataset_splits_whole_runs_and_keeps_future_targets_out_of_both_actor_views(tmp_path):
    from fh5.demonstration_dataset import DemonstrationDataset

    request = DemonstrationDataset(dataset_fixture(tmp_path), tmp_path / "dataset")
    result = run_experiment(request)
    dataset = result.summary["demonstration_dataset"]
    examples = dataset["examples"]
    assert {e["split"] for e in examples if e["bc_eligible"]} == {"train", "holdout"}
    first = next(e for e in examples if e["split"] == "train" and e["bc_eligible"])
    assert first["supervision"]["action"] == [1, 128 / 255]
    assert first["supervision"]["label_delay_ms"] == 150
    assert first["supervision"]["future_waypoints_m"][0] == pytest.approx([0, 3.1])
    assert first["views"]["no_reference"]["reference"]["mask"] == [False] * 3
    assert first["views"]["reference_assisted"]["reference"]["mask"] == [True] * 3
    assert first["views"]["no_reference"]["actions"][-1] == [-1, 128 / 255]
    last = [e for e in examples if e["split"] == "train"][-1]
    assert last["supervision"]["future_mask"] == [False, False]
    assert all("supervision" not in e["views"]["no_reference"] for e in examples)


@pytest.mark.parametrize("problem", ["duplicate", "review_hash", "overlap", "conditions"])
def test_dataset_rejects_untrustworthy_splits_and_reviews(tmp_path, problem):
    from fh5.demonstration_dataset import DemonstrationDataset

    manifest = dataset_fixture(tmp_path)
    config = json.loads(manifest.read_text())
    if problem == "duplicate":
        config["runs"][1]["directory"] = config["runs"][0]["directory"]
        config["runs"][1]["review"] = config["runs"][0]["review"]
    elif problem == "conditions":
        path = tmp_path / "holdout/session.json"
        metadata = json.loads(path.read_text())
        metadata["snapshot"]["tune"]["value"] = "different tune"
        path.write_text(json.dumps(metadata))
    else:
        path = tmp_path / "train-review.json"
        review = json.loads(path.read_text())
        if problem == "review_hash":
            review["vision_sha256"] = "0" * 64
        else:
            review["intervals"].append(dict(review["intervals"][0], quality="failed"))
        path.write_text(json.dumps(review))
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        run_experiment(DemonstrationDataset(manifest, tmp_path / "dataset"))


def test_failed_interval_and_intent_change_cut_supervision(tmp_path):
    from fh5.demonstration_dataset import DemonstrationDataset

    manifest = dataset_fixture(tmp_path)
    review_file = tmp_path / "train-review.json"
    review = json.loads(review_file.read_text())
    review["intervals"] = [
        {
            "start_ns": 900_000_000,
            "end_ns": 1_400_000_000,
            "quality": "trusted",
            "evidence": "synthetic",
        },
        {
            "start_ns": 1_400_000_000,
            "end_ns": 2_600_000_000,
            "quality": "failed",
            "evidence": "synthetic offroad",
        },
    ]
    review["intent_changes"] = [
        {"observed_ns": 1_200_000_000, "evidence": "synthetic destination change"}
    ]
    review_file.write_text(json.dumps(review))
    result = run_experiment(DemonstrationDataset(manifest, tmp_path / "dataset"))
    examples = [
        e for e in result.summary["demonstration_dataset"]["examples"] if e["split"] == "train"
    ]
    assert not any(e["supervision"]["future_mask"][1] for e in examples)
    assert not examples[0]["bc_eligible"]  # label crosses intent change
    assert "label_discontinuity" in examples[0]["supervision"]["reasons"]
    assert not any(e["bc_eligible"] for e in examples if e["decision_ns"] >= 1_300_000_000)


def test_late_input_delivery_is_rejected_and_environment_closed(tmp_path):
    from fh5.demonstrations import DemonstrationRecord

    env = Observations(
        [
            VisionInput(
                packets=(motion_packet(990),),
                frame=frame(),
                events=(raw_input(available_ns=1_500_000_000),),
            )
        ]
    )
    with pytest.raises(ValueError, match="Input clock"):
        run_experiment(
            DemonstrationRecord(
                VisionRecord(config_file(tmp_path), tmp_path / "demo"), profile_file(tmp_path)
            ),
            vision_environment=env,
        )
    assert env.closed


def test_invalid_input_cuts_previous_action_history_even_with_injected_environment(tmp_path):
    from test_navigation_observations import navigation_fixture

    from fh5.demonstrations import DemonstrationRecord

    setup = navigation_fixture(tmp_path, max_telemetry_age_ms=200)
    result = run_experiment(
        DemonstrationRecord(
            VisionRecord(
                config_file(tmp_path), tmp_path / "demo", observation_config=setup.config_file
            ),
            profile_file(tmp_path),
        ),
        vision_environment=Observations(
            [
                VisionInput(packets=(motion_packet(990),), frame=frame(), events=(raw_input(),)),
                VisionInput(
                    packets=(motion_packet(1190),),
                    frame=frame(1200, 1220, 1230),
                    events=(raw_input(1250, other_input=True),),
                ),
            ]
        ),
    )
    decision = result.summary["observations"]["decisions"][-1]
    assert not any(decision["actor"]["action_mask"])


def test_future_targets_do_not_bridge_old_origin_telemetry(tmp_path):
    from fh5.demonstration_dataset import DemonstrationDataset

    result = run_experiment(
        DemonstrationDataset(dataset_fixture(tmp_path, boundary=True), tmp_path / "dataset")
    )
    first = result.summary["demonstration_dataset"]["examples"][0]
    assert first["supervision"]["future_mask"] == [False, False]


def test_reference_cannot_be_sourced_from_another_held_out_recording(tmp_path):
    from fh5.demonstration_dataset import DemonstrationDataset

    manifest = dataset_fixture(tmp_path)
    # Construct a hash-consistent training recording whose reference source is the holdout.
    route = tmp_path / "train/observation-route/route.json"
    value = json.loads(route.read_text())
    value["source"]["packets_sha256"] = hashlib.sha256(
        (tmp_path / "holdout/packets.jsonl").read_bytes()
    ).hexdigest()
    route.write_text(json.dumps(value))
    bound = tmp_path / "train/demonstration-session.json"
    session = json.loads(bound.read_text())
    session["hashes"]["observation-route/route.json"] = hashlib.sha256(
        route.read_bytes()
    ).hexdigest()
    bound.write_text(json.dumps(session))
    with pytest.raises(ValueError, match="reference.*demonstration"):
        run_experiment(DemonstrationDataset(manifest, tmp_path / "dataset"))


@pytest.mark.parametrize("break_kind", ["intent", "review"])
def test_future_origin_cannot_precede_trusted_review_or_intent(tmp_path, break_kind):
    from fh5.demonstration_dataset import DemonstrationDataset

    manifest = dataset_fixture(tmp_path)
    path = tmp_path / "train-review.json"
    review = json.loads(path.read_text())
    if break_kind == "intent":
        review["intent_changes"] = [{"observed_ns": 1_060_000_000, "evidence": "new intent"}]
    else:
        review["intervals"][0]["start_ns"] = 1_060_000_000
    path.write_text(json.dumps(review))
    result = run_experiment(DemonstrationDataset(manifest, tmp_path / "dataset"))
    assert result.summary["demonstration_dataset"]["examples"][0]["supervision"]["future_mask"] == [
        False,
        False,
    ]
