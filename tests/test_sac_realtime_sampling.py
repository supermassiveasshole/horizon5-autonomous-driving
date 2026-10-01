"""Bounded stochastic SAC execution through the public experiment boundary."""

import json

from test_evaluation import sha
from test_evaluation_execution import PacketGame
from test_sac_evaluation import sac_policy as sac_policy

from fh5.experiment import run_experiment
from fh5.numeric_images import PixelContract
from fh5.realtime import RealtimeConfig, RealtimeNumericReplay, RealtimeRun


def test_frozen_exploration_replays_after_worker_warmup_without_advancing_a_random_stream(
    tmp_path, sac_policy
):
    from fh5.sac_sampling_actor import SACSamplingActor

    pixels = PixelContract(size=(64, 36))
    before = (sac_policy / "policy.pt").read_bytes()

    def actor(seed=37):
        return SACSamplingActor(
            sac_policy, pixels, sha(sac_policy / "policy.json"), exploration_seed=seed
        )

    game = PacketGame()
    root = tmp_path / "sampling"
    report = run_experiment(
        RealtimeRun(root, RealtimeConfig(pixels=pixels, reference_count=1), seconds=0.7),
        realtime_environment=game,
        numeric_actor_factory=actor,
    ).summary["realtime"]
    assert report["stop_reason"] == "time_limit", report
    assert report["model"]["exploration"] is True
    assert report["inference"]["inference_device"] == "cpu"
    accepted = [d for d in report["decisions"] if d["status"] == "accepted"]
    assert len(accepted) >= 3
    assert len({tuple(d["prediction"]) for d in accepted}) >= 2
    for row in accepted:
        sent = next(c for c in report["commands"] if c["decision_id"] == row["decision_id"])
        assert sent["owner"] == "policy" and sent["status"] == "sent"
        assert abs(sent["sent"]["steer_i16"]) <= 13107
        assert 0 <= sent["sent"]["throttle_u8"] <= 64
        assert 0 <= sent["sent"]["brake_u8"] <= 128
        assert not (sent["sent"]["throttle_u8"] and sent["sent"]["brake_u8"])

    # Replay starts a fresh model without the worker's synthetic warmup call.
    verified = run_experiment(
        RealtimeNumericReplay(root, tmp_path / "verified.html"), numeric_actor=actor()
    ).summary["realtime_numeric_replay"]
    assert verified["verified"], verified
    assert verified["verified_predictions"] >= len(accepted)
    assert verified["commands_sent_to_game"] is False
    wrong_seed = run_experiment(
        RealtimeNumericReplay(root, tmp_path / "wrong-seed.html"), numeric_actor=actor(38)
    ).summary["realtime_numeric_replay"]
    assert wrong_seed["verified"] is False
    assert "model differs" in str(wrong_seed["errors"])
    assert (sac_policy / "policy.pt").read_bytes() == before

    # Even matching rewritten identity hashes cannot make another seed reproduce
    # these outputs. This also distinguishes exploration from zero-noise evaluation.
    report["model"]["noise"]["seed"] = 38
    (root / "report.json").write_text(json.dumps(report))
    (root / "realtime-manifest.json").write_text(
        json.dumps({"version": 1, "report_sha256": sha(root / "report.json")})
    )
    changed_noise = run_experiment(
        RealtimeNumericReplay(root, tmp_path / "changed-noise.html"), numeric_actor=actor(38)
    ).summary["realtime_numeric_replay"]
    assert changed_noise["verified"] is False
    assert "Prediction differs" in str(changed_noise["errors"])
