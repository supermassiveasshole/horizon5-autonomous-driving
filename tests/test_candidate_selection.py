"""Candidate comparisons from independently replayed evidence at the experiment seam."""

import json
import struct
from copy import deepcopy

import pytest
from test_attempts import evidence
from test_evaluation import entry, prepare, sha
from test_route_check import record
from test_temporal_bc import temporal_fixture

from fh5.evaluation import EvaluationPrepare
from fh5.experiment import run_experiment
from fh5.temporal_bc import TemporalBCTrain


@pytest.fixture(scope="module")
def policies(tmp_path_factory):
    root = tmp_path_factory.mktemp("selection-models")
    config, _ = temporal_fixture(root)
    models = []
    for seed in (7, 8):
        options = json.loads(config.read_bytes())
        options["seed"] = seed
        config.write_text(json.dumps(options))
        path = root / f"model-{seed}"
        run_experiment(TemporalBCTrain(config, path))
        models.append(path)
    return models


def comparison(
    tmp_path,
    policies,
    *,
    incumbent=("valid",),
    candidate=("wall",),
    modes=None,
    candidate_periods=None,
):
    modes = modes or ["no_reference"] * len(incumbent)
    baseline = tmp_path / "incumbent"
    baseline.mkdir()
    _, config = prepare(baseline, policies[0], modes)
    options = json.loads(config.read_bytes())
    options["model"] = {
        "directory": str(policies[1]),
        "manifest_sha256": sha(policies[1] / "model.json"),
    }
    contender = tmp_path / "candidate"
    contender.mkdir()
    next_config = contender / "evaluation.json"
    next_config.write_text(json.dumps(options))
    run_experiment(EvaluationPrepare(next_config, contender / "frozen"))
    bindings = {}
    for side, folder, outcomes, period in (
        ("incumbent", baseline, incumbent, 100),
        ("candidate", contender, candidate, 50),
    ):
        rows = []
        for i, outcome in enumerate(outcomes):
            if outcome == "unstarted":
                continue
            sample = folder / f"source-{i}"
            sample.mkdir()
            source = record(
                sample,
                "drive",
                [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2)],
                speed=30 if outcome == "failed" else 4,
            )
            packets = source / "packets.jsonl"
            values = [json.loads(line) for line in packets.read_text().splitlines()]
            offset = (10 if side == "candidate" else 0) + i
            step_ms = candidate_periods[i] if side == "candidate" and candidate_periods else period
            for j, row in enumerate(values):
                row["received_monotonic_ns"] = (
                    10 + offset
                ) * 1_000_000_000 + j * step_ms * 1_000_000
                raw = bytearray.fromhex(row["payload_hex"])
                struct.pack_into("<I", raw, 4, 1000 + j * step_ms)
                row["payload_hex"] = raw.hex()
            packets.write_text("".join(json.dumps(row) + "\n" for row in values))
            events = (
                [{"packet_index": 1, "kind": "wall_riding", "status": "confirmed"}]
                if outcome == "wall"
                else []
            )
            proof = evidence(sample, source, events) if outcome != "unknown" else None
            rows.append(entry(f"run-{i}", source, proof))
        ledger = folder / "ledger.json"
        ledger.write_text(
            json.dumps(
                {"version": 1, "batch_sha256": sha(folder / "frozen/batch.json"), "entries": rows}
            )
        )
        bindings[side] = {
            "batch": str(folder / "frozen"),
            "batch_sha256": sha(folder / "frozen/batch.json"),
            "ledger": str(ledger),
            "ledger_sha256": sha(ledger),
        }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps({"version": 1, **bindings}))
    return path


def test_faster_confirmed_violation_cannot_displace_incumbent_or_erase_candidate(
    tmp_path, policies
):
    from fh5.candidate_selection import CandidateCompare

    config = comparison(tmp_path, policies)
    before = [sha(path / "actor.pt") for path in policies]
    result = run_experiment(CandidateCompare(config, tmp_path / "decision")).summary[
        "candidate_selection"
    ]
    assert result["local_recommendation"] == "retain_incumbent"
    assert "candidate:invalid_attempts" in result["reasons"]
    assert result["promotion_allowed"] is False
    assert result["candidate_discarded"] is False
    assert result["training_state_modified"] is False
    assert result["reviews"]["candidate"]["metrics"]["outcomes"]["invalid"] == 1
    assert result["reviews"]["incumbent"]["metrics"]["valid_duration_s"]["median"] == 0.3
    assert [sha(path / "actor.pt") for path in policies] == before
    assert result["selected_model_sha256"] == sha(policies[0] / "model.json")


def test_faster_less_reliable_candidate_is_kept_separately_with_every_failure(tmp_path, policies):
    from fh5.candidate_selection import CandidateCompare

    config = comparison(
        tmp_path, policies, incumbent=("valid",) * 3, candidate=("valid", "valid", "failed")
    )
    result = run_experiment(CandidateCompare(config, tmp_path / "decision")).summary[
        "candidate_selection"
    ]
    assert result["local_recommendation"] == "retain_incumbent"
    assert result["aggressive_by_reference"] == {"no_reference": sha(policies[1] / "model.json")}
    assert "candidate:reliability_regressed:no_reference" in result["reasons"]
    group = result["by_reference"]["no_reference"]
    assert group["incumbent"]["valid_fraction_all_attempts"] == 1
    assert group["candidate"]["valid_fraction_all_attempts"] == pytest.approx(2 / 3)
    assert group["candidate"]["all_attempts"] == 3
    assert group["candidate"]["outcomes"]["driving_failed"] == 1
    assert group["candidate"]["valid_duration_s"]["median"] == 0.15
    assert result["default_changed"] is False


def test_locally_qualified_candidate_is_recommended_without_promoting_diagnostic_evidence(
    tmp_path, policies
):
    from fh5.candidate_selection import CandidateCompare

    config = comparison(
        tmp_path, policies, incumbent=("valid", "valid"), candidate=("valid", "valid")
    )
    result = run_experiment(CandidateCompare(config, tmp_path / "decision")).summary[
        "candidate_selection"
    ]
    assert result["local_recommendation"] == "prefer_candidate_locally"
    assert result["selected_model_sha256"] == sha(policies[1] / "model.json")
    assert result["by_reference"]["no_reference"]["median_improvement_fraction"] == 0.5
    assert result["reasons"] == ["local_thresholds_met"]
    assert result["promotion_allowed"] is False
    assert result["default_changed"] is False
    assert "closed_loop_not_verified" in result["promotion_blockers"]
    assert "diagnostic_evidence" in result["promotion_blockers"]
    assert "independence_not_proven" in result["promotion_blockers"]
    assert result["reviews"]["candidate"]["metrics"]["all_attempts"] == 2


def test_reused_recording_cannot_supply_comparison_evidence_for_two_models(tmp_path, policies):
    from fh5.candidate_selection import CandidateCompare

    config = comparison(tmp_path, policies, candidate=("valid",))
    binding = json.loads(config.read_bytes())
    old = json.loads((tmp_path / "incumbent/ledger.json").read_bytes())
    path = tmp_path / "candidate/ledger.json"
    candidate = json.loads(path.read_bytes())
    candidate["entries"] = deepcopy(old["entries"])
    path.write_text(json.dumps(candidate))
    binding["candidate"]["ledger_sha256"] = sha(path)
    config.write_text(json.dumps(binding))
    result = run_experiment(CandidateCompare(config, tmp_path / "decision")).summary[
        "candidate_selection"
    ]
    assert "shared_recording_origins" in result["reasons"]
    assert result["local_recommendation"] == "retain_incumbent"
    assert result["aggressive_by_reference"] == {}
    assert result["reviews"]["incumbent"]["metrics"]["all_attempts"] == 1
    assert result["reviews"]["candidate"]["metrics"]["all_attempts"] == 1


def test_known_training_reuse_blocks_comparison_and_registers_actual_selection_use(
    tmp_path, policies
):
    from fh5.candidate_selection import CandidateCompare
    from fh5.evidence_usage import RecordUsage

    config = comparison(tmp_path, policies, candidate=("valid",))
    registry = tmp_path / "usage.sqlite"
    run_experiment(
        RecordUsage(
            registry,
            (tmp_path / "incumbent/source-0/drive",),
            "training",
            tmp_path / "training-use",
        )
    )
    result = run_experiment(CandidateCompare(config, tmp_path / "decision", registry)).summary[
        "candidate_selection"
    ]
    assert "incumbent:known_evidence_reuse" in result["reasons"]
    assert result["local_recommendation"] == "retain_incumbent"
    snapshot = json.loads((tmp_path / "decision/candidate/usage-snapshot.json").read_bytes())
    uses = [u for u in snapshot["uses"] if u["role"] == "selection"]
    assert {u["model"] for u in uses} == {sha(p / "model.json") for p in policies}
    assert result["reviews"]["incumbent"]["metrics"]["all_attempts"] == 1
    assert result["reviews"]["candidate"]["metrics"]["all_attempts"] == 1


@pytest.mark.parametrize(
    "outcome,reason",
    [("unknown", "candidate:pending_review_attempts"), ("unstarted", "candidate:incomplete_plan")],
)
def test_incomplete_evidence_keeps_the_candidate_out_of_both_comparison_slots(
    tmp_path, policies, outcome, reason
):
    from fh5.candidate_selection import CandidateCompare

    config = comparison(
        tmp_path, policies, incumbent=("valid", "valid"), candidate=("valid", outcome)
    )
    result = run_experiment(CandidateCompare(config, tmp_path / "decision")).summary[
        "candidate_selection"
    ]
    assert result["local_recommendation"] == "retain_incumbent"
    assert reason in result["reasons"]
    assert result["aggressive_by_reference"] == {}
    assert result["reviews"]["candidate"]["metrics"]["all_attempts"] == (
        1 if outcome == "unstarted" else 2
    )


def test_cli_compares_from_sources_without_devices_or_training(tmp_path, policies, capsys):
    from fh5.cli import main

    config = comparison(tmp_path, policies, candidate=("valid",))
    assert (
        main(["candidate-compare", "--config", str(config), "--output", str(tmp_path / "cli")]) == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["local_recommendation"] == "prefer_candidate_locally"
    assert result["default_changed"] is False
    assert result["commands_sent"] is False
    assert result["training_state_modified"] is False
    assert (tmp_path / "cli/report.html").is_file()


def test_exact_reliability_tolerance_is_not_rejected_by_binary_rounding(tmp_path, policies):
    from fh5.candidate_selection import CandidateCompare

    config = comparison(
        tmp_path, policies, incumbent=("valid",) * 20, candidate=("valid",) * 19 + ("failed",)
    )
    result = run_experiment(CandidateCompare(config, tmp_path / "decision")).summary[
        "candidate_selection"
    ]
    assert result["local_recommendation"] == "prefer_candidate_locally"
    assert result["by_reference"]["no_reference"]["reliability_delta"] == pytest.approx(-0.05)
    assert result["aggressive_by_reference"] == {}


def test_reference_group_regression_cannot_be_hidden_by_a_faster_pooled_median(tmp_path, policies):
    from fh5.candidate_selection import CandidateCompare

    config = comparison(
        tmp_path,
        policies,
        incumbent=("valid", "valid"),
        candidate=("valid", "valid"),
        modes=["no_reference", "reference_assisted"],
        candidate_periods=[50, 110],
    )
    result = run_experiment(CandidateCompare(config, tmp_path / "decision")).summary[
        "candidate_selection"
    ]
    assert result["reviews"]["candidate"]["metrics"]["valid_duration_s"]["median"] == 0.24
    assert result["local_recommendation"] == "retain_incumbent"
    assert "candidate:median_time_regressed:reference_assisted" in result["reasons"]
    assert result["by_reference"]["no_reference"]["median_improvement_fraction"] == 0.5


@pytest.mark.parametrize(
    "change,reason",
    [
        ("conditions", "conditions"),
        ("criteria", "criteria"),
        ("final", "Final acceptance"),
        ("ledger", "ledger changed"),
    ],
)
def test_comparison_refuses_changed_conditions_thresholds_final_use_or_bound_ledger(
    tmp_path, policies, change, reason
):
    from fh5.candidate_selection import CandidateCompare

    config = comparison(tmp_path, policies, candidate=("valid",))
    binding = json.loads(config.read_bytes())
    ledger = tmp_path / "candidate/ledger.json"
    if change == "ledger":
        ledger.write_text("{}")
    else:
        source = tmp_path / "candidate/evaluation.json"
        options = json.loads(source.read_bytes())
        if change == "conditions":
            options["conditions"]["camera"] = "cockpit"
        elif change == "criteria":
            options["criteria"]["reliability_tolerance"] = 0.2
        else:
            options["purpose"] = "final"
        source.write_text(json.dumps(options))
        frozen = tmp_path / "candidate/changed-frozen"
        run_experiment(EvaluationPrepare(source, frozen))
        values = json.loads(ledger.read_bytes())
        values["batch_sha256"] = sha(frozen / "batch.json")
        ledger.write_text(json.dumps(values))
        binding["candidate"].update(
            batch=str(frozen), batch_sha256=sha(frozen / "batch.json"), ledger_sha256=sha(ledger)
        )
        config.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match=reason):
        run_experiment(CandidateCompare(config, tmp_path / "decision"))
    assert not (tmp_path / "decision").exists()
