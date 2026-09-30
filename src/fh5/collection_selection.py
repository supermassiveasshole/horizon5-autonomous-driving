"""Stream sealed blocks and select unmodified labels with explicit eligibility."""

from __future__ import annotations

import hashlib
import json
import random
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fh5.collection_review import _references
from fh5.collection_store import read_bounded
from fh5.demonstrations import _mapped
from fh5.numeric_images import PixelContract, asset, validate_frame_history
from fh5.numeric_recording import read_numeric_frame


def sealed_rows(
    source: dict[str, Any], contract: PixelContract
) -> Iterator[tuple[str, dict[str, Any]]]:
    root, binding = Path(source["recording"]), source["session_sha256"]
    refs = _references(
        {"version": 1, "session_sha256": binding, "blocks": source["blocks"]}, binding
    )
    if not refs or list(refs) != sorted(refs):
        raise ValueError("Frozen sealed blocks must be nonempty and ordered")
    previous = -1
    previous_ns = -1
    for relative, ref in refs.items():
        block = asset(root, relative)
        raw = read_bounded(block / "manifest.json", 4096)
        manifest = json.loads(raw)
        if hashlib.sha256(raw).hexdigest() != ref["sha256"] or (
            manifest["version"] != 1
            or manifest["session_sha256"] != binding
            or manifest["index"] != int(block.name)
            or manifest["row_count"] != ref["rows"]
        ):
            raise ValueError("Frozen sealed block reference changed")
        payload = read_bounded(block / "rows.jsonl", 256 * 1024**2)
        if hashlib.sha256(payload).hexdigest() != manifest["rows_sha256"]:
            raise ValueError("Frozen sealed rows changed")
        lines = payload.splitlines()
        if len(lines) != ref["rows"]:
            raise ValueError("Frozen sealed row count differs")
        rows = [json.loads(line) for line in lines]
        if (
            rows[0]["sequence"] != manifest["first_sequence"]
            or rows[-1]["sequence"] != manifest["last_sequence"]
        ):
            raise ValueError("Sealed sequence bounds differ")
        for row in rows:
            seq, tick = row["sequence"], row["at_ns"]
            if (
                type(seq) is not int
                or seq <= previous
                or type(tick) is not int
                or tick <= previous_ns
            ):
                raise ValueError("Frozen source clocks and sequence must advance")
            previous, previous_ns = seq, tick
            if row["frames"]:
                frames = tuple(
                    read_numeric_frame(block, f, byte_limit=contract.size[0] * contract.size[1] * 3)
                    for f in row["frames"]
                )
                reason = validate_frame_history(row["capture_epoch"], tick, frames, contract)
                if reason or any(f.source_time_ns < row["segment_start_ns"] for f in frames):
                    raise ValueError("Frozen image history is unavailable or crosses a segment")
            yield relative, row


def _active_interval(attempt: dict[str, Any], sequence: int) -> dict[str, Any] | None:
    intervals = attempt["intervals"]
    index = bisect_right([i["start_sequence"] for i in intervals], sequence) - 1
    if index >= 0 and sequence < intervals[index]["end_sequence"]:
        return dict(intervals[index], index=index)
    return None


def _with_following(
    rows: Iterator[tuple[str, dict[str, Any]]],
) -> Iterator[tuple[tuple[str, dict[str, Any]], tuple[str, dict[str, Any]] | None]]:
    current = next(rows, None)
    while current is not None:
        following = next(rows, None)
        yield current, following
        current = following


def _behaviors(
    action: list[float], speed: float, previous: tuple[list[float], float] | None
) -> set[str]:
    steer, pedal = action
    names = {
        name
        for name, yes in (
            ("left", steer < -0.1),
            ("right", steer > 0.1),
            ("throttle", pedal > 0),
            ("brake", pedal < 0),
            ("coast", pedal == 0),
            ("stationary", speed < 1 / 3.6),
            ("low_speed", 1 / 3.6 <= speed < 15 / 3.6),
            ("medium_speed", 15 / 3.6 <= speed < 100 / 3.6),
            ("high_speed", speed >= 100 / 3.6),
        )
        if yes
    }
    if previous is not None:
        if previous[0][1] > 0 and pedal <= 0:
            names.add("release_rt")
        if previous[1] < 1 / 3.6 <= speed and (pedal > 0 or previous[0][1] > 0):
            names.add("startup")
    return names


def select_source(
    source: dict[str, Any], session: dict[str, Any], rules: dict[str, Any], rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from fh5.experiment import Packet, _decode

    cfg = session["configuration"]
    contract = PixelContract.from_metadata(cfg["pixels"])
    attempts = source["review"]["attempts"]
    stats = {
        a["id"]: {
            "attempt": a["id"],
            "group": a["group"],
            "split": a["split"],
            "trusted_events": Counter(),
            "exclusions": Counter(),
            "observations": 0,
            "selected": 0,
            "eligible_before_sampling": 0,
            "sequence_gaps": 0,
            "road_kind_unknown_polls": 0,
            "first_ns": None,
            "last_ns": None,
        }
        for a in attempts
    }
    chosen: dict[str, list[dict[str, Any]]] = {a["id"]: [] for a in attempts}
    totals: Counter[str] = Counter()
    latest = None
    last_sequence = -1
    last_key = None
    history_floor = 0
    previous_behavior: tuple[list[float], float] | None = None
    active_events: set[str] = set()
    seen_attempts: set[str] = set()
    for (block, row), following in _with_following(sealed_rows(source, contract)):
        tick, seq = row["at_ns"], row["sequence"]
        decoded = []
        source_faults = []
        for packet in row["packets"]:
            try:
                sample = _decode(
                    Packet(
                        packet["received_monotonic_ns"],
                        packet["received_utc"],
                        bytes.fromhex(packet["payload_hex"]),
                    )
                )
            except ValueError:
                source_faults.append("invalid_telemetry")
                continue
            if sample["received_monotonic_ns"] > tick or (
                latest is not None
                and sample["received_monotonic_ns"] <= latest["received_monotonic_ns"]
            ):
                raise ValueError("Recorded telemetry must be causal and ordered")
            latest = sample
            decoded.append(sample)
        if decoded != row["telemetry"]:
            raise ValueError("Decoded telemetry differs from raw source evidence")
        try:
            mapped = (
                _mapped(row["human_input"], session["profile"])
                if row["human_input"] is not None
                else None
            )
        except (ValueError, TypeError, KeyError):
            mapped = None
            source_faults.append("invalid_human_input")
        if mapped != row["mapped_input"]:
            raise ValueError("Mapped label differs from raw source evidence")
        index = bisect_right([a["start_sequence"] for a in attempts], seq) - 1
        attempt = attempts[index] if index >= 0 and seq < attempts[index]["end_sequence"] else None
        gap = seq != last_sequence + 1
        last_sequence = seq
        if attempt is None:
            last_key = None
            previous_behavior, active_events = None, set()
            continue
        identifier = attempt["id"]
        seen_attempts.add(identifier)
        result = stats[identifier]
        if result["first_ns"] is None:
            result["first_ns"] = tick
        result["last_ns"] = tick
        interval = _active_interval(attempt, seq)
        key = (identifier, interval["index"] if interval else None, row["segment"])
        if key != last_key or gap:
            history_floor = tick
            previous_behavior, active_events = None, set()
        last_key = key
        result["sequence_gaps"] += gap
        reasons = [*row["reasons"], *source_faults]
        if not row["input_usable"] or mapped is None or not mapped["mapping_valid"]:
            reasons.append("unusable_input")
        if mapped is not None and not (
            0 <= tick - mapped["poll_ns"] <= cfg["max_age_ms"] * 1_000_000
            and mapped["available_ns"] <= tick
        ):
            reasons.append("label_clock")
        if latest is None or not (
            0 <= tick - latest["received_monotonic_ns"] <= cfg["max_age_ms"] * 1_000_000
        ):
            reasons.append("missing_telemetry")
        elif (
            not latest["is_race_on"]
            or latest["motion"] is None
            or latest["car_ordinal"] != cfg["expected_car_ordinal"]
            or latest["car_performance_index"] != cfg["expected_pi"]
        ):
            reasons.append("invalid_telemetry")
        if interval is None or interval["quality"] != "trusted":
            reasons.append("unreviewed_or_failed")
        if not source["review"]["conditions_verified"]:
            reasons.append("unverified_conditions")
        action = mapped["mapped"] if mapped is not None else None
        speed = latest["speed_mps"] if latest is not None else None
        if action is not None and (
            abs(action[0]) > rules["steering_limit"] or abs(action[1]) > rules["longitudinal_limit"]
        ):
            reasons.append("action_outside_envelope")
        if (
            speed is not None
            and not rules["speed_range_mps"][0] <= speed <= rules["speed_range_mps"][1]
        ):
            reasons.append("speed_outside_envelope")
        if not reasons and action is not None and speed is not None:
            behaviors = _behaviors(action, speed, previous_behavior)
            if interval is not None and interval["road_kind"] != "unknown":
                behaviors.add(interval["road_kind"])
            else:
                result["road_kind_unknown_polls"] += 1
            result["trusted_events"].update(behaviors - active_events)
            previous_behavior, active_events = (action, speed), behaviors
        else:
            previous_behavior, active_events = None, set()
        if not row["observation_due"]:
            continue
        result["observations"] += 1
        if not row["frames"] or any(f["source_time_ns"] < history_floor for f in row["frames"]):
            result["exclusions"]["incomplete_or_cross_review_history"] += 1
            continue
        if following is None:
            result["exclusions"]["no_following_label"] += 1
            continue
        label_block, label_row = following
        try:
            label = _mapped(label_row["human_input"], session["profile"])
        except (ValueError, KeyError, TypeError):
            result["exclusions"]["unavailable_label"] += 1
            continue
        if (
            not label_row["input_usable"]
            or not label["mapping_valid"]
            or label_row["sequence"] != seq + 1
            or label_row["segment"] != row["segment"]
            or label_row["sequence"] >= attempt["end_sequence"]
            or _active_interval(attempt, label_row["sequence"]) != interval
            or not 0 <= label["poll_ns"] - tick <= rules["max_label_delay_ms"] * 1_000_000
            or label["available_ns"] > label_row["at_ns"]
        ):
            result["exclusions"]["unavailable_or_cross_boundary_label"] += 1
            continue
        action = label["mapped"]
        if abs(action[0]) > rules["steering_limit"] or abs(action[1]) > rules["longitudinal_limit"]:
            reasons.append("action_outside_envelope")
        unique_reasons = sorted(set(reasons))
        result["exclusions"].update(unique_reasons)
        result["eligible_before_sampling"] += not unique_reasons
        sample = {
            "id": source["session_sha256"] + ":" + str(seq),
            "source_sha256": source["session_sha256"],
            "block": block,
            "sequence": seq,
            "attempt": identifier,
            "group": attempt["group"],
            "decision_ns": tick,
            "target_action": action,
            "label_block": label_block,
            "label_sequence": label_row["sequence"],
            "label_poll_ns": label["poll_ns"],
            "label_available_ns": label["available_ns"],
            "bc_eligible": not unique_reasons,
            "reasons": unique_reasons,
            "history_floor_ns": history_floor,
            "q_eligible": False,
            "q_reason": "reward_and_termination_unverified",
            "dynamics_eligible": False,
            "dynamics_reason": "action_response_alignment_pending",
        }
        totals[identifier] += 1
        pool = chosen[identifier]
        if len(pool) < rules["max_samples_per_attempt"]:
            pool.append(sample)
        else:
            candidate = rng.randrange(totals[identifier])
            if candidate < len(pool):
                pool[candidate] = sample
    # Reviews may describe an ongoing attempt whose tail is not yet sealed; only
    # published rows are selected. A completely absent attempt is a configuration error.
    if seen_attempts != set(stats):
        raise ValueError("Reviewed attempt has no published rows")
    selected = []
    for identifier, pool in chosen.items():
        stats[identifier]["selected"] = len(pool)
        stats[identifier]["reservoir_omitted"] = totals[identifier] - len(pool)
        selected.extend(pool)
    return sorted(selected, key=lambda s: s["sequence"]), list(stats.values())
