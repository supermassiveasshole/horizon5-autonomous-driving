"""Keep decision support separate from full asynchronous command hold time."""

from typing import Any


def next_action_elapsed(row: dict[str, Any]) -> float:
    if row.get("action_time_basis") != "asynchronous_send_return_proxy_v1":
        if "next_action_elapsed_s" in row or "execution_timing" in row:
            raise ValueError("SAC asynchronous timing requires its explicit versioned basis")
        return float(row["hold_dt_s"])
    t = row["execution_timing"]
    ordered = [
        t[k]
        for k in (
            "previous_returned_ns",
            "decision_ns",
            "issued_ns",
            "returned_ns",
            "next_issued_ns",
            "next_returned_ns",
        )
    ]
    if (
        any(type(v) is not int or v < 0 for v in ordered)
        or any(a > b for a, b in zip(ordered, ordered[1:]))
        or t["previous_returned_ns"] >= t["decision_ns"]
        or t["returned_ns"] >= t["next_issued_ns"]
        or row["current"]["decision_ns"] != t["decision_ns"]
        or row["action_elapsed_s"] != (t["decision_ns"] - t["previous_returned_ns"]) / 1e9
        or row["hold_dt_s"] != (t["next_returned_ns"] - t["returned_ns"]) / 1e9
    ):
        raise ValueError("SAC asynchronous command timing is inconsistent")
    if not row["bootstrap"]:
        if t["next_decision_ns"] is not None or row["next_action_elapsed_s"] is not None:
            raise ValueError("SAC terminal cannot claim a bootstrap decision")
        return 0.0
    now = t["next_decision_ns"]
    if (
        type(now) is not int
        or not t["returned_ns"] < now <= t["next_issued_ns"]
        or row["next"] is None
        or row["next"]["decision_ns"] != now
        or row["next"]["epoch"] != row["current"]["epoch"]
        or row["next_action_elapsed_s"] != (now - t["returned_ns"]) / 1e9
    ):
        raise ValueError("SAC bootstrap timing differs from its actual decision")
    return float(row["next_action_elapsed_s"])
