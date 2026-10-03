"""Keep hypothetical SAC proposals distinct from successful actuator sends."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from fh5.learning.sac.replay import command_action

SEND_CONTEXT = "successful-send-return-proxy-v1"
PROPOSAL_CONTEXT = "counterfactual-proposal-v1"


def proposal_context(proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "kind": PROPOSAL_CONTEXT,
        **{
            key: deepcopy(proposal[key])
            for key in ("proposal_index", "proposed_ns", "proposal", "owner")
        },
    }


def context_action(context: dict[str, Any], kind: str, decision_ns: int) -> tuple[list[float], int]:
    """Return the preceding action and its declared time basis, without relabeling it."""
    if kind == PROPOSAL_CONTEXT:
        keys = {"version", "kind", "proposal_index", "proposed_ns", "proposal", "owner"}
        if set(context) != keys or context["kind"] != kind:
            raise ValueError("Invalid counterfactual proposal context")
        index, at, value = context["proposal_index"], context["proposed_ns"], context["proposal"]
    elif kind == SEND_CONTEXT:
        keys = {"version", "command_index", "issued_ns", "returned_ns", "sent", "owner"}
        if (
            set(context) != keys
            or type(context["issued_ns"]) is not int
            or type(context["returned_ns"]) is not int
            or not 0 <= context["issued_ns"] <= context["returned_ns"]
        ):
            raise ValueError("Invalid successful command context")
        index, at, value = context["command_index"], context["returned_ns"], context["sent"]
    else:
        raise ValueError("Unsupported SAC command context")
    if (
        type(context["version"]) is not int
        or context["version"] != 1
        or type(index) is not int
        or index < 0
        or type(at) is not int
        or not 0 <= at < decision_ns
        or context["owner"] not in ("initial_neutral", "policy", "lease_expiry", "warmup")
    ):
        raise ValueError("Invalid SAC action context time or owner")
    return command_action(value), at
