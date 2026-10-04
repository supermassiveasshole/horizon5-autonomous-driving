"""Read archived v1 policy evidence without an online driving implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_policy(directory: Path) -> dict[str, Any]:
    value = json.loads((directory / "policy.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Unsupported policy session")
    expected = {
        "packets.jsonl",
        "commands.jsonl",
        "control.json",
        "vision.jsonl",
        "vision-session.json",
        "policy-decisions.jsonl",
    }
    errors = []
    hashes = value.get("hashes", {})
    if set(hashes) != expected:
        errors.append("Incomplete policy evidence hashes")
    for name in expected:
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(name):
            errors.append("Policy evidence changed: " + name)
    journal = directory / "policy-decisions.jsonl"
    if journal.is_file():
        try:
            if [json.loads(line) for line in journal.read_bytes().splitlines()] != value.get(
                "decisions"
            ):
                errors.append("Policy decisions differ from the finalized journal")
        except ValueError:
            errors.append("Invalid policy decision journal")
    value["artifact_errors"] = errors
    value["formal_validity"] = "pending_independent_review"
    return value
