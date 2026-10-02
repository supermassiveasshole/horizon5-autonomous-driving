"""Read the public checkpoint artifacts used by experiment acceptance checks."""

import hashlib
import json


def history_entries(root, manifest=None):
    if manifest is None:
        name = "policy.json" if (root / "policy.json").exists() else "critic.json"
        manifest = json.loads((root / name).read_text())
    binding = manifest["history"]
    if isinstance(binding, list):
        return binding
    entries = []
    while binding["head"] is not None:
        node = json.loads((root / binding["head"]["path"]).read_text())
        entries.append(node["entry"])
        binding = node["previous"]
    return list(reversed(entries))


def update_records(root):
    """Read and authenticate update records from either public report format."""
    descriptor = json.loads((root / "training-report.json").read_bytes())["updates"]
    if isinstance(descriptor, list):
        return descriptor
    assert descriptor["status"] == "complete"
    raw = (root / descriptor["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
    records = [json.loads(line) for line in raw.splitlines()]
    assert len(records) == descriptor["records"]
    return records
