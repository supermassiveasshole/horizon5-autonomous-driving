"""Read the public checkpoint artifacts used by experiment acceptance checks."""

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
