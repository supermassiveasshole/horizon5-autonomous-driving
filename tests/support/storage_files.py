"""Independent reader for the public storage-plan file-detail format."""

import json

from tests.evaluation.test_evaluation import sha


def storage_files(root, plan):
    binding = plan["files"]
    assert binding["format"] == "storage-files-v1"
    path = root / binding["path"]
    assert sha(path) == binding["sha256"]
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            count += 1
            yield json.loads(line)
    assert count == binding["count"] == plan["protected_files"]
