"""Inspect committed prefixes of public learning stage segments."""

import hashlib
import json


def stage_records(root, summary):
    binding = summary["stages"]
    if isinstance(binding, list):
        return binding
    expected = binding["events"]
    descriptor = binding["diagnostic"]
    segments = []
    while descriptor is not None:
        assert descriptor["status"] == "complete"
        with (root / descriptor["path"]).open("rb") as stream:
            lines = [stream.readline() for _ in range(descriptor["records"])]
        assert all(lines) and hashlib.sha256(b"".join(lines)).hexdigest() == descriptor["sha256"]
        header, *events = [json.loads(line) for line in lines]
        assert header["kind"] == "segment"
        assert header["base_events"] + len(events) == expected
        expected = header["base_events"]
        segments.append(events)
        descriptor = header["previous"]
    assert expected == 0
    records = [row for segment in reversed(segments) for row in segment]
    assert records[-2:] == binding["tail"]
    return records


def update_bindings(root, row):
    binding = row.get("update_segments", [])
    if isinstance(binding, list):
        return binding
    previous = None
    entries = []
    for number in range(binding["count"]):
        path = root / f"round-{row['number']:03d}/update-history/{number:06d}.json"
        raw = path.read_bytes()
        node = json.loads(raw)
        assert node["format"] == "learning-update-node-v1"
        assert node["previous_sha256"] == previous
        previous = hashlib.sha256(raw).hexdigest()
        entries.append(node["entry"])
    assert previous == binding["head_sha256"]
    return entries
