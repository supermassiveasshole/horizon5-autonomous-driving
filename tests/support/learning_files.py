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
    entries = []
    while True:
        directory = root / f"round-{row['number']:03d}/updates-{len(entries):03d}"
        checkpoint = directory / "policy.json"
        if not checkpoint.is_file():
            return entries
        entries.append(
            {
                "directory": str(directory),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )
