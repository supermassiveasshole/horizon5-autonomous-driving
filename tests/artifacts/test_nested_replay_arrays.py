"""Nested replay arrays stay indexed without changing their public JSON meaning."""

import hashlib
import json
import tracemalloc

import pytest

from fh5.artifacts.document import ReplayArray, replay_document, write_replay_document
from fh5.artifacts.io import VerifiedFile


def verified(path):
    with path.open("rb") as stream:
        return VerifiedFile(path, hashlib.file_digest(stream, "sha256").hexdigest())


@pytest.mark.parametrize("index_attempts", [False, True])
def test_selected_nested_arrays_roundtrip_with_literal_metadata_and_default_lists(
    tmp_path, index_attempts
):
    literal = {"$replay_array": "0", "section": "1", "length": 2, "refs": [[[], "0", 1]]}
    expected = {
        "sources": [
            {
                "blocks": [{"id": "block-1"}],
                "review": {
                    "attempts": [
                        {"id": "a", "intervals": [literal, None, [1, 2]]},
                        {"id": "b", "intervals": []},
                    ]
                },
            }
        ],
        "environment": {"source_samples": {"records": [{"text": '道路 😀 \\ "'}]}},
        "config": {"sources": [{"path": "source-1"}]},
        "model": {"shape": [3, 36, 64]},
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")
    source = verified(path)
    with replay_document(source) as original:
        assert isinstance(original["sources"], ReplayArray)
        assert isinstance(original["sources"][0]["review"]["attempts"], list)
        assert isinstance(original["environment"]["source_samples"]["records"], list)
    paths = {
        ("sources", "*", "blocks"),
        ("sources", "*", "review", "attempts", "*", "intervals"),
        ("environment", "source_samples", "records"),
        ("config", "sources"),
    }
    if index_attempts:
        paths.add(("sources", "*", "review", "attempts"))
    with replay_document(source, nested_arrays=paths) as document:
        attempts = document["sources"][0]["review"]["attempts"]
        assert isinstance(attempts, ReplayArray if index_attempts else list)
        assert len(attempts) == 2
        blocks = document["sources"][0]["blocks"]
        assert isinstance(blocks, ReplayArray) and list(blocks) == [{"id": "block-1"}]
        configured = document["config"]["sources"]
        assert isinstance(configured, ReplayArray) and list(configured) == [{"path": "source-1"}]
        intervals = attempts[0]["intervals"]
        assert isinstance(intervals, ReplayArray) and len(intervals) == 3
        assert intervals[-1] == [1, 2]
        assert intervals[:2] == [literal, None]
        assert list(intervals) == list(intervals) == [literal, None, [1, 2]]
        assert list(attempts[1]["intervals"]) == []
        records = document["environment"]["source_samples"]["records"]
        assert isinstance(records, ReplayArray)
        assert list(records) == expected["environment"]["source_samples"]["records"]
        assert document["model"]["shape"] == [3, 36, 64]
        output = tmp_path / "rewritten.json"
        write_replay_document(output, document)
    assert json.loads(output.read_bytes()) == expected
    assert (
        output.read_bytes()
        == (json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )


def test_nested_growth_is_indexed_and_rewritten_without_materializing_parent_objects(tmp_path):
    source = tmp_path / "growing.json"
    item = json.dumps({"evidence": "x" * 32768}).encode()
    count = 1024
    with source.open("wb") as stream:
        stream.write(b'{"sources":[{"review":{"attempts":[{"intervals":[')
        for index in range(count):
            stream.write((b"," if index else b"") + item)
        stream.write(b']}]}}],"environment":{"source_samples":{"records":[')
        for index in range(count):
            stream.write((b"," if index else b"") + item)
        stream.write(b']}},"model":{"shape":[3,36,64]}}')
    binding = verified(source)
    output = tmp_path / "rewritten.json"
    tracemalloc.start()
    try:
        with replay_document(
            binding,
            nested_arrays={
                ("sources", "*", "review", "attempts"),
                ("sources", "*", "review", "attempts", "*", "intervals"),
                ("environment", "source_samples", "records"),
            },
        ) as document:
            intervals = document["sources"][0]["review"]["attempts"][0]["intervals"]
            records = document["environment"]["source_samples"]["records"]
            for values in (intervals, records):
                assert isinstance(values, ReplayArray) and len(values) == count
                assert sum(len(row["evidence"]) for row in values) == count * 32768
                assert values[-1] == {"evidence": "x" * 32768}
            write_replay_document(output, document)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < source.stat().st_size / 2, "An enclosing object must not load its growing array"
    with replay_document(
        verified(output), nested_arrays={("environment", "source_samples", "records")}
    ) as rewritten:
        assert len(rewritten["environment"]["source_samples"]["records"]) == count
        assert rewritten["model"]["shape"] == [3, 36, 64]
    assert verified(source) == binding


def test_nested_array_elements_remain_indexed_without_object_wrappers(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"arrays":[[[],[1,2]],[[3]]]}')
    with replay_document(
        verified(source), nested_arrays={("arrays", "*"), ("arrays", "*", "*")}
    ) as document:
        first = document["arrays"][0]
        assert isinstance(first, ReplayArray)
        assert isinstance(first[1], ReplayArray)
        assert list(first[0]) == [] and list(first[1]) == [1, 2]
        output = tmp_path / "output.json"
        write_replay_document(output, document)
    assert json.loads(output.read_bytes()) == {"arrays": [[[], [1, 2]], [[3]]]}


@pytest.mark.parametrize(
    "raw",
    [
        '{"a":{"rows":[0],"rows":[1,2]}}',
        '{"a":{"rows":[0]},"a":{"rows":[3]}}',
        '{"a":{"rows":[0]},"a":null}',
        '{"a":{"rows":null,"rows":[1]}}',
        '{"sources":[{"rows":[0]}],"sources":[{"rows":[1]}]}',
        '{"sources":[{"rows":[0]}],"sources":{"rows":[1]}}',
        '{"a":{"rows":[0],"rows":{"keep":[1,2]}}}',
    ],
)
def test_nested_array_selection_preserves_last_duplicate_value(tmp_path, raw):
    source = tmp_path / "duplicates.json"
    source.write_text(raw)
    with replay_document(
        verified(source), nested_arrays={("a", "rows"), ("sources", "*", "rows")}
    ) as document:
        output = tmp_path / "output.json"
        write_replay_document(output, document)
    assert json.loads(output.read_bytes()) == json.loads(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"a":{"rows":[1,]}}',
        '{"a":{"rows":[{"key":1,}]}}',
        '{"a":{"rows":[1]}',
        '{"a":{"rows":[1]}} tail',
        '{"a":{"rows":[1]},}',
        '{"a":{"rows":[1x]}}',
        '{"a":{"rows":["bad\\q"]}}',
        '{"a":{"rows":[1]},"unchecked":{"bad":[2,]}}',
        '{"a":{"rows":[1,]},"a":null}',
    ],
)
def test_invalid_nested_or_unselected_json_is_never_hidden_by_indexing(tmp_path, raw):
    source = tmp_path / "invalid.json"
    source.write_text(raw)
    with (
        pytest.raises(ValueError),
        replay_document(verified(source), nested_arrays={("a", "rows")}),
    ):
        pytest.fail("Malformed JSON was admitted")


def test_nested_indexing_keeps_hash_binding_and_context_lifetime(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"a":{"rows":[1,2]}}')
    binding = verified(source)
    source.write_text('{"a":{"rows":[3,4]}}')
    with pytest.raises(ValueError), replay_document(binding, nested_arrays={("a", "rows")}):
        pytest.fail("Changed source was admitted")
    with replay_document(verified(source), nested_arrays={("a", "rows")}) as document:
        rows = document["a"]["rows"]
        assert list(rows) == [3, 4]
    with pytest.raises(OSError, match="replay document index"):
        rows[0]
