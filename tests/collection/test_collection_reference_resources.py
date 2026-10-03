"""Independent BC references grow through the public preparation interface."""

import hashlib
import io
import json
import tracemalloc
from pathlib import Path

import pytest

from fh5.collection.bc import CollectionBCPrepare
from fh5.experiment import run_experiment
from fh5.observation.routes import BuildRoute
from tests.collection.test_collection_bc import prepare_inputs
from tests.observation.test_routes import recording


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def independent_reference(tmp_path):
    config = prepare_inputs(tmp_path)
    source = recording(tmp_path, [(x, 0) for x in range(0, 101, 5)])
    route = tmp_path / "route"
    run_experiment(BuildRoute(source, route, 0, 20))
    options = json.loads(config.read_text())
    options["reference"] = {
        "route_file": str(route / "route.json"),
        "independence_evidence": ["Separate synthetic recording; no real-game claim"],
    }
    config.write_text(json.dumps(options))
    return config, route


def prepare_with_peak(config, output):
    tracemalloc.start()
    try:
        result = run_experiment(CollectionBCPrepare(config, output))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, peak


@pytest.mark.parametrize(
    "growth",
    [
        "manifest",
        "asset",
        "asset_diagnostics",
        "bundle",
        "manifest_string",
        "asset_string",
        "manifest_key",
        "asset_key",
    ],
)
def test_independent_reference_growth_preserves_causal_inputs_without_payload_residency(
    tmp_path, growth, record_testsuite_property
):
    config, route = independent_reference(tmp_path)
    _, baseline_peak = prepare_with_peak(config, tmp_path / "baseline")
    expected = json.loads((tmp_path / "baseline/dataset.json").read_text())["decisions"]
    manifest_path = route / "route.json"
    manifest = json.loads(manifest_path.read_text())
    block = b" " * 1024**2
    if growth not in ("asset", "bundle"):
        # Diagnostic history is valid old-format JSON but has no navigation role.
        target = (
            manifest_path
            if growth.startswith("manifest")
            else route / manifest["assets"]["reference"]["path"]
        )
        original = json.loads(target.read_text())
        with target.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(original)[:-1] + ',"diagnostics":')
            if growth.endswith(("_string", "_key")):
                stream.write('{"' if growth.endswith("_key") else '"')
                for _ in range(5):
                    stream.write("x" * 1024**2)
                stream.write('":null}}' if growth.endswith("_key") else '"}')
            else:
                stream.write("[")
                entry = json.dumps({"note": "x" * 1024})
                for index in range(5000):
                    stream.write(("," if index else "") + entry)
                stream.write("]}")
        assert target.stat().st_size > 4 * 1024**2
        added_bytes = target.stat().st_size
        if growth.startswith("asset"):
            manifest["assets"]["reference"]["sha256"] = digest(target)
            manifest_path.write_text(json.dumps(manifest))
    elif growth == "asset":
        reference = route / manifest["assets"]["reference"]["path"]
        with reference.open("ab") as stream:
            for _ in range(33):
                stream.write(block)
        manifest["assets"]["reference"]["sha256"] = digest(reference)
        manifest_path.write_text(json.dumps(manifest))
        assert reference.stat().st_size > 32 * 1024**2
        added_bytes = 33 * len(block)
    else:
        for index in range(3):
            path = route / f"independent-survey-{index}.bin"
            with path.open("wb") as stream:
                for _ in range(23):
                    stream.write(block)
            manifest["evidence"].append({"path": path.name, "sha256": digest(path)})
        manifest_path.write_text(json.dumps(manifest))
        added_bytes = 69 * len(block)
        assert added_bytes > 64 * 1024**2
    output = tmp_path / "grown"
    result, peak = prepare_with_peak(config, output)
    record_testsuite_property(f"reference_{growth}_baseline_python_peak_bytes", baseline_peak)
    record_testsuite_property(f"reference_{growth}_grown_python_peak_bytes", peak)
    record_testsuite_property(f"reference_{growth}_added_payload_bytes", added_bytes)
    data = json.loads((output / "dataset.json").read_text())
    assert data["decisions"] == expected
    assert any(any(row["views"]["reference_assisted"]["reference"]["mask"]) for row in expected)
    assert result.summary["collection_bc"]["diagnostic_only"] is True
    assert not result.summary["collection_bc"]["commands_sent"]
    assert digest(output / "reference/route.json") == digest(manifest_path)
    for entry in [*manifest["assets"].values(), *manifest["evidence"]]:
        assert digest(output / "reference" / entry["path"]) == entry["sha256"]
    # Exclude retention of even one full added serialized payload; this is a
    # measured regression assertion, not a runtime memory admission threshold.
    assert peak - baseline_peak < added_bytes, (
        "Preparation retained the growing serialized reference or diagnostic payload",
        baseline_peak,
        peak,
        added_bytes,
    )


@pytest.mark.parametrize("fault", ["asset", "evidence", "geometry", "path", "held_out_source"])
def test_reference_streaming_preserves_integrity_and_independence_rejection(tmp_path, fault):
    config, route = independent_reference(tmp_path)
    manifest_path = route / "route.json"
    manifest = json.loads(manifest_path.read_text())
    reference = route / manifest["assets"]["reference"]["path"]
    if fault == "asset":
        reference.write_bytes(b"changed after route was frozen")
    elif fault in ("evidence", "path"):
        evidence = route / "survey.txt" if fault == "evidence" else tmp_path / "outside.txt"
        evidence.write_text("Independent synthetic survey")
        manifest["evidence"].append(
            {
                "path": "survey.txt" if fault == "evidence" else "../outside.txt",
                "sha256": digest(evidence),
            }
        )
        if fault == "evidence":
            evidence.write_text("Changed after the evidence hash was frozen")
    elif fault == "geometry":
        data = json.loads(reference.read_text())
        data["points"][1]["s_m"] = -1
        reference.write_text(json.dumps(data))
        manifest["assets"]["reference"]["sha256"] = digest(reference)
    else:
        held_out = Path(json.loads(config.read_text())["sources"][-1]["recording"])
        manifest["source"]["session_sha256"] = digest(held_out / "session.json")
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_experiment(CollectionBCPrepare(config, output))
    assert not (output / "dataset.json").exists()
    assert not (output / "evaluation.json").exists()


def test_reference_copy_checks_the_bytes_written_before_publishing(tmp_path, corrupt_private_reads):
    config, route = independent_reference(tmp_path)
    reference = route / "reference.json"
    expected = digest(reference)
    output = tmp_path / "changed-copy"
    with corrupt_private_reads(reference, b'"version": 1', b'"version": 9') as changed:
        with pytest.raises(ValueError, match="changed during copy"):
            run_experiment(CollectionBCPrepare(config, output))
    assert changed
    assert digest(reference) == expected
    assert not (output / "dataset.json").exists()
    assert not (output / "evaluation.json").exists()


def test_reference_copy_io_failure_keeps_sources_and_does_not_publish_dataset(
    tmp_path, monkeypatch
):
    config, route = independent_reference(tmp_path)
    before = {path.name: digest(path) for path in route.iterdir() if path.is_file()}
    output = tmp_path / "failed-copy"
    original_open = Path.open

    def unavailable_destination(path, mode="r", *args, **kwargs):
        if path == output / "reference/reference.json" and mode == "xb":
            raise OSError("reference destination unavailable")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", unavailable_destination)
        with pytest.raises(OSError, match="reference destination unavailable"):
            run_experiment(CollectionBCPrepare(config, output))
    assert {path.name: digest(path) for path in route.iterdir() if path.is_file()} == before
    assert not (output / "dataset.json").exists()
    assert not (output / "evaluation.json").exists()
    retry = run_experiment(CollectionBCPrepare(config, tmp_path / "retry"))
    assert retry.summary["collection_bc"]["reference"]["status"] == "loaded"


@pytest.mark.parametrize("location", ["value", "key"])
@pytest.mark.parametrize("boundary_offset", range(7))
def test_discarded_reference_strings_validate_escapes_across_buffer_boundaries(
    tmp_path, location, boundary_offset
):
    config, route = independent_reference(tmp_path)
    path = route / "route.json"
    prefix = path.read_text().rstrip()[:-1] + ',"diagnostics":'
    if location == "key":
        prefix += "{"
    prefix += '"'
    padding = (io.DEFAULT_BUFFER_SIZE - 1 - len(prefix) - boundary_offset) % io.DEFAULT_BUFFER_SIZE
    # Place every character in the six-character Unicode escape at a read edge,
    # followed by all simple escapes, surrogate escapes, and raw Unicode text.
    text = prefix + "x" * padding + r"\u4e2d\"\\\/\b\f\n\r\t\ud83d\ude00道路 😀"
    text += '":null}}' if location == "key" else '"}'
    json.loads(text)  # Independent standard-library syntax oracle for the fixture.
    path.write_text(text, encoding="utf-8")
    output = tmp_path / "escaped"
    result = run_experiment(CollectionBCPrepare(config, output))
    assert result.summary["collection_bc"]["reference"]["status"] == "loaded"
    assert digest(output / "reference/route.json") == digest(path)
    decisions = json.loads((output / "dataset.json").read_text())["decisions"]
    assert any(any(row["views"]["reference_assisted"]["reference"]["mask"]) for row in decisions)


@pytest.mark.parametrize("location", ["value", "key"])
@pytest.mark.parametrize(
    "invalid",
    [
        '"raw\ncontrol"',
        '"raw\x00control"',
        r'"bad\q"',
        r'"bad\u12"',
        r'"bad\u12G4"',
        '"open',
        '"open\\',
    ],
    ids=["newline", "nul", "escape", "short_unicode", "invalid_unicode", "open", "open_escape"],
)
def test_discarded_reference_strings_still_reject_invalid_json(tmp_path, location, invalid):
    config, route = independent_reference(tmp_path)
    path = route / "route.json"
    prefix = path.read_text().rstrip()[:-1] + ',"diagnostics":'
    if location == "key":
        prefix += "{"
    # Exercise the same malformed text after a consumed block, rather than only
    # errors in the initial buffer. Whitespace here is outside the string.
    padding = (io.DEFAULT_BUFFER_SIZE - 1 - len(prefix)) % io.DEFAULT_BUFFER_SIZE
    suffix = "" if invalid.endswith("\\") else ":null}}" if location == "key" else "}"
    text = prefix + " " * padding + invalid + suffix
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    path.write_text(text, encoding="utf-8")
    output = tmp_path / "invalid-string"
    with pytest.raises(ValueError):
        run_experiment(CollectionBCPrepare(config, output))
    assert not (output / "dataset.json").exists()
    assert not (output / "evaluation.json").exists()
