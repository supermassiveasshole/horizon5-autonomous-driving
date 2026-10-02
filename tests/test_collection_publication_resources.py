"""Control publication must not weaken durable collection index recovery."""

import ctypes
import json
import os
from pathlib import Path

import pytest
from test_collection import Stream, input_at, request

from fh5.collection import CollectionReview
from fh5.experiment import run_experiment


@pytest.mark.skipif(os.name != "nt", reason="ReplaceFileW is a Windows publication API")
def test_durable_collection_index_never_uses_control_only_partial_replacement(
    tmp_path, monkeypatch
):
    req = request(tmp_path, block_rows=1)
    index_path = req.output_dir / "index.json"
    partial_replacements = []
    windll = ctypes.WinDLL

    class ReplaceCall:
        def __init__(self, operation):
            self.operation = operation

        @property
        def argtypes(self):
            return self.operation.argtypes

        @argtypes.setter
        def argtypes(self, value):
            self.operation.argtypes = value

        @property
        def restype(self):
            return self.operation.restype

        @restype.setter
        def restype(self, value):
            self.operation.restype = value

        def __call__(self, target, source, *args):
            target_path = Path(target)
            if target_path.name == "index.json" and target_path.samefile(index_path):
                partial_replacements.append(index_path.read_bytes())
                # Documented Win32 error 1176 with no backup: the old target
                # disappeared and the new source remains under its temporary
                # name. This is an external OS-API fault, never a learner mock.
                target_path.unlink()
                ctypes.set_last_error(1176)
                return 0
            return self.operation(target, source, *args)

    class Kernel:
        def __init__(self, library):
            self.library = library
            self.ReplaceFileW = ReplaceCall(library.ReplaceFileW)

        def __getattr__(self, name):
            return getattr(self.library, name)

    def inject_partial_replacement(name, *args, **kwargs):
        library = windll(name, *args, **kwargs)
        if str(name).lower() in ("kernel32", "kernel32.dll"):
            return Kernel(library)
        return library

    monkeypatch.setattr(ctypes, "WinDLL", inject_partial_replacement)
    result = run_experiment(
        req, collection_environment=Stream([input_at(250), input_at(300)])
    ).summary["collection"]
    assert partial_replacements == []
    assert result["archive_error"] is None and result["complete"] is True
    assert result["written_rows"] == 2 and result["sealed_blocks"] == 2
    index = json.loads(index_path.read_bytes())
    assert len(index["blocks"]) == 2
    recovered = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html"))
    assert recovered.summary["collection"]["verified_blocks"] == 2
    assert recovered.summary["collection"]["complete"] is True


def test_failed_durable_index_replacement_keeps_previous_state_and_sealed_blocks(
    tmp_path, monkeypatch
):
    req = request(tmp_path, block_rows=1)
    index_path = req.output_dir / "index.json"
    prior_publications = []
    replacing = os.replace

    def failed_index_replace(source, target, *args, **kwargs):
        if Path(target) == index_path and index_path.exists():
            prior_publications.append(index_path.read_bytes())
            raise OSError("injected durable index publication failure")
        return replacing(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", failed_index_replace)
    result = run_experiment(
        req, collection_environment=Stream([input_at(250), input_at(300)])
    ).summary["collection"]
    assert len(prior_publications) == 1
    assert index_path.read_bytes() == prior_publications[0]
    assert len(json.loads(index_path.read_bytes())["blocks"]) == 1
    assert index_path.with_suffix(".tmp").is_file()
    assert result["stop_reason"] == "archive_failure" and result["complete"] is False
    assert "injected durable index publication failure" in result["archive_error"]
    recovered = run_experiment(CollectionReview(req.output_dir, tmp_path / "review.html"))
    evidence = recovered.summary["collection"]
    assert evidence["verified_blocks"] == 2 and evidence["errors"] == []
    assert evidence["complete"] is False
    assert (
        "injected durable index publication failure" in evidence["source_status"]["archive_error"]
    )
