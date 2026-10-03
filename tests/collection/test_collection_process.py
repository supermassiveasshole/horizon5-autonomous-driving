"""Frozen independent collector lifecycle through the experiment interface."""

import hashlib
import json
import os
import py_compile
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fh5.collection.model import CollectionControl, CollectionReview
from fh5.experiment import run_experiment
from tests.collection.test_collection import request


def install_test_environment(project, *, flat_worker=False):
    # External installer substitute: a real isolated interpreter with a copied
    # package, no system site packages and no real acquisition dependencies.
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(project / ".venv")], check=True
    )
    python = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    site = Path(
        subprocess.check_output(
            [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
    )
    shutil.copytree(project / "src" / "fh5", site / "fh5")
    if flat_worker:
        # Older frozen collectors start at this installed module path. Keep the
        # actual worker implementation so this exercises its whole lifecycle.
        (site / "fh5/collection/worker.py").rename(site / "fh5/collection_worker.py")


def prepare(tmp_path, seconds=3, source="synthetic", *, flat_worker=False):
    from fh5.collection.process import CollectionPrepare

    req = request(tmp_path, seconds=seconds, block_rows=20)
    capture = json.loads(Path("configs/capture-dxgi.example.json").read_bytes())
    capture["pixels"] = req.config.pixels.metadata()
    config_path = tmp_path / "capture.json"
    config_path.write_text(json.dumps(capture))
    bundle = tmp_path / "bundle"
    developer = tmp_path / "developer"
    shutil.copytree(
        Path("src/fh5"), developer / "src" / "fh5", ignore=shutil.ignore_patterns("__pycache__")
    )
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(name, developer / name)
    result = run_experiment(
        CollectionPrepare(
            developer, bundle, config_path, req.input_profile, req.config, source=source
        ),
        collection_installer=lambda project: install_test_environment(
            project, flat_worker=flat_worker
        ),
    )
    return bundle, config_path, result


def test_prepare_freezes_copied_source_config_and_isolated_dependencies_without_devices(tmp_path):
    bundle, config, result = prepare(tmp_path)
    value = result.summary["collection"]
    assert value["state"] == "prepared" and not (bundle / "recording").exists()
    assert not value["commands_sent"] and not (bundle / "process.json").exists()
    config.write_text("changed developer configuration")
    frozen = json.loads((bundle / "frozen.json").read_bytes())
    assert frozen["source"] == "synthetic" and frozen["files"]
    assert frozen["version"] == 3
    assert Path(frozen["project"]) == (bundle / "project").resolve()
    assert set(frozen["inputs"]) == {"capture.json", "input-profile.json"}
    assert all(name.startswith(frozen["collector_package"] + "/") for name in frozen["files"])
    assert json.loads((bundle / "project" / "capture.json").read_bytes())["version"] == 1
    python = (
        bundle / "project" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    module_path = subprocess.check_output(
        [str(python), "-I", "-c", "import fh5; print(fh5.__file__)"], text=True
    ).strip()
    assert Path(module_path).is_relative_to(bundle / "project" / ".venv")


@pytest.mark.parametrize("flat_worker", [False, True])
def test_prepared_collector_runs_twice_without_reinstalling_or_changing_the_first_run(
    tmp_path, flat_worker
):
    from fh5.collection.process import CollectionStart

    bundle, config, _ = prepare(tmp_path, seconds=15, flat_worker=flat_worker)
    frozen_bytes = (bundle / "frozen.json").read_bytes()
    manifest_sha256 = hashlib.sha256(frozen_bytes).hexdigest()
    frozen = json.loads(frozen_bytes)
    forbidden = bundle / "project" / "recording-run"
    with pytest.raises(ValueError, match="outside the frozen installation"):
        run_experiment(CollectionStart(bundle, output_dir=forbidden))
    assert not forbidden.exists()
    cache = bundle / "project/.venv/installer-cache"
    cache.mkdir()
    (cache / "last-check.txt").write_text("Unrelated installer cache is not collection input")
    python = (
        bundle / "project" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    sessions, first_run = [], {}
    for index, output in enumerate((tmp_path / "run-a", tmp_path / "run-b")):
        if index:
            (tmp_path / "developer/src/fh5/collection/worker.py").write_text(
                "raise RuntimeError('developer edits must not run')"
            )
            config.write_text("developer config changed between collection runs")
        state = {}
        try:
            if index == 0:
                # This launcher exits before we query the detached worker.
                launched = json.loads(
                    subprocess.check_output(
                        [
                            str(python),
                            "-I",
                            "-B",
                            "-m",
                            "fh5",
                            "collection-start",
                            str(bundle),
                            "--output",
                            str(output),
                        ],
                        text=True,
                        timeout=10,
                    )
                )
            else:
                launched = run_experiment(CollectionStart(bundle, output_dir=output)).summary[
                    "collection"
                ]
            assert launched["pid"] != os.getpid() and launched["state"] == "launched"
            assert launched["manifest_sha256"] == manifest_sha256
            deadline = time.monotonic() + 7
            while time.monotonic() < deadline:
                state = run_experiment(CollectionControl(output)).summary["collection"]
                if state.get("seen_rows", 0) >= 40:
                    break
                time.sleep(0.05)
            assert state["process_liveness"] == "running"
            assert state["seen_rows"] >= 40 and not state["commands_sent"]
            assert not state["stop_requested"]
            with pytest.raises(FileExistsError):
                run_experiment(CollectionStart(bundle, output_dir=output))
            live_review = run_experiment(
                CollectionReview(output / "recording", tmp_path / f"live-{index}.html")
            )
            assert live_review.summary["collection"]["verified_blocks"] > 0
        finally:
            if (output / "start.claim").exists():
                run_experiment(CollectionControl(output, stop=True))
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    state = run_experiment(CollectionControl(output)).summary["collection"]
                    if state["process_liveness"] == "exited":
                        break
                    time.sleep(0.05)
                if state["process_liveness"] == "running":
                    # Only the synthetic worker identified by this run can remain.
                    os.kill(state["pid"], signal.SIGTERM)
        assert state["process_liveness"] == "exited" and state["final_status_present"]
        assert state["complete"] and state["stop_reason"] == "requested_stop"
        assert state["software_snapshot_verified"]
        review = run_experiment(
            CollectionReview(output / "recording", tmp_path / f"sealed-{index}.html")
        ).summary["collection"]
        assert review["complete"] and review["verified_blocks"] >= 2
        assert (output / "frozen.json").read_bytes() == frozen_bytes
        session = json.loads((output / "recording/session.json").read_bytes())
        assert session["software_snapshot"]["manifest_sha256"] == manifest_sha256
        assert session["software_snapshot"]["runtime"] == frozen["runtime"]
        assert (
            Path(session["software_snapshot"]["runtime"]["prefix"]).resolve()
            == (bundle / "project/.venv").resolve()
        )
        sessions.append(session["session_id"])
        if index == 0:
            first_run = {
                output / name: (output / name).read_bytes()
                for name in (
                    "recording/session.json",
                    "recording/final.json",
                    "stop.request",
                    "process.json",
                )
            }
        else:
            assert all(path.read_bytes() == original for path, original in first_run.items())
        assert not any(
            (bundle / name).exists()
            for name in (
                "start.claim",
                "worker.log",
                "process.json",
                "worker-state.json",
                "recording",
                "stop.request",
            )
        )
        assert (bundle / "frozen.json").read_bytes() == frozen_bytes
    assert len(set(sessions)) == 2


@pytest.mark.parametrize("changed", ["capture.json", "input-profile.json", "installed-collector"])
def test_modified_frozen_asset_is_rejected_before_a_worker_starts(tmp_path, changed):
    from fh5.collection.process import CollectionStart

    bundle, _, result = prepare(tmp_path)
    path = (
        Path(result.summary["collection"]["runtime"]["package"])
        if changed == "installed-collector"
        else bundle / "project" / changed
    )
    with path.open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="snapshot changed"):
        run_experiment(CollectionStart(bundle))
    assert not (bundle / "process.json").exists() and not (bundle / "recording").exists()


def test_old_prepared_bundle_requests_reprepare_without_rewriting_it(tmp_path):
    from fh5.collection.process import CollectionStart

    bundle = tmp_path / "old-bundle"
    bundle.mkdir()
    path = bundle / "frozen.json"
    original = json.dumps({"version": 1, "kind": "frozen-passive-collection-v1"})
    path.write_text(original)
    with pytest.raises(ValueError, match="collection-prepare in a new directory"):
        run_experiment(CollectionStart(bundle))
    assert path.read_text() == original
    assert not (bundle / "start.claim").exists()


def test_native_bundle_requires_explicit_live_start_and_locked_installer(tmp_path):
    from fh5.collection.process import CollectionStart

    bundle, _, _ = prepare(tmp_path, source="native")
    with pytest.raises(ValueError, match="explicit --live"):
        run_experiment(CollectionStart(bundle))
    with pytest.raises(ValueError, match="locked, independently installed"):
        run_experiment(CollectionStart(bundle, live=True))
    assert not (bundle / "start.claim").exists()


def test_unrecorded_executable_bytecode_is_rejected_before_start(tmp_path):
    from fh5.collection.process import CollectionStart

    bundle, _, _ = prepare(tmp_path)
    module = bundle / "project/.venv/Lib/site-packages/fh5/collection/worker.py"
    if os.name != "nt":
        module = next(
            (bundle / "project/.venv/lib").glob("*/site-packages/fh5/collection/worker.py")
        )
    original, stat = module.read_bytes(), module.stat()
    module.write_bytes(b"raise RuntimeError('unverified cached worker')\n".ljust(len(original)))
    os.utime(module, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    py_compile.compile(str(module), doraise=True)
    module.write_bytes(original)
    os.utime(module, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    try:
        with pytest.raises(ValueError, match="snapshot changed"):
            run_experiment(CollectionStart(bundle))
        assert not (bundle / "start.claim").exists()
    finally:
        if (bundle / "start.claim").exists():
            run_experiment(CollectionControl(bundle, stop=True))


def test_terminated_worker_is_not_mistaken_for_live_heartbeat_and_seals_recover(tmp_path):
    from fh5.collection.process import CollectionStart

    bundle, _, _ = prepare(tmp_path, seconds=15)
    manifest_path = bundle / "frozen.json"
    legacy = json.loads(manifest_path.read_bytes())
    legacy.update(version=2, kind="frozen-passive-collection-v2")
    legacy.pop("project")
    manifest_path.write_text(json.dumps(legacy))
    original_manifest = manifest_path.read_bytes()
    separate = tmp_path / "separate"
    with pytest.raises(ValueError, match="collection-prepare"):
        run_experiment(CollectionStart(bundle, output_dir=separate))
    assert manifest_path.read_bytes() == original_manifest
    assert not separate.exists() and not (bundle / "start.claim").exists()
    run_experiment(CollectionStart(bundle))
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = run_experiment(CollectionControl(bundle)).summary["collection"]
            if state.get("sealed_blocks", 0) >= 2:
                break
            time.sleep(0.05)
        assert state["process_liveness"] == "running" and state["sealed_blocks"] >= 2
        worker = json.loads((bundle / "worker-state.json").read_bytes())
        assert state["pid"] == worker["pid"]
        with pytest.raises(FileExistsError):
            run_experiment(CollectionStart(bundle))
        # This is the synthetic child just created and verified by OS identity.
        os.kill(state["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = run_experiment(CollectionControl(bundle)).summary["collection"]
            if state["process_liveness"] == "exited":
                break
            time.sleep(0.05)
        assert state["process_liveness"] == "exited"
        assert state["abnormal_exit"] and not state["complete"]
        recovered = run_experiment(
            CollectionReview(bundle / "recording", tmp_path / "recovered.html")
        )
        assert recovered.summary["collection"]["verified_blocks"] >= 2
        assert not recovered.summary["collection"]["complete"]
    finally:
        run_experiment(CollectionControl(bundle, stop=True))


def test_worker_reverification_failure_reports_the_actual_failed_process(tmp_path, monkeypatch):
    from fh5.collection.process import CollectionStart

    bundle, _, _ = prepare(tmp_path)
    start_process = subprocess.Popen

    def change_at_launch(*args, **kwargs):
        # Real process boundary: parent verification has finished, child has not
        # started. A second verification must reject this otherwise valid JSON.
        with (bundle / "project/capture.json").open("a") as stream:
            stream.write(" ")
        return start_process(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", change_at_launch)
    run_experiment(CollectionStart(bundle))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = run_experiment(CollectionControl(bundle)).summary["collection"]
        if "worker-state" in state and state["launcher_liveness"] == "exited":
            break
        time.sleep(0.02)
    assert state["worker-state"]["state"] == "failed"
    assert state["process_liveness"] == "exited"
    assert state["state"] == "interrupted_or_start_failed" and state["abnormal_exit"]
    assert not state["complete"] and not state["software_snapshot_verified"]
    assert not (bundle / "recording").exists()
