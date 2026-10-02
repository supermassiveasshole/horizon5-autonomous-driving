"""Frozen independent collector lifecycle through the experiment interface."""

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
from test_collection import request

from fh5.collection import CollectionControl, CollectionReview
from fh5.experiment import run_experiment


def install_test_environment(project):
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


def prepare(tmp_path, seconds=3, source="synthetic"):
    from fh5.collection_process import CollectionPrepare

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
        collection_installer=install_test_environment,
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
    assert frozen["version"] == 2
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


def test_detached_collector_survives_launcher_and_can_be_queried_stopped_and_replayed(tmp_path):
    bundle, config, _ = prepare(tmp_path, seconds=15)
    cache = bundle / "project/.venv/installer-cache"
    cache.mkdir()
    (cache / "last-check.txt").write_text("Unrelated installer cache is not collection input")
    (tmp_path / "developer/src/fh5/collection_worker.py").write_text(
        "raise RuntimeError('developer edits must not run')"
    )
    python = (
        bundle / "project" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    # This launcher process actually exits; the collector must outlive it.
    launched = json.loads(
        subprocess.check_output(
            [str(python), "-I", "-B", "-m", "fh5", "collection-start", str(bundle)],
            text=True,
            timeout=10,
        )
    )
    assert launched["pid"] != os.getpid() and launched["state"] == "launched"
    config.write_text("developer config changed while collector runs")
    deadline = time.monotonic() + 7
    state = {}
    try:
        while time.monotonic() < deadline:
            state = run_experiment(CollectionControl(bundle)).summary["collection"]
            if state.get("seen_rows", 0) >= 40:
                break
            time.sleep(0.05)
        assert state["process_liveness"] == "running"
        assert state["seen_rows"] >= 40 and not state["commands_sent"]
        live_review = run_experiment(CollectionReview(bundle / "recording", tmp_path / "live.html"))
        assert live_review.summary["collection"]["verified_blocks"] > 0
    finally:
        run_experiment(CollectionControl(bundle, stop=True))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = run_experiment(CollectionControl(bundle)).summary["collection"]
        if state.get("final_status_present") and state["process_liveness"] == "exited":
            break
        time.sleep(0.05)
    assert state["process_liveness"] == "exited" and state["complete"]
    assert state["stop_reason"] == "requested_stop"
    assert state["software_snapshot_verified"]


@pytest.mark.parametrize("changed", ["capture.json", "input-profile.json", "installed-collector"])
def test_modified_frozen_asset_is_rejected_before_a_worker_starts(tmp_path, changed):
    from fh5.collection_process import CollectionStart

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
    from fh5.collection_process import CollectionStart

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
    from fh5.collection_process import CollectionStart

    bundle, _, _ = prepare(tmp_path, source="native")
    with pytest.raises(ValueError, match="explicit --live"):
        run_experiment(CollectionStart(bundle))
    with pytest.raises(ValueError, match="locked, independently installed"):
        run_experiment(CollectionStart(bundle, live=True))
    assert not (bundle / "start.claim").exists()


def test_unrecorded_executable_bytecode_is_rejected_before_start(tmp_path):
    from fh5.collection_process import CollectionStart

    bundle, _, _ = prepare(tmp_path)
    module = bundle / "project/.venv/Lib/site-packages/fh5/collection_worker.py"
    if os.name != "nt":
        module = next(
            (bundle / "project/.venv/lib").glob("*/site-packages/fh5/collection_worker.py")
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
    from fh5.collection_process import CollectionStart

    bundle, _, _ = prepare(tmp_path, seconds=15)
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
    from fh5.collection_process import CollectionStart

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
