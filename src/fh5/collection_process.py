"""Prepare a copied, non-editable collector environment without opening devices."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fh5.capture_config import parse_capture_config
from fh5.collection import CollectionConfig, CollectionControl
from fh5.collection_host import process_identity
from fh5.collection_store import atomic_json, read_bounded
from fh5.demonstrations import _profile

if TYPE_CHECKING:
    from fh5.experiment import RunResult

CollectionInstaller = Callable[[Path], None]


@dataclass(frozen=True)
class CollectionPrepare:
    repository: Path
    output_dir: Path
    capture_config: Path
    input_profile: Path
    config: CollectionConfig = field(default_factory=CollectionConfig)
    source: Literal["native", "synthetic"] = "native"
    port: int = 5300
    uv: Path | None = None
    offline: bool = False


@dataclass(frozen=True)
class CollectionStart:
    bundle: Path
    live: bool = False


def bundle_python(project: Path) -> Path:
    return project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _clean_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("PYTHON", "UV_"))
        and key.upper() not in ("VIRTUAL_ENV", "CONDA_PREFIX")
    }


def _install(request: CollectionPrepare, project: Path) -> None:
    uv = request.uv or shutil.which("uv")
    if uv is None:
        raise OSError("uv is required to prepare the frozen collector environment")
    command = [
        str(uv),
        "sync",
        "--project",
        str(project),
        "--locked",
        "--no-dev",
        "--no-editable",
        "--link-mode",
        "copy",
        "--python",
        sys.executable,
        "--no-python-downloads",
    ]
    if request.source == "native":
        command.extend(["--extra", "capture"])
    if request.offline:
        command.append("--offline")
    with (project.parent / "install.log").open("xb") as log:
        subprocess.run(
            command,
            cwd=project,
            env=_clean_environment(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=600,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )


def snapshot_files(project: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    total = 0
    for path in sorted(project.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(project.resolve()):
            raise ValueError("Frozen collector files must be copied, not linked")
        if path.is_dir():
            continue
        total += path.stat().st_size
        if len(result) >= 20000 or total > 2 * 1024**3:
            raise ValueError("Frozen collector exceeds file or byte budget")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while data := stream.read(1024**2):
                digest.update(data)
        result[path.relative_to(project).as_posix()] = digest.hexdigest()
    return result


def prepare_collection(
    request: CollectionPrepare, installer: CollectionInstaller | None = None
) -> RunResult:
    from fh5.experiment import RunResult

    capture_bytes = read_bounded(request.capture_config, 1024**2)
    capture, _ = parse_capture_config(json.loads(capture_bytes))
    profile_bytes = read_bounded(request.input_profile, 1024**2)
    profile = _profile(profile_bytes)
    if profile["calibration"]["status"] != "verified":
        raise ValueError("Frozen collection requires a verified input profile")
    request.config.validate_capture(capture)
    if (
        request.source not in ("native", "synthetic")
        or (request.source == "synthetic" and request.config.seconds > 60)
        or type(request.port) is not int
        or not 1024 <= request.port <= 65535
    ):
        raise ValueError("Frozen collection configuration or source mismatch")
    root, repository = request.output_dir.resolve(), request.repository.resolve()
    sources = repository / "src" / "fh5"
    source_paths = [p for p in sources.rglob("*") if p.suffix in (".py", ".html")]
    if not source_paths or len(source_paths) > 1000:
        raise ValueError("Collector source package missing or too large")
    root.mkdir(parents=True, exist_ok=False)
    project = root / "project"
    project.mkdir()
    try:
        for path in [repository / "pyproject.toml", repository / "uv.lock", *source_paths]:
            relative = path.relative_to(repository)
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(read_bounded(path, 16 * 1024**2))
        (project / "capture.json").write_bytes(capture_bytes)
        (project / "input-profile.json").write_bytes(profile_bytes)
        if installer is None:
            _install(request, project)
        else:
            installer(project)
        python = bundle_python(project)
        runtime = json.loads(
            subprocess.check_output(
                [
                    str(python),
                    "-I",
                    "-B",
                    "-c",
                    "import sys,json,importlib.metadata as m,fh5; "
                    "print(json.dumps(dict(version=sys.version,base_prefix=sys.base_prefix,"
                    "prefix=sys.prefix,package=fh5.__file__,"
                    "packages=sorted((d.metadata['Name'],d.version) for d in m.distributions()))))",
                ],
                cwd=project,
                env=_clean_environment(),
                timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        )
        if not Path(runtime["package"]).resolve().is_relative_to(project / ".venv"):
            raise ValueError("Collector package resolved outside its isolated environment")
        manifest = {
            "version": 1,
            "kind": "frozen-passive-collection-v1",
            "source": request.source,
            "port": request.port,
            "collection": {**asdict(request.config), "pixels": request.config.pixels.metadata()},
            "installer": "uv_locked_copy" if installer is None else "injected_installer",
            "runtime": runtime,
            "files": snapshot_files(project),
            "commands_sent": False,
        }
        atomic_json(root / "frozen.json", manifest)
    except Exception as error:
        atomic_json(root / "prepare-failed.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    return RunResult(
        {}, [], [], {"collection": {"state": "prepared", **manifest}}, root / "frozen.json"
    )


def verify_bundle(root: Path) -> tuple[dict[str, Any], str]:
    payload = read_bounded(root / "frozen.json", 8 * 1024**2)
    manifest = json.loads(payload)
    if (
        manifest.get("kind") != "frozen-passive-collection-v1"
        or manifest.get("version") != 1
        or manifest.get("commands_sent") is not False
        or snapshot_files(root / "project") != manifest["files"]
    ):
        raise ValueError("Frozen collector snapshot changed or is unsupported")
    return manifest, hashlib.sha256(payload).hexdigest()


def start_collection(request: CollectionStart) -> RunResult:
    from fh5.experiment import RunResult

    root = request.bundle.resolve()
    manifest, digest = verify_bundle(root)
    if manifest["source"] == "native" and not request.live:
        raise ValueError("Native collection requires explicit --live; preparation opens no devices")
    if manifest["source"] == "native" and manifest["installer"] != "uv_locked_copy":
        raise ValueError("Native collection requires locked, independently installed dependencies")
    if (root / "recording").exists():
        raise FileExistsError(root / "recording")
    token = uuid.uuid4().hex
    with (root / "start.claim").open("x", encoding="utf-8") as claim:
        claim.write(token)
    try:
        with (root / "worker.log").open("xb") as log:
            process = subprocess.Popen(
                [
                    str(bundle_python(root / "project")),
                    "-I",
                    "-B",
                    "-m",
                    "fh5.collection_worker",
                    str(root),
                    digest,
                    token,
                ],
                cwd=root,
                env=_clean_environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                close_fds=True,
                creationflags=(
                    subprocess.DETACHED_PROCESS
                    | subprocess.CREATE_NEW_PROCESS_GROUP
                    | subprocess.CREATE_NO_WINDOW
                )
                if os.name == "nt"
                else 0,
                start_new_session=os.name != "nt",
            )
        identity = process_identity(process.pid)
        value = {
            "pid": process.pid,
            "birth": identity["birth"],
            "token": token,
            "manifest_sha256": digest,
            "state": "launched",
            "commands_sent": False,
        }
        atomic_json(root / "process.json", value)
    except Exception as error:
        atomic_json(root / "launch-failed.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    return RunResult({}, [], [], {"collection": value}, root / "process.json")


def _process_liveness(process: dict[str, Any]) -> str:
    identity = process_identity(process["pid"])
    state = str(identity["state"])
    if state == "running" and process.get("birth") is None:
        return "unknown"
    if state == "running" and identity["birth"] != process["birth"]:
        return "exited"  # PID was reused; the original process is gone.
    return state


def control_bundle(request: CollectionControl) -> RunResult:
    from fh5.collection_runtime import control_collection
    from fh5.experiment import RunResult

    root = request.recording_dir
    value: dict[str, Any] = {
        "state": "prepared",
        "commands_sent": False,
        "process_liveness": "not_started",
        "final_status_present": False,
    }
    if request.stop:
        (root / "stop.request").touch(exist_ok=True)
    recording = root / "recording"
    if (recording / "session.json").is_file():
        value.update(control_collection(CollectionControl(recording)).summary["collection"])
    process_path = root / "process.json"
    process = None
    if process_path.is_file():
        process = json.loads(read_bounded(process_path, 16384))
        state = _process_liveness(process)
        value.update(
            process_liveness=state,
            pid=process["pid"],
            process_birth=process["birth"],
            process_role="launcher",
            launcher_pid=process["pid"],
            launcher_liveness=state,
        )
    for filename in ("worker-state.json", "launch-failed.json"):
        if (root / filename).is_file():
            child = json.loads(read_bounded(root / filename, 65536))
            value[filename.removesuffix(".json")] = child
            if filename == "worker-state.json":
                matching = (
                    process is not None
                    and child.get("token") == process["token"]
                    and child.get("manifest_sha256") == process["manifest_sha256"]
                )
                value["worker_identity_matches"] = matching
                value["software_snapshot_verified"] = matching and child.get(
                    "software_snapshot_verified", False
                )
                if matching:
                    value.update(
                        process_liveness=_process_liveness(child),
                        pid=child["pid"],
                        process_birth=child.get("birth"),
                        process_role="worker",
                    )
                else:
                    value["process_liveness"] = "unknown"
    value["stop_requested"] = (root / "stop.request").exists()
    value["abnormal_exit"] = (
        value["process_liveness"] == "exited" and not value["final_status_present"]
    )
    if value["abnormal_exit"]:
        value.update(state="interrupted_or_start_failed", complete=False)
    return RunResult({}, [], [], {"collection": value}, process_path)
