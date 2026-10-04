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

from fh5.artifacts.io import atomic_json, sha256_file
from fh5.capture.config import parse_capture_config
from fh5.collection.demonstrations import _profile
from fh5.collection.host import process_identity
from fh5.collection.model import CollectionConfig, CollectionControl
from fh5.collection.status import (
    PROCESS_FIELDS,
    STATUS_FIELDS,
    control_source,
    read_collection_session,
    read_control_status,
)
from fh5.collection.store import atomic_control_json

if TYPE_CHECKING:
    from fh5.result import RunResult

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
    output_dir: Path | None = None


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
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )


def collector_files(project: Path, package: Path) -> dict[str, str]:
    """Bind executable collector files, not the surrounding Python installation."""
    result: dict[str, str] = {}
    for path in sorted(package.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(project.resolve()):
            raise ValueError("Frozen collector files must be copied, not linked")
        if path.is_dir():
            continue
        result[path.relative_to(project).as_posix()] = sha256_file(path)
    return result


def prepare_collection(
    request: CollectionPrepare, installer: CollectionInstaller | None = None
) -> RunResult:
    from fh5.result import RunResult

    capture_bytes = request.capture_config.read_bytes()
    capture, _ = parse_capture_config(json.loads(capture_bytes))
    profile_bytes = request.input_profile.read_bytes()
    profile = _profile(profile_bytes)
    if profile["calibration"]["status"] != "verified":
        raise ValueError("Frozen collection requires a verified input profile")
    request.config.validate_capture(capture)
    if (
        request.source not in ("native", "synthetic")
        or type(request.port) is not int
        or not 1024 <= request.port <= 65535
    ):
        raise ValueError("Frozen collection configuration or source mismatch")
    root, repository = request.output_dir.resolve(), request.repository.resolve()
    sources = repository / "src" / "fh5"
    source_paths = [p for p in sources.rglob("*") if p.suffix in (".py", ".html")]
    if not source_paths:
        raise ValueError("Collector source package missing")
    root.mkdir(parents=True, exist_ok=False)
    project = root / "project"
    project.mkdir()
    try:
        for path in [repository / "pyproject.toml", repository / "uv.lock", *source_paths]:
            relative = path.relative_to(repository)
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
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
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        )
        if not Path(runtime["package"]).resolve().is_relative_to(project / ".venv"):
            raise ValueError("Collector package resolved outside its isolated environment")
        package = Path(runtime["package"]).resolve().parent
        manifest = {
            "version": 3,
            "kind": "frozen-passive-collection-v3",
            "project": str(project),
            "source": request.source,
            "port": request.port,
            "collection": {**asdict(request.config), "pixels": request.config.pixels.metadata()},
            "installer": "uv_locked_copy" if installer is None else "injected_installer",
            "runtime": runtime,
            "collector_package": package.relative_to(project).as_posix(),
            "files": collector_files(project, package),
            "inputs": {
                name: sha256_file(project / name) for name in ("capture.json", "input-profile.json")
            },
            "commands_sent": False,
        }
        atomic_json(root / "frozen.json", manifest)
    except Exception as error:
        atomic_json(root / "prepare-failed.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    return RunResult(
        {}, [], [], {"collection": {"state": "prepared", **manifest}}, root / "frozen.json"
    )


def collector_project(root: Path, manifest: dict[str, Any]) -> Path:
    """Resolve the fixed installation independently of a recording's output directory."""
    if manifest["version"] == 2:
        return root / "project"
    project = Path(manifest["project"])
    if not project.is_absolute():
        raise ValueError("Frozen collector installation requires an absolute project path")
    return project


def read_bundle(root: Path) -> tuple[dict[str, Any], str]:
    """Read the launch identity and verify the two inputs before opening devices."""
    payload = (root / "frozen.json").read_bytes()
    manifest = json.loads(payload)
    if (manifest.get("version"), manifest.get("kind")) not in (
        (2, "frozen-passive-collection-v2"),
        (3, "frozen-passive-collection-v3"),
    ) or manifest.get("commands_sent") is not False:
        raise ValueError("Unsupported collector bundle; run collection-prepare in a new directory")
    project = collector_project(root, manifest)
    inputs = {name: sha256_file(project / name) for name in ("capture.json", "input-profile.json")}
    if inputs != manifest["inputs"]:
        raise ValueError("Frozen collector input snapshot changed")
    return manifest, hashlib.sha256(payload).hexdigest()


def verify_bundle(root: Path) -> tuple[dict[str, Any], str]:
    """Check the installed collector once at launch; dependencies stay isolated by copy."""
    manifest, digest = read_bundle(root)
    project = collector_project(root, manifest).resolve()
    package = (project / manifest["collector_package"]).resolve()
    if (
        not package.is_relative_to(project / ".venv")
        or collector_files(project, package) != manifest["files"]
    ):
        raise ValueError("Frozen collector code snapshot changed")
    return manifest, digest


def start_collection(request: CollectionStart) -> RunResult:
    from fh5.result import RunResult

    bundle = request.bundle.resolve()
    manifest, digest = verify_bundle(bundle)
    if manifest["source"] == "native" and not request.live:
        raise ValueError("Native collection requires explicit --live; preparation opens no devices")
    if manifest["source"] == "native" and manifest["installer"] != "uv_locked_copy":
        raise ValueError("Native collection requires locked, independently installed dependencies")
    installed = Path(manifest["collector_package"])
    if (installed / "collection/worker.py").as_posix() in manifest["files"]:
        worker_module = "fh5.collection.worker"
    elif (installed / "collection_worker.py").as_posix() in manifest["files"]:
        worker_module = "fh5.collection_worker"
    else:
        raise ValueError("Frozen collector worker is missing from its verified installation")
    project = collector_project(bundle, manifest).resolve()
    root = request.output_dir.resolve() if request.output_dir is not None else bundle
    if root != bundle:
        if manifest["version"] == 2:
            raise ValueError("Separate output requires a new collection-prepare installation")
        if root.is_relative_to(project):
            raise ValueError("Collection output must be outside the frozen installation")
        root.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(bundle / "frozen.json", root / "frozen.json")
    if (root / "recording").exists():
        raise FileExistsError(root / "recording")
    token = uuid.uuid4().hex
    with (root / "start.claim").open("x", encoding="utf-8") as claim:
        claim.write(token)
    try:
        with (root / "worker.log").open("xb") as log:
            process = subprocess.Popen(
                [
                    str(bundle_python(project)),
                    "-I",
                    "-B",
                    "-m",
                    worker_module,
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
        atomic_control_json(root / "process.json", value)
    except Exception as error:
        atomic_control_json(
            root / "launch-failed.json", {"error": f"{type(error).__name__}: {error}"}
        )
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


def control_collection(request: CollectionControl) -> RunResult:
    if (request.recording_dir / "frozen.json").is_file():
        return control_bundle(request)
    return _control_recording(request)


def _control_recording(request: CollectionControl) -> RunResult:
    from fh5.result import RunResult

    root = request.recording_dir
    read_collection_session(root / "session.json")
    final = root / "final.json"
    status = final if final.exists() else root / "status.json"
    if status.is_file():
        value = read_control_status(status, STATUS_FIELDS)
    else:
        value = {"state": "starting", "commands_sent": False}
    if request.stop and not final.exists():
        (root / "stop.request").touch(exist_ok=True)
    value.update(
        stop_requested=(root / "stop.request").exists(),
        process_liveness="not_checked; status file alone does not prove a running process",
        final_status_present=final.exists(),
        source_documents={
            "session": control_source(root / "session.json"),
            "status": control_source(status),
        },
    )
    return RunResult({"source_kind": "collection_control"}, [], [], {"collection": value}, status)


def control_bundle(request: CollectionControl) -> RunResult:
    from fh5.result import RunResult

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
        value.update(_control_recording(CollectionControl(recording)).summary["collection"])
    process_path = root / "process.json"
    process = None
    if process_path.is_file():
        process = read_control_status(process_path, PROCESS_FIELDS)
        value.setdefault("source_documents", {})["process"] = control_source(process_path)
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
            child = read_control_status(root / filename, PROCESS_FIELDS)
            value[filename.removesuffix(".json")] = child
            value.setdefault("source_documents", {})[filename.removesuffix(".json")] = (
                control_source(root / filename)
            )
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
