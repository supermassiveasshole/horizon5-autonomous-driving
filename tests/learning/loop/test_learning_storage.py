"""Storage dependency accounting through the public experiment-run interface."""

import json
import os
import shutil
import sqlite3
import subprocess
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from fh5.cli import main
from fh5.experiment import run_experiment
from fh5.learning.storage import LearningStoragePlan
from tests.driving.test_event_run import verified_config
from tests.evaluation.test_candidate_store import candidates as candidates
from tests.evaluation.test_evaluation import sha
from tests.learning.loop.test_learning_loop import SharedBackend, loop_request
from tests.learning.loop.test_learning_loop import seeded_loop as seeded_loop
from tests.support.storage_files import storage_files


@pytest.fixture(scope="module")
def recorded_storage(tmp_path_factory, seeded_loop):
    root = tmp_path_factory.mktemp("storage-session")
    request = loop_request(root, seeded_loop, rounds=1)
    result = run_experiment(request, learning_environment=SharedBackend(seeded_loop[0])).summary[
        "learning_loop"
    ]
    assert result["stop_reason"] == "budget_completed"
    return request.output_dir, tmp_path_factory.getbasetemp()


def storage_request(tmp_path, recorded_storage, budget_bytes=2**40):
    run, scope = recorded_storage
    config = tmp_path / "storage.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "root": str(scope),
                "learning": {"directory": str(run), "state_sha256": sha(run / "state.json")},
                "budget_bytes": budget_bytes,
            }
        )
    )
    return LearningStoragePlan(config, tmp_path / "storage-plan")


def test_plan_protects_shared_learners_and_the_original_evaluation_without_mutation(
    tmp_path, recorded_storage
):
    run, scope = recorded_storage
    state = json.loads((run / "state.json").read_bytes())
    state_hash = sha(run / "state.json")
    operation = storage_request(tmp_path, recorded_storage)
    plan = run_experiment(operation).summary["storage"]
    files = list(storage_files(operation.output_dir, plan))
    assert plan["status"] == "within_budget"
    assert plan["files_deleted"] == 0
    assert plan["cleanup_authorized"] is False
    paths = [entry["path"] for entry in files]
    assert len(paths) == len(set(paths))
    weight = Path(state["latest_learner"]["directory"]) / "policy.pt"
    entry = next(row for row in files if row["path"] == str(weight.resolve()))
    assert {"latest_learner", "explorer"} <= set(entry["roles"])
    assert entry["bytes"] == weight.stat().st_size
    manifest = json.loads((weight.parent / "policy.json").read_bytes())
    history_node = (weight.parent / manifest["history"]["head"]["path"]).resolve()
    node_entry = next(row for row in files if row["path"] == str(history_node))
    assert {"latest_learner", "metadata"} <= set(node_entry["roles"])
    assert node_entry["bytes"] == history_node.stat().st_size
    for original in (
        run / "state.json",
        run / "round-000/evaluation/attempt-0000/recording/packets.jsonl",
        next((run / "round-000/evaluation/attempt-0000/execution/pixels").glob("*.rgb")),
        run / "round-000/evaluation/attempt-0000/ready/start-manifest.json",
    ):
        assert str(original.resolve()) in paths
    assert plan["protected_bytes"] > entry["bytes"]
    assert plan["protected_bytes"] == sum(entry["bytes"] for entry in files)
    assert sha(run / "state.json") == state_hash
    assert all(Path(path).is_relative_to(scope.resolve()) for path in paths)


def test_plan_retains_original_route_needed_by_continuation(tmp_path, recorded_storage):
    run, _ = recorded_storage
    config = json.loads((run / "config.json").read_bytes())
    task_path = Path(config["task"])
    task = json.loads(task_path.read_bytes())
    route_path = (task_path.parent / task["route_file"]).resolve()
    route = json.loads(route_path.read_bytes())
    originals = [route_path] + [
        route_path.parent / item["path"] for item in [*route["assets"].values(), *route["evidence"]]
    ]
    request = storage_request(tmp_path, recorded_storage)
    plan = run_experiment(request).summary["storage"]
    paths = {entry["path"] for entry in storage_files(request.output_dir, plan)}
    assert all(str(path.resolve()) in paths for path in originals)


def test_cli_reports_over_budget_without_deleting_retained_files(tmp_path, recorded_storage):
    run, _ = recorded_storage
    original = sha(run / "state.json")
    request = storage_request(tmp_path, recorded_storage, budget_bytes=1)
    output = StringIO()
    with redirect_stdout(output):
        code = main(
            [
                "learning-storage-plan",
                "--config",
                str(request.config_file),
                "--output",
                str(request.output_dir),
            ]
        )
    result = json.loads(output.getvalue())
    assert code == 4
    assert result["status"] == "over_budget"
    assert result["files_deleted"] == 0
    assert result["cleanup_authorized"] is False
    plan = json.loads((request.output_dir / "storage-plan.json").read_bytes())
    assert all(Path(row["path"]).is_file() for row in storage_files(request.output_dir, plan))
    assert sha(run / "state.json") == original


def session_with_task(tmp_path, recorded_storage, task_path):
    """A changed input binding, retaining the real saved learners and evaluations."""
    run, scope = recorded_storage
    root = tmp_path / "session"
    root.mkdir()
    state = json.loads((run / "state.json").read_bytes())
    config = json.loads((run / "config.json").read_bytes())
    del state["source_files"][config["task"]]
    state["source_files"][str(task_path)] = sha(task_path)
    config["task"] = str(task_path)
    (root / "config.json").write_text(json.dumps(config))
    state["config_sha256"] = sha(root / "config.json")
    (root / "state.json").write_text(json.dumps(state))
    return root, scope


def test_plan_retains_original_automatic_start_templates(tmp_path, recorded_storage):
    run, _ = recorded_storage
    config = json.loads((run / "config.json").read_bytes())
    task = json.loads(Path(config["task"]).read_bytes())
    menu = tmp_path / "menu"
    menu.mkdir()
    event = verified_config(menu)
    event_config = json.loads(event.read_bytes())["event_run"]
    task.update(
        version=2,
        start_mode="automatic_event_ready",
        automatic_start={
            "event_file": str(event),
            "event_sha256": sha(event),
            "handoff_timeout_s": 5,
        },
    )
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))
    source = session_with_task(tmp_path, recorded_storage, task_path)
    request = storage_request(tmp_path, source)
    plan = run_experiment(request).summary["storage"]
    paths = {entry["path"] for entry in storage_files(request.output_dir, plan)}
    originals = (
        [event]
        + [
            event.parent / patch["template"]
            for patches in event_config["signatures"].values()
            for patch in patches
        ]
        + [event.parent / name for name in event_config["verification_evidence"]]
    )
    assert all(str(path.resolve()) in paths for path in originals)


def test_unreadable_retained_directory_cannot_publish_an_incomplete_plan(
    tmp_path, recorded_storage
):
    run, _ = recorded_storage
    config = json.loads((run / "config.json").read_bytes())
    source = session_with_task(tmp_path, recorded_storage, Path(config["task"]))
    denied = source[0] / "retained-extra"
    denied.mkdir()
    (denied / "original.bin").write_bytes(b"unreadable original evidence")
    request = storage_request(tmp_path, source)
    scan = os.scandir

    def unavailable(path):
        if Path(path).resolve() == denied.resolve():
            raise PermissionError("retained storage unavailable")
        return scan(path)

    with pytest.MonkeyPatch.context() as filesystem:
        filesystem.setattr(os, "scandir", unavailable)
        with pytest.raises(PermissionError, match="retained storage unavailable"):
            run_experiment(request)
    assert not request.output_dir.exists()


@pytest.mark.parametrize("fault", ["stale_state", "outside_root", "output_inside_session"])
def test_invalid_storage_boundary_publishes_nothing(tmp_path, recorded_storage, fault):
    run, _ = recorded_storage
    operation = storage_request(tmp_path, recorded_storage)
    config = json.loads(operation.config_file.read_bytes())
    if fault == "stale_state":
        config["learning"]["state_sha256"] = "0" * 64
    elif fault == "outside_root":
        config["root"] = str(tmp_path)
    else:
        operation = LearningStoragePlan(operation.config_file, run / "forbidden-storage-plan")
    operation.config_file.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        run_experiment(operation)
    assert not operation.output_dir.exists()


def test_missing_original_route_asset_stops_accounting(tmp_path, recorded_storage):
    run, _ = recorded_storage
    config = json.loads((run / "config.json").read_bytes())
    task = json.loads(Path(config["task"]).read_bytes())
    route = json.loads(Path(task["route_file"]).read_bytes())
    route_dir = tmp_path / "incomplete-route"
    route_dir.mkdir()
    route_path = route_dir / "route.json"
    route_path.write_text(json.dumps(route))
    task.update(route_file=str(route_path), route_sha256=sha(route_path))
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))
    source = session_with_task(tmp_path, recorded_storage, task_path)
    operation = storage_request(tmp_path, source)
    with pytest.raises(FileNotFoundError):
        run_experiment(operation)
    assert not operation.output_dir.exists()


def test_original_route_cannot_hide_a_directory_link(tmp_path, recorded_storage):
    run, _ = recorded_storage
    config = json.loads((run / "config.json").read_bytes())
    task = json.loads(Path(config["task"]).read_bytes())
    original_route = Path(task["route_file"])
    directory = tmp_path / "route"
    directory.mkdir()
    contents = directory / "original-assets"
    shutil.copytree(original_route.parent, contents)
    linked = directory / "linked-assets"
    if os.name == "nt":
        subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(linked), str(contents)],
            check=True,
            capture_output=True,
        )
    else:
        linked.symlink_to(contents, target_is_directory=True)
    route = json.loads(original_route.read_bytes())
    for entry in [*route["assets"].values(), *route["evidence"]]:
        entry["path"] = "linked-assets/" + entry["path"]
    route_path = directory / "route.json"
    route_path.write_text(json.dumps(route))
    task.update(route_file=str(route_path), route_sha256=sha(route_path))
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))
    source = session_with_task(tmp_path, recorded_storage, task_path)
    operation = storage_request(tmp_path, source)
    with pytest.raises(ValueError, match="links"):
        run_experiment(operation)
    assert not operation.output_dir.exists()


@pytest.mark.parametrize("component", ["experience/replay.json", "source_replay"])
def test_changed_continuation_manifest_cannot_shrink_dependency_accounting(
    tmp_path, recorded_storage, component
):
    run, _ = recorded_storage
    config = json.loads((run / "config.json").read_bytes())
    source = session_with_task(tmp_path, recorded_storage, Path(config["task"]))
    state_path = source[0] / "state.json"
    state = json.loads(state_path.read_bytes())
    learner = tmp_path / "learner"
    shutil.copytree(Path(state["latest_learner"]["directory"]), learner)
    state["latest_learner"]["directory"] = str(learner)
    state_path.write_text(json.dumps(state))
    if component == "source_replay":
        replay = json.loads((learner / "experience/replay.json").read_bytes())
        path = learner / "experience" / replay["source_inventory"][0]["path"]
    else:
        path = learner / component
    contents = json.loads(path.read_bytes())
    contents["transitions"] = []
    path.write_text(json.dumps(contents))
    operation = storage_request(tmp_path, source)
    with pytest.raises(ValueError, match="manifest changed"):
        run_experiment(operation)
    assert not operation.output_dir.exists()


def test_metadata_read_count_is_distinct_from_streamed_hashes(tmp_path, recorded_storage):
    request = storage_request(tmp_path, recorded_storage)
    open_file = Path.open
    observed = []
    streamed = []

    class CountedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, *args):
            contents = self.stream.read(*args)
            observed.append(len(contents))
            return contents

        def readable(self):
            return self.stream.readable()

        def readinto(self, buffer):
            size = self.stream.readinto(buffer)
            streamed.append(size)
            return size

    def counting_open(path, *args, **kwargs):
        stream = open_file(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if mode == "rb" and path.suffix == ".json" and path != request.config_file:
            return CountedReader(stream)
        return stream

    with pytest.MonkeyPatch.context() as filesystem:
        filesystem.setattr(Path, "open", counting_open)
        plan = run_experiment(request).summary["storage"]
    assert plan["metadata_read_bytes"] >= sum(observed) > 0
    assert sum(streamed) > 0
    assert "metadata_budget_bytes" not in plan


def test_valid_metadata_is_not_rejected_by_the_old_size_limits(tmp_path, recorded_storage):
    run, _ = recorded_storage
    config = json.loads((run / "config.json").read_bytes())
    session = session_with_task(tmp_path, recorded_storage, Path(config["task"]))
    state = session[0] / "state.json"
    # Valid JSON padding crosses both old 128 MiB and 256 MiB gates. This is
    # legacy input compatibility, not a claim of 256 MiB of training history.
    with state.open("ab") as stream:
        for _ in range(257):
            stream.write(b" " * 1024**2)
    original = sha(state)
    request = storage_request(tmp_path, session)
    plan = run_experiment(request).summary["storage"]
    assert plan["status"] == "within_budget"
    assert plan["metadata_read_bytes"] > 256 * 1024**2
    assert sha(state) == original


def test_candidate_database_link_is_rejected_before_sqlite_access(tmp_path, recorded_storage):
    run, _ = recorded_storage
    config = json.loads((run / "config.json").read_bytes())
    database = Path(config["store"]["directory"]) / "state.sqlite"
    request = storage_request(tmp_path, recorded_storage)
    is_link = Path.is_symlink
    connect = sqlite3.connect

    def database_link(path):
        return path == database or is_link(path)

    def reject_database_open(*args, **kwargs):
        if str(database) in str(args[0]) or database.as_uri() in str(args[0]):
            pytest.fail("SQLite opened before the candidate database link was checked")
        return connect(*args, **kwargs)

    with pytest.MonkeyPatch.context() as filesystem:
        filesystem.setattr(Path, "is_symlink", database_link)
        filesystem.setattr(sqlite3, "connect", reject_database_open)
        with pytest.raises(ValueError, match="links"):
            run_experiment(request)
    assert not request.output_dir.exists()
