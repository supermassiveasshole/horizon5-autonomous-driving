"""Read-only host and frozen-collector metrics; no capture or input devices."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from fh5.artifact_io import VerifiedFile, sha256_file
from fh5.capture_resources import WindowsResources
from fh5.collection import CollectionControl
from fh5.collection_process import control_bundle
from fh5.collection_status import read_collection_session
from fh5.learning_observation import resource_observation


class NativeLearningResources:
    source_kind = "native_resources"

    def __init__(
        self,
        bundle: Path,
        digest: str,
        disk_path: Path,
        sample_interval_s: float,
        *,
        include_gpu: bool,
    ) -> None:
        self.bundle, self.digest, self.disk_path = bundle, digest, disk_path
        self.resources = WindowsResources(include_gpu=include_gpu)
        self.sample_interval_s = sample_interval_s
        self.cached: dict[str, Any] | None = None
        self._check_manifest()

    def _check_manifest(self) -> None:
        if sha256_file(self.bundle / "frozen.json") != self.digest:
            raise ValueError("Collector manifest differs from scheduled resource binding")

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def sample(self) -> dict[str, Any]:
        if self.cached is not None and (
            self.now_ns() - self.cached["observed_ns"] < self.sample_interval_s * 1_000_000_000
        ):
            return self.cached
        self._check_manifest()
        status = control_bundle(CollectionControl(self.bundle)).summary["collection"]
        if "session_sha256" in status:
            session = read_collection_session(
                VerifiedFile(self.bundle / "recording/session.json", status["session_sha256"])
            )
            snapshot = session.get("software_snapshot") or {}
            worker = status.get("worker-state") or {}
            for name, binding in (("session", snapshot), ("worker", worker)):
                digest = binding.get("manifest_sha256")
                if digest is not None and digest != self.digest:
                    raise ValueError(f"Collector {name} snapshot differs from scheduled binding")
                status[name + "_manifest_sha256"] = digest
            if (
                snapshot.get("verified") is not True
                or snapshot.get("manifest_sha256") != self.digest
                or worker.get("manifest_sha256") != self.digest
            ):
                status = dict(status, software_snapshot_verified=False)
        else:
            status = dict(status, software_snapshot_verified=False)
        # Keep expensive OS/process queries in the learner, never in acquisition.
        values = self.resources()
        # Status may carry a large final manifest. Retain only admission evidence.
        keys = (
            "process_liveness",
            "software_snapshot_verified",
            "state",
            "heartbeat_ns",
            "last_poll_ns",
            "latest_image_source_ns",
            "pending_bytes",
            "dropped_rows",
            "seen_rows",
            "archive_error",
            "abnormal_exit",
            "final_status_present",
            "complete",
            "pid",
            "process_birth",
            "session_sha256",
            "session_manifest_sha256",
            "worker_manifest_sha256",
            "error",
            "stop_reason",
        )
        result = {
            **values,
            "collector": {k: status[k] for k in keys if k in status},
            "free_disk_bytes": shutil.disk_usage(self.disk_path).free,
            "observed_ns": self.now_ns(),
        }
        # Admission consumes its schema, not arbitrary host diagnostic payloads.
        self.cached = resource_observation(result, include_gpu=self.resources.include_gpu)
        return self.cached

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)

    def close(self) -> None:
        pass  # This adapter owns no long-running worker or game resources.
