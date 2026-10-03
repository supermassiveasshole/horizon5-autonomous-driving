"""Detached entry point. Synthetic mode cannot open devices or control the game."""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from fh5.capture.config import parse_capture_config
from fh5.collection.host import process_identity
from fh5.collection.model import CollectionConfig, CollectionInput, CollectionRun
from fh5.collection.process import collector_project, read_bundle
from fh5.collection.store import atomic_control_json
from fh5.observation.numeric import NumericFrame, PixelContract


class SyntheticCollection:
    source_kind = "synthetic"

    def __init__(self, request: CollectionRun) -> None:
        self.config = request.config
        self.serial = 0
        self.history: deque[NumericFrame] = deque(maxlen=64)

    def read(self, period_s: float) -> CollectionInput:
        from fh5.telemetry.packet import Packet

        time.sleep(period_s)
        now = time.perf_counter_ns()
        cfg = self.config
        raw = bytearray(324)
        struct.pack_into("<iI", raw, 0, 1, (now // 1_000_000) % 2**32)
        struct.pack_into("<iii", raw, 212, cfg.expected_car_ordinal, 6, cfg.expected_pi)
        struct.pack_into("<f", raw, 256, 2.0)
        self.history.append(
            NumericFrame(
                "synthetic",
                str(self.serial),
                now,
                now,
                now,
                "synthetic_clock",
                0,
                cfg.pixels.size,
                memoryview(bytes([17, 34, 51]) * (cfg.pixels.size[0] * cfg.pixels.size[1])),
                {"source": "synthetic_generated"},
                preprocess_version=cfg.pixels.resize,
            )
        )
        self.serial += 1
        frames: list[NumericFrame] = []
        for offset in cfg.pixels.history_offsets_ms:
            eligible = [f for f in self.history if f.source_time_ns <= now - offset * 1_000_000]
            if not eligible:
                frames = []
                break
            frames.append(eligible[-1])
        human = {
            "observed_ns": now,
            "available_ns": now,
            "device_index": 0,
            "connected": True,
            "focused": True,
            "other_input": False,
            "raw": {
                "packet_number": self.serial,
                "buttons": 0,
                "left_trigger": 0,
                "right_trigger": 64,
                "thumb_lx": 4000,
                "thumb_ly": 0,
                "thumb_rx": 0,
                "thumb_ry": 0,
            },
        }
        return CollectionInput(
            time.perf_counter_ns(),
            packets=(Packet(now, "synthetic", bytes(raw)),),
            human_input=human,
            frames=tuple(frames),
            capture_epoch="synthetic",
            focused=True,
        )

    def close(self) -> dict[str, Any]:
        self.history.clear()
        return {"resources_released": True, "synthetic_only": True}


def main() -> int:
    from fh5.experiment import run_experiment

    root, expected, token = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
    state: dict[str, Any] = {
        "pid": os.getpid(),
        "birth": process_identity(os.getpid())["birth"],
        "token": token,
        "manifest_sha256": expected,
        "state": "starting",
        "commands_sent": False,
        "software_snapshot_verified": False,
    }
    code = 4
    try:
        manifest, digest = read_bundle(root)
        if digest != expected or (root / "start.claim").read_text(encoding="utf-8") != token:
            raise ValueError("Collector launch identity or snapshot changed")
        state.update(state="recording", software_snapshot_verified=True, manifest_sha256=digest)
        atomic_control_json(root / "worker-state.json", state)
        project = collector_project(root, manifest)
        document = json.loads((project / "capture.json").read_bytes())
        capture, target = parse_capture_config(document)
        options = dict(manifest["collection"])
        options["pixels"] = PixelContract.from_metadata(options["pixels"])
        request = CollectionRun(
            root / "recording",
            project / "input-profile.json",
            CollectionConfig(**options),
            software_snapshot={
                "manifest_sha256": digest,
                "bundle": str(root),
                "runtime": manifest["runtime"],
                "verified": True,
            },
            input_conditions=document["input_conditions"],
            stop_path=root / "stop.request",
        )
        if manifest["source"] == "synthetic":
            environment = SyntheticCollection(request)
            result = run_experiment(request, collection_environment=environment)
        elif manifest["source"] == "native":
            from fh5.collection.live import native_collection_environment

            native = native_collection_environment(request, capture, target, manifest["port"])
            try:
                result = run_experiment(request, collection_environment=native)
            finally:
                native.close()
        else:
            raise ValueError("Unsupported collector source")
        code = 0 if result.summary["collection"]["complete"] else 4
        state.update(
            state="finished", complete=result.summary["collection"]["complete"], exit_code=code
        )
    except Exception as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}", exit_code=4)
    finally:
        atomic_control_json(root / "worker-state.json", state)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
