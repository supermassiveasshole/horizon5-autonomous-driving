"""One-time legacy preparation and a bounded, codec-free offline numerical source."""

from __future__ import annotations

import hashlib
import json
import threading
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any

from fh5.numeric_images import NumericDecision, NumericFrame, PixelContract, asset
from fh5.numeric_report import preview_png

if TYPE_CHECKING:
    from fh5.experiment import RunResult


@dataclass(frozen=True)
class LegacyNumericImport:
    model_dir: Path
    dataset_file: Path
    output_dir: Path
    max_decisions: int = 200
    view: str = "no_reference"


def prepare_legacy(request: LegacyNumericImport) -> RunResult:
    from PIL import Image

    from fh5.numeric_recording import _encode, _result

    if (
        request.view not in ("no_reference", "reference_assisted")
        or not 1 <= request.max_decisions <= 5000
    ):
        raise ValueError("Invalid numerical import view or decision limit")
    raw = request.dataset_file.read_bytes()
    model = json.loads((request.model_dir / "model.json").read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != model["dataset_sha256"]:
        raise ValueError("Legacy dataset differs from the frozen model's bound export")
    dataset = json.loads(raw)
    contract = PixelContract(
        size=tuple(model["contract"]["image_size"]),
        origin="legacy_offline",
        history_offsets_ms=tuple(model["contract"]["observation"]["history_offsets_ms"]),
    )
    selected = [
        (index, row)
        for index, row in enumerate(dataset["examples"])
        if row["bc_eligible"]
        and row["views"][request.view]["ego_mask"]
        and all(row["views"][request.view]["image_mask"])
    ][: request.max_decisions]
    if not selected:
        raise ValueError("No eligible complete numerical observations to import")
    sources = {}
    for run_index in {row["run_index"] for _, row in selected}:
        source = dataset["sources"][run_index]
        directory = Path(source["directory"])
        if (
            hashlib.sha256((directory / "packets.jsonl").read_bytes()).hexdigest()
            != source["packets_sha256"]
        ):
            raise ValueError("Legacy source telemetry changed")
        report = request.dataset_file.parent / f"run-{run_index}-targets.json"
        observation = json.loads(report.read_text(encoding="utf-8"))["summary"]["observations"]
        if observation["source"]["packets_sha256"] != source["packets_sha256"]:
            raise ValueError("Legacy observation provenance mismatch")
        journal = [
            json.loads(line)
            for line in (directory / "vision.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        sources[run_index] = (
            directory,
            {f["path"]: f for f in journal if f.get("kind") == "frame"},
            observation["history_boundaries_ns"],
            {d["decision_ns"]: d for d in observation["decisions"]},
        )
    request.output_dir.mkdir(parents=True, exist_ok=False)
    (request.output_dir / "pixels").mkdir()
    (request.output_dir / "previews").mkdir()
    converted: dict[tuple[int, str], dict[str, Any]] = {}
    records = []
    for index, example in selected:
        run_index, tick = example["run_index"], example["decision_ns"]
        directory, frames, boundaries, observations = sources[run_index]
        if observations[tick]["actor"] != example["views"]["reference_assisted"]:
            raise ValueError("Legacy exported actor differs from its observation evidence")
        epoch = f"source-{run_index}:segment-{bisect_right(boundaries, tick)}"
        chosen = []
        actor = example["views"][request.view]
        for origin in actor["images"]:
            key = (run_index, origin["path"])
            if key not in converted:
                original = frames[origin["path"]]
                encoded = asset(directory, origin["path"])
                if hashlib.sha256(encoded.read_bytes()).hexdigest() != origin["sha256"]:
                    raise ValueError("Legacy encoded source hash mismatch")
                with Image.open(encoded) as image:
                    if list(image.size) != original["size"] or image.mode != "RGB":
                        raise ValueError("Legacy source pixel layout mismatch")
                    pixels = image.resize(contract.size, Image.Resampling.BILINEAR).tobytes()
                digest = hashlib.sha256(pixels).hexdigest()
                path = f"pixels/{digest}.rgb"
                (request.output_dir / path).write_bytes(pixels)
                (request.output_dir / f"previews/{digest}.png").write_bytes(
                    preview_png(pixels, contract.size)
                )
                converted[key] = {
                    "epoch": f"source-{run_index}:segment-{bisect_right(boundaries, original['capture_start_ns'])}",
                    "frame_id": f"source-{run_index}:{origin['path']}",
                    "source_time_ns": original["capture_start_ns"],
                    "capture_received_ns": original["capture_end_ns"],
                    "preprocess_ready_ns": original["delivered_ns"],
                    "time_quality": "capture_start_proxy",
                    "uncertainty_ns": None,
                    "availability_kind": "legacy_encoded_delivery_proxy",
                    "preprocess_version": contract.resize,
                    "size": list(contract.size),
                    "source_layout": {
                        "size": original["size"],
                        "client_size": original["client_size"],
                        "format": "RGB",
                        "stride_bytes": original["size"][0] * 3,
                        "codec": original["codec"],
                        "resize_method": original["resize_method"],
                        "encoded_sha256": origin["sha256"],
                        "source_clock": "recorded_perf_counter_ns",
                        "color_space": "unknown",
                    },
                    "path": path,
                    "sha256": digest,
                }
            chosen.append(converted[key])
        records.append(
            {
                "decision_id": f"legacy:{index}",
                "epoch": epoch,
                "decision_ns": tick,
                "frames": chosen,
                "actor": actor,
                "split": example["split"],
                "view": request.view,
                "previews": [f"previews/{f['sha256']}.png" for f in chosen],
            }
        )
    summary = {
        "version": 1,
        "contract": contract.metadata(),
        "decisions": records,
        "decoded_unique_frames": len(converted),
        "dataset_sha256": model["dataset_sha256"],
        "commands_sent": False,
        "diagnostic_only": True,
        "availability_note": "Historical encoded delivery is a proxy, not measured numerical preprocessing latency.",
    }
    (request.output_dir / "prepared.json").write_bytes(_encode(summary))
    return _result(request.output_dir / "report.html", summary, section="numeric_import")


class PreparedNumericSource:
    """Offline disk reads are isolated from the numerical prediction caller."""

    def __init__(self, directory: Path, max_cache_bytes: int = 16 * 1024**2) -> None:
        if type(max_cache_bytes) is not int or not 1 <= max_cache_bytes <= 1024**3:
            raise ValueError("Invalid numerical source cache budget")
        self.directory = directory
        self.manifest = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
        self.contract = PixelContract.from_metadata(self.manifest["contract"])
        if self.manifest.get("version") != 1 or not 1 <= len(self.manifest["decisions"]) <= 5000:
            raise ValueError("Invalid prepared numerical source")
        identities: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self.manifest["decisions"]:
            if len(row["frames"]) != len(self.contract.history_offsets_ms):
                raise ValueError("Prepared frame count differs from contract")
            for frame in row["frames"]:
                identity = (frame["epoch"], frame["frame_id"])
                if identity in identities and identities[identity] != frame:
                    raise ValueError("Prepared frame identity reused with different metadata")
                identities[identity] = frame
        self.max_cache_bytes = max_cache_bytes
        self.queue: Queue[NumericDecision | Exception | None] = Queue(2)
        self.stopped = threading.Event()
        self.closed = False
        self.peak_cached_bytes = 0
        self.worker: threading.Thread | None = None

    def _put(self, item: NumericDecision | Exception | None) -> bool:
        while not self.stopped.is_set():
            try:
                self.queue.put(item, timeout=0.02)
                return True
            except Full:
                pass
        return False

    def _read(self) -> None:
        cache: OrderedDict[tuple[str, str], NumericFrame] = OrderedDict()
        size = 0
        try:
            for row in self.manifest["decisions"]:
                if self.stopped.is_set():
                    break
                chosen = []
                for frame in row["frames"]:
                    key = (frame["epoch"], frame["frame_id"])
                    if key not in cache:
                        pixels = asset(self.directory, frame["path"]).read_bytes()
                        if len(pixels) > self.max_cache_bytes:
                            raise ValueError("Numerical frame exceeds source cache budget")
                        if hashlib.sha256(pixels).hexdigest() != frame["sha256"]:
                            raise ValueError("Prepared numerical pixel hash mismatch")
                        metadata = {k: v for k, v in frame.items() if k not in ("path", "sha256")}
                        metadata["size"] = tuple(metadata["size"])
                        while cache and size + len(pixels) > self.max_cache_bytes:
                            _, previous = cache.popitem(last=False)
                            size -= previous.pixels.nbytes
                        cache[key] = NumericFrame(pixels=memoryview(pixels), **metadata)
                        size += len(pixels)
                        self.peak_cached_bytes = max(self.peak_cached_bytes, size)
                    cache.move_to_end(key)
                    chosen.append(cache[key])
                if not self._put(
                    NumericDecision(
                        row["decision_id"],
                        row["epoch"],
                        row["decision_ns"],
                        tuple(chosen),
                        row["actor"],
                    )
                ):
                    break
        except Exception as error:
            self._put(error)
        finally:
            cache.clear()
            self._put(None)

    def __iter__(self) -> Iterator[NumericDecision]:
        if self.worker is not None or self.closed:
            raise ValueError("A prepared numerical source can only be consumed once")
        self.worker = threading.Thread(target=self._read, name="fh5-numeric-source", daemon=True)
        self.worker.start()
        while True:
            try:
                item = self.queue.get(timeout=0.1)
            except Empty:
                if not self.worker.is_alive():
                    raise RuntimeError("Numerical source stopped without completion") from None
                continue
            if isinstance(item, Exception):
                raise item
            if item is None:
                break
            yield item

    def close(self) -> bool:
        self.stopped.set()
        if self.worker is not None:
            self.worker.join(timeout=2)
        while not self.queue.empty():
            self.queue.get_nowait()
        self.closed = self.worker is None or not self.worker.is_alive()
        return self.closed
