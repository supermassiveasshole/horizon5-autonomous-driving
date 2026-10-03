"""Frozen numerical actor; loading is explicit and prediction never opens an image."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fh5.artifacts.document import read_document_fields
from fh5.artifacts.io import VerifiedFile, sha256_file
from fh5.learning.bc.features import (
    TEMPORAL_ARCHITECTURE,
    TEMPORAL_METADATA_KEYS,
    actor_shape,
    temporal_features,
    time_contract,
)
from fh5.learning.bc.legacy_training import ARCHITECTURE, MODEL_METADATA_KEYS, _numeric
from fh5.learning.bc.network import make_actor
from fh5.observation.numeric import NumericFrame, PixelContract


class FrozenNumericActor:
    kind = "frozen-numeric-bc-compatibility-v1"

    def __init__(
        self,
        model_dir: Path,
        contract: PixelContract | None = None,
        device: str = "cpu",
        *,
        legacy_diagnostic: bool = False,
        expected_manifest_sha256: str | None = None,
    ) -> None:
        path = model_dir / "model.json"
        digest = sha256_file(path)
        if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
            raise ValueError("Frozen numerical actor manifest changed from its bound batch")
        self.manifest_file = VerifiedFile(path, digest)
        original = read_document_fields(
            self.manifest_file, {*TEMPORAL_METADATA_KEYS, "weights_sha256"}
        )
        temporal = original.get("version") == 2
        if not temporal and not legacy_diagnostic:
            raise ValueError("Old BC weights require explicit legacy_diagnostic=True")
        if (original.get("version"), original.get("architecture")) not in (
            (1, ARCHITECTURE),
            (2, TEMPORAL_ARCHITECTURE),
        ):
            raise ValueError("Unsupported frozen numerical actor model")
        # Checkpoint consumers derive the pixel contract from verified metadata;
        # live adapters may additionally require their explicit external contract.
        if contract is None:
            contract = PixelContract.from_metadata(original["numeric_contract"])
        c = original["contract"]
        self.temporal = c.get("temporal") if temporal else None
        if temporal and (
            not self.temporal
            or self.temporal != time_contract(self.temporal["mode"], contract)
            or original["numeric_contract"] != contract.metadata()
        ):
            raise ValueError("Frozen temporal actor contract mismatch")
        if temporal:
            self.kind = "frozen-numeric-temporal-bc-v2"
        if (
            c["image_size"] != list(contract.size)
            or c["observation"]["history_offsets_ms"] != list(contract.history_offsets_ms)
            or c["normalization"]
            != "fixed-v1:speed/100,velocity/100,angular/5,age/1000,waypoints/100"
            or original["preprocessing"]
            != {
                "version": 1,
                "rgb": "full frame bilinear resize, float32 / 255; no crop",
                "size": list(contract.size),
            }
            or contract.resize != "full-frame-pillow-bilinear-v1"
        ):
            raise ValueError("Frozen numerical actor preprocessing contract mismatch")
        weights = VerifiedFile(model_dir / "actor.pt", original["weights_sha256"])
        self.torch = importlib.import_module("torch")
        self.device = device
        if device not in ("cpu", "cuda") or (
            device == "cuda" and not self.torch.cuda.is_available()
        ):
            raise ValueError("Requested numerical actor device unavailable")
        with weights.snapshot() as stream:
            saved = self.torch.load(stream, map_location=device, weights_only=True)
        keys = TEMPORAL_METADATA_KEYS if temporal else MODEL_METADATA_KEYS
        if saved["metadata"] != {k: original[k] for k in keys}:
            raise ValueError("Frozen numerical actor metadata mismatch")
        self.model = make_actor(c).to(device)
        self.model.load_state_dict(saved["actor"], strict=True)
        if not all(self.torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError("Non-finite numerical actor weights")
        self.model.eval()
        self.contract = contract
        self.original_contract = c
        self.cache: OrderedDict[int, tuple[NumericFrame, Any]] = OrderedDict()
        self.manifest = {
            "weights_sha256": original["weights_sha256"],
            "architecture": original["architecture"],
            "numeric_contract": contract.metadata(),
            "model_contract": c,
            "original_preprocessing": original["preprocessing"],
            "diagnostic_only": (
                not temporal
                or contract.origin == "legacy_offline"
                or original.get("provenance", {}).get("diagnostic_only") is not False
            ),
            "new_capture_distribution_validated": False,
            "explicit_dt_model": temporal,
        }
        if temporal:
            self.manifest.update(
                dataset_sha256=original["dataset_sha256"],
                provenance=original["provenance"],
                groups=original["groups"],
            )

    def input_features(
        self, actor: dict[str, Any], frames: tuple[NumericFrame, ...]
    ) -> list[float]:
        if self.temporal is None:
            return _numeric(actor)
        if actor_shape(actor, len(frames)) != self.original_contract["actor_shape"]:
            raise ValueError("Temporal actor feature shape mismatch")
        return temporal_features(actor, frames, self.temporal)

    def predict(self, actor: dict[str, Any], frames: tuple[NumericFrame, ...]) -> list[float]:
        return self.predict_with_features(actor, frames)[1]

    def predict_with_features(
        self, actor: dict[str, Any], frames: tuple[NumericFrame, ...]
    ) -> tuple[Any, list[float]]:
        """Return frozen features and their prediction from one numerical encoding."""
        if (
            set(actor) != set(self.original_contract["actor_fields"])
            or len(frames) != self.original_contract["image_count"]
            or not actor["ego_mask"]
            or not all(actor["image_mask"])
            or any(f.size != self.contract.size for f in frames)
        ):
            raise ValueError("Incomplete numerical actor observation")
        values = self.input_features(actor, frames)
        if len(values) != self.original_contract["numeric_size"]:
            raise ValueError("Numerical actor feature shape mismatch")
        tensors = []
        for frame in frames:
            key = id(frame)
            if key not in self.cache:
                width, height = frame.size
                tensor = self.torch.frombuffer(bytearray(frame.pixels), dtype=self.torch.uint8)
                self.cache[key] = (frame, tensor.reshape(height, width, 3).permute(2, 0, 1))
            self.cache.move_to_end(key)
            tensors.append(self.cache[key][1])
            while len(self.cache) > 32:
                self.cache.popitem(last=False)
        with self.torch.inference_mode():
            pixels = self.torch.stack(tensors).unsqueeze(0).to(self.device).float() / 255
            state = self.torch.tensor([values], dtype=self.torch.float32, device=self.device)
            features = self.model.features(pixels, state)
            prediction: list[float] = self.model.fusion(features)[0].cpu().tolist()
        return features[0], prediction

    def clear_input_cache(self) -> None:
        self.cache.clear()
