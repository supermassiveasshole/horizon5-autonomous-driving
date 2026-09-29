"""Frozen BC assets and identical online preprocessing; no driver imports or training."""

from __future__ import annotations

import hashlib
import importlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from fh5.bc_learning import ARCHITECTURE, MODEL_METADATA_KEYS, _checked_config, _numeric
from fh5.observations import validate_settings


class FrozenActor:
    kind = "frozen_bc_v1"

    def __init__(self, config_file: Path) -> None:
        from fh5.bc_network import make_actor
        from fh5.policy import validate_policy_file

        root, _ = validate_policy_file(config_file)
        self.bound_config = root
        p = root["policy"]
        directory = config_file.parent / p["model_dir"]
        self.manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        m = self.manifest
        if (
            not isinstance(m, dict)
            or m.get("version") != 1
            or m.get("architecture") != ARCHITECTURE
        ):
            raise ValueError("Unsupported frozen model version")
        try:
            c = m["contract"]
            observation = validate_settings(c["observation"])
            trained_config = _checked_config(m["config"])
            if (
                observation["version"] != 2
                or c["action"] != "xinput-lx-rt-lt-v1"
                or c["normalization"]
                != "fixed-v1:speed/100,velocity/100,angular/5,age/1000,waypoints/100"
                or c["image_count"] != len(observation["history_offsets_ms"])
                or c["numeric_size"]
                != 9
                + 2 * c["image_count"]
                + 4 * len(observation["action_history_offsets_ms"])
                + 3 * len(observation["waypoint_distances_m"])
                or c["image_size"] != trained_config["image_size"]
                or c["camera"]
                != {"camera_mode": p["camera_mode"], "camera_pose": "dynamic_unknown"}
                or c["conditions"] != root["snapshot"]
                or c["observed_vehicle"][0] != p["expected_car_ordinal"]
                or c["observed_vehicle"][2] != p["expected_pi"]
                or m["preprocessing"]
                != {
                    "version": 1,
                    "rgb": "full frame bilinear resize, float32 / 255; no crop",
                    "size": c["image_size"],
                }
            ):
                raise ValueError("Incompatible frozen actor contract")
            if any(
                m["training"]["train_examples_by_view"].get(view, 0) < 1
                for view in ("no_reference", "reference_assisted")
            ):
                raise ValueError("Both reference conditions must have trained examples")
            weight_bytes = (directory / "actor.pt").read_bytes()
            digest = hashlib.sha256(weight_bytes).hexdigest()
            if digest != m["weights_sha256"] or digest != p.get("expected_model_sha256"):
                raise ValueError("Frozen model weights hash mismatch")
            self.torch = importlib.import_module("torch")
            self.device = p["device"]
            if self.device == "cuda" and not self.torch.cuda.is_available():
                raise ValueError("CUDA requested but unavailable")
            # Thread count is bounded before the first online inference.
            self.torch.set_num_threads(2)
            saved = self.torch.load(
                BytesIO(weight_bytes), map_location=self.device, weights_only=True
            )
            if saved["metadata"] != {k: m[k] for k in MODEL_METADATA_KEYS}:
                raise ValueError("Frozen model metadata mismatch")
            self.model = make_actor(c).to(self.device)
            self.model.load_state_dict(saved["actor"], strict=True)
            if not all(self.torch.isfinite(v).all() for v in self.model.parameters()):
                raise ValueError("Non-finite model weights")
            self.model.eval()
            self.size = c["image_size"]
            self.cache: dict[str, Any] = {}
            with self.torch.inference_mode():
                for _ in range(3):
                    self.model(
                        self.torch.zeros(
                            1, c["image_count"], 3, self.size[1], self.size[0], device=self.device
                        ),
                        self.torch.zeros(1, c["numeric_size"], device=self.device),
                    )
                if self.device == "cuda":
                    self.torch.cuda.synchronize()
        except (KeyError, TypeError, RuntimeError, IndexError) as error:
            raise ValueError("Invalid frozen model metadata or weights") from error

    def predict(self, actor: dict[str, Any], images: list[bytes]) -> list[float]:
        from PIL import Image

        if (
            set(actor) != set(self.manifest["contract"]["actor_fields"])
            or not actor["ego_mask"]
            or not all(actor["image_mask"])
            or len(images) != self.manifest["contract"]["image_count"]
        ):
            raise ValueError("Incomplete or incompatible actor observation")
        tensors = []
        for image, origin in zip(images, actor["images"]):
            digest = hashlib.sha256(image).hexdigest()
            if digest != origin["sha256"]:
                raise ValueError("Actor RGB source hash mismatch")
            if digest not in self.cache:
                with Image.open(BytesIO(image)) as source:
                    pixels = source.convert("RGB").resize(
                        tuple(self.size), Image.Resampling.BILINEAR
                    )
                self.cache[digest] = (
                    self.torch.frombuffer(bytearray(pixels.tobytes()), dtype=self.torch.uint8)
                    .reshape(self.size[1], self.size[0], 3)
                    .permute(2, 0, 1)
                )
            tensors.append(self.cache[digest])
        if len(self.cache) > 32:
            self.cache.clear()
        with self.torch.inference_mode():
            rgb = self.torch.stack(tensors).unsqueeze(0).to(self.device).float() / 255
            state = self.torch.tensor(
                [_numeric(actor)], dtype=self.torch.float32, device=self.device
            )
            prediction: list[float] = self.model(rgb, state)[0].cpu().tolist()
        return prediction
