"""Optional, local-only Cityscapes inference adapter. No game/controller imports."""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any

from fh5.observation.perception import PixelPrediction, file_hash


class SegformerRoadModel:
    def __init__(self, model_dir: Path, protocol_file: Path, device: str = "cuda") -> None:
        protocol = json.loads(protocol_file.read_text(encoding="utf-8-sig"))
        for name, digest in protocol["model_files"].items():
            path = (model_dir / name).resolve()
            if not path.is_relative_to(model_dir.resolve()) or file_hash(path) != digest:
                raise ValueError(f"Frozen model artifact mismatch: {name}")
        self.torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        self.device = device
        if device not in ("cpu", "cuda") or (
            device == "cuda" and not self.torch.cuda.is_available()
        ):
            raise ValueError("Requested inference device is unavailable")
        started = time.perf_counter_ns()
        self.processor = transformers.SegformerImageProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
            size={"width": protocol["input_size"][0], "height": protocol["input_size"][1]},
            do_reduce_labels=False,
        )
        self.model = (
            transformers.SegformerForSemanticSegmentation.from_pretrained(
                model_dir,
                local_files_only=True,
                weights_only=True,
                use_safetensors=False,
            )
            .to(device)
            .eval()
        )
        if device == "cuda":
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        self.metadata: dict[str, Any] = {
            "method": protocol["method"],
            "model_id": protocol["model_id"],
            "revision": protocol["revision"],
            "weights_sha256": protocol["model_files"]["pytorch_model.bin"],
            "protocol_file_sha256": file_hash(protocol_file),
            "torch": self.torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
            "device_name": self.torch.cuda.get_device_name() if device == "cuda" else "CPU",
            "load_ms": (time.perf_counter_ns() - started) / 1e6,
            "preprocessing": self.processor.to_dict(),
            "dtype": "float32",
            "batch_size": 1,
            "scores": "uncalibrated maximum softmax, rounded to uint8",
        }

    def predict(self, image: Path) -> PixelPrediction:
        from PIL import Image

        with Image.open(image) as source:
            rgb = source.convert("RGB")
            size = rgb.size
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = self.torch.nn.functional.interpolate(
                logits, size=(size[1], size[0]), mode="bilinear", align_corners=False
            )
            scores, classes = logits.softmax(dim=1).max(dim=1)
            class_bytes = classes[0].to(self.torch.uint8).cpu().numpy().tobytes()
            score_bytes = (scores[0] * 255).round().to(self.torch.uint8).cpu().numpy().tobytes()
        if self.device == "cuda":
            self.torch.cuda.synchronize()
            self.metadata["cuda_peak_allocated_bytes"] = self.torch.cuda.max_memory_allocated()
        return PixelPrediction(size, class_bytes, score_bytes)
