"""A pinned PyTorch SegFormer adapter with an explicit Cityscapes label contract."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from PIL import Image

CITYSCAPES_LABELS = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)
MODEL_SPECS = {
    "b1": {
        "id": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
        "revision": "ec86afeba68e656629ccf47e0c8d2902f964917b",
    },
    "b5": {
        "id": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        "revision": "2c6f153e4c23c229e2fa2b188eb250607e030cd8",
    },
}


def validate_cityscapes_labels(id2label: Any) -> dict[int, str]:
    """Reject VOC/ADE20K or generic LABEL_N outputs instead of relabeling them."""
    if not isinstance(id2label, dict):
        raise ValueError("The model must provide its Cityscapes id2label mapping.")
    try:
        labels = {int(key): str(value).strip().lower() for key, value in id2label.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid model label mapping.") from error
    if len(id2label) != 19 or labels != dict(enumerate(CITYSCAPES_LABELS)):
        raise ValueError("Expected exactly the 19 Cityscapes train IDs; VOC/ADE labels cannot be substituted.")
    return labels


def mask_from_logits(logits: Any, original_size: tuple[int, int]) -> np.ndarray:
    """Resize scores before argmax; return original-size uint8 class IDs."""
    import torch
    if not torch.is_tensor(logits) or logits.ndim != 4 or tuple(logits.shape[:2]) != (1, 19):
        raise ValueError("Expected one segmentation tensor with exactly 19 class channels.")
    width, height = original_size
    if width < 1 or height < 1 or logits.shape[-1] < 1 or logits.shape[-2] < 1:
        raise ValueError("Image and logits dimensions must be positive.")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("The model returned non-finite segmentation scores.")
    scores = torch.nn.functional.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)
    return scores.argmax(dim=1)[0].to("cpu", dtype=torch.uint8).numpy()


class SegFormerPredictor:
    """Downloads only a fixed NVIDIA checkpoint; arbitrary pickle paths are not accepted."""
    def __init__(self, model_size: str = "b1", device: str = "cpu", max_edge: int = 1024,
                 cpu_threads: int = 4):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        if model_size not in MODEL_SPECS:
            raise ValueError("Choose model b1 or b5.")
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("Choose device cpu, cuda or mps.")
        if not isinstance(max_edge, int) or not 32 <= max_edge <= 2048:
            raise ValueError("max_edge must be an integer between 32 and 2048.")
        if not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 32:
            raise ValueError("cpu_threads must be between 1 and 32.")
        version = re.match(r"(\d+)\.(\d+)", torch.__version__)
        if not version or tuple(map(int, version.groups())) < (2, 6):
            raise RuntimeError("PyTorch 2.6 or later is required for weights-only loading of this pinned checkpoint.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available; choose --device cpu.")
        if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS is not available; choose --device cpu.")
        if device == "cpu":
            torch.set_num_threads(cpu_threads)
        self.spec = MODEL_SPECS[model_size]
        self.device = torch.device(device)
        self.max_edge = max_edge
        self.cpu_threads = cpu_threads
        # A slow PIL processor avoids an unnecessary torchvision dependency.
        self.processor = AutoImageProcessor.from_pretrained(
            self.spec["id"], revision=self.spec["revision"], use_fast=False,
            trust_remote_code=False,
        )
        # These official pinned repositories currently distribute .bin weights.
        # Modern PyTorch's restricted weights-only loader handles their tensors.
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            self.spec["id"], revision=self.spec["revision"], trust_remote_code=False,
            use_safetensors=False, weights_only=True, dtype=torch.float32,
        )
        if getattr(self.model.config, "model_type", None) != "segformer":
            raise ValueError("The checkpoint is not a SegFormer model.")
        self.labels = validate_cityscapes_labels(self.model.config.id2label)
        self.model.to(self.device)
        self.model.eval()
        self.last_processed_size: tuple[int, int] | None = None

    def predict(self, image: Image.Image) -> np.ndarray:
        import torch
        original_size = image.size
        scale = min(1.0, self.max_edge / max(original_size))
        size = tuple(max(1, round(value * scale)) for value in original_size)
        processed = image if size == original_size else image.resize(size, Image.Resampling.LANCZOS)
        self.last_processed_size = size
        # The CLI edge limit is real: do not let the checkpoint processor resize
        # the bounded image back to its training resolution behind the scenes.
        inputs = self.processor(images=processed, return_tensors="pt", do_resize=False)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            return mask_from_logits(outputs.logits, original_size)

    def metadata(self) -> dict[str, Any]:
        import torch
        import transformers
        return {
            "model_id": self.spec["id"], "revision": self.spec["revision"],
            "architecture": "SegFormer", "label_dataset": "Cityscapes", "num_labels": 19,
            "id2label": self.labels, "device": str(self.device), "max_edge": self.max_edge,
            "cpu_threads": self.cpu_threads, "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "preprocessing": "RGB; aspect-preserving downscale only; no processor resize; logits resized before argmax",
            "historical_preprocessing_identical": False,
            "checkpoint_loading": "pinned NVIDIA revision; trust_remote_code=False; weights_only=True",
        }
