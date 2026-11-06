"""Segment local street/fisheye images and export auditable masks and metrics."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .inputs import image_metadata
from .metrics import aggregate_rows, compute_metrics, valid_region
from .model import CITYSCAPES_PALETTE, SegFormerPredictor

SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Unsupported metadata value: {type(value).__name__}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                   default=_json_value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fallback_fields: list[str]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or fallback_fields
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_visuals(image: Image.Image, mask: np.ndarray, relative: str, output: Path) -> dict[str, str]:
    if mask.shape != (image.height, image.width) or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("The predictor must return an original-size integer class mask.")
    if not np.all(((mask >= 0) & (mask < 19)) | (mask == 255)):
        raise ValueError("The mask contains labels outside Cityscapes IDs 0–18/ignore 255.")
    mask = mask.astype(np.uint8, copy=False)
    valid = mask != 255
    colors = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colors[valid] = CITYSCAPES_PALETTE[mask[valid]]
    original = np.asarray(image, dtype=np.uint8)
    overlay = original.copy()
    overlay[valid] = np.rint(original[valid] * 0.6 + colors[valid] * 0.4).astype(np.uint8)
    # Append .png rather than replacing suffix: same-name .jpg and .png inputs
    # remain distinct, and nested point directories never overwrite each other.
    paths = {kind: Path(kind) / (relative + ".png") for kind in ["raw_masks", "color_masks", "overlays"]}
    for target in paths.values():
        (output / target).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output / paths["raw_masks"])
    Image.fromarray(colors).save(output / paths["color_masks"])
    Image.fromarray(overlay).save(output / paths["overlays"])
    return {kind: target.as_posix() for kind, target in paths.items()}


def run_inference(input_path: str | Path, output_path: str | Path, *, view: str,
                  model_size: str = "b1", device: str = "cpu", max_edge: int = 1024,
                  cpu_threads: int = 4, predictor: Any = None) -> dict[str, Any]:
    """Run a local batch; predictor injection is for offline tests, not a CLI option."""
    supplied_input = Path(input_path)
    source = supplied_input.resolve(strict=True)
    if not source.is_file() and not source.is_dir():
        raise ValueError("Input must be an image file or directory.")
    if view not in {"street", "fisheye"}:
        raise ValueError("Choose view street or fisheye.")
    root = source if source.is_dir() else source.parent
    output = Path(output_path).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("The output directory must be outside the input image directory.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("The output directory must be new or empty; previous runs are not overwritten.")
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_IMAGES:
            raise ValueError("Input is not a supported image type.")
        images = [source]
    else:
        images = sorted((path for path in source.rglob("*")
                         if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGES),
                        key=lambda path: path.relative_to(root).as_posix())
    if not images:
        raise ValueError("No supported images were found in the input directory.")
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_name": source.name, "view": view, "metric_policy": "historical",
        "metric_interpretation": (
            "Historical proxy metrics: vegetation includes terrain; street sky fraction is not fisheye SVF. "
            "Fisheye factors retain the archived sin(theta) weighting and are not a calibrated physical view-factor measurement."
        ),
        "discovered_images": len(images), "processed_images": 0, "point_count": 0,
        "status": "running", "failures": [], "images": [],
    }
    if predictor is None:
        try:
            predictor = SegFormerPredictor(model_size=model_size, device=device,
                                           max_edge=max_edge, cpu_threads=cpu_threads)
        except Exception as error:
            report.update(status="failed", failure_stage="model_initialization")
            report["failures"].append({"image": None, "error": f"{type(error).__name__}: {error}"})
            report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(output / "run.json", report)
            return report
    report["model"] = predictor.metadata()
    rows: list[dict[str, Any]] = []
    for path in images:
        relative = path.relative_to(root).as_posix()
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError("Input symlinks may not point outside the input directory.")
            metadata = image_metadata(path, root)
            if (view == "street") != (metadata.get("heading") is not None):
                raise ValueError("The filename view does not match --view; street inputs need coordinates/heading, fisheye inputs need POINT_fisheye.ext.")
            with Image.open(path) as opened:
                if getattr(opened, "n_frames", 1) != 1:
                    raise ValueError("Only single-frame images are supported.")
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.load()
            mask = np.asarray(predictor.predict(image))
            if view == "fisheye":
                mask = mask.copy()
                mask[~valid_region(mask.shape, view="fisheye")] = 255
            artifacts = _save_visuals(image, mask, relative, output)
            values = compute_metrics(mask, view=view, policy="historical")
            rows.append({**metadata, **values})
            report["images"].append({
                "image": relative, "original_size": list(image.size),
                "processed_size": list(getattr(predictor, "last_processed_size", None) or image.size),
                **artifacts,
            })
        except Exception as error:
            report["failures"].append({"image": relative, "error": f"{type(error).__name__}: {error}"})
    points: list[dict[str, Any]] = []
    try:
        points = aggregate_rows(rows, view=view) if rows else []
    except Exception as error:
        report["failures"].append({"image": None, "stage": "point_aggregation",
                                   "error": f"{type(error).__name__}: {error}"})
    _write_csv(output / "metrics.csv", rows, ["point_id", "image"])
    _write_csv(output / "point_metrics.csv", points, ["point_id", "image_count"])
    report["processed_images"] = len(rows)
    report["point_count"] = len(points)
    report["status"] = ("partial_failed" if rows else "failed") if report["failures"] else "completed"
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "run.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Image or point-directory tree.")
    parser.add_argument("--output", required=True, type=Path, help="New/empty directory outside input.")
    parser.add_argument("--view", required=True, choices=["street", "fisheye"])
    parser.add_argument("--model", choices=["b1", "b5"], default="b1")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--max-edge", type=int, default=1024, help="Maximum processed edge, 32–2048; smaller values reduce compute and may change predictions.")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        result = run_inference(args.input, args.output, view=args.view, model_size=args.model,
                               device=args.device, max_edge=args.max_edge, cpu_threads=args.cpu_threads)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({key: result[key] for key in ["status", "processed_images", "point_count", "failures"]}, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
