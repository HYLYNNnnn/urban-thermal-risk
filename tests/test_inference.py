"""Offline inference and output tests; these never download a checkpoint."""
from __future__ import annotations

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from segmentation.infer import main, run_inference
from segmentation.model import CITYSCAPES_LABELS, SegFormerPredictor, mask_from_logits, validate_cityscapes_labels


class FakePredictor:
    def metadata(self):
        return {"model_id": "offline-test-double", "revision": "not-a-real-checkpoint", "max_edge": 32}

    def predict(self, image):
        self.last_processed_size = image.size
        mask = np.full((image.height, image.width), 10, dtype=np.uint8)
        mask[:, :image.width // 2] = 8
        return mask


class InferenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input"
        (self.source / "1").mkdir(parents=True)

    def image(self, name="1_121.1_38.1_0.png"):
        path = self.source / "1" / name
        Image.new("RGB", (20, 12), (100, 120, 140)).save(path)
        return path

    def test_label_mapping_rejects_wrong_datasets_and_generic_labels(self):
        expected = dict(enumerate(CITYSCAPES_LABELS))
        self.assertEqual(expected, validate_cityscapes_labels({str(k): v for k, v in expected.items()}))
        for invalid in [None, {i: f"LABEL_{i}" for i in range(19)}, {i: str(i) for i in range(21)},
                        {**expected, 10: "cow"}, {**expected, 19: "extra"}]:
            with self.subTest(labels=invalid), self.assertRaises(ValueError):
                validate_cityscapes_labels(invalid)

    def test_logits_are_resized_before_argmax_and_invalid_channels_are_rejected(self):
        logits = torch.zeros((1, 19, 2, 3))
        logits[:, 8] = 2
        result = mask_from_logits(logits, (12, 8))
        self.assertEqual((8, 12), result.shape)
        self.assertEqual(np.uint8, result.dtype)
        self.assertTrue(np.all(result == 8))
        for bad in [torch.zeros(1, 21, 2, 3), torch.zeros(2, 19, 2, 3), torch.full((1, 19, 2, 3), float("nan"))]:
            with self.assertRaises(ValueError):
                mask_from_logits(bad, (12, 8))

    def test_processor_cannot_silently_undo_the_bounded_input_size(self):
        # Exercise the real predict/postprocess path using local tensor scores.
        predictor = SegFormerPredictor.__new__(SegFormerPredictor)
        predictor.device = torch.device("cpu")
        predictor.max_edge = 32
        seen = {}
        def process(**kwargs):
            seen.update(kwargs)
            return {"pixel_values": torch.zeros((1, 3, kwargs["images"].height, kwargs["images"].width))}
        def predict(**kwargs):
            self.assertFalse(torch.is_grad_enabled())
            logits = torch.zeros((1, 19, 4, 4)); logits[:, 10] = 3
            return SimpleNamespace(logits=logits)
        predictor.processor = process
        predictor.model = predict
        result = predictor.predict(Image.new("RGB", (90, 60)))
        self.assertEqual((32, 21), predictor.last_processed_size)
        self.assertFalse(seen["do_resize"])
        self.assertEqual((60, 90), result.shape)
        self.assertTrue(np.all(result == 10))

    def test_raw_masks_artifacts_metadata_and_suffixes_stay_distinct(self):
        self.image("1_121.1_38.1_0.png")
        self.image("1_121.1_38.1_0.jpg")
        output = self.root / "output"
        report = run_inference(self.source, output, view="street", predictor=FakePredictor())
        # Two extensions of the same view must keep separate raw artifacts,
        # while the point aggregator correctly rejects their duplicate heading.
        self.assertEqual("partial_failed", report["status"])
        self.assertEqual(2, report["processed_images"])
        self.assertEqual(0, report["point_count"])
        masks = sorted((output / "raw_masks" / "1").glob("*.png"))
        self.assertEqual(2, len(masks))
        self.assertEqual({"1_121.1_38.1_0.jpg.png", "1_121.1_38.1_0.png.png"}, {p.name for p in masks})
        with Image.open(masks[0]) as image:
            self.assertEqual("L", image.mode)
            self.assertEqual({8, 10}, set(np.asarray(image).ravel()))
        with (output / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(2, len(rows))
        self.assertTrue(all(row["point_id"] == "1" for row in rows))
        saved = json.loads((output / "run.json").read_text())
        self.assertEqual("offline-test-double", saved["model"]["model_id"])
        self.assertEqual("historical", saved["metric_policy"])
        self.assertFalse(any(str(self.source) in json.dumps(item) for item in saved["images"]))

    def test_complete_four_direction_batch_has_a_successful_point_export(self):
        for heading in [0, 90, 180, 270]:
            self.image(f"1_121.1_38.1_{heading}.png")
        output = self.root / "output"
        with patch("segmentation.infer.SegFormerPredictor", return_value=FakePredictor()), redirect_stdout(io.StringIO()):
            code = main(["--input", str(self.source), "--output", str(output), "--view", "street"])
        self.assertEqual(0, code)
        with (output / "point_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            points = list(csv.DictReader(stream))
        self.assertEqual(1, len(points))
        self.assertEqual("True", points[0]["complete_four_views"])
        self.assertEqual("4", points[0]["image_count"])
        self.assertAlmostEqual(0.5, float(points[0]["green_view_index"]))

    def test_bad_images_produce_partial_failed_and_a_nonzero_cli_exit(self):
        self.image()
        (self.source / "1" / "1_121.1_38.1_90.png").write_bytes(b"not an image")
        with patch("segmentation.infer.SegFormerPredictor", return_value=FakePredictor()), redirect_stdout(io.StringIO()):
            exit_code = main(["--input", str(self.source), "--output", str(self.root / "output"), "--view", "street"])
        self.assertEqual(1, exit_code)
        report = json.loads((self.root / "output/run.json").read_text())
        self.assertEqual("partial_failed", report["status"])
        self.assertEqual(1, report["processed_images"])
        self.assertEqual(1, len(report["failures"]))
        self.assertTrue(report["failures"][0]["image"].endswith("90.png"))

    def test_inputs_and_prior_output_are_never_overwritten(self):
        self.image()
        with self.assertRaises(ValueError):
            run_inference(self.source, self.source / "output", view="street", predictor=FakePredictor())
        output = self.root / "output"; output.mkdir()
        (output / "keep.txt").write_text("keep")
        with self.assertRaises(ValueError):
            run_inference(self.source, output, view="street", predictor=FakePredictor())
        self.assertEqual("keep", (output / "keep.txt").read_text())

    def test_model_initialization_errors_are_recorded_without_fake_metrics(self):
        self.image()
        output = self.root / "output"
        with patch("segmentation.infer.SegFormerPredictor", side_effect=RuntimeError("checkpoint unavailable")):
            report = run_inference(self.source, output, view="street")
        self.assertEqual("failed", report["status"])
        self.assertEqual("model_initialization", report["failure_stage"])
        self.assertEqual(0, report["processed_images"])
        self.assertIn("checkpoint unavailable", (output / "run.json").read_text())

    def test_fisheye_exports_ignore_labels_outside_the_analysis_circle(self):
        self.image("1_fisheye.png")
        output = self.root / "output"
        report = run_inference(self.source, output, view="fisheye", predictor=FakePredictor())
        self.assertEqual("completed", report["status"])
        with Image.open(output / "raw_masks/1/1_fisheye.png.png") as image:
            mask = np.asarray(image)
            self.assertEqual(255, int(mask[0, 0]))
            self.assertEqual(10, int(mask[6, 10]))
        with Image.open(output / "color_masks/1/1_fisheye.png.png") as image:
            self.assertEqual((0, 0, 0), image.getpixel((0, 0)))
        with Image.open(output / "overlays/1/1_fisheye.png.png") as image:
            self.assertEqual((100, 120, 140), image.getpixel((0, 0)))
        with (output / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(int(np.count_nonzero(mask != 255)), int(row["valid_pixels"]))

    def test_street_and_fisheye_filenames_cannot_be_silently_mixed(self):
        self.image("1_fisheye.png")
        output = self.root / "output"
        report = run_inference(self.source, output, view="street", predictor=FakePredictor())
        self.assertEqual("failed", report["status"])
        self.assertEqual(0, report["processed_images"])
        self.assertIn("does not match", report["failures"][0]["error"])

    def test_duplicate_fisheye_points_are_not_silently_averaged(self):
        self.image("1_fisheye.png")
        self.image("1_fisheye.jpg")
        output = self.root / "output"
        report = run_inference(self.source, output, view="fisheye", predictor=FakePredictor())
        self.assertEqual("partial_failed", report["status"])
        self.assertEqual(2, report["processed_images"])
        self.assertEqual(0, report["point_count"])
        self.assertTrue(any(f.get("stage") == "point_aggregation" for f in report["failures"]))


if __name__ == "__main__":
    unittest.main()
