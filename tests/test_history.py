import csv
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from segmentation.history import main, reconcile


class HistoricalReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.features = self.root / "features.csv"
        self.base = {"序号": "1", "avg_green_view_index": ".3", "avg_sky_visibility": ".4",
                     "avg_enclosure_index": ".2", "SVF": ".4", "TVF": ".2", "BVF": ".1", "hw_value": "1.5"}
        self.write(self.features, [self.base])
        self.write(self.root / "folder_average_results.csv", [{"folder": "1", **{key: self.base[key] for key in self.base if key.startswith("avg_")}}])
        self.write(self.root / "fisheye_view_factors.csv", [{"folder_id": "1", "SVF": ".4", "TVF": ".2", "BVF": ".1"}])
        self.write(self.root / "street_canyon_hw_analysis_results.csv", [{"point_id": "1", "hw_value": "1.5"}])
        self.write(self.root / "all_image_results.csv", [
            {"folder": "1", "image": "1_0.png", "green_view_index": ".2", "sky_visibility": ".3", "enclosure_index": ".1"},
            {"folder": "1", "image": "1_90.png", "green_view_index": ".4", "sky_visibility": ".5", "enclosure_index": ".3"},
        ])

    @staticmethod
    def write(path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_reconciles_seven_metrics_and_raw_image_averages(self):
        report, rows = reconcile(self.root, self.features)
        self.assertTrue(report["passed"], report["issues"])
        self.assertEqual(7, len(rows))
        self.assertTrue(all(row["status"] == "matched" for row in rows))
        self.assertEqual({"compared": 3, "matched": 3, "mismatched": 0}, report["raw_street_average_check"])
        self.assertEqual({"1": 2}, report["points_with_nonfour_image_count"])

    def test_duplicate_and_conflicting_ids_fail_without_silent_overwrite(self):
        self.write(self.features, [self.base, {**self.base, "hw_value": "9"}])
        report, _ = reconcile(self.root, self.features)
        self.assertFalse(report["passed"])
        duplicate = next(issue for issue in report["issues"] if issue["code"] == "DUPLICATE_POINT_ID")
        self.assertTrue(duplicate["conflicting"])
        self.assertEqual(1, report["datasets"]["features"]["unique_point_ids"])
        self.assertEqual(1, report["metrics"]["hw_value"]["matched"])

    def test_missing_point_and_nonfinite_values_are_explicit_failures(self):
        self.write(self.root / "fisheye_view_factors.csv", [{"folder_id": "2", "SVF": ".4", "TVF": ".2", "BVF": ".1"}])
        self.write(self.root / "street_canyon_hw_analysis_results.csv", [{"point_id": "1", "hw_value": "NaN"}])
        report, _ = reconcile(self.root, self.features)
        self.assertFalse(report["passed"])
        self.assertEqual(1, report["metrics"]["SVF"]["missing"])
        self.assertEqual(1, report["metrics"]["hw_value"]["invalid"])

    def test_absolute_tolerance_is_inclusive_and_mismatch_has_nonzero_exit(self):
        self.write(self.features, [{**self.base, "hw_value": "1.625"}])
        report, _ = reconcile(self.root, self.features, atol=.125)
        self.assertTrue(report["passed"], report["issues"])
        output = self.root / "report.json"
        with redirect_stdout(StringIO()):
            status = main(["--data-dir", str(self.root), "--features", str(self.features),
                           "--output", str(output), "--atol", ".124"])
        self.assertEqual(1, status)
        self.assertFalse(json.loads(output.read_text())["passed"])
        self.assertTrue(output.with_suffix(".csv").exists())

    def test_raw_average_disagreement_fails_even_when_final_feature_matches_saved_average(self):
        self.write(self.root / "all_image_results.csv", [{"folder": "1", "image": "1_0.png",
            "green_view_index": ".9", "sky_visibility": ".4", "enclosure_index": ".2"}])
        report, _ = reconcile(self.root, self.features)
        self.assertFalse(report["passed"])
        self.assertEqual(1, report["metrics"]["avg_green_view_index"]["matched"])
        self.assertEqual(1, report["raw_street_average_check"]["mismatched"])

    def test_cli_refuses_to_overwrite_any_input(self):
        before = self.features.read_bytes()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            main(["--data-dir", str(self.root), "--features", str(self.features), "--output", str(self.features)])
        self.assertEqual(2, raised.exception.code)
        self.assertEqual(before, self.features.read_bytes())


if __name__ == "__main__":
    unittest.main()
