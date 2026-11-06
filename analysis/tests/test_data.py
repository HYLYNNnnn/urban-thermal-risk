import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from analysis.data import FEATURES, RISK_LABELS, TARGET, calculate_thir, load_features


def sample_features(count=60, seed=7):
    random = np.random.default_rng(seed)
    frame = pd.DataFrame(random.uniform(.01, .8, size=(count, len(FEATURES))), columns=FEATURES)
    frame["ID"] = np.arange(1, count + 1)
    frame["longitude"] = 121.6 + np.arange(count) * .00001
    frame["latitude"] = 38.9 + np.arange(count) * .00001
    return frame


class IndexTests(unittest.TestCase):
    def test_original_weights_give_known_extremes_and_zero_receives_low_risk(self):
        frame = pd.DataFrame({name: [0.0, .5, 1.0] for name in FEATURES})
        result = calculate_thir(frame)
        np.testing.assert_allclose(result.THIR, [.8, 1, 1.2])
        np.testing.assert_allclose(result[TARGET], [0, .5, 1])
        self.assertEqual(result.risk_level.astype(str).tolist(), ["Low Risk", "Medium Risk", "Extreme Risk"])
        self.assertEqual(sum((result.risk_level == name).sum() for name in RISK_LABELS), len(result))
        self.assertNotIn("THIR", frame.columns)

    def test_constant_study_is_finite_and_fully_classified(self):
        result = calculate_thir(pd.DataFrame({name: [0.2] * 5 for name in FEATURES}))
        self.assertTrue(np.isfinite(result[TARGET]).all())
        self.assertEqual(result[TARGET].tolist(), [0] * 5)
        self.assertEqual(result.risk_level.astype(str).tolist(), ["Low Risk"] * 5)

    def test_rejects_missing_invalid_and_out_of_range_features(self):
        frame = sample_features()
        for changed in [frame.drop(columns=[FEATURES[0]]), frame.assign(green_view_index=np.nan),
                        frame.assign(sky_visibility=1.01), frame.assign(population_density=-1)]:
            with self.subTest(columns=changed.columns.tolist()):
                with self.assertRaises(ValueError):
                    calculate_thir(changed)

    def test_csv_aliases_keep_unique_point_identity_and_reject_duplicate_ids(self):
        frame = sample_features().rename(columns={"ID": "序号", "longitude": "经度", "latitude": "纬度", "sky_view_factor": "SVF"})
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "features.csv"
            frame.to_csv(source, index=False, encoding="utf-8-sig")
            result = load_features(source)
            self.assertEqual(result.ID.tolist(), list(range(1, 61)))
            self.assertEqual(result.shape, (60, 16))
            frame.loc[1, "序号"] = frame.loc[0, "序号"]
            frame.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "unique positive"):
                load_features(source)

    def test_historical_data_preserves_report_index_when_available(self):
        source = Path(__file__).resolve().parents[2] / "data/processed/historical_features.csv"
        if not source.exists():
            self.skipTest("The root integration has not installed the historical data yet.")
        result = calculate_thir(load_features(source))
        self.assertEqual(len(result), 238)
        self.assertAlmostEqual(result[TARGET].mean(), .555652834, places=8)
        self.assertAlmostEqual(result[TARGET].std(), .176055582, places=8)
        self.assertEqual(result.risk_level.isna().sum(), 0)
        self.assertEqual((result.risk_level == "Low Risk").sum(), 19)


if __name__ == "__main__":
    unittest.main()
