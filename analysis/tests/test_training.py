from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis.data import FEATURES, TARGET, calculate_thir
from analysis.tests.test_data import sample_features
from analysis.train import Split, fit_model, run_analysis, split_rows, validate_split


class PartitionTests(unittest.TestCase):
    def test_splits_are_reproducible_disjoint_and_cover_every_row(self):
        split = split_rows(60, 42)
        self.assertEqual(split, split_rows(60, 42))
        self.assertNotEqual(split, split_rows(60, 13))
        validate_split(split, 60)
        self.assertEqual((len(split.train), len(split.validation), len(split.test)), (36, 12, 12))
        with self.assertRaises(ValueError):
            split_rows(19)
        invalid = Split(split.train, split.validation, tuple([split.train[0], *split.test[1:]]))
        with self.assertRaises(ValueError):
            validate_split(invalid, 60)

    def test_existing_output_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("existing results")
            with self.assertRaisesRegex(ValueError, "existing results"):
                run_analysis(Path(temp) / "does-not-exist.csv", output)
            self.assertEqual(sentinel.read_text(), "existing results")


class ModelTests(unittest.TestCase):
    def test_test_targets_cannot_change_training_early_stopping_or_cv(self):
        frame = calculate_thir(sample_features())
        split = split_rows(len(frame))
        options = dict(max_estimators=30, early_stopping_rounds=4, cv_folds=3)
        first, predictions, metrics = fit_model(frame, split, **options)
        perturbed = frame.copy()
        perturbed.loc[list(split.test), TARGET] = 1000
        second, changed_predictions, changed_metrics = fit_model(perturbed, split, **options)
        self.assertEqual(first.get_booster().get_dump(), second.get_booster().get_dump())
        self.assertEqual(first.best_iteration, second.best_iteration)
        self.assertEqual(first.evals_result(), second.evals_result())
        np.testing.assert_allclose(predictions, changed_predictions)
        self.assertEqual(metrics["cross_validation"], changed_metrics["cross_validation"])
        self.assertNotEqual(metrics["partition_metrics"]["test"], changed_metrics["partition_metrics"]["test"])
        self.assertEqual(list(first.evals_result()), ["validation_0"])

    def test_real_training_shap_and_saved_model_outputs_agree(self):
        from xgboost import XGBRegressor

        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            source = temp / "features.csv"
            sample_features().to_csv(source, index=False)
            output = temp / "run"
            metrics = run_analysis(source, output, max_estimators=30, early_stopping_rounds=4,
                                   cv_folds=3, max_shap_samples=18)
            self.assertEqual(metrics["sample_count"], 60)
            self.assertEqual(sum(metrics["risk_counts"].values()), 60)
            self.assertEqual(metrics["shap"]["sample_count"], 18)
            result = pd.read_csv(output / "point_results.csv")
            model = XGBRegressor()
            model.load_model(output / "xgboost_model.json")
            np.testing.assert_allclose(model.predict(result.loc[:, list(FEATURES)]), result.predicted_THIR_normalized, rtol=1e-6)
            explanations = pd.read_csv(output / "shap_values.csv")
            selected = result.set_index("ID").loc[explanations.ID]
            reconstructed = explanations.loc[:, list(FEATURES)].sum(axis=1) + explanations.base_value
            np.testing.assert_allclose(reconstructed, selected.predicted_THIR_normalized, rtol=1e-4, atol=1e-5)
            self.assertTrue((output / "shap_importance.png").stat().st_size > 1000)
            self.assertTrue((output / "shap_summary.png").stat().st_size > 1000)
            self.assertEqual(len(pd.read_csv(output / "split_ids.csv")), 60)


if __name__ == "__main__":
    unittest.main()
