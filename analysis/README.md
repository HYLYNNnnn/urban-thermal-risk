# THIR model analysis

This module preserves the historical HEVA weights and constructs the study's THIR index. It then fits XGBoost as a surrogate for that index and uses SHAP to explain model predictions. The target is not independently measured temperature.

From the repository root, after installing the analysis dependencies:

```sh
python -m analysis.train --input data/processed/historical_features.csv --output outputs/analysis
```

Both paths are explicit and may be absolute. Run the module from the repository root or install the repository package so Python can find `analysis`. Choose a new output directory for each run; existing results are not overwritten.

The original Chinese CSV column names and documented English aliases are accepted. Missing values, duplicate point IDs, invalid coordinates and invalid pixel fractions are rejected. No imputation is fitted using validation or test data.

`data.py` preserves the H/E/V/A formulas, including normalization of the combined living-service and education POI value. THIR and feature ranges use the complete input study table, as in the historical index definition. This normalization is part of constructing the research target; it does not make the target independent of its inputs. Risk bins include zero, so all points are classified without filling missing classes with the majority label.

The seeded point split is approximately 60% training, 20% validation and 20% test. Only the validation partition selects the early-stopping iteration. Five-fold cross-validation runs only on the outer training partition, using fixed parameters without early stopping; failures are reported rather than replaced by another method. The final test is used only for reporting metrics. This random point split does not establish spatial generalization.

Outputs include:

- `point_results.csv`: all points, HEVA dimensions, THIR, corrected risk classes, partition and model predictions.
- `split_ids.csv`: reproducible point-to-partition assignments.
- `metrics.json`: RMSE/MAE/R², training-only CV, selected tree count, index normalization ranges, versions and input hash.
- `xgboost_model.json`: the fitted model including early-stopping attributes.
- `shap_values.csv`, `shap_importance.csv`, `shap_importance.png` and `shap_summary.png`: descriptive explanations for up to 256 study points. SHAP uses only the selected boosting rounds, and additivity is checked against model predictions.

The default parameters retain the historical depth, learning rate and sampling ratios; the estimator cap defaults to 300 with 20-round patience. `--max-estimators`, `--early-stopping-rounds`, `--seed`, `--cv-folds` and `--max-shap-samples` can be set explicitly. `--cv-folds 0` or `--skip-shap` explicitly skips that operation and is recorded in the results. No archived result is presented as newly reproduced by this module.

Run the tests with `python -m unittest discover -s analysis/tests -t .`. Tests include real small-model training, invariance to held-out test targets, saved-model prediction agreement and SHAP additivity. They do not download data or contact a map API.
