"""Train an index surrogate using disjoint train/validation/test partitions.

Example:
    python -m analysis.train --input data/processed/historical_features.csv --output outputs/analysis
"""

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .data import FEATURES, RISK_LABELS, SCOPE_NOTE, TARGET, calculate_thir, load_features, normalization_ranges


@dataclass(frozen=True)
class Split:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


def split_rows(count: int, seed: int = 42) -> Split:
    """Reproducible random point split, approximately 60/20/20.

    This is not a spatial-block split. Neighboring points can share surroundings.
    """
    if count < 20:
        raise ValueError("Use at least 20 points for train/validation/test evaluation.")
    order = np.random.default_rng(seed).permutation(count).tolist()
    test_size = max(2, int(np.ceil(count * .2)))
    validation_size = max(2, int(np.ceil(count * .2)))
    return Split(tuple(order[test_size + validation_size:]), tuple(order[test_size:test_size + validation_size]), tuple(order[:test_size]))


def validate_split(split: Split, count: int) -> None:
    parts = [set(split.train), set(split.validation), set(split.test)]
    if any(len(part) < 2 for part in parts) or any(parts[i] & parts[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError("Train, validation and test must be nonempty disjoint partitions.")
    flattened = [*split.train, *split.validation, *split.test]
    if len(flattened) != count or set(flattened) != set(range(count)):
        raise ValueError("Split must assign each row exactly once.")


def regression_metrics(actual, predicted) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mae": float(mean_absolute_error(actual, predicted)), "r2": float(r2_score(actual, predicted))}


def fit_model(frame: pd.DataFrame, split: Split, *, seed: int = 42, max_estimators: int = 300,
              early_stopping_rounds: int = 20, cv_folds: int = 5):
    """Fit only train rows; use only validation rows for early stopping.

    The fixed-parameter CV runs only on the training partition and has no early
    stopping. It neither sees the outer validation/test rows nor changes the
    fitted model. No fallback silently substitutes a different CV procedure.
    """
    from sklearn.model_selection import KFold
    from xgboost import XGBRegressor

    validate_split(split, len(frame))
    if max_estimators < 1 or not 1 <= early_stopping_rounds <= max_estimators:
        raise ValueError("Early stopping rounds must be positive and no larger than max estimators.")
    if cv_folds != 0 and (cv_folds < 2 or len(split.train) < 2 * cv_folds):
        raise ValueError("Each cross-validation fold needs at least two validation points; use 0 to skip CV.")
    X, y = frame.loc[:, list(FEATURES)], frame[TARGET]
    parameters = dict(n_estimators=max_estimators, max_depth=6, learning_rate=.1, subsample=.8,
                      colsample_bytree=.8, random_state=seed, objective="reg:squarederror", tree_method="hist", n_jobs=1)
    model = XGBRegressor(**parameters, eval_metric="rmse", early_stopping_rounds=early_stopping_rounds)
    model.fit(X.iloc[list(split.train)], y.iloc[list(split.train)],
              eval_set=[(X.iloc[list(split.validation)], y.iloc[list(split.validation)])], verbose=False)
    predictions = model.predict(X)
    metrics = {name: regression_metrics(y.iloc[list(rows)], predictions[list(rows)])
               for name, rows in [("train", split.train), ("validation", split.validation), ("test", split.test)]}
    scores = []
    if cv_folds:
        training_X, training_y = X.iloc[list(split.train)], y.iloc[list(split.train)]
        for train_rows, held_out_rows in KFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(training_X):
            fold_model = XGBRegressor(**parameters, eval_metric="rmse")
            fold_model.fit(training_X.iloc[train_rows], training_y.iloc[train_rows], verbose=False)
            scores.append(regression_metrics(training_y.iloc[held_out_rows], fold_model.predict(training_X.iloc[held_out_rows])))
    cv = {"folds": cv_folds, "population": "outer_training_partition_only", "early_stopping": False, "scores": scores}
    if scores:
        cv["mean_r2"] = float(np.mean([score["r2"] for score in scores]))
        cv["std_r2"] = float(np.std([score["r2"] for score in scores]))
    return model, predictions, {"partition_metrics": metrics, "cross_validation": cv,
                                "selected_tree_count": int(model.best_iteration + 1),
                                "early_stopping_rounds": early_stopping_rounds,
                                "evaluation_metric": "rmse", "model_parameters": parameters}


def write_shap(model, frame: pd.DataFrame, output: Path, *, seed: int, max_samples: int) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    if max_samples < 1:
        raise ValueError("SHAP sample limit must be positive.")
    positions = np.arange(len(frame))
    if len(positions) > max_samples:
        positions = np.sort(np.random.default_rng(seed).choice(positions, max_samples, replace=False))
    sample = frame.iloc[positions]
    X = sample.loc[:, list(FEATURES)]
    # Explicitly slice to the selected boosting round: patience trees must not
    # be explained when sklearn's predict() uses best_iteration automatically.
    booster = model.get_booster()[:model.best_iteration + 1]
    explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
    values = np.asarray(explainer.shap_values(X), dtype=float)
    baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    if values.shape != X.shape or not np.isfinite(values).all():
        raise ValueError("SHAP returned an invalid explanation matrix.")
    if not np.allclose(values.sum(axis=1) + baseline, model.predict(X), rtol=1e-4, atol=1e-5):
        raise ValueError("SHAP explanations do not match the selected model predictions.")
    exported = pd.DataFrame(values, columns=list(FEATURES))
    exported.insert(0, "ID", sample.ID.to_numpy())
    exported["base_value"] = baseline
    exported.to_csv(output / "shap_values.csv", index=False)
    importance = pd.DataFrame({"feature": FEATURES, "mean_absolute_shap": np.abs(values).mean(axis=0)}).sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(output / "shap_importance.csv", index=False)
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.barh(importance.feature.iloc[::-1], importance.mean_absolute_shap.iloc[::-1])
    axis.set(xlabel="Mean absolute SHAP value", title="Feature importance for the constructed THIR index")
    fig.tight_layout(); fig.savefig(output / "shap_importance.png", dpi=160); plt.close(fig)
    shap.summary_plot(values, X, show=False, max_display=len(FEATURES), plot_size=(10, 7))
    plt.title("SHAP distribution for the constructed THIR index")
    plt.tight_layout(); plt.savefig(output / "shap_summary.png", dpi=160, bbox_inches="tight"); plt.close("all")
    return {"sample_count": len(sample), "population": "study_table_descriptive_explanation", "selected_tree_count": int(model.best_iteration + 1)}


def run_analysis(input_path: str | Path, output_path: str | Path, *, seed: int = 42,
                 max_estimators: int = 300, early_stopping_rounds: int = 20,
                 cv_folds: int = 5, max_shap_samples: int = 256, with_shap: bool = True) -> dict:
    source, output = Path(input_path).expanduser().resolve(), Path(output_path).expanduser().resolve()
    if output.exists():
        raise ValueError("Choose a new output directory; existing results are never overwritten.")
    frame = calculate_thir(load_features(source))
    split = split_rows(len(frame), seed)
    model, predictions, metrics = fit_model(frame, split, seed=seed, max_estimators=max_estimators,
                                           early_stopping_rounds=early_stopping_rounds, cv_folds=cv_folds)
    assignment = np.empty(len(frame), dtype=object)
    for name, indices in [("train", split.train), ("validation", split.validation), ("test", split.test)]:
        assignment[list(indices)] = name
    frame["split"] = assignment
    frame["predicted_THIR_normalized"] = predictions
    metrics.update({"sample_count": len(frame), "seed": seed, "target": TARGET, "scope_note": SCOPE_NOTE,
                    "input_file": source.name, "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "split_counts": {name: len(rows) for name, rows in [("train", split.train), ("validation", split.validation), ("test", split.test)]},
                    "risk_counts": {name: int((frame.risk_level == name).sum()) for name in RISK_LABELS},
                    "normalization_population": "complete_input_study_table", "normalization_ranges": normalization_ranges(frame),
                    "early_stopping_population": "validation_only", "spatial_block_validation": False,
                    "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost"]}})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".thir-analysis-", dir=output.parent) as temp:
        stage = Path(temp) / "results"; stage.mkdir()
        frame.to_csv(stage / "point_results.csv", index=False, encoding="utf-8-sig")
        frame.loc[:, ["ID", "split"]].to_csv(stage / "split_ids.csv", index=False)
        model.save_model(stage / "xgboost_model.json")
        if with_shap:
            metrics["shap"] = write_shap(model, frame, stage, seed=seed, max_samples=max_shap_samples)
            metrics["versions"].update({name: importlib.metadata.version(name) for name in ["shap", "matplotlib"]})
        else:
            metrics["shap"] = {"performed": False, "reason": "explicit_skip_shap_option"}
        (stage / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Historical feature CSV (Chinese or English schema)")
    parser.add_argument("--output", required=True, type=Path, help="A new directory for generated results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-estimators", type=int, default=300)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=5, help="Training-only folds, or 0 to skip")
    parser.add_argument("--max-shap-samples", type=int, default=256)
    parser.add_argument("--skip-shap", action="store_true", help="Explicitly skip SHAP, recorded in metrics")
    args = parser.parse_args(argv)
    try:
        metrics = run_analysis(args.input, args.output, seed=args.seed, max_estimators=args.max_estimators,
                               early_stopping_rounds=args.early_stopping_rounds, cv_folds=args.cv_folds,
                               max_shap_samples=args.max_shap_samples, with_shap=not args.skip_shap)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"Analysis failed: {error}\n")
    print(json.dumps({"output": str(args.output.resolve()), "samples": metrics["sample_count"],
                      "test": metrics["partition_metrics"]["test"], "scope_note": SCOPE_NOTE}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
