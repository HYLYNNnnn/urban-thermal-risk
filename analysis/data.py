"""Validate historical feature tables and preserve the original HEVA definition."""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_ALIASES = {
    "序号": "ID", "经度": "longitude", "纬度": "latitude",
    "SVF": "sky_view_factor", "TVF": "tree_view_factor", "BVF": "building_view_factor",
    "avg_green_view_index": "green_view_index", "avg_sky_visibility": "sky_visibility",
    "avg_enclosure_index": "enclosure_index", "hw_value": "height_width_ratio",
    "房价": "housing_price", "popdensity": "population_density", "poi医疗": "medical_poi",
    "poi生活服务": "living_service_poi", "poi教育": "education_poi", "路网密度": "road_network_density",
}
FEATURES = (
    "sky_view_factor", "tree_view_factor", "building_view_factor", "green_view_index",
    "sky_visibility", "enclosure_index", "height_width_ratio", "housing_price",
    "population_density", "medical_poi", "living_service_poi", "education_poi",
    "road_network_density",
)
RISK_LABELS = ("Low Risk", "Medium Risk", "High Risk", "Extreme Risk")
TARGET = "THIR_normalized"
SCOPE_NOTE = (
    "THIR is constructed from these environmental features using the original fixed HEVA "
    "weights and full-study-table min/max normalization. Scores describe a surrogate for "
    "that constructed index, not independent temperature measurements, causal cooling "
    "effects, or spatial generalization. The held-out test partition is never used for "
    "early stopping or cross-validation."
)


def load_features(path: str | Path) -> pd.DataFrame:
    """Accept the original Chinese schema or the documented English column names.

    Missing/invalid values are rejected rather than imputed using the full dataset.
    """
    frame = pd.read_csv(Path(path), encoding="utf-8-sig").rename(columns=COLUMN_ALIASES)
    if frame.columns.duplicated().any():
        raise ValueError("The table has duplicate columns after applying feature aliases.")
    columns = ["ID", "longitude", "latitude", *FEATURES]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    frame = frame.loc[:, columns].copy()
    for name in columns:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("The input must contain finite numeric values without missing rows.")
    if (frame.ID < 1).any() or (frame.ID % 1 != 0).any() or frame.ID.duplicated().any():
        raise ValueError("Point IDs must be unique positive integers.")
    if not frame.longitude.between(-180, 180).all() or not frame.latitude.between(-90, 90).all():
        raise ValueError("Coordinates must be longitude/latitude in degrees.")
    validate_environment(frame)
    frame["ID"] = frame.ID.astype("int64")
    return frame.reset_index(drop=True)


def validate_environment(frame: pd.DataFrame) -> None:
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing environmental features: {', '.join(missing)}")
    values = frame.loc[:, list(FEATURES)].to_numpy(dtype=float)
    if not len(frame) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Environmental features must be finite, nonnegative numbers.")
    # Six image ratios have an explicit pixel-fraction interpretation.
    if (frame.loc[:, list(FEATURES[:6])].to_numpy(dtype=float) > 1).any():
        raise ValueError("Image-derived fractions must be within 0 and 1.")


def normalize(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    width = values.max() - values.min()
    if width == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.min()) / width


def calculate_thir(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct the historical study's HEVA/THIR definition, without mutating input.

    Normalization uses the complete study table intentionally to reproduce the
    research index. It is not a fitted input preprocessing step or an independent
    observed target. A new study area must document its normalization population.
    """
    validate_environment(frame)
    result = frame.copy()
    n = lambda name: normalize(result[name])
    result["H_hazard"] = .4 * n("sky_view_factor") + .3 * (1 - n("tree_view_factor")) + .3 * n("height_width_ratio")
    result["E_exposure"] = .4 * n("population_density") + .3 * normalize(result.living_service_poi + result.education_poi) + .3 * n("road_network_density")
    result["V_vulnerability"] = .4 * (1 - n("green_view_index")) + .3 * n("sky_visibility") + .2 * n("enclosure_index") + .1 * (1 - n("medical_poi"))
    result["A_adaptability"] = .25 * (n("housing_price") + n("medical_poi") + n("tree_view_factor") + n("green_view_index"))
    result["THIR"] = result.H_hazard + result.E_exposure + result.V_vulnerability - result.A_adaptability
    result[TARGET] = normalize(result.THIR)
    result["risk_level"] = pd.cut(result[TARGET], bins=[0, .3, .5, .7, 1], labels=list(RISK_LABELS), include_lowest=True)
    if result.risk_level.isna().any():
        raise ValueError("Every THIR value must receive a risk level.")
    result["risk_level_num"] = result.risk_level.map(dict(zip(RISK_LABELS, range(1, 5)))).astype("int64")
    return result


def normalization_ranges(frame: pd.DataFrame) -> dict:
    """Record the study population's ranges for auditability."""
    columns = [*FEATURES, "THIR"]
    ranges = {name: {"min": float(frame[name].min()), "max": float(frame[name].max())} for name in columns}
    poi = frame.living_service_poi + frame.education_poi
    ranges["living_plus_education_poi"] = {"min": float(poi.min()), "max": float(poi.max())}
    return ranges
