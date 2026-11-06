"""Reconcile recovered segmentation results with the historical ML feature table.

This module reads CSVs only. It does not rerun inference, change historical
definitions, reconstruct missing points, or overwrite the input feature table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SOURCE_FILES = {
    "street_images": "all_image_results.csv",
    "street_points": "folder_average_results.csv",
    "fisheye_points": "fisheye_view_factors.csv",
    "hw_points": "street_canyon_hw_analysis_results.csv",
}
METRICS = {
    "street_points": ("folder", (
        "avg_green_view_index", "avg_sky_visibility", "avg_enclosure_index")),
    "fisheye_points": ("folder_id", ("SVF", "TVF", "BVF")),
    "hw_points": ("point_id", ("hw_value",)),
}
RAW_METRICS = {
    "green_view_index": "avg_green_view_index",
    "sky_visibility": "avg_sky_visibility",
    "enclosure_index": "avg_enclosure_index",
}


def _point_id(value: str | None) -> int:
    try:
        number = Decimal((value or "").strip())
        if not number.is_finite() or number != number.to_integral_value() or number <= 0:
            raise ValueError
        return int(number)
    except (InvalidOperation, ValueError):
        raise ValueError("Point IDs must be positive integers.") from None


def _number(value: str | None) -> float:
    try:
        number = float(value or "")
    except (ValueError, TypeError):
        raise ValueError("A finite numeric value is required.") from None
    if not math.isfinite(number):
        raise ValueError("A finite numeric value is required.")
    return number


def reconcile(data_dir: str | Path, features: str | Path, *, atol: float = 1e-6) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a JSON-compatible audit report and per-metric comparison rows.

    Extra historical points are reported, rather than treated as invented ML
    rows. Every point actually in ``features`` must exist in all three recovered
    point-level sources. Duplicate IDs are errors, even if their values agree.
    """
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative.")
    data_dir, features = Path(data_dir), Path(features)
    issues: list[dict[str, Any]] = []
    datasets: dict[str, dict[str, Any]] = {}

    def issue(code: str, dataset: str, message: str, **extra: Any) -> None:
        issues.append({"code": code, "dataset": dataset, "message": message, **extra})

    def load(path: Path, dataset: str, required: set[str]) -> list[dict[str, str]]:
        meta: dict[str, Any] = {"file": path.name, "rows": 0, "unique_point_ids": 0}
        datasets[dataset] = meta
        try:
            raw = path.read_bytes()
            meta["sha256"] = hashlib.sha256(raw).hexdigest()
            text = raw.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text), strict=True)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)):
                issue("DUPLICATE_HEADERS", dataset, "CSV column names must be unique.")
            missing = sorted(required - set(headers))
            if missing:
                issue("MISSING_COLUMNS", dataset, "Required columns are absent.", columns=missing)
                return []
            rows = list(reader)
            meta["rows"] = len(rows)
            if not rows:
                issue("EMPTY_DATASET", dataset, "The dataset contains no data rows.")
            if any(None in row for row in rows):
                issue("MALFORMED_CSV", dataset, "A CSV row has more fields than its header.")
            return rows
        except (OSError, UnicodeError, csv.Error) as error:
            issue("UNREADABLE_DATASET", dataset, f"Cannot read {path.name}: {type(error).__name__}.")
            return []

    def index(rows: list[dict[str, str]], key: str, dataset: str) -> dict[int, dict[str, str]]:
        result: dict[int, dict[str, str]] = {}
        for row_number, row in enumerate(rows, start=2):
            try:
                point = _point_id(row.get(key))
            except ValueError:
                issue("INVALID_POINT_ID", dataset, "Point ID is not a positive integer.", row=row_number)
                continue
            if point in result:
                issue("DUPLICATE_POINT_ID", dataset, "A point occurs more than once; no row was silently replaced.",
                      point_id=point, row=row_number, conflicting=row != result[point])
                continue
            result[point] = row
        datasets[dataset]["unique_point_ids"] = len(result)
        return result

    all_metrics = {metric for _, metrics in METRICS.values() for metric in metrics}
    feature_rows = load(features, "features", {"序号", *all_metrics})
    feature_index = index(feature_rows, "序号", "features")
    indexes: dict[str, dict[int, dict[str, str]]] = {}
    for name, (key, metrics) in METRICS.items():
        rows = load(data_dir / SOURCE_FILES[name], name, {key, *metrics})
        indexes[name] = index(rows, key, name)

    comparison_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for source, (_, metrics) in METRICS.items():
        for metric in metrics:
            summary = {"source": source, "compared": 0, "matched": 0,
                       "mismatched": 0, "missing": 0, "invalid": 0, "max_absolute_difference": None}
            summaries[metric] = summary
            for point, feature in sorted(feature_index.items()):
                original = indexes[source].get(point)
                detail: dict[str, Any] = {"point_id": point, "metric": metric, "source": source,
                    "source_value": None, "feature_value": None, "absolute_difference": None, "status": "missing"}
                if original is None:
                    summary["missing"] += 1
                    issue("MISSING_SOURCE_POINT", source, "An ML point is absent from this source.", point_id=point, metric=metric)
                else:
                    try:
                        observed, expected = _number(original.get(metric)), _number(feature.get(metric))
                    except ValueError:
                        summary["invalid"] += 1
                        detail["status"] = "invalid"
                        issue("INVALID_METRIC", source, "Source or ML metric is missing/nonfinite.", point_id=point, metric=metric)
                    else:
                        difference = abs(observed - expected)
                        matched = difference <= atol
                        detail.update(source_value=observed, feature_value=expected,
                                      absolute_difference=difference, status="matched" if matched else "mismatched")
                        summary["compared"] += 1
                        summary["matched" if matched else "mismatched"] += 1
                        maximum = summary["max_absolute_difference"]
                        summary["max_absolute_difference"] = difference if maximum is None else max(maximum, difference)
                        if not matched:
                            issue("METRIC_MISMATCH", source, "Metric differs beyond the absolute tolerance.",
                                  point_id=point, metric=metric, absolute_difference=difference)
                comparison_rows.append(detail)

    raw_rows = load(data_dir / SOURCE_FILES["street_images"], "street_images", {"folder", "image", *RAW_METRICS})
    image_groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    image_keys: set[tuple[int, str]] = set()
    for row_number, row in enumerate(raw_rows, start=2):
        try:
            point = _point_id(row.get("folder"))
        except ValueError:
            issue("INVALID_POINT_ID", "street_images", "Image has no positive point ID.", row=row_number)
            continue
        image = row.get("image") or ""
        if not image.strip():
            issue("INVALID_IMAGE_ID", "street_images", "Image filename is empty.", point_id=point, row=row_number)
            continue
        if (point, image) in image_keys:
            issue("DUPLICATE_IMAGE", "street_images", "An image occurs more than once.", point_id=point, image=image)
            continue
        image_keys.add((point, image))
        image_groups[point].append(row)
    datasets["street_images"]["unique_point_ids"] = len(image_groups)

    aggregation = {"compared": 0, "matched": 0, "mismatched": 0}
    for point in sorted(set(image_groups) | set(indexes["street_points"])):
        images, point_row = image_groups.get(point), indexes["street_points"].get(point)
        if not images or point_row is None:
            issue("MISSING_STREET_AGGREGATION_POINT", "street_images", "Image and point-level street IDs disagree.", point_id=point)
            continue
        for raw_metric, mean_metric in RAW_METRICS.items():
            try:
                values = [_number(row.get(raw_metric)) for row in images]
                mean = math.fsum(values) / len(values)
                expected = _number(point_row.get(mean_metric))
            except ValueError:
                issue("INVALID_METRIC", "street_images", "A raw metric or saved average is nonfinite.", point_id=point, metric=raw_metric)
                continue
            difference = abs(mean - expected)
            matched = difference <= atol
            aggregation["compared"] += 1
            aggregation["matched" if matched else "mismatched"] += 1
            if not matched:
                issue("STREET_AVERAGE_MISMATCH", "street_points", "Raw image average differs from the saved point average.",
                      point_id=point, metric=mean_metric, absolute_difference=difference)

    complete_ids = set.intersection(*(set(rows) for rows in indexes.values()))
    report = {
        "passed": not issues,
        "absolute_tolerance": atol,
        "datasets": datasets,
        "metrics": summaries,
        "raw_street_average_check": aggregation,
        "image_count_distribution": dict(sorted(Counter(len(rows) for rows in image_groups.values()).items())),
        "points_with_nonfour_image_count": {str(point): len(rows) for point, rows in sorted(image_groups.items()) if len(rows) != 4},
        "extra_historical_point_ids": {name: sorted(set(rows) - set(feature_index)) for name, rows in indexes.items()},
        "historical_complete_point_count": len(complete_ids),
        "complete_historical_ids_not_in_ml": sorted(complete_ids - set(feature_index)),
        "issues": issues,
        "provenance": {
            "street": "Historical Cityscapes metrics: GVI=vegetation+terrain, sky=sky, enclosure=building+wall+fence; per-point image means.",
            "fisheye": "Recovered CSV uses the original SegFormer/Cityscapes circle and sin(zenith) weighting; values are not recomputed or treated as calibrated physical view factors.",
            "hw": "Recovered hw_value is the original empirical pixel-ratio estimate, not a trained or surveyed height/width measurement.",
            "features": "The existing 238-point study table remains the historical modeling input; extra recovered points are not automatically added.",
        },
    }
    return report, comparison_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the four recovered CSVs.")
    parser.add_argument("--features", type=Path, required=True, help="Historical ML CSV with 序号 and seven recovered metrics.")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path; a sibling CSV contains per-point comparisons.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute comparison tolerance (default: 1e-6).")
    args = parser.parse_args(argv)
    json_path = args.output.with_suffix(".json") if args.output.suffix.lower() == ".csv" else args.output
    csv_path = json_path.with_suffix(".csv")
    if json_path.resolve() == csv_path.resolve():
        parser.error("Choose a JSON or non-CSV output path.")
    inputs = {args.features.resolve(), *( (args.data_dir / filename).resolve() for filename in SOURCE_FILES.values())}
    if json_path.resolve() in inputs or csv_path.resolve() in inputs:
        parser.error("Reports must not overwrite any historical input file.")
    try:
        report, details = reconcile(args.data_dir, args.features, atol=args.atol)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            fields = ["point_id", "metric", "source", "source_value", "feature_value", "absolute_difference", "status"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(details)
    except (OSError, ValueError) as error:
        print(f"History reconciliation failed: {error}", file=sys.stderr)
        return 2
    print(f"Historical reconciliation {'passed' if report['passed'] else 'FAILED'}: {len(report['issues'])} issue(s).")
    print(f"JSON: {json_path}\nComparisons: {csv_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
