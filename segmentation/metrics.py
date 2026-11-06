"""Metrics matching the recovered Cityscapes pipeline, with policy recorded explicitly.

The historical GVI/TVF include terrain as well as vegetation. Terrain is not a
synonym for plants, so vegetation-only values are always exported separately.
Historical fisheye values use the recovered sin(theta) per-pixel weights; these
are reproducible proxies, not calibrated physical radiation view factors.
"""
from collections import defaultdict
import math
import numpy as np

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
           'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
           'truck', 'bus', 'train', 'motorcycle', 'bicycle')
POLICIES = ('historical', 'vegetation-only')


def valid_region(shape: tuple[int, int], view: str) -> np.ndarray:
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError('Expected a nonempty two-dimensional label mask.')
    if view == 'street':
        return np.ones(shape, dtype=bool)
    if view != 'fisheye':
        raise ValueError('View must be street or fisheye.')
    h, w = shape
    cy, cx = h // 2, w // 2
    radius = min(cx, cy)
    if radius < 1:
        raise ValueError('Fisheye images need a radius of at least one pixel.')
    y, x = np.ogrid[:h, :w]
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2


def compute_metrics(mask: np.ndarray, *, view: str, policy: str = 'historical') -> dict:
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError('Use a two-dimensional integer class-ID mask, not RGB colors.')
    if policy not in POLICIES:
        raise ValueError(f'Metric policy must be one of {POLICIES}.')
    region = valid_region(mask.shape, view)
    valid = region & (mask != 255)
    if not valid.any():
        raise ValueError('The analyzed region has no valid labels.')
    if (mask[valid] < 0).any() or (mask[valid] >= len(CLASSES)).any():
        raise ValueError('Labels must match Cityscapes IDs 0–18; 255 means ignored.')
    weights = np.ones(mask.shape, dtype=np.float64)
    if view == 'fisheye':
        h, w = mask.shape
        cy, cx = h // 2, w // 2
        radius = min(cx, cy)
        y, x = np.ogrid[:h, :w]
        theta = np.hypot(x - cx, y - cy) / radius * (np.pi / 2)
        weights = np.sin(theta)
    denominator = float(weights[valid].sum())
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError('The analyzed region has no positive finite weight.')
    ratios = np.bincount(mask[valid].astype(np.int64), weights=weights[valid], minlength=len(CLASSES)) / denominator
    result = {f'class_{name.replace(" ", "_")}_fraction': float(ratios[i]) for i, name in enumerate(CLASSES)}
    vegetation, terrain = float(ratios[8]), float(ratios[9])
    result.update(vegetation_only_fraction=vegetation, terrain_fraction=terrain,
                  valid_pixels=int(valid.sum()), total_pixels=int(mask.size),
                  analyzed_region_pixels=int(region.sum()), weight_sum=denominator,
                  metric_policy=policy)
    green = vegetation + terrain if policy == 'historical' else vegetation
    if view == 'street':
        result.update(green_view_index=green, sky_visibility=float(ratios[10]),
                      enclosure_index=float(ratios[2] + ratios[3] + ratios[4]))
    else:
        result.update(SVF=float(ratios[10]), TVF=green, BVF=float(ratios[2] + ratios[3]))
    return result


def aggregate_rows(rows: list[dict], *, view: str) -> list[dict]:
    """Equal-image averaging matches the original four-direction calculation."""
    if view not in ('street', 'fisheye'):
        raise ValueError('View must be street or fisheye.')
    groups = defaultdict(list)
    seen_images = set()
    for row in rows:
        identity = (str(row['point_id']), str(row['image']))
        if identity in seen_images:
            raise ValueError(f'Duplicate image result: {identity}')
        seen_images.add(identity)
        groups[str(row['point_id'])].append(row)
    output = []
    for point_id, items in sorted(groups.items(), key=lambda x: int(x[0])):
        if view == 'fisheye' and len(items) != 1:
            raise ValueError(f'Expected one fisheye image per point, got {len(items)} for {point_id}.')
        coordinates = [(r.get('longitude'), r.get('latitude')) for r in items if r.get('longitude') is not None]
        if coordinates and any(abs(a[0] - coordinates[0][0]) > 1e-6 or abs(a[1] - coordinates[0][1]) > 1e-6 for a in coordinates):
            raise ValueError(f'Inconsistent coordinates for point {point_id}.')
        headings = [float(r['heading']) % 360 for r in items if r.get('heading') is not None]
        if len(headings) != len(set(headings)):
            raise ValueError(f'Duplicate headings for point {point_id}.')
        result = {'point_id': point_id, 'image_count': len(items),
                  'longitude': coordinates[0][0] if coordinates else None,
                  'latitude': coordinates[0][1] if coordinates else None,
                  'valid_pixels_total': sum(int(r['valid_pixels']) for r in items)}
        policies = {r.get('metric_policy', 'historical') for r in items}
        if len(policies) != 1:
            raise ValueError(f'Mixed metric policies for point {point_id}.')
        result['metric_policy'] = policies.pop()
        if view == 'street':
            result.update(heading_count=len(headings), complete_four_views=set(headings) == {0., 90., 180., 270.})
        names = [key for key in items[0] if key.startswith('class_') or key in
                 ['green_view_index', 'sky_visibility', 'enclosure_index', 'SVF', 'TVF', 'BVF', 'vegetation_only_fraction', 'terrain_fraction']]
        for key in names:
            values = [float(r[key]) for r in items]
            if not all(map(math.isfinite, values)):
                raise ValueError(f'Non-finite metric {key} for point {point_id}.')
            result[key] = sum(values) / len(values)
        output.append(result)
    return output
