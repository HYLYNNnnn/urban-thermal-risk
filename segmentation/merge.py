"""Explicitly replace image-derived columns, preserving independently supplied GIS context."""
import argparse
import csv
import json
import math
from pathlib import Path

ALIASES = {'序号': 'ID', '经度': 'longitude', '纬度': 'latitude', 'SVF': 'sky_view_factor',
           'TVF': 'tree_view_factor', 'BVF': 'building_view_factor',
           'avg_green_view_index': 'green_view_index', 'avg_sky_visibility': 'sky_visibility',
           'avg_enclosure_index': 'enclosure_index', 'hw_value': 'height_width_ratio',
           '房价': 'housing_price', 'popdensity': 'population_density', 'poi医疗': 'medical_poi',
           'poi生活服务': 'living_service_poi', 'poi教育': 'education_poi', '路网密度': 'road_network_density'}
CONTEXT = ('longitude', 'latitude', 'height_width_ratio', 'housing_price', 'population_density',
           'medical_poi', 'living_service_poi', 'education_poi', 'road_network_density')
IMAGE_FEATURES = ('sky_view_factor', 'tree_view_factor', 'building_view_factor',
                  'green_view_index', 'sky_visibility', 'enclosure_index')


def number(value, label):
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f'Missing/non-numeric {label}.') from error
    if not math.isfinite(result):
        raise ValueError(f'Non-finite {label}.')
    return result


def read_table(path, id_column):
    rows = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = [ALIASES.get(name, name) for name in (reader.fieldnames or [])]
        if len(fields) != len(set(fields)):
            raise ValueError(f'Duplicate columns after applying aliases in {path}.')
        for row in reader:
            row = {ALIASES.get(k, k): v for k, v in row.items()}
            value = number(row.get(id_column), 'point ID')
            if not value.is_integer() or value < 1:
                raise ValueError('Point IDs must be positive integers.')
            identity = str(int(value))
            if identity in rows:
                raise ValueError(f'Duplicate point ID {identity} in {path}.')
            rows[identity] = row
    if not rows:
        raise ValueError(f'Empty table: {path}.')
    return rows


def merge_tables(street, fisheye, context, *, allow_partial=False):
    selected = set(street) & set(fisheye) & set(context)
    missing = sorted(set(context) - selected, key=int)
    extra = sorted((set(street) | set(fisheye)) - set(context), key=int)
    if extra:
        raise ValueError(f'Segmentation points missing from context: {extra}.')
    if missing and not allow_partial:
        raise ValueError(f'Missing segmentation for {len(missing)} context points. Use --allow-partial explicitly for a sample subset.')
    if not selected:
        raise ValueError('No common point IDs across street, fisheye and context tables.')
    output = []
    for identity in sorted(selected, key=int):
        s, f, c = street[identity], fisheye[identity], context[identity]
        if s.get('metric_policy') not in ('historical', 'vegetation-only') or f.get('metric_policy') != s.get('metric_policy'):
            raise ValueError(f'Missing/mixed metric policy for point {identity}.')
        if str(s.get('complete_four_views', '')).lower() not in ('true', '1'):
            raise ValueError(f'Point {identity} needs four distinct cardinal street views.')
        row = {'ID': int(identity)}
        for name in CONTEXT:
            row[name] = number(c.get(name), f'{name} for {identity}')
        for name in ('longitude', 'latitude'):
            if s.get(name) not in (None, '') and abs(number(s[name], name) - row[name]) > 1e-6:
                raise ValueError(f'{name} disagrees with context for point {identity}.')
        for name in ('green_view_index', 'sky_visibility', 'enclosure_index'):
            row[name] = number(s.get(name), name)
        for source, destination in [('SVF', 'sky_view_factor'), ('TVF', 'tree_view_factor'), ('BVF', 'building_view_factor')]:
            row[destination] = number(f.get(destination, f.get(source)), source)
        if any(not 0 <= row[name] <= 1 for name in IMAGE_FEATURES):
            raise ValueError(f'Image-derived fractions must lie in [0,1] for point {identity}.')
        row['metric_policy'] = s['metric_policy']
        output.append(row)
    return output, {'output_points': len(output), 'omitted_context_ids': missing,
                    'context_rows': len(context), 'partial': bool(missing),
                    'height_width_source': 'supplied context; no new H/W inference',
                    'gis_source': 'supplied context; not regenerated from images'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--street', type=Path, required=True)
    parser.add_argument('--fisheye', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.with_suffix('.provenance.json').exists():
            raise ValueError('Choose a new output file to avoid overwriting prior results.')
        rows, report = merge_tables(read_table(args.street, 'point_id'), read_table(args.fisheye, 'point_id'),
                                    read_table(args.context, 'ID'), allow_partial=args.allow_partial)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        report.update(street=str(args.street), fisheye=str(args.fisheye), context=str(args.context))
        args.output.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Merged {len(rows)} points; omitted {len(report["omitted_context_ids"])} explicitly reported context points.')
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
