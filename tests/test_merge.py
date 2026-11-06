import copy
import csv
import tempfile
import unittest
from pathlib import Path
from segmentation.merge import merge_tables, main, read_table


class MergeTest(unittest.TestCase):
    def fixtures(self):
        street = {'1': {'green_view_index': .3, 'sky_visibility': .4, 'enclosure_index': .2,
                        'complete_four_views': 'True', 'metric_policy': 'historical',
                        'longitude': '121', 'latitude': '38'}}
        fish = {'1': {'SVF': .4, 'TVF': .3, 'BVF': .2, 'metric_policy': 'historical'}}
        context = {'1': dict(longitude=121, latitude=38, height_width_ratio=.8, housing_price=10000,
                            population_density=7000, medical_poi=.1, living_service_poi=.2,
                            education_poi=.1, road_network_density=.5,
                            green_view_index=.99, sky_view_factor=.99)}
        return street, fish, context

    def test_only_image_columns_are_replaced_and_context_is_kept(self):
        street, fish, context = self.fixtures()
        original = copy.deepcopy(context)
        rows, report = merge_tables(street, fish, context)
        self.assertEqual(rows[0]['green_view_index'], .3)
        self.assertEqual(rows[0]['sky_view_factor'], .4)
        self.assertEqual(rows[0]['height_width_ratio'], .8)
        self.assertEqual(context, original)
        self.assertFalse(report['partial'])

    def test_partial_join_and_incomplete_views_are_never_silent(self):
        s, f, c = self.fixtures(); c['2'] = dict(c['1'])
        with self.assertRaises(ValueError): merge_tables(s, f, c)
        rows, report = merge_tables(s, f, c, allow_partial=True)
        self.assertEqual(len(rows), 1); self.assertEqual(report['omitted_context_ids'], ['2'])
        s['1']['complete_four_views'] = 'False'
        with self.assertRaises(ValueError): merge_tables(s, f, c, allow_partial=True)

    def test_rejects_mixed_policies_and_mismatched_coordinates(self):
        s, f, c = self.fixtures(); f['1']['metric_policy'] = 'vegetation-only'
        with self.assertRaises(ValueError): merge_tables(s, f, c)

    def test_actual_csv_cli_maps_fisheye_aliases_without_losing_the_values(self):
        s, f, c = self.fixtures()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for filename, records, identity in [('street.csv', s, 'point_id'), ('fish.csv', f, 'point_id'), ('context.csv', c, 'ID')]:
                rows = [{identity: key, **row} for key, row in records.items()]
                with (root / filename).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            main(['--street', str(root/'street.csv'), '--fisheye', str(root/'fish.csv'),
                  '--context', str(root/'context.csv'), '--output', str(root/'joined.csv')])
            row = read_table(root/'joined.csv', 'ID')['1']
            self.assertEqual(float(row['sky_view_factor']), .4)
            self.assertEqual(float(row['tree_view_factor']), .3)
            self.assertTrue((root/'joined.provenance.json').is_file())
        f['1']['metric_policy'] = 'historical'; s['1']['longitude'] = '120'
        with self.assertRaises(ValueError): merge_tables(s, f, c)


if __name__ == '__main__': unittest.main()
