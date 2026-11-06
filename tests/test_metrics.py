import unittest
import numpy as np
from segmentation.metrics import compute_metrics, aggregate_rows
from segmentation.inputs import image_metadata
from pathlib import Path


class MetricsTest(unittest.TestCase):
    def test_real_class_ids_and_terrain_policy_are_explicit(self):
        mask = np.array([[8, 9], [10, 2]], dtype=np.uint8)
        old = compute_metrics(mask, view='street')
        new = compute_metrics(mask, view='street', policy='vegetation-only')
        self.assertEqual(old['green_view_index'], .5)
        self.assertEqual(new['green_view_index'], .25)
        self.assertEqual(old['vegetation_only_fraction'], .25)
        self.assertEqual(old['sky_visibility'], .25)
        self.assertEqual(old['enclosure_index'], .25)
        self.assertAlmostEqual(sum(v for k, v in old.items() if k.startswith('class_')), 1)

    def test_fisheye_excludes_corners_and_keeps_unassigned_categories(self):
        mask = np.full((5, 5), 10, dtype=np.uint8)
        mask[0, 0] = 8  # Outside circle: must not change the sky fraction.
        result = compute_metrics(mask, view='fisheye')
        self.assertAlmostEqual(result['SVF'], 1)
        self.assertEqual(result['TVF'], 0)
        road = compute_metrics(np.zeros((5, 5), dtype=np.uint8), view='fisheye')
        self.assertEqual(road['SVF'] + road['TVF'] + road['BVF'], 0)

    def test_rejects_color_masks_wrong_taxonomy_and_empty_support(self):
        for mask in [np.zeros((2, 2, 3), dtype=np.uint8), np.full((2, 2), 21),
                     np.full((2, 2), 255, dtype=np.uint8), np.full((2, 2), .1)]:
            with self.assertRaises(ValueError): compute_metrics(mask, view='street')

    def test_aggregation_uses_equal_direction_weights_and_rejects_duplicates(self):
        rows = []
        for heading, shape, label in [(0, (2, 2), 8), (90, (10, 10), 10), (180, (2, 2), 2), (270, (2, 2), 0)]:
            rows.append({'point_id': '1', 'image': str(heading), 'heading': heading,
                         **compute_metrics(np.full(shape, label), view='street')})
        point = aggregate_rows(rows, view='street')[0]
        self.assertEqual(point['green_view_index'], .25)
        self.assertTrue(point['complete_four_views'])
        with self.assertRaises(ValueError): aggregate_rows(rows + [rows[0]], view='street')
        with self.assertRaises(ValueError): aggregate_rows(rows[:2], view='fisheye')

    def test_metadata_preserves_point_coordinates_and_detects_folder_mismatch(self):
        root = Path('/tmp/sample')
        row = image_metadata(root/'1/1_121.620895_38.918283_270.png', root)
        self.assertEqual(row['point_id'], '1'); self.assertEqual(row['heading'], 270)
        self.assertAlmostEqual(row['longitude'], 121.620895)
        self.assertEqual(image_metadata(root/'1/1_fisheye.jpg', root)['point_id'], '1')
        with self.assertRaises(ValueError): image_metadata(root/'2/1_fisheye.jpg', root)
        with self.assertRaises(ValueError): image_metadata(root/'unidentified.jpg', root)


if __name__ == '__main__': unittest.main()
