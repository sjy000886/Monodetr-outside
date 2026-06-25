import unittest

import numpy as np

from tools.evaluate_outside_center_compensation import center_error_metrics
from tools.evaluate_outside_center_compensation import pairwise_iou
from tools.evaluate_outside_center_compensation import summarize_records
from tools.evaluate_outside_center_compensation import (
    summarize_unique_source_objects,
)


class OutsideCenterCompensationTest(unittest.TestCase):
    def test_pairwise_iou_handles_overlap_and_degenerate_boxes(self):
        boxes1 = np.array([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        boxes2 = np.array([
            [0.5, 0.0, 1.5, 1.0],
        ])
        overlaps = pairwise_iou(boxes1, boxes2)
        np.testing.assert_allclose(overlaps[:, 0], [1.0 / 3.0, 0.0])

    def test_center_metrics_report_successful_compensation(self):
        metrics = center_error_metrics(
            predicted_representative=[0.9, 0.5],
            predicted_offset=[0.2, 0.0],
            target_center=[1.1, 0.5],
            target_offset=[0.2, 0.0],
        )
        self.assertAlmostEqual(
            metrics["representative_error_px"], 256.0, places=5
        )
        self.assertAlmostEqual(metrics["full_error_px"], 0.0, places=5)
        self.assertTrue(metrics["improved"])
        self.assertAlmostEqual(metrics["offset_cosine"], 1.0, places=6)

    def test_summary_uses_only_requested_iou_matches(self):
        base = {
            "assigned": True,
            "representative_error_px": 10.0,
            "full_error_px": 5.0,
            "error_delta_px": -5.0,
            "target_offset_norm_px": 8.0,
            "predicted_offset_norm_px": 7.0,
            "offset_error_px": 1.0,
            "offset_cosine": 1.0,
            "improved": True,
        }
        records = [
            dict(base, bbox_iou=0.6),
            dict(base, bbox_iou=0.4),
        ]
        summary = summarize_records(records, match_iou=0.5)
        self.assertEqual(summary["targets"], 2)
        self.assertEqual(summary["matched_iou_0.5"], 1)
        self.assertEqual(summary["metric_matches"], 1)
        self.assertEqual(summary["improved_fraction"], 1.0)

    def test_unique_source_summary_averages_repeated_crops(self):
        base = {
            "image_id": 1,
            "target_slot": 2,
            "assigned": True,
            "bbox_iou": 0.8,
            "representative_error_px": 10.0,
            "full_error_px": 5.0,
            "error_delta_px": -5.0,
            "target_offset_norm_px": 8.0,
            "predicted_offset_norm_px": 7.0,
            "offset_error_px": 1.0,
            "offset_cosine": 1.0,
            "improved": True,
        }
        records = [
            dict(base),
            dict(base, full_error_px=7.0, error_delta_px=-3.0),
        ]
        summary = summarize_unique_source_objects(records, match_iou=0.5)
        self.assertEqual(summary["source_objects_total"], 1)
        self.assertEqual(summary["source_objects_matched"], 1)
        self.assertEqual(summary["metric_matches"], 1)
        self.assertAlmostEqual(summary["full_error_px_mean"], 6.0)
        self.assertAlmostEqual(summary["error_delta_px_mean"], -4.0)


if __name__ == "__main__":
    unittest.main()
