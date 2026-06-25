import unittest

import numpy as np

from lib.datasets.kitti.outside_center_utils import (
    compute_boundary_intersection,
    is_point_inside_image,
)


class BoundaryIntersectionTest(unittest.TestCase):
    width = 100
    height = 60
    bbox_center = np.array([50.0, 30.0], dtype=np.float64)

    def assert_geometry(self, projected_center, expected=None):
        projected_center = np.asarray(projected_center, dtype=np.float64)
        representative = compute_boundary_intersection(
            self.bbox_center,
            projected_center,
            self.width,
            self.height,
        ).astype(np.float64)

        self.assertTrue(
            is_point_inside_image(representative, self.width, self.height)
        )
        if not is_point_inside_image(
            projected_center, self.width, self.height, eps=0.0
        ):
            on_boundary = (
                np.isclose(representative[0], 0.0, atol=1e-5)
                or np.isclose(representative[0], self.width - 1, atol=1e-5)
                or np.isclose(representative[1], 0.0, atol=1e-5)
                or np.isclose(representative[1], self.height - 1, atol=1e-5)
            )
            self.assertTrue(on_boundary)

            direction = projected_center - self.bbox_center
            nonzero_axis = int(np.argmax(np.abs(direction)))
            t = (
                representative[nonzero_axis] - self.bbox_center[nonzero_axis]
            ) / direction[nonzero_axis]
            self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(t, 1.0 + 1e-6)
            np.testing.assert_allclose(
                representative,
                self.bbox_center + t * direction,
                atol=1e-5,
            )

        if expected is not None:
            np.testing.assert_allclose(representative, expected, atol=1e-5)

        resolution = np.array([self.width, self.height], dtype=np.float64)
        representative_norm = representative / resolution
        projected_norm = projected_center / resolution
        offset_norm = projected_norm - representative_norm
        reconstructed = representative_norm + offset_norm
        np.testing.assert_allclose(reconstructed, projected_norm, atol=1e-7)
        np.testing.assert_allclose(
            reconstructed * resolution, projected_center, atol=1e-5
        )

    def test_left(self):
        self.assert_geometry([-20.0, 30.0], [0.0, 30.0])

    def test_right(self):
        self.assert_geometry([130.0, 30.0], [99.0, 30.0])

    def test_top(self):
        self.assert_geometry([50.0, -20.0], [50.0, 0.0])

    def test_bottom(self):
        self.assert_geometry([50.0, 90.0], [50.0, 59.0])

    def test_left_top(self):
        self.assert_geometry([-20.0, -20.0])

    def test_right_top(self):
        self.assert_geometry([130.0, -20.0])

    def test_left_bottom(self):
        self.assert_geometry([-20.0, 90.0])

    def test_right_bottom(self):
        self.assert_geometry([130.0, 90.0])

    def test_ray_through_corner(self):
        direction = np.array([-50.0, -30.0])
        projected_center = self.bbox_center + 2.0 * direction
        self.assert_geometry(projected_center, [0.0, 0.0])

    def test_dx_near_zero(self):
        self.assert_geometry([50.0 + 1e-12, -20.0], [50.0, 0.0])

    def test_dy_near_zero(self):
        self.assert_geometry([-20.0, 30.0 + 1e-12], [0.0, 30.0])

    def test_projected_center_inside(self):
        self.assert_geometry([70.0, 40.0], [70.0, 40.0])

    def test_bbox_center_near_boundary(self):
        bbox_center = self.bbox_center.copy()
        self.bbox_center = np.array([0.001, 10.0])
        try:
            self.assert_geometry([-5.0, 10.0], [0.0, 10.0])
        finally:
            self.bbox_center = bbox_center

    def test_float_coordinates(self):
        self.assert_geometry([-12.75, 17.125])

    def test_horizontal_flip_reverses_offset_x(self):
        bbox_center = np.array([20.0, 30.0])
        projected_center = np.array([-20.0, 42.0])
        representative = compute_boundary_intersection(
            bbox_center, projected_center, self.width, self.height
        )
        offset = projected_center - representative

        flipped_bbox = np.array([self.width - 1 - bbox_center[0], bbox_center[1]])
        flipped_projected = np.array(
            [self.width - 1 - projected_center[0], projected_center[1]]
        )
        flipped_representative = compute_boundary_intersection(
            flipped_bbox, flipped_projected, self.width, self.height
        )
        flipped_offset = flipped_projected - flipped_representative

        self.assertAlmostEqual(flipped_offset[0], -offset[0], places=5)
        self.assertAlmostEqual(flipped_offset[1], offset[1], places=5)

    def test_strict_subpixel_outside(self):
        self.assert_geometry([-1e-7, 30.0], [0.0, 30.0])

    def test_invalid_bbox_center_raises(self):
        with self.assertRaisesRegex(ValueError, "bbox_center must be inside"):
            compute_boundary_intersection(
                [-1.0, 20.0], [-10.0, 20.0], self.width, self.height
            )

    def test_non_finite_input_raises(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_boundary_intersection(
                self.bbox_center,
                [np.nan, 20.0],
                self.width,
                self.height,
            )


if __name__ == "__main__":
    unittest.main()
