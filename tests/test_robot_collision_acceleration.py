"""Compare optional native kernels with reference geometry, including boundaries."""

import unittest
from unittest.mock import patch

import numpy as np

from astrabot.robot import collision
from astrabot.robot.config import Config
from astrabot.robot.model import RobotModel, overlaps


class NativeCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original = collision._FAST
        try:
            if not collision.enable_acceleration():
                raise unittest.SkipTest("optional numba dependency unavailable")
            cls.native = collision._FAST
        finally:
            collision._FAST = original

    def compare(self, function, *args):
        with patch.object(collision, "_FAST", None):
            expected = function(*args)
        with patch.object(collision, "_FAST", self.native):
            actual = function(*args)
        if isinstance(expected, bool):
            self.assertEqual(actual, expected)
        else:
            self.assertAlmostEqual(actual, expected, delta=1e-12)

    def test_hulls_points_and_degenerate_faces_match_reference(self):
        rng = np.random.default_rng(913)
        for count in (1, 2, 4, 8):
            for _ in range(25):
                first = rng.uniform(-0.1, 0.1, (count, 3))
                second = rng.uniform(-0.1, 0.1, (8, 3)) + rng.uniform(-0.1, 0.1, 3)
                for margin in (0.0, 0.004, 0.03):
                    self.compare(collision.convex_hulls_within_distance, first, second, margin)
        for epsilon in (-1e-9, 0, 1e-9, 1e-6):
            self.compare(collision.convex_hulls_within_distance, [[0, 0, 0]], [[0.004 + epsilon, 0, 0]], 0.004)
        for first in (np.zeros((4, 3)), np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]], float)):
            self.compare(collision.convex_hulls_within_distance, first, first + [0, 0, 0.001], 0.001)

    def test_capsules_and_segments_match_reference_with_strided_inputs(self):
        rng = np.random.default_rng(625)
        lower, upper = np.full(3, -0.1), np.full(3, 0.1)
        for _ in range(200):
            a, b, c, d = rng.uniform(-0.3, 0.3, (4, 6))[:, ::2]
            for end in (b, a):
                self.compare(collision.segment_box_distance, a, end, lower, upper)
                self.compare(collision.capsule_hits_box, a, end, lower, upper, 0.035)
                self.compare(collision.segment_distance, a, end, c, d)
        for epsilon in (0.0, 1e-10, 1e-6):
            self.compare(
                collision.segment_distance,
                np.array([0.0, 0, 0]),
                np.array([1.0, 0, 0]),
                np.array([0.0, 0.01, 0]),
                np.array([1.0, 0.01 + epsilon, 0]),
            )

    def test_batched_bounds_preserve_touching_and_margin(self):
        rng = np.random.default_rng(27)
        for count in (1, 6, 13):
            for _ in range(30):
                lower = rng.normal(size=(count, 3))
                other = rng.normal(size=(4, 3))
                self.compare(overlaps, lower, lower + 0.1, other, other + 0.2, 0.005)
        self.compare(overlaps, [0, 0, 0], [1, 1, 1], [1, 0, 0], [2, 1, 1], 0.0)

    def test_native_fk_and_wrist_subset_match_full_tree_after_geometry_change(self):
        rng = np.random.default_rng(672)
        model = RobotModel(Config(tcp_offset=[0.01, 0.02, 0.03]))
        for index in range(30):
            body, hands = rng.uniform(-1.5, 1.5, 29), rng.uniform(0, 255, 12)
            if index == 15:
                joint = next(j for j in model.joints if j["name"] == "left_elbow_joint")
                joint["origin"][0, 3] += 0.01
            with patch.object(collision, "_FAST", None):
                reference = model.poses(body, hands)
            with patch.object(collision, "_FAST", self.native):
                actual = model.poses(body, hands)
                for name in reference:
                    np.testing.assert_allclose(actual[name], reference[name], atol=2e-14, rtol=0)
                for side in ("left", "right"):
                    position, rotation = model.tcp(body, hands, side)
                    pose = reference[f"{side}_wrist_yaw_link"]
                    expected = pose[:3, 3] + pose[:3, :3] @ model.config.tcp_offset
                    np.testing.assert_allclose(position, expected, atol=2e-14, rtol=0)
                    np.testing.assert_allclose(rotation, pose[:3, :3], atol=2e-14, rtol=0)


if __name__ == "__main__":
    unittest.main()
