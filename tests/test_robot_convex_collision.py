"""Collision regressions against analytic distances and independent optimization."""

import unittest
from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from astrabot.robot.collision import convex_hulls_within_distance, segment_box_distance


class ConvexCollisionTests(unittest.TestCase):
    def test_segment_box_distance_in_arbitrary_frames(self):
        rng = np.random.default_rng(426)
        cube = np.array(list(product((-0.1, 0.1), repeat=3)))
        for _ in range(40):
            segment = rng.uniform(-0.3, 0.3, (2, 3))
            distance = segment_box_distance(*segment, np.full(3, -0.1), np.full(3, 0.1))
            rotation = Rotation.random(random_state=rng).as_matrix()
            offset = rng.normal(size=3)
            a, b = cube @ rotation.T + offset, segment @ rotation.T + offset
            self.assertTrue(convex_hulls_within_distance(a, b, distance + 1e-5))
            if distance > 2e-5:
                self.assertFalse(convex_hulls_within_distance(a, b, distance - 1e-5))

    def test_mesh_empty_corner_and_real_contact(self):
        tetra = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        self.assertFalse(convex_hulls_within_distance(tetra, [[0.8, 0.8, 0.8]], 0.1))
        self.assertTrue(convex_hulls_within_distance(tetra, [[0.1, 0.1, 0.1]], 0))
        self.assertTrue(convex_hulls_within_distance([[0, 0, 0]], [[0, 0, 0]], 0))
        self.assertTrue(convex_hulls_within_distance([[0, 0, 0]], [[0.035, 0, 0]], 0.035))

    def test_random_hulls_match_independent_constrained_minimization(self):
        rng = np.random.default_rng(18)
        for _ in range(15):
            a = rng.uniform(-0.05, 0.05, (8, 3))
            b = rng.uniform(-0.05, 0.05, (7, 3)) + rng.uniform(-0.1, 0.1, 3)

            def objective(w, a=a, b=b):
                d = w[:8] @ a - w[8:] @ b
                return d @ d

            result = minimize(
                objective,
                np.r_[np.full(8, 1 / 8), np.full(7, 1 / 7)],
                bounds=[(0, 1)] * 15,
                constraints=[
                    {"type": "eq", "fun": lambda w: w[:8].sum() - 1},
                    {"type": "eq", "fun": lambda w: w[8:].sum() - 1},
                ],
                method="SLSQP",
                options={"ftol": 1e-14, "maxiter": 500},
            )
            self.assertTrue(result.success)
            distance = np.sqrt(result.fun)
            self.assertTrue(convex_hulls_within_distance(a, b, distance + 2e-5))
            if distance > 3e-5:
                self.assertFalse(convex_hulls_within_distance(a, b, distance - 2e-5))

    def test_invalid_geometry_fails_closed(self):
        for a, margin in (([], 0), ([[float("nan"), 0, 0]], 0), ([[0, 0, 0]], -1)):
            with self.assertRaises(ValueError):
                convex_hulls_within_distance(a, [[0, 0, 0]], margin)


class SignedClearanceTests(unittest.TestCase):
    def test_penetration_limit_and_rigid_transform(self):
        from itertools import product

        from scipy.spatial.transform import Rotation

        from astrabot.robot.collision import convex_hulls_violate_clearance

        box = np.array(list(product((-0.01, 0.01), repeat=3)))
        rotation = Rotation.from_euler("xyz", [0.3, 0.7, -0.2]).as_matrix()
        for depth, rejected in ((-0.001, False), (0.0005, False), (0.000999, False), (0.001001, True), (0.002, True)):
            other = box + [0.02 - depth, 0, 0]
            for r in (np.eye(3), rotation):
                with self.subTest(depth=depth, rotated=not np.array_equal(r, np.eye(3))):
                    self.assertEqual(convex_hulls_violate_clearance(box @ r.T, other @ r.T, -0.001), rejected)
        self.assertTrue(convex_hulls_violate_clearance(box, box * 0.01, -0.001))
        self.assertTrue(convex_hulls_violate_clearance(box, box + [0.0201, 0, 0], 0.00025))
