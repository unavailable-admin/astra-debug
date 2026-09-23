"""Tilted table volume and carried-cube contact, independent analytic checks."""

import unittest

import numpy as np

from astrabot.robot.collision import convex_hulls_within_distance
from astrabot.robot.config import Config
from astrabot.robot.table import cube_penetrates, height, vertices


class TableTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            table_min=[0.2, -0.6, -0.7],
            table_max=[1, 0.6, 0.1],
            table_plane=[0.066, 0, -0.024],
            table_uncertainty=0.005,
        )

    def test_volume_covers_sloping_top_without_filling_empty_upper_corner(self):
        table = vertices(self.config)
        self.assertTrue(convex_hulls_within_distance(table, [[0.4, 0, -0.01]], 0))
        self.assertFalse(convex_hulls_within_distance(table, [[0.4, 0, 0.025]], 0.005))
        self.assertTrue(convex_hulls_within_distance(table, [[0.8, 0, 0.025]], 0))

    def test_table_aligned_cube_can_rest_and_lift_but_not_penetrate(self):
        normal = np.array([-0.066, 0, 1])
        normal /= np.linalg.norm(normal)
        contact = np.array([0.45, 0, height(self.config, 0.45, 0)])
        center = contact + 0.02 * normal
        self.assertFalse(cube_penetrates(self.config, center))
        self.assertFalse(cube_penetrates(self.config, center + [0, 0, 0.02]))
        self.assertFalse(cube_penetrates(self.config, center - [0, 0, 0.0029]))
        self.assertTrue(cube_penetrates(self.config, center - [0, 0, 0.0031]))

    def test_legacy_horizontal_table_and_invalid_plane(self):
        c = Config(table_min=[0.2, -0.6, -0.7], table_max=[1, 0.6, 0.1])
        self.assertFalse(cube_penetrates(c, [0.4, 0, 0.12]))
        self.assertFalse(cube_penetrates(c, [0.4, 0, 0.1171]))
        self.assertTrue(cube_penetrates(c, [0.4, 0, 0.1169]))
        with self.assertRaises(ValueError):
            Config(table_plane=[float("nan"), 0, 0])
        with self.assertRaises(ValueError):
            Config(table_plane=[0.3, 0, 0])

    def test_oriented_footprint_excludes_only_its_actual_convex_volume(self):
        config = Config(
            table_min=[0, 0, -1],
            table_max=[1, 1, 0.1],
            table_plane=[0, 0, 0.1],
            table_footprint=[[0, 0.5], [0.5, 0], [1, 0.5], [0.5, 1]],
        )
        table = vertices(config)
        self.assertFalse(convex_hulls_within_distance(table, [[0.05, 0.05, 0]], 0.01))
        self.assertTrue(convex_hulls_within_distance(table, [[0.5, 0.5, 0]], 0))
        with self.assertRaisesRegex(ValueError, "ordered_convex"):
            Config(
                table_min=[0, 0, -1],
                table_max=[1, 1, 0.1],
                table_plane=[0, 0, 0.1],
                table_footprint=[[0, 0], [1, 1], [1, 0], [0, 1]],
            )
