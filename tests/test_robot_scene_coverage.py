"""Visible task workspace gates, independent of whether the far table corners fit."""

import unittest
from types import SimpleNamespace

import numpy as np

from astrabot.robot.config import Config
from astrabot.robot.scene_coverage import check_observed_box, halfplanes, observed_table


class SceneCoverageTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            scene_measured=True,
            table_min=[-1, -1, -0.8],
            table_max=[1, 1, 0],
            table_plane=[0, 0, 0],
            table_observed=[[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
        )

    def test_visible_route_passes_while_unseen_table_and_carried_cube_are_blocked(self):
        check_observed_box(self.config, [-0.1, -0.1, 0.1], [0.1, 0.1, 0.2], 0.01)
        for height in (0.04, 0.5, 5):
            with self.assertRaisesRegex(ValueError, "unobserved_table_region"):
                check_observed_box(self.config, [0.49, 0, height], [0.51, 0.1, height + 0.1], 0.01)
        # Space below the table and outside its footprint is checked by other
        # collision geometry; cropped tabletop visibility does not forbid it.
        check_observed_box(self.config, [0.6, 0, -0.3], [0.7, 0.1, -0.2])
        check_observed_box(self.config, [1.1, 0, 0.1], [1.2, 0.1, 0.2])

    def test_swept_box_crossing_unseen_strip_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "unobserved_table_region"):
            check_observed_box(self.config, [-1.2, -0.1, 0.1], [1.2, 0.1, 0.2])

    def test_full_workspace_then_changed_geometry_does_not_reuse_authorization(self):
        self.config.table_footprint = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        self.config.table_observed = [point[:] for point in self.config.table_footprint]
        check_observed_box(self.config, [-2, -2, 0.1], [2, 2, 0.3], 0.05)
        # In-place changes must invalidate cached containment, including when
        # only the physical footprint changes and the workspace stays fixed.
        for point in self.config.table_observed:
            point[0] *= 0.5
        with self.assertRaisesRegex(ValueError, "unobserved_table_region"):
            check_observed_box(self.config, [0.7, 0, 0.1], [0.8, 0.1, 0.2])
        self.config.table_footprint = [point[:] for point in self.config.table_observed]
        check_observed_box(self.config, [-2, -2, 0.1], [2, 2, 0.3])
        for point in self.config.table_footprint:
            point[0] *= 2
        with self.assertRaisesRegex(ValueError, "unobserved_table_region"):
            check_observed_box(self.config, [0.7, 0, 0.1], [0.8, 0.1, 0.2])

    def test_stereo_domain_excludes_cropped_corners_but_keeps_central_target(self):
        intrinsic = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
        geometry = SimpleNamespace(
            image_size_wh=(640, 480),
            K_left=intrinsic,
            K_right=intrinsic,
            D_left=np.zeros(5),
            D_right=np.zeros(5),
            R=np.eye(3),
            T=np.array([-0.06, 0, 0]),
        )
        transform = np.diag([1.0, -1.0, -1.0, 1.0])
        transform[:3, 3] = [0, 0, 1]
        footprint = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        region = observed_table(footprint, [0, 0, 0], geometry, transform)
        normals, offsets = halfplanes(region)
        self.assertTrue(np.all(offsets > 0))
        self.assertTrue(np.any(np.asarray(footprint) @ normals.T + offsets < 0))
        self.assertGreater(np.min(np.asarray(region)[:, 0]), -0.55)
        self.assertLess(np.max(np.asarray(region)[:, 1]), 0.4)

    def test_concave_or_degenerate_domain_is_rejected(self):
        for polygon in (
            [[0, 0], [0.5, 0], [0.1, 0.1], [0.5, 0.5], [0, 0.5]],
            [[0, 0], [0, 0], [0.5, 0.5]],
            [[0, 0], [0.1, 0.1], [0.2, 0.2]],
        ):
            with self.assertRaises(ValueError):
                Config(table_plane=[0, 0, 0], table_observed=polygon)
