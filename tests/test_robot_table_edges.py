"""Analytic rounded-rectangle reconstruction and rejection of wrong boundaries."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from astrabot.robot.collision import convex_hulls_within_distance
from astrabot.robot.config import Config
from astrabot.robot.model import RobotModel
from astrabot.robot.scene_builder import on_plane, project
from astrabot.robot.table import vertices
from astrabot.robot.table_edges import fit_table_edges, object_extent_report
from astrabot.robot.trajectory import Planner


class TableEdgeTests(unittest.TestCase):
    def test_recorded_outer_edge_samples_fix_extrapolation_without_relaxing_limits(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/robot_outer_table_edge.json").read_text())
        with self.assertRaisesRegex(ValueError, "table_corner_extrapolation_too_far:near:356.9mm"):
            fit_table_edges(
                fixture["original_near_m"],
                fixture["original_left_m"],
                fixture["plane"],
                fixture["width"],
                fixture["depth"],
            )
        _, report = fit_table_edges(
            fixture["reselected_near_m"],
            fixture["reselected_left_m"],
            fixture["plane"],
            fixture["width"],
            fixture["depth"],
        )
        self.assertAlmostEqual(report["corner_extrapolation_mm"]["near"], 68.03, places=2)
        self.assertLess(report["rectangle_residual_max_mm"], 10)
        self.assertLess(report["perpendicular_deviation_deg"], 5)
        self.assertGreater(report["span_mm"]["near"], 1000)

    def test_recorded_rectangle_residual_becomes_checked_boundary_volume(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/robot_uncertain_table_edge.json").read_text())
        corners, report = fit_table_edges(
            fixture["near"], fixture["left"], fixture["plane"], fixture["width"], fixture["depth"]
        )
        self.assertAlmostEqual(report["rectangle_residual_max_mm"], 13.5656, places=3)
        self.assertTrue(report["rectangle_residual_above_10mm"])
        self.assertLess(report["footprint_padding_m"], 0.05)
        padded = np.array(report["padded_corners_m"])
        lo, hi = padded.min(axis=0), padded.max(axis=0)
        lo[2] -= 0.74
        config = Config(
            table_min=lo.tolist(),
            table_max=hi.tolist(),
            table_plane=fixture["plane"],
            table_footprint=padded[:, :2].tolist(),
        )
        uncertain = vertices(config)
        nominal = Config(
            table_min=lo.tolist(),
            table_max=hi.tolist(),
            table_plane=fixture["plane"],
            table_footprint=corners[:, :2].tolist(),
        )
        inward = (corners[3] - corners[0]) / fixture["depth"]
        edge_point = (corners[0] + corners[1]) / 2 - 0.02 * inward - [0, 0, 0.02]
        # A path outside the nominal edge but inside its uncertainty must still collide.
        self.assertFalse(convex_hulls_within_distance([edge_point], vertices(nominal), 0))
        self.assertTrue(convex_hulls_within_distance([edge_point], uncertain, 0))
        center = corners.mean(axis=0)
        self.assertFalse(convex_hulls_within_distance([center + [0, 0, 0.06]], uncertain, 0.014))
        self.assertTrue(convex_hulls_within_distance([center - [0, 0, 0.02]], uncertain, 0))

    def test_operator_extent_only_requires_target_but_keeps_outlier_evidence(self):
        corners = np.array([[0, 0, 0], [1.2, 0, 0], [1.2, 0.6, 0], [0, 0.6, 0]])
        letters = {
            "A": {"estimated_xyz": [0.5664, 0.2547, 0.02]},
            "cube_012": {"estimated_xyz": [0.8917, 0.6886, -0.25]},
        }
        report = object_extent_report(letters, corners, 1.2, 0.6, target_only=True)
        self.assertTrue(report["passed"])
        self.assertEqual(report["diagnostic_only_outside"], ["cube_012"])
        strict = object_extent_report(letters, corners, 1.2, 0.6)
        self.assertFalse(strict["passed"])
        self.assertEqual(strict["failed_objects"], ["cube_012"])
        letters["A"]["estimated_xyz"] = [0.5, 0.59, 0.02]
        self.assertFalse(object_extent_report(letters, corners, 1.2, 0.6, target_only=True)["passed"])
        del letters["A"]
        self.assertFalse(object_extent_report(letters, corners, 1.2, 0.6, target_only=True)["passed"])

    def setUp(self):
        self.plane = np.array([0.0, 0.0, 1.0])
        self.near = np.array([[u, 0, 1] for u in (0.05, 0.4, 0.8, 1.1)])
        self.left = np.array([[0, v, 1] for v in (0.05, 0.2, 0.35, 0.55)])
        self.expected = np.array([[0, 0, 1], [1.2, 0, 1], [1.2, 0.6, 1], [0, 0.6, 1]])

    def test_rounding_needs_no_sample_at_virtual_corner(self):
        corners, report = fit_table_edges(self.near, self.left, self.plane, 1.2, 0.6)
        np.testing.assert_allclose(corners, self.expected, atol=1e-10)
        self.assertGreater(report["footprint_padding_m"], 0.01)
        self.assertAlmostEqual(report["corner_extrapolation_mm"]["near"], 50)
        self.assertTrue(report["rounded_corners_enclosed"])
        # A 40 mm rounded near-left arc lies inside the reconstructed rectangle;
        # the virtual corner itself is outside the physical rounded tabletop.
        theta = np.linspace(np.pi, 1.5 * np.pi, 30)
        arc = np.c_[0.04 + 0.04 * np.cos(theta), 0.04 + 0.04 * np.sin(theta), np.ones(len(theta))]
        self.assertTrue(np.all(arc >= corners.min(axis=0) - 1e-10))
        self.assertTrue(np.all(arc <= corners.max(axis=0) + 1e-10))

    def test_perspective_and_tilt_fit_in_world_plane(self):
        plane = np.array([0.1, 0.05, 1.0])
        normal = np.r_[-plane[:2], 1]
        normal /= np.linalg.norm(normal)
        across = np.array([1.0, 0.0, plane[0]])
        across /= np.linalg.norm(across)
        inward = np.cross(normal, across)
        origin = np.array([0.15, -0.25, 1 + 0.1 * 0.15 - 0.05 * 0.25])
        near = origin + self.near[:, :1] * across
        left = origin + self.left[:, 1:2] * inward
        geometry = SimpleNamespace(
            K_left=np.array([[900.0, 0, 960], [0, 920.0, 600], [0, 0, 1]]),
            D_left=np.array([-0.01, 0.001, 0, 0, 0]),
        )
        transform = np.eye(4)
        transform[:3, 3] = [-0.3, 0.1, -0.5]
        projected = [project(points, geometry, transform) for points in (near, left)]
        samples = [np.array([on_plane(pixel, plane, geometry, transform) for pixel in pixels]) for pixels in projected]
        corners, report = fit_table_edges(*samples, plane, 1.2, 0.6)
        expected = [origin, origin + 1.2 * across, origin + 1.2 * across + 0.6 * inward, origin + 0.6 * inward]
        np.testing.assert_allclose(corners, expected, atol=1e-6)
        self.assertLess(report["rectangle_residual_max_mm"], 0.001)

    def test_small_sample_errors_expand_obstacle_footprint(self):
        near, left = self.near.copy(), self.left.copy()
        near[:, 1] += [0.001, -0.001, 0.0015, -0.001]
        left[:, 0] += [-0.001, 0.001, -0.0015, 0.001]
        corners, report = fit_table_edges(near, left, self.plane, 1.2, 0.6)
        padding = report["footprint_padding_m"]
        self.assertTrue(np.all(self.expected[:, :2] >= corners.min(axis=0)[:2] - padding))
        self.assertTrue(np.all(self.expected[:, :2] <= corners.max(axis=0)[:2] + padding))
        self.assertGreater(report["rectangle_residual_max_mm"], 0)

    def test_wrong_edges_and_unstable_extrapolation_are_rejected(self):
        cases = []
        cases.append((self.near, self.left[:2], "invalid_samples"))
        cases.append((self.near, self.left[::-1], "outside_measured_extent"))
        cases.append((self.near, self.near + [0, 0.2, 0], "not_perpendicular"))
        cases.append((self.near, self.left + [1.2, 0, 0], "outside_measured_extent"))
        cases.append((self.near + [0, 0.6, 0], self.left, "outside_measured_extent"))
        cases.append((self.near, self.left * [1, 0.1, 1], "span_below_150mm"))
        cases.append((self.near[2:].tolist() + [[1.15, 0, 1]], self.left, "extrapolation_too_far"))
        duplicate = self.near.copy()
        duplicate[1] = duplicate[0]
        cases.append((duplicate, self.left, "unordered_or_too_close"))
        curved = self.near.copy()
        curved[1, 1] = 0.06
        cases.append((curved, self.left, "line_residual"))
        for near, left, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                fit_table_edges(near, left, self.plane, 1.2, 0.6)

    def test_directional_padding_does_not_apply_far_side_error_to_near_edge(self):
        left = self.left.copy()
        left[:, 0] += [0.002, -0.003, 0.003, -0.002]
        corners, report = fit_table_edges(self.near, left, self.plane, 1.2, 0.6)
        self.assertLess(report["boundary_padding_m"]["near_far"], report["boundary_padding_m"]["left_right"])
        padded = np.asarray(report["padded_corners_m"])
        # The full nominal rectangle, including rounded corner arcs, remains enclosed.
        self.assertTrue(np.all(corners >= padded.min(axis=0) - 1e-10))
        self.assertTrue(np.all(corners <= padded.max(axis=0) + 1e-10))

    def test_recorded_low_entry_table_false_positive_removed_without_thigh_bypass(self):
        source = json.loads((Path(__file__).parent / "fixtures/robot_round_table_low_entry.json").read_text())
        body, hands = np.asarray(source["body"]), np.asarray(source["hands"])
        old = RobotModel(Config(scene_measured=True, **source["original_table"]))
        with self.assertRaisesRegex(ValueError, "hand_obstacle:left"):
            old.check(body, hands)
        config = Config(scene_measured=True, **source["corrected_table"])
        model = RobotModel(config)
        with self.assertRaisesRegex(ValueError, "hand_thigh:left"):
            model.check(body, hands)
        segment = Planner(model, config)._thigh_escape_segment(body, hands)
        final = body.copy()
        final[15:] = segment.target
        model.check(final, hands)
        with self.assertRaisesRegex(ValueError, "hand_thigh:left"):
            model.check(body, hands)


if __name__ == "__main__":
    unittest.main()
