"""Constrained escape regressions; no hardware imports or movement."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize_scalar

from astrabot.robot.collision import segment_box_distance
from astrabot.robot.config import GRIP, Config
from astrabot.robot.executor import Snapshot
from astrabot.robot.model import RobotModel
from astrabot.robot.trajectory import Planner


class ThighEscapeTests(unittest.TestCase):
    def setUp(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/robot_thigh_margin_entry.json").read_text())
        self.body, self.hands = np.array(fixture["body"]), np.array(fixture["hands"])
        self.config = Config(scene_measured=True, table_min=[0.28, -0.85, -0.735], table_max=[0.95, 0.7, 0.025])
        self.model = RobotModel(self.config)
        self.planner = Planner(self.model, self.config)
        self.target = self.planner._clear_thighs_target(self.body, self.hands)

    def test_real_margin_entry_can_escape_but_normal_move_still_rejects(self):
        snapshot = Snapshot(self.body, self.hands, 0, 0)
        with self.assertRaisesRegex(ValueError, "hand_thigh:right"):
            self.planner.move(snapshot, {"arms": self.target.tolist()})
        (segment,) = self.planner.move(snapshot, {"action": "clear_thighs"})
        np.testing.assert_array_equal(segment.hands, segment.hand_target)
        np.testing.assert_array_equal(np.flatnonzero(segment.target != segment.arms), [1, 8])
        self.assertLessEqual(1.5 * np.max(np.abs(segment.target - segment.arms)) / segment.duration, 0.03 + 1e-12)
        final = self.body.copy()
        final[15:] = segment.target
        self.model.check(final, self.hands)
        # The exception is scoped to this path; it cannot leak to later moves.
        with self.assertRaisesRegex(ValueError, "hand_thigh:right"):
            self.model.check(self.body, self.hands)

    def test_inward_other_joint_large_and_unfinished_moves_rejected(self):
        for index, amount in ((1, -0.01), (8, 0.01), (0, 0.01), (1, 0.21), (2, 0.041)):
            target = self.body[15:].copy()
            target[index] += amount
            with self.assertRaisesRegex(ValueError, "shoulder_motion"):
                self.model.check_thigh_escape(self.body, self.hands, target)
        with self.assertRaisesRegex(ValueError, "hand_thigh:right"):
            self.model.check_thigh_escape(self.body, self.hands, self.body[15:])

    def test_second_measured_posture_needs_verified_shoulder_yaw_correction(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/robot_thigh_yaw_entry.json").read_text())
        body, hands = np.array(fixture["body"]), np.array(fixture["hands"])
        target = self.planner._clear_thighs_target(body, hands)
        with self.assertRaisesRegex(ValueError, "clearance_decreased"):
            self.model.check_thigh_escape(body, hands, target)
        (segment,) = self.planner.move(Snapshot(body, hands, 0, 0), {"action": "clear_thighs"})
        self.assertLess(segment.target[9], body[24])
        self.assertLessEqual(abs(segment.target[9] - body[24]), 0.04)
        fixed = [0, 3, 4, 5, 6, 7, 10, 11, 12, 13]
        np.testing.assert_array_equal(segment.target[fixed], body[15:][fixed])
        final = body.copy()
        final[15:] = segment.target
        self.model.check(final, hands)

    def test_hand_motion_holding_and_request_overrides_rejected(self):
        closed = self.hands.copy()
        closed[0] -= 1
        with self.assertRaisesRegex(ValueError, "fixed_empty_hands"):
            self.planner.segment(self.body, self.hands, self.target, closed, name="clear_thighs")
        with self.assertRaisesRegex(ValueError, "fixed_empty_hands"):
            self.planner.segment(self.body, self.hands, self.target, self.hands, name="clear_thighs", holding=True)
        for key, value in (
            ("arms", self.target.tolist()),
            ("hands", closed.tolist()),
            ("holding", True),
            ("obstacles", []),
            ("thigh_minimums", {}),
        ):
            with self.assertRaisesRegex(ValueError, "invalid_thigh_escape_request"):
                self.planner.move(Snapshot(self.body, self.hands, 0, 0), {"action": "clear_thighs", key: value})

    def test_reboot_wrist_posture_reaches_ready_before_curling(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/robot_low_compact_contact.json").read_text())
        body, hands = np.array(fixture["body"]), np.array(fixture["hands"])
        config = Config(scene_measured=True, **fixture["table"])
        model = RobotModel(config)
        planner = Planner(model, config)
        with self.assertRaisesRegex(ValueError, "hand_thigh"):
            model.check(body, GRIP)
        segments = planner.ready(Snapshot(body, hands, 0, 0))
        self.assertEqual([s.name for s in segments], ["raise", "prepare", "ready_pinch"])
        for segment in segments[:-1]:
            np.testing.assert_array_equal(segment.hands, hands)
            np.testing.assert_array_equal(segment.hand_target, hands)
        np.testing.assert_array_equal(segments[-1].arms, config.ready_arms)
        np.testing.assert_array_equal(segments[-1].target, config.ready_arms)
        np.testing.assert_array_equal(segments[-1].hand_target, config.ready_hands)
        body[15:] = segments[-1].target
        model.check(body, GRIP)
        lo, hi = model.boxes(model.poses(body, GRIP), "left")[0]
        with self.assertRaisesRegex(ValueError, "hand_obstacle:left"):
            model.check(body, GRIP, extra_obstacles=[(lo - 0.01, hi + 0.01)])

    def test_capsule_overlap_requires_source_mesh_certificate(self):
        key = ("right", "right", "right_thumb_distal")
        for distance in (0.06, 0.068, 0.069):
            with (
                patch.object(self.model, "_thigh_distances", return_value={key: distance}),
                self.assertRaisesRegex(ValueError, "thigh_escape_mesh_collision"),
            ):
                self.model.check_thigh_escape(self.body, self.hands, self.target)

    def test_initial_distance_decrease_is_rejected_even_if_outward_angles(self):
        key = ("right", "right", "right_thumb_distal")
        distance = self.model._thigh_distances(self.model.poses(self.body, self.hands))[key]
        with (
            patch.object(
                self.model, "_thigh_distances", side_effect=[{key: distance}, {key: distance}, {key: distance - 0.0001}]
            ),
            self.assertRaisesRegex(ValueError, "clearance_decreased"),
        ):
            self.model.check_thigh_escape(self.body, self.hands, self.target)

    def test_other_obstacles_and_unmeasured_scene_still_rejected(self):
        bounds = self.model.boxes(self.model.poses(self.body, self.hands), "left")[0]
        with self.assertRaisesRegex(ValueError, "hand_obstacle:left"):
            self.model.check_thigh_escape(
                self.body, self.hands, self.target, extra_obstacles=[(bounds[0] - 0.01, bounds[1] + 0.01)]
            )
        self.config.scene_measured = False
        with self.assertRaisesRegex(ValueError, "scene_not_measured"):
            self.model.check_thigh_escape(self.body, self.hands, self.target)

    def test_distance_matches_independent_numeric_minimization(self):
        random = np.random.default_rng(16)
        for _ in range(50):
            start, end = random.normal(size=(2, 3))
            lo, hi = -random.uniform(0.01, 0.4, 3), random.uniform(0.01, 0.4, 3)

            def distance(t, start=start, end=end, lo=lo, hi=hi):
                point = start + t * (end - start)
                return np.linalg.norm(point - np.clip(point, lo, hi))

            numeric = minimize_scalar(distance, bounds=(0, 1), method="bounded", options={"xatol": 1e-13})
            expected = min(distance(0), distance(1), numeric.fun)
            self.assertAlmostEqual(segment_box_distance(start, end, lo, hi), expected, places=7)
        self.assertEqual(segment_box_distance(np.zeros(3), np.zeros(3), -np.ones(3), np.ones(3)), 0)
        self.assertAlmostEqual(
            segment_box_distance(np.ones(3) * 2, np.ones(3) * 2, -np.ones(3), np.ones(3)), np.sqrt(3)
        )


if __name__ == "__main__":
    unittest.main()
