"""Regressions for rotated hand bounds and retained head clearance."""

import json
import unittest
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from astrabot.robot.collision import oriented_box_hits_box
from astrabot.robot.config import GRIP, Config
from astrabot.robot.executor import Snapshot
from astrabot.robot.model import RobotModel, overlaps
from astrabot.robot.trajectory import Planner


class HeadCollisionTests(unittest.TestCase):
    def recorded(self):
        return json.loads((Path(__file__).parent / "fixtures/robot_head_prepare.json").read_text())

    def test_recorded_virtual_wall_pose_clears_source_head(self):
        fixture = self.recorded()
        pose = fixture["old_head_collision"]
        body, hands = np.array(pose["body"]), np.array(pose["hands"])
        model = RobotModel(Config(**fixture["scene"]))
        poses = model.poses(body, hands)
        entry = next(e for e in model.bounds["left"] if e["frame"] == pose["frame"])
        local = np.array(entry["corners"])
        self.assertTrue(
            oriented_box_hits_box(
                local.min(0),
                local.max(0),
                poses[entry["frame"]],
                [-0.15, -0.12, 0.35],
                [0.18, 0.12, 0.68],
                0.009,
            )
        )
        model.check(body, hands, thigh_mesh=True)

    def test_recorded_complete_preparation_with_post_takeover_waist(self):
        fixture = self.recorded()
        config = Config(**fixture["scene"])
        snapshot = Snapshot(np.array(fixture["body"]), np.array(fixture["hands"]), 0, 0)
        segments = Planner(RobotModel(config), config).ready(snapshot)
        self.assertEqual([s.name for s in segments], ["raise", "prepare", "ready_pinch"])
        for segment in segments[:-1]:
            np.testing.assert_array_equal(segment.hands, snapshot.hands)
            np.testing.assert_array_equal(segment.hand_target, snapshot.hands)

    def test_real_arm_head_collision_still_blocks_after_waist_rotation(self):
        fixture = self.recorded()
        model = RobotModel(Config(scene_measured=True, table_min=[2, -1, -1], table_max=[3, 1, 0]))
        body, hands = np.array(fixture["body"]), np.array(fixture["hands"])
        body[15:22] = [2.22013138, -0.68311949, 0.00067369, -0.20228550, 1.83098562, 0.40221153, 0.84797047]
        for waist in ([0, 0, 0], [0.3, 0.2, -0.4], [-0.3, -0.2, 0.4]):
            with self.subTest(waist=waist):
                body[12:15] = waist
                with self.assertRaisesRegex(ValueError, "head_collision:left:arm_0"):
                    model.check(body, hands)

    def test_hand_contact_and_clearance_in_rotated_head_frame(self):
        model = RobotModel(Config())
        head = model.head_bounds[0]
        local = np.array(list(product((-0.01, 0.01), repeat=3)))
        for side in ("left", "right"):
            model.bounds[side] = [{"frame": "probe", "corners": local}]
            for angles in ([0, 0, 0], [0.3, -0.4, 0.8]):
                head_pose = np.eye(4)
                head_pose[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
                head_pose[:3, 3] = [0.1, -0.2, 0.3]
                corners = np.array(head["corners"]) @ head_pose[:3, :3].T + head_pose[:3, 3]
                head_boxes = [(head, np.linalg.inv(head_pose), corners.min(0), corners.max(0))]
                for gap in (-0.01, 0, 0.008, 0.010):
                    with self.subTest(side=side, angles=angles, gap=gap):
                        probe = np.eye(4)
                        probe[:3, 3] = (np.array(head["lower"]) + head["upper"]) / 2
                        probe[0, 3] = head["upper"][0] + 0.01 + gap
                        poses = {"probe": head_pose @ probe}
                        boxes = model.boxes(poses, side)
                        if gap <= 0.009:
                            with self.assertRaisesRegex(ValueError, f"head_collision:{side}:probe"):
                                model._check_head(poses, side, boxes, [], head_boxes, 0.009)
                        else:
                            model._check_head(poses, side, boxes, [], head_boxes, 0.009)

    def test_rotated_box_empty_aabb_corner_is_not_a_collision(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        local = np.array([1.0, 0.02, 0.02])
        world_extent = np.abs(pose[:3, :3]) @ local
        lower, upper = np.array([0.55, -0.65, -0.03]), np.array([0.65, -0.55, 0.03])
        self.assertTrue(overlaps(-world_extent, world_extent, lower, upper, 0.009))
        self.assertFalse(oriented_box_hits_box(-local, local, pose, lower, upper, 0.009))

    def test_real_intersection_contact_and_clearance_are_rejected(self):
        pose = np.eye(4)
        lower, upper = np.zeros(3), np.ones(3)
        for x in (0.5, 1.0, 1.008):
            pose[0, 3] = x
            self.assertTrue(oriented_box_hits_box(lower, upper, pose, lower, upper, 0.009))
        pose[0, 3] = 1.010
        self.assertFalse(oriented_box_hits_box(lower, upper, pose, lower, upper, 0.009))

    def test_matches_independent_linear_feasibility_for_rotated_boxes(self):
        random = np.random.default_rng(20260916)
        for _ in range(40):
            pose = np.eye(4)
            pose[:3, :3] = Rotation.random(random_state=random).as_matrix()
            pose[:3, 3] = random.uniform(-0.6, 0.6, 3)
            extent = random.uniform(0.02, 0.4, 3)
            other = random.uniform(0.02, 0.4, 3)
            margin = 0.009
            rotation, center = pose[:3, :3], pose[:3, 3]
            feasible = linprog(
                np.zeros(3),
                A_ub=np.vstack((rotation, -rotation)),
                b_ub=np.r_[other + margin - center, other + margin + center],
                bounds=list(zip(-extent, extent)),
                method="highs",
            )
            self.assertIn(feasible.status, (0, 2))
            self.assertEqual(
                oriented_box_hits_box(-extent, extent, pose, -other, other, margin),
                feasible.success,
            )

    def test_measured_rotated_left_hand_pose_no_longer_false_alarms(self):
        body = np.array(
            [
                -0.27389,
                -0.01349,
                0.00170,
                0.60995,
                -0.40972,
                0.00741,
                -0.28020,
                0.01894,
                -0.01162,
                0.61278,
                -0.39296,
                -0.00522,
                -0.00257,
                0.00292,
                -0.01378,
                -0.07026,
                0.20461,
                0.08091,
                -0.57189,
                0.38013,
                -0.02203,
                0.01626,
                0.27386,
                -0.12641,
                0.01136,
                0.98275,
                -0.17540,
                0.07787,
                -0.07584,
            ]
        )
        hands = GRIP.copy()
        hands[6:] = 254
        model = RobotModel(Config(scene_measured=True, table_min=[1, -1, -1], table_max=[2, 1, 0]))
        boxes = model.boxes(model.poses(body, hands), "left")
        self.assertTrue(overlaps(boxes[:, 0], boxes[:, 1], [-0.15, -0.12, 0.35], [0.18, 0.12, 0.68], 0.009))
        model.check(body, hands)


if __name__ == "__main__":
    unittest.main()
