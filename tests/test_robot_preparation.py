"""Regression for the measured G1 standing entry and both checked paths."""

import json
import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

from astrabot.pinch import raw_hand
from astrabot.robot.config import GRIP, Config
from astrabot.robot.executor import Snapshot
from astrabot.robot.model import RobotModel
from astrabot.robot.trajectory import Planner

# 2026-09-15: actual standing feedback; no hardware or output-file dependency.
STANDING = np.array(
    [
        -0.282985,
        -0.009962,
        0.001178,
        0.615459,
        -0.350135,
        -0.006291,
        -0.267040,
        -0.000426,
        -0.009173,
        0.632767,
        -0.368920,
        -0.000155,
        -0.002277,
        0.000661,
        -0.004609,
        0.285033,
        0.130257,
        0.015160,
        0.976810,
        0.033556,
        0.010894,
        0.026257,
        0.289635,
        -0.123246,
        -0.007862,
        0.989070,
        -0.105329,
        0.017677,
        -0.030188,
    ]
)


class StandingPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(scene_measured=True)
        self.model = RobotModel(self.config)
        self.open = np.full(12, 255.0)

    def test_rotated_open_palms_clear_but_compacted_fingers_hit_thighs(self) -> None:
        self.model.check(STANDING, self.open)
        with self.assertRaisesRegex(ValueError, "hand_thigh"):
            self.model.check(STANDING, GRIP)

    def test_source_mesh_rejects_old_entry_despite_capsule_clearance(self) -> None:
        # The old thigh capsule misses this recorded thumb/hip hull contact.
        # Positive clearance still rejects the original contact after adding
        # the explicitly configured signed-clearance preparation policy.
        config = replace(
            self.config, motion_checks=replace(self.config.motion_checks, prepare_thigh_clearance_m=0.00025)
        )
        with self.assertRaisesRegex(ValueError, "prepare_hand_thigh_mesh_collision:right_thumb_distal"):
            Planner(RobotModel(config), config).ready(Snapshot(STANDING.copy(), self.open.copy(), 0.0, 0.0))

    def test_one_mm_policy_accepts_recorded_shallow_initial_mesh_contact(self):
        poses = self.model.poses(STANDING, self.open)
        self.model._check_escape_mesh_sweep(poses, poses, np.zeros(1), preparation=True)

    def test_recorded_scene_reaches_final_ready_before_changing_all_fingers(self) -> None:
        fixture = json.loads((Path(__file__).parent / "fixtures/robot_open_hand_prepare.json").read_text())
        config = Config(**fixture["scene"])
        body, hands = np.array(fixture["body"]), np.array(fixture["hands"])
        model = RobotModel(config)
        ready = Planner(model, config).ready(Snapshot(body.copy(), hands.copy(), 0, 0))
        self.assertEqual([s.name for s in ready], ["raise", "prepare", "ready_pinch"])
        for segment in ready[:-1]:
            np.testing.assert_array_equal(segment.hands, hands)
            np.testing.assert_array_equal(segment.hand_target, hands)
        np.testing.assert_array_equal(ready[-1].arms, config.ready_arms)
        np.testing.assert_array_equal(ready[-1].target, config.ready_arms)
        np.testing.assert_array_equal(ready[-1].hand_target, config.ready_hands)

    def test_return_preserves_clearance_through_hand_transitions(self) -> None:
        planner = Planner(self.model, self.config)
        body = STANDING.copy()
        body[15:] = self.config.ready_arms
        park = planner.park(Snapshot(body, np.tile(raw_hand("open"), 2), 0.0, 0.0), STANDING[15:])
        self.assertEqual(park[-2].name, "park_open")
        self.assertEqual(park[-1].name, "park_takeover_pose")
        np.testing.assert_array_equal(park[-1].target, STANDING[15:])
        np.testing.assert_array_equal(park[-1].hands, self.open)
        # The production planner preflights every arm/finger segment.

    def test_preparation_mesh_resolves_capsule_false_positive_but_rejects_contact(self) -> None:
        fixture = json.loads((Path(__file__).parent / "fixtures/robot_low_compact_contact.json").read_text())
        body, hands = np.array(fixture["body"]), np.array(fixture["hands"])
        model = RobotModel(Config(scene_measured=True, **fixture["table"]))
        with self.assertRaisesRegex(ValueError, "hand_thigh"):
            model.check(body, hands)
        model.check(body, hands, thigh_mesh=True)
        with self.assertRaisesRegex(ValueError, "prepare_hand_thigh_mesh_collision"):
            model.check(body, GRIP, thigh_mesh=True)

    def test_swept_mesh_rejects_collision_between_two_clear_endpoints(self) -> None:
        # Small synthetic hulls isolate the continuous sweep from the robot's
        # other collision checks. Both endpoint poses are separated by 30 mm.
        cube = np.array(list(product((-0.005, 0.005), repeat=3)))
        self.model.escape_geometry = [("right_thumb_distal", cube), ("right_hip_pitch_link", cube)]
        previous = {frame: np.eye(4) for frame, _ in self.model.escape_geometry}
        current = {frame: pose.copy() for frame, pose in previous.items()}
        previous["right_thumb_distal"][0, 3] = -0.04
        current["right_thumb_distal"][0, 3] = 0.04
        for pose in (previous, current):
            self.model._check_escape_mesh_sweep(pose, pose, np.zeros(14), preparation=True)
        with self.assertRaisesRegex(ValueError, "prepare_hand_thigh_mesh_collision"):
            self.model._check_escape_mesh_sweep(previous, current, np.zeros(14), preparation=True)

    def test_source_mesh_radius_certificate_rejects_oversized_model(self) -> None:
        joint = next(j for j in self.model.joints if j["name"] == "right_elbow_joint")
        joint["origin"][0, 3] = 2
        with self.assertRaisesRegex(ValueError, "escape_mesh_outside_radius_bound"):
            self.model._load_escape_geometry()

    def test_finger_sweep_bound_includes_driven_and_mimic_angles(self) -> None:
        closed = self.open.copy()
        closed[3] -= 1
        steps = self.model._path_joint_steps(STANDING[15:], STANDING[15:], self.open, closed)
        self.assertGreater(np.count_nonzero(steps), 1)
        self.assertGreater(np.sum(np.abs(steps)), 1.6 / 255)


if __name__ == "__main__":
    unittest.main()
