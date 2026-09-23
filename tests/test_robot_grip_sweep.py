"""Conservative envelopes preserve sampled finger coverage and route geometry."""

import json
import unittest
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from astrabot.robot.collision import enable_acceleration
from astrabot.robot.config import Config
from astrabot.robot.grip_sweep import envelope_model
from astrabot.robot.model import RobotModel
from astrabot.robot.trajectory import Planner
from astrabot.robot.trial_runtime import pinch_pose


class GripSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enable_acceleration()
        cls.data = json.loads((Path(__file__).parent / "fixtures/robot_transfer_latency.json").read_text())

    def setUp(self):
        self.model = RobotModel(Config(**self.data["config"]))
        self.body = np.array(self.data["body"])
        self.hands = np.array(self.data["hands"])
        self.command = np.array(self.data["request"]["hands"])

    def test_envelope_contains_all_original_finger_samples_after_arm_rotation(self):
        envelope = envelope_model(self.model, self.body, self.hands, self.command)
        count = int(np.ceil(np.max(np.abs(self.command - self.hands)) / 255 * 1.6 / 0.001))
        for rotation in (0.0, 0.1):
            body = self.body.copy()
            body[15:22] += rotation
            reference = self.model.poses(body, self.hands)
            for alpha in np.linspace(0, 1, count + 1):
                poses = self.model.poses(body, self.hands + alpha * (self.command - self.hands))
                for side in ("left", "right"):
                    for original, expanded in zip(self.model.bounds[side], envelope.bounds[side]):
                        pose = poses[original["frame"]]
                        base = reference[original["frame"]]
                        points = np.array(original["corners"]) @ pose[:3, :3].T + pose[:3, 3]
                        local = (points - base[:3, 3]) @ base[:3, :3]
                        box = np.array(expanded["corners"])
                        self.assertTrue(np.all(local >= box.min(0) - 1e-12))
                        self.assertTrue(np.all(local <= box.max(0) + 1e-12))

    def test_optimized_sweep_matches_original_samples_targets_and_durations(self):
        from astrabot.robot import trial_runtime

        planner = Planner(self.model, self.model.config)
        state = SimpleNamespace(body=self.body, hands=self.hands)
        fast = planner.move(state, self.data["request"])
        sweep = trial_runtime.check_sweep

        def reference(*args, **kwargs):
            kwargs.pop("pose_cache", None)
            return sweep(*args, **kwargs)

        with (
            patch("astrabot.robot.grip_sweep.covers_grip", return_value=False),
            patch("astrabot.robot.trial_runtime.check_sweep", side_effect=reference),
        ):
            slow = planner.move(state, self.data["request"])
        self.assertEqual(len(fast), len(slow))
        for first, second in zip(fast, slow):
            np.testing.assert_allclose(first.target, second.target, atol=1e-12, rtol=0)
            self.assertAlmostEqual(first.duration, second.duration, places=9)
            np.testing.assert_array_equal(first.hand_target, self.command)
        body = self.body.copy()
        body[15:] = fast[-1].target
        center, _ = pinch_pose(self.model, body, self.hands)
        self.assertLess(np.linalg.norm(center - self.data["request"]["goal_center"]), 0.0001)

    def test_conservative_overlap_falls_back_and_real_obstacle_still_rejects(self):
        request = copy(self.data["request"])
        poses = self.model.poses(self.body, self.hands)
        box = self.model.boxes(poses, "left")[0]
        request["obstacles"] = [(box[0] - 0.002).tolist(), (box[1] + 0.002).tolist()]
        request["obstacles"] = [request["obstacles"]]
        with self.assertRaisesRegex(ValueError, "hand_obstacle"):
            Planner(self.model, self.model.config).move(SimpleNamespace(body=self.body, hands=self.hands), request)

    def test_identical_grip_does_not_repeat_collision_certificate(self):
        from astrabot.robot.trial_runtime import pinch_step

        request = copy(self.data["request"])
        request["hands"] = self.hands.tolist()
        planner = Planner(self.model, self.model.config)
        state = SimpleNamespace(body=self.body, hands=self.hands)
        with (
            patch.object(self.model, "check_path", wraps=self.model.check_path) as sweep,
            patch("astrabot.robot.grip_sweep.covers_grip", side_effect=AssertionError("duplicate sweep")),
        ):
            segment = pinch_step(planner, state, request, carry_cache={"envelopes": {}, "poses": {}})
        self.assertEqual(sweep.call_count, 1)
        np.testing.assert_array_equal(segment.hand_target, self.hands)

    def test_recorded_boundary_pose_reaches_six_cm_edge_slot(self):
        from astrabot.robot.transfer import front_slots

        data = json.loads((Path(__file__).parent / "fixtures/robot_transfer_step_boundary.json").read_text())
        config = Config(**data["config"])
        model = RobotModel(config)
        request = data["request"]
        request["goal_center"][0] = front_slots(config, 1)[0][0]
        request["center"][0] = request["goal_center"][0]
        state = SimpleNamespace(body=np.array(data["body"]), hands=np.array(data["hands"]))
        segments = Planner(model, config).move(state, request)
        body = state.body.copy()
        body[15:] = segments[-1].target
        center, _ = pinch_pose(model, body, state.hands)
        self.assertLessEqual(np.linalg.norm(center - request["goal_center"]), config.motion_checks.transfer_arrival_m)
        np.testing.assert_allclose(segments[-1].pinch_center, request["goal_center"])
        self.assertTrue(all(segment.continuous_carry for segment in segments))
