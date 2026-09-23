"""Real entry regression and refusal cases; no hardware communication."""

import json
import unittest
from pathlib import Path

import numpy as np

from astrabot.robot.config import RAISE, Config
from astrabot.robot.executor import Snapshot
from astrabot.robot.model import RobotModel
from astrabot.robot.trajectory import Planner


class WristRecoveryTests(unittest.TestCase):
    def setUp(self):
        data = json.loads((Path(__file__).parent / "fixtures/robot_wrist_limit_entry.json").read_text())
        self.body, self.hands = np.array(data["body"]), np.array(data["hands"])
        self.config = Config(scene_measured=True, table_min=[0.28, -0.85, -0.735], table_max=[0.95, 0.7, 0.025])
        self.model = RobotModel(self.config)
        self.planner = Planner(self.model, self.config)

    def test_real_entry_recovers_then_closed_ready_passes(self):
        with self.assertRaisesRegex(ValueError, "joint_limit"):
            self.model.check(self.body, self.hands)
        (segment,) = self.planner.move(Snapshot(self.body, self.hands, 0, 0), {"action": "recover_wrists"})
        np.testing.assert_array_equal(np.flatnonzero(segment.target != segment.arms), [5, 12])
        np.testing.assert_array_equal(segment.hand_target, self.hands)
        self.assertLessEqual(1.5 * np.max(abs(segment.target - segment.arms)) / segment.duration, 0.02000001)
        end = self.body.copy()
        end[15:] = segment.target
        self.model.check(end, self.hands)
        self.planner.move(Snapshot(end, self.hands, 0, 0), {"arms": self.config.ready_arms})
        with self.assertRaisesRegex(ValueError, "joint_limit"):
            self.model.check(self.body, self.hands)

    def test_large_overshoot_and_non_wrist_rejected(self):
        for joint, value in ((27, self.model.lower[12] - 0.0021), (15, self.model.lower[0] - 0.0001)):
            body = self.body.copy()
            body[joint] = value
            with self.assertRaisesRegex(ValueError, "entry_limit"):
                self.model.wrist_recovery_target(body)

    def test_outward_or_custom_target_rejected(self):
        target = self.model.wrist_recovery_target(self.body)
        for joint, amount in ((12, -0.06), (0, 0.01), (5, 0.01)):
            bad = target.copy()
            bad[joint] += amount
            with self.assertRaisesRegex(ValueError, "target_mismatch"):
                self.model.check_wrist_recovery(self.body, self.hands, bad)
        for key in ("arms", "hands", "holding", "obstacles", "recovery_start"):
            with self.assertRaisesRegex(ValueError, "invalid_wrist_recovery_request"):
                self.planner.move(Snapshot(self.body, self.hands, 0, 0), {"action": "recover_wrists", key: []})

    def test_actual_start_geometry_and_scene_still_checked(self):
        boxes = self.model.boxes(self.model.poses(self.body, self.hands), "right")
        self.config.obstacles = [(boxes[0, 0] - 0.01, boxes[0, 1] + 0.01)]
        with self.assertRaisesRegex(ValueError, "hand_obstacle:right"):
            self.planner.move(Snapshot(self.body, self.hands, 0, 0), {"action": "recover_wrists"})
        self.config.obstacles = []
        self.config.scene_measured = False
        with self.assertRaisesRegex(ValueError, "scene_not_measured"):
            self.planner.move(Snapshot(self.body, self.hands, 0, 0), {"action": "recover_wrists"})

    def test_new_raise_has_wrist_limit_margin(self):
        for i in (4, 5, 6, 11, 12, 13):
            self.assertGreater(min(RAISE[i] - self.model.lower[i], self.model.upper[i] - RAISE[i]), 0.05)
