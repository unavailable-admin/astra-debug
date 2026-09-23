"""Recorded rotated-pinky corner overlap must not invent contact with W."""

import json
import unittest
from pathlib import Path

import numpy as np

from astrabot.robot.config import Config
from astrabot.robot.model import RobotModel


class ObstacleCollisionTests(unittest.TestCase):
    def test_recorded_empty_corner_passes_but_real_link_obstacle_fails(self):
        data = json.loads((Path(__file__).parent / "fixtures/robot_pinky_cube_corner.json").read_text())
        model = RobotModel(Config(**data["config"]))
        model.check(data["body"], data["hands"], extra_obstacles=data["obstacles"])
        poses = model.poses(data["body"], data["hands"])
        entry = next(x for x in model.bounds["left"] if x["frame"] == "left_pinky_distal")
        local = np.array(entry["corners"]).mean(0)
        pose = poses[entry["frame"]]
        center = pose[:3, :3] @ local + pose[:3, 3]
        with self.assertRaisesRegex(ValueError, "hand_obstacle:left"):
            model.check(data["body"], data["hands"], extra_obstacles=[(center - 0.005, center + 0.005)])
