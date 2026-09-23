"""Recorded sloped-table closure and complete orientation failure diagnostics."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from astrabot.robot.config import Config
from astrabot.robot.model import RobotModel
from astrabot.robot.trial_plan import TrialPlanner, TrialPlanningError


class TrialTableAlignmentTests(unittest.TestCase):
    def test_recorded_sloped_table_full_trial_passes_without_relaxing_checks(self):
        from astrabot.robot.collision import enable_acceleration

        enable_acceleration()
        data = json.loads((Path(__file__).parent / "fixtures/robot_sloped_table_trial.json").read_text())
        config = Config(**data["config"])
        model = RobotModel(config)
        planner = TrialPlanner(model, config)
        start = SimpleNamespace(body=np.array(data["body"]), hands=np.array(data["hands"]))
        result = planner.plan(start, data["grasp_center_m"])
        self.assertEqual(result["yaw_reference"], "measured_table_tangent")
        self.assertEqual(config.clearance, 0.005)
        self.assertEqual(config.table_uncertainty, 0.005)
        self.assertEqual(config.motion_checks.ik_position_error_m, 0.003)
        self.assertEqual(result["phases"][-1]["phase"], "return_ready")
        self.assertTrue(any(p["phase"] == "trial_lift" for p in result["phases"]))
        self.assertGreater(result["reversed_segments_reused"], 0)
        previous, hands = start.body.copy(), start.hands.copy()
        for phase in result["phases"]:
            target, hand_target = np.array(phase["arms"]), np.array(phase["hands"])
            if phase["phase"] in ("return", "return_ready"):
                # Independently sweep every reused reverse segment again.
                model.check_path(previous, previous[15:], target, hands, hand_target)
            previous[15:], hands = target, hand_target
        a, b, _ = config.table_plane
        normal = np.array([-a, -b, 1.0])
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(np.array(result["table_rotation"])[:, 2], normal, atol=1e-12)
        np.testing.assert_allclose(result["phases"][-1]["arms"], start.body[15:], atol=1e-12)

    def test_all_rejections_are_preserved_instead_of_only_last_ik(self):
        planner = TrialPlanner(None, Config())

        def reject(*args, **kwargs):
            planner.last_phase = "close" if kwargs["contact_yaw"] == 0 else "approach_compact"
            raise ValueError(
                "hand_obstacle:left:table:left_index_distal" if kwargs["contact_yaw"] == 0 else "IK_residual"
            )

        with patch.object(planner, "plan", side_effect=reject), self.assertRaises(TrialPlanningError) as caught:
            planner.plan_adaptive(None, [0, 0, 0])
        self.assertEqual(len(caught.exception.orientation_attempts), 11)
        self.assertIn("close:hand_obstacle:left:table:left_index_distal", str(caught.exception))
        self.assertIn("approach_compact:IK_residual", str(caught.exception))
