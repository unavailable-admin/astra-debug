"""Physical-profile FK correspondence and complete offline trial regressions."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from astrabot.pinch import CLOSED, MOTOR_RANGES, OPEN, raw_hand
from astrabot.robot.alignment import compare_models, physical_simulator_profile, simulation_joints
from astrabot.robot.config import Config
from astrabot.robot.model import RobotModel
from astrabot.robot.trial_plan import TrialPlanner


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads((Path(__file__).parent / "fixtures/robot_aligned_trial.json").read_text())
        cls.config = Config(
            scene_measured=True,
            table_min=cls.source["table_min"],
            table_max=cls.source["table_max"],
            joint_speed=0.15,
            cartesian_speed=0.01,
            hand_speed=40,
        )
        cls.model = RobotModel(cls.config)
        cls.snapshot = SimpleNamespace(body=np.array(cls.source["body"]), hands=np.array(cls.source["hands"]))
        cls.trial = TrialPlanner(cls.model, cls.config)
        cls.plan = cls.trial.plan(cls.snapshot, cls.source["grasp_center_m"], obstacles=cls.source["obstacles"])

    def test_native_discrepancy_is_visible_and_physical_profile_agrees(self):
        samples = [(self.snapshot.body, self.snapshot.hands)]
        native = compare_models(self.model, samples, physical_profile=False)
        self.assertFalse(native["passed"])
        self.assertAlmostEqual(native["position_max_m"], 0.003, places=6)
        self.assertTrue(compare_models(self.model, samples)["passed"])

    def test_presets_and_right_thumb_use_physical_values(self):
        for label, preset in [("open", OPEN), ("closed", CLOSED)]:
            angles = (1 - raw_hand(label) / 255) * MOTOR_RANGES
            np.testing.assert_allclose(angles, [1.3, 1.3, 1.3, preset[0], preset[3], preset[2]])
        joints = simulation_joints(self.snapshot.body, np.tile(raw_hand("closed"), 2), self.model)
        self.assertAlmostEqual(joints["rh_thumb_ip"], joints["rh_thumb_cmc_pitch"] * 1.86)
        self.assertAlmostEqual(joints["lh_thumb_ip"], joints["lh_thumb_cmc_pitch"] * 2.29)

    def test_complete_trace_return_contact_gates_and_independent_fk(self):
        plan = self.plan
        self.assertTrue(plan["geometry_passed"])
        self.assertFalse(plan["hardware_passed"])
        self.assertTrue(plan["execution_policy"]["offline_only"])
        phases = plan["phases"]
        names = [phase["phase"] for phase in phases]
        for name in (
            "approach_compact",
            "preshape",
            "clearance",
            "turn",
            "descend",
            "close",
            "trial_lift",
            "replace",
            "release",
            "retract",
            "return_orientation",
            "compact_for_return",
            "return_ready",
        ):
            self.assertIn(name, names)
        self.assertEqual(names[plan["gates"]["opposing_contact_before_phase"]], "trial_lift")
        self.assertEqual(names[plan["gates"]["visual_lift_check_before_phase"]], "replace")
        np.testing.assert_array_equal(phases[-1]["arms"], self.snapshot.body[15:])
        np.testing.assert_array_equal(phases[-1]["hands"], self.snapshot.hands)
        sim, _ = physical_simulator_profile(self.model)
        previous, old_hands = self.snapshot.body.copy(), self.snapshot.hands.copy()
        for phase in phases:
            q, h = self.snapshot.body.copy(), np.array(phase["hands"])
            q[15:] = phase["arms"]
            np.testing.assert_array_equal(q[22:], self.snapshot.body[22:])
            np.testing.assert_array_equal(h[6:], self.snapshot.hands[6:])
            self.assertLessEqual(
                1.5 * np.max(np.abs(q[15:] - previous[15:])) / phase["duration"], self.config.joint_speed + 1e-10
            )
            self.assertLessEqual(
                1.5 * np.max(np.abs(h - old_hands)) / phase["duration"], self.config.hand_speed + 1e-10
            )
            if phase["phase"] == "close":
                self.assertLessEqual(np.max(np.abs(h - old_hands)), 2 + 1e-10)
            joints = simulation_joints(q, h, self.model)
            from astrabot.pinch import INDEX_TIP, THUMB_TIP

            center = (
                (sim.fk(joints, "lh_index_distal") @ INDEX_TIP)[:3]
                + (sim.fk(joints, "lh_thumb_distal") @ THUMB_TIP)[:3]
            ) / 2
            if "pinch_center_m" in phase:
                self.assertLess(np.linalg.norm(center - phase["pinch_center_m"]), 0.003)
            # Independently evaluate the midpoint of every saved joint/hand segment.
            self.model.check((previous + q) / 2, (old_hands + h) / 2, extra_obstacles=self.source["obstacles"])
            previous, old_hands = q, h

    def test_target_obstacle_rejected_and_tcp_offset_not_double_counted(self):
        center = np.array(self.source["grasp_center_m"])
        with self.assertRaisesRegex(ValueError, "trial_cube_obstacle"):
            self.trial.check_cube(center, [(center - 0.01, center + 0.01)])
        self.config.tcp_offset = [0, 0, 0.01]
        try:
            with self.assertRaisesRegex(ValueError, "trial_requires_wrist_origin_tcp"):
                self.trial.plan(self.snapshot, center)
        finally:
            self.config.tcp_offset = [0, 0, 0]
