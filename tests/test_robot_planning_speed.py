"""Preserve collision coverage and FK results when reusing planning work."""

import json
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from astrabot.robot import collision
from astrabot.robot.alignment import physical_simulator_profile, simulation_joints, simulation_poses
from astrabot.robot.config import Config
from astrabot.robot.model import RobotModel


class PlanningSpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not collision.enable_acceleration():
            raise unittest.SkipTest("numba unavailable")
        cls.native = collision.native_backend()

    def test_batched_capsules_match_independent_pairs_including_contact(self):
        rng = np.random.default_rng(418)
        for index in range(60):
            poses = np.tile(np.eye(4), (4, 1, 1))
            poses[:, :3, :3] = Rotation.from_rotvec(rng.normal(size=(4, 3))).as_matrix()
            poses[:, :3, 3] = rng.uniform(-0.2, 0.2, (4, 3))
            lower, upper = np.full((4, 3), -0.04), np.full((4, 3), 0.04)
            capsules = rng.uniform(-0.3, 0.3, (3, 2, 3))
            radius = 0.04
            if index < 3:
                poses[:] = np.eye(4)
                capsules[:] = [[0.08 + (index - 1) * 1e-8, 0, 0], [0.08 + (index - 1) * 1e-8, 0, 0]]
            with patch.object(collision, "_FAST", None):
                expected = any(
                    collision.capsule_hits_box(
                        pose[:3, :3].T @ (a - pose[:3, 3]),
                        pose[:3, :3].T @ (b - pose[:3, 3]),
                        lo,
                        hi,
                        radius,
                    )
                    for a, b in capsules
                    for pose, lo, hi in zip(poses, lower, upper)
                )
                candidates = [collision.capsule_hits_box(*capsules[0], lo, hi, radius) for lo, hi in zip(lower, upper)]
            self.assertEqual(self.native.capsules_hit_local_boxes(capsules, poses, lower, upper, radius), expected)
            np.testing.assert_array_equal(
                self.native.capsule_box_candidates(*capsules[0], np.stack((lower, upper), axis=1), radius), candidates
            )

    def test_fk_subsets_and_simulator_cache_reflect_geometry_changes(self):
        model = RobotModel(Config())
        rng = np.random.default_rng(710)
        for _ in range(5):
            next(j for j in model.joints if j["name"] == "left_thumb_cmc_pitch")["origin"][0, 3] += 0.001
            body, hands = rng.uniform(-0.2, 0.2, 29), rng.uniform(0, 255, 12)
            full = model.poses(body, hands)
            for name, value in model.pinch_poses(body, hands).items():
                np.testing.assert_allclose(value, full[name], atol=1e-14, rtol=0)
            sim, _ = physical_simulator_profile(model)
            q = simulation_joints(body, hands, model)
            for name, value in simulation_poses(sim, q).items():
                np.testing.assert_allclose(value, sim.fk(q, name), atol=1e-14, rtol=0)

    def test_path_reuse_matches_every_sample_and_does_not_cross_calls(self):
        data = json.loads((Path(__file__).parent / "fixtures/robot_sloped_table_trial.json").read_text())
        model = RobotModel(Config(**data["config"]))
        body, hands = np.array(data["body"]), np.array(data["hands"])
        rng = np.random.default_rng(382)

        def outcome(fn):
            try:
                fn()
                return None
            except ValueError as exc:
                return str(exc)

        def reference(target, hand_target, obstacles):
            start = body[15:]
            delta = max(np.max(np.abs(target - start)), np.max(np.abs(hand_target - hands)) / 255 * 1.6)
            count = max(1, int(np.ceil(delta / model.config.motion_checks.path_step_rad)))
            for t in np.linspace(0, 1, count + 1):
                q = body.copy()
                q[15:] = start + t * (target - start)
                model.check(q, hands + t * (hand_target - hands), extra_obstacles=obstacles)

        for index in range(12):
            target, hand_target = body[15:].copy(), hands.copy()
            # Alternate static right, static left, both moving and hand-only paths.
            active = slice(0, 7) if index % 3 == 0 else slice(7, 14) if index % 3 == 1 else slice(0, 14)
            target[active] += rng.uniform(-0.005, 0.005, len(target[active]))
            if index % 4 == 0:
                hand_target[3] += 0.5
            boxes = model.boxes(model.poses(body, hands), "right")
            obstacles = [] if index < 6 else [(boxes[0, 0] - 0.01, boxes[0, 1] + 0.01)]
            expected = outcome(partial(reference, target, hand_target, obstacles))
            actual = outcome(
                partial(model.check_path, body, body[15:], target, hands, hand_target, extra_obstacles=obstacles)
            )
            self.assertEqual(actual, expected)
            if index >= 6:
                self.assertIsNotNone(actual)
