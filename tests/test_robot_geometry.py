"""Physical geometry checks against an independent MuJoCo representation."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from astrabot.robot.calibration import fit_rigid, load
from astrabot.robot.config import ASSETS, GRIP, RAISE, READY, REST, Config
from astrabot.robot.model import RobotModel
from astrabot.robot.trajectory import Segment


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(scene_measured=True, table_min=[1, -1, -1], table_max=[2, 1, 0])
        self.model = RobotModel(self.config)

    def test_known_entry_poses_pass_and_actual_table_intersection_fails(self):
        for q in (REST, RAISE, READY):
            self.model.check(np.r_[np.zeros(15), q], GRIP)
        pose = self.model.poses(np.r_[np.zeros(15), REST], GRIP)
        bounds = self.model.boxes(pose, "left")[0]
        with self.assertRaisesRegex(ValueError, "obstacle"):
            self.model.check(np.r_[np.zeros(15), REST], GRIP, extra_obstacles=[(bounds[0] - 0.01, bounds[1] + 0.01)])

    def test_unmeasured_scene_is_not_a_motion_permission(self):
        with self.assertRaisesRegex(ValueError, "scene_not_measured"):
            RobotModel(Config()).check(np.r_[np.zeros(15), REST], GRIP)

    def test_stereo_candidate_cannot_enable_motion_without_extrinsic(self):
        with self.assertRaisesRegex(ValueError, "extrinsic_missing"):
            load(ASSETS / "camera_candidate.json")

    def test_urdf_matches_mujoco_at_multiple_arm_postures(self):
        try:
            import mujoco

            from astrabot.robot.hardware import Gravity
        except ImportError:
            self.skipTest("optional mujoco unavailable")
        dynamics = Gravity()
        root = mujoco.mj_name2id(dynamics.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        for q in (REST, RAISE, READY, (REST + READY) / 2):
            dynamics.data.qpos[dynamics.qidx] = q
            mujoco.mj_forward(dynamics.model, dynamics.data)
            poses = self.model.poses(np.r_[np.zeros(15), q], np.full(12, 255.0))
            for side in ("left", "right"):
                frame = f"{side}_wrist_yaw_link"
                i = mujoco.mj_name2id(dynamics.model, mujoco.mjtObj.mjOBJ_BODY, frame)
                np.testing.assert_allclose(
                    poses[frame][:3, 3], dynamics.data.xpos[i] - dynamics.data.xpos[root], atol=2e-6
                )
                np.testing.assert_allclose(poses[frame][:3, :3], dynamics.data.xmat[i].reshape(3, 3), atol=2e-6)

    def test_local_ik_and_joint_limits(self):
        q = np.r_[np.zeros(15), READY]
        target, rotation = self.model.tcp(q, GRIP)
        target = target + [0.004, 0, 0]
        solved = self.model.solve(q, GRIP, target, rotation)
        np.testing.assert_array_equal(solved[:15], q[:15])
        np.testing.assert_array_equal(solved[22:], q[22:])
        self.assertLess(np.linalg.norm(self.model.tcp(solved, GRIP)[0] - target), 0.003)

    def test_shoulder_leads_and_speed_is_bounded(self):
        start = np.zeros(14)
        end = np.ones(14) * 0.3
        segment = Segment(start, np.full(12, 255.0), end, np.full(12, 255.0), 2.5, profile="raise")
        at_quarter, _ = segment.sample(0.625)
        self.assertGreater(at_quarter[0], 0)
        self.assertEqual(at_quarter[1], 0)
        sampled = np.array([segment.sample(t)[0] for t in np.linspace(0, 2.5, 2501)])
        self.assertLessEqual(np.max(np.abs(np.diff(sampled, axis=0))) / 0.001, 0.3 + 1e-6)

    def test_candidate_rigid_fit_and_degeneracy(self):
        a = np.random.default_rng(10).normal(size=(12, 3)) * 0.1
        r = Rotation.from_euler("xyz", [0.1, 0.3, -0.2]).as_matrix()
        b = a @ r.T + [0.2, 0.1, 0.6]
        t = fit_rigid(a, b)
        np.testing.assert_allclose(a @ t[:3, :3].T + t[:3, 3], b, atol=1e-10)
        with self.assertRaises(ValueError):
            fit_rigid(np.zeros((6, 3)), np.zeros((6, 3)))


if __name__ == "__main__":
    unittest.main()
