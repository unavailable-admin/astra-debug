import unittest

import cv2
import numpy as np
import yaml

from astrabot.geometry import DEFAULT_CALIBRATION, StereoGeometry


class StereoGeometryTests(unittest.TestCase):
    def test_profiles_and_relative_mounts(self):
        (sim, real) = (StereoGeometry.load("sim"), StereoGeometry.load("real"))
        np.testing.assert_allclose(sim.K_left[:2, 2], [320, 240])
        self.assertEqual(sim.K_left[0, 0], sim.K_left[1, 1])
        np.testing.assert_array_equal(sim.D_left, np.zeros(5))
        self.assertGreater(np.linalg.norm(real.D_left), 0)
        self.assertNotEqual(real.K_left[0, 2], 320)
        self.assertIsNone(real.T_head_left_cv)
        np.testing.assert_allclose(sim.R, real.R, atol=1e-09)
        np.testing.assert_allclose(sim.T, real.T, atol=1e-09)
        with open(DEFAULT_CALIBRATION) as stream:
            mount = yaml.safe_load(stream)["stereo_head_camera"]["left"]["isaac_render_offset"]
        np.testing.assert_allclose(sim.T_head_left_cv[:3, 3], mount["translation_m"])
        self.assertGreater(sim.T_head_left_cv[0, 2], 0)
        self.assertLess(sim.T_head_left_cv[2, 2], 0)

    def test_roundtrip_raw_pixels_with_and_without_distortion(self):
        xyz = np.array([[-0.12, -0.1, 0.6], [0.1, 0.15, 0.9], [0.02, 0.03, 1.3]])
        for mode in ("sim", "real"):
            with self.subTest(mode=mode):
                g = StereoGeometry.load(mode)
                left = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), g.K_left, g.D_left)[
                    0
                ].reshape(-1, 2)
                right = cv2.projectPoints(xyz, cv2.Rodrigues(g.R)[0], g.T, g.K_right, g.D_right)[
                    0
                ].reshape(-1, 2)
                recovered = g.triangulate(left, right, image_size_wh=(640, 480))
                np.testing.assert_allclose(recovered, xyz, atol=1e-06)
                with self.assertRaises(ValueError):
                    g.triangulate(left, right + [0, 30], image_size_wh=(640, 480))
                with self.assertRaises(ValueError):
                    g.triangulate(left, right, image_size_wh=(1280, 720))

    def test_explicit_mode(self):
        with self.assertRaises(ValueError):
            StereoGeometry.load("auto")


if __name__ == "__main__":
    unittest.main()
