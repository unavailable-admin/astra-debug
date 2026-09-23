"""Real printed cyan survives bright camera exposure without including white paper."""

import unittest

import cv2
import numpy as np

from astrabot.vision import known_glyph_mask, mask


class RealGlyphMaskTests(unittest.TestCase):
    def test_known_ink_does_not_merge_with_bright_blue_cast_paper(self):
        hsv = np.zeros((40, 40, 3), np.uint8)
        hsv[:] = [100, 85, 150]
        hsv[10:25, 10:25] = [100, 210, 170]
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        selected = known_glyph_mask(image, [[17, 17]])
        self.assertEqual(np.count_nonzero(selected), 225)
        self.assertTrue(np.all(selected[10:25, 10:25] == 255))

    def test_real_ink_with_red_above_simulation_threshold_is_retained(self):
        hsv = np.array([[[100, 120, 240], [100, 90, 230], [100, 70, 255]]], np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.assertTrue(np.all(image[:, :, 2] > 100))
        self.assertTrue(np.all(mask(image, "real") == 255))
        self.assertTrue(np.all(mask(image, "sim") == 0))

    def test_dark_saturated_ink_is_recovered_only_in_identified_windows(self):
        hsv = np.zeros((20, 100, 3), np.uint8)
        hsv[:] = [100, 150, 70]
        hsv[5:15, 5:15] = [100, 210, 75]
        hsv[5:15, 80:90] = [100, 210, 75]
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = known_glyph_mask(image, [[10, 10]], radius=10)
        self.assertTrue(np.all(result[5:15, 5:15] == 255))
        self.assertTrue(np.all(result[:, 30:] == 0))
        self.assertEqual(result[0, 0], 0)
        self.assertTrue(np.all(mask(image, "real") == 0))

    def test_white_paper_skin_and_dark_robot_shell_are_excluded(self):
        hsv = np.array([[[90, 20, 255], [20, 100, 220], [100, 150, 70]]], np.uint8)
        self.assertTrue(np.all(mask(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "real") == 0))
        with self.assertRaises(ValueError):
            mask(np.zeros((1, 1, 3), np.uint8), "unknown")


if __name__ == "__main__":
    unittest.main()
