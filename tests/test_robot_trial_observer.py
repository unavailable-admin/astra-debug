"""Stereo visibility and projection-seed regressions without hardware writes."""

import copy
import tempfile
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from astrabot.robot.observation_policy import cube_obstacles
from astrabot.robot.trial_observer import TrialObserver
from astrabot.stereo import StereoTracker


class ObserverTests(unittest.TestCase):
    def test_right_camera_can_be_occluded_while_left_is_clear(self):
        intrinsic = np.array([[100.0, 0, 100], [0, 100, 100], [0, 0, 1]])
        g = SimpleNamespace(
            K_left=intrinsic,
            K_right=intrinsic,
            D_left=np.zeros(5),
            D_right=np.zeros(5),
            R=np.eye(3),
            T=np.array([-0.2, 0, 0]),
        )
        observer = TrialObserver.__new__(TrialObserver)
        observer.model = SimpleNamespace(
            bounds={"hand": [{"frame": "hand", "corners": list(product((0.08, 0.12), (-0.02, 0.02), (0.98, 1.02)))}]}
        )
        point, pose = [0, 0, 2], np.eye(4)
        self.assertFalse(observer.occluded(observer.project(point, g, pose), {"hand": pose}, g, pose))
        self.assertTrue(
            observer.occluded(observer.project(point, g, pose, right=True), {"hand": pose}, g, pose, right=True)
        )
        # The center can be clear while the edge of a projected top is hidden.
        self.assertTrue(observer.occluded([[100, 100], [110, 105]], {"hand": pose}, g, pose))

    def test_projection_seed_is_resolved_to_detected_box_and_duplicates_rejected(self):
        hsv = np.zeros((100, 100, 3), np.uint8)
        hsv[45:55, 45:55] = [100, 210, 75]
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        found, _ = TrialObserver.reacquire_pixels(image, {"A": [70, 50]})
        np.testing.assert_array_equal(found["A"], [50, 50])
        with self.assertRaisesRegex(ValueError, "ambiguous_or_missing"):
            TrialObserver.reacquire_pixels(image, {"A": [50, 50], "B": [55, 50]})
        with self.assertRaisesRegex(ValueError, "ambiguous_or_missing"):
            TrialObserver.reacquire_pixels(image, {"A": [80, 50]})

    def test_coarse_boxes_enclose_20mm_drift_but_target_neighbors_stay_precise(self):
        letters = {
            name: {"estimated_xyz": center}
            for name, center in (("A", [0.4, 0, 0.02]), ("near", [0.5, 0, 0.02]), ("far", [0.4, -0.4, 0.02]))
        }
        boxes = cube_obstacles(letters)
        self.assertEqual(letters["A"]["depth_spread_limit_m"], 0.008)
        self.assertEqual(letters["near"]["tracking_tolerance_m"], 0.01)
        self.assertEqual(letters["far"]["tracking_tolerance_m"], 0.02)
        lo, hi = np.asarray(boxes[1])
        center = np.asarray(letters["far"]["estimated_xyz"])
        # A yawed cube translated by 20 mm must remain inside the planned box.
        half = np.array([0.04 / np.sqrt(2)] * 2 + [0.02])
        for axis in np.eye(3):
            for sign in (-1, 1):
                shifted = center + sign * 0.02 * axis
                self.assertTrue(np.all(shifted - half >= lo))
                self.assertTrue(np.all(shifted + half <= hi))

    def test_target_cannot_use_coarse_depth_mode(self):
        tracker = StereoTracker.__new__(StereoTracker)
        with self.assertRaisesRegex(ValueError, "invalid_coarse"):
            tracker.locate_images(None, None, {"A": [0, 0]}, None, coarse_objects={"A"})


class ObserverToleranceTests(unittest.IsolatedAsyncioTestCase):
    async def test_far_drift_passes_20mm_but_near_and_missing_cubes_still_reject(self):
        letters = {
            name: {"estimated_xyz": center, "top_center_xyz": center, "pixel": [30, 30]}
            for name, center in (("A", [0.4, 0, 1]), ("near", [0.5, 0, 1]), ("far", [0.4, -0.4, 1]))
        }
        cube_obstacles(letters)
        for name, drift, error in (
            ("far", 0.019, None),
            ("far", 0.021, "scene_object_moved"),
            ("near", 0.011, "scene_object_moved"),
            ("missing", 0, "current_scene_incomplete"),
        ):
            with self.subTest(name=name, drift=drift), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                image_path = root / "image.jpg"
                cv2.imwrite(str(image_path), np.zeros((64, 64, 3), np.uint8))
                observer = TrialObserver.__new__(TrialObserver)
                observer.client, observer.config = None, SimpleNamespace(calibration="unused")
                observer.output, observer.index, observer.initial = root, 0, letters
                observer.template, observer.image_hash = np.ones((5, 5), np.uint8), "hash"
                observer.model = Mock()
                observer.model.poses.return_value = {
                    name: np.eye(4) for name in ("left_index_distal", "left_thumb_distal")
                }
                measured = {"letters": copy.deepcopy(letters), "errors": {}}
                if name == "missing":
                    del measured["letters"]["far"]
                else:
                    measured["letters"][name]["estimated_xyz"][0] += drift
                tracker = Mock()
                tracker.locate_images.return_value = measured

                def capture(_config, output, _client, image_path=image_path):
                    output.mkdir()
                    return image_path, image_path, {"body_after": {"body": [], "hands": []}, "acquired_monotonic": 1}

                with (
                    patch("astrabot.robot.trial_observer.capture", side_effect=capture),
                    patch("astrabot.robot.trial_observer.load", return_value=(None, None, None)),
                    patch("astrabot.robot.trial_observer.StereoTracker", return_value=tracker),
                    patch.object(observer, "project", return_value=np.array([30, 30])),
                    patch.object(observer, "occluded", return_value=False),
                    patch.object(observer, "reacquire_pixels", return_value=({n: [30, 30] for n in letters}, None)),
                    patch.object(observer, "_glyph", return_value=observer.template),
                ):
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            await observer("before_preparation")
                    else:
                        result = await observer("before_preparation")
                        self.assertTrue(result["scene_clear"])
                        self.assertEqual(result["coarse_objects"], ["far"])
                        self.assertEqual(tracker.locate_images.call_args.kwargs["coarse_objects"], {"far"})
