"""Verify unknown rigid marker offsets are observable and independently checked."""

import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from astrabot.robot.marker_calibration import fit_rigid_hand_markers


def observations():
    """Generate independent, nondegenerate wrist poses with stereo-like noise."""
    random = np.random.default_rng(90421)
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler("xyz", [-135, 0, -90], degrees=True).as_matrix()
    camera[:3, 3] = [0.03, 0.015, 0.46]
    markers = np.array([[-0.02, 0, 0.03], [-0.02, 0, 0.07]])
    records = []
    for index in range(16):
        hand = np.eye(4)
        hand[:3, :3] = Rotation.from_rotvec(random.uniform(-0.6, 0.6, 3)).as_matrix()
        hand[:3, 3] = random.uniform([0.2, 0.02, 0.05], [0.4, 0.2, 0.25])
        world = markers @ hand[:3, :3].T + hand[:3, 3]
        points = (world - camera[:3, 3]) @ camera[:3, :3]
        points += random.normal(0, 0.0002, points.shape)
        records.append(
            {
                "sample_id": str(index),
                "frame": "left_hand_base_link",
                "marker_ids": ["A", "B"],
                "hand_to_pelvis": hand.tolist(),
                "camera_points_m": points.tolist(),
            }
        )
    return records[:12], records[12:], camera, markers


class MarkerCalibrationTests(unittest.TestCase):
    def test_recovers_camera_and_unknown_marker_locations(self):
        training, validation, camera, markers = observations()
        report = fit_rigid_hand_markers(training, validation, 0.04)
        self.assertTrue(report["geometric_validation_passed"], report["blockers"])
        self.assertFalse(report["calibrated_for_manipulation"])
        np.testing.assert_allclose(report["camera_to_pelvis"], camera, atol=0.002)
        np.testing.assert_allclose(report["local_points"], markers, atol=0.001)

    def test_validation_errors_do_not_refit_camera(self):
        training, validation, _, _ = observations()
        first = fit_rigid_hand_markers(training, validation, 0.04)
        changed = copy.deepcopy(validation)
        points = np.array(changed[0]["camera_points_m"]) + [0.02, 0, 0]
        changed[0]["camera_points_m"] = points.tolist()
        second = fit_rigid_hand_markers(training, changed, 0.04)
        np.testing.assert_array_equal(first["camera_to_pelvis"], second["camera_to_pelvis"])
        self.assertIn("held_out_marker_error", second["blockers"])

    def test_constant_orientation_cannot_identify_camera_translation(self):
        training, validation, _, _ = observations()
        for record in training:
            transform = np.array(record["hand_to_pelvis"])
            transform[:3, :3] = np.eye(3)
            record["hand_to_pelvis"] = transform.tolist()
        with self.assertRaisesRegex(ValueError, "rotation_diversity_is_degenerate"):
            fit_rigid_hand_markers(training, validation, 0.04)

    def test_repeated_training_pose_cannot_be_validation(self):
        training, validation, _, _ = observations()
        validation[0] = copy.deepcopy(training[0])
        validation[0]["sample_id"] = "different-label-same-pose"
        with self.assertRaisesRegex(ValueError, "too_close_to_training"):
            fit_rigid_hand_markers(training, validation, 0.04)

    def test_ruler_measurement_checks_scale_without_changing_fit(self):
        training, validation, _, _ = observations()
        reference = fit_rigid_hand_markers(training, validation, 0.04)
        wrong_scale = fit_rigid_hand_markers(training, validation, 0.05)
        np.testing.assert_array_equal(reference["camera_to_pelvis"], wrong_scale["camera_to_pelvis"])
        self.assertIn("fitted_marker_scale_error", wrong_scale["blockers"])
        self.assertIn("held_out_marker_scale_error", wrong_scale["blockers"])

    def test_torso_reference_is_explicit_and_never_labeled_as_pelvis(self):
        training, validation, camera, _ = observations()
        for record in training + validation:
            record["hand_to_torso_link"] = record.pop("hand_to_pelvis")
        report = fit_rigid_hand_markers(training, validation, 0.04, reference_frame="torso_link")
        self.assertTrue(report["geometric_validation_passed"], report["blockers"])
        np.testing.assert_allclose(report["camera_to_torso_link"], camera, atol=0.002)
        self.assertNotIn("camera_to_pelvis", report)
        self.assertFalse(report["calibrated_for_manipulation"])
        with self.assertRaises(KeyError):
            fit_rigid_hand_markers(training, validation, 0.04)
        with self.assertRaisesRegex(ValueError, "unsupported_camera_reference"):
            fit_rigid_hand_markers(training, validation, 0.04, reference_frame="world")


if __name__ == "__main__":
    unittest.main()
