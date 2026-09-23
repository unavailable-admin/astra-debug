"""Finger and tactile evidence must bracket exposure without actuator writes."""

import copy
import unittest

import numpy as np

from astrabot.robot.manual_grasp_capture import check_hand_trace


class ManualGraspEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "recorded_monotonic": float(t),
                "hand_time": float(t - 0.01),
                "tactile_time": float(t - 0.02),
                "hands": [100.0] * 12,
                "tactile": [3.0, 4.0] + [0.0] * 8,
                "command_writes": 0,
                "fault": "",
            }
            for t in np.linspace(9.8, 10.2, 21)
        ]

    def test_valid_evidence_preserves_raw_contacts_without_claiming_force_units(self):
        result = check_hand_trace(self.records, 10, 0.25)
        self.assertTrue(result["passed"])
        self.assertEqual(result["tactile_at_exposure"][:2], [3.0, 4.0])
        self.assertIn("no Newton", result["tactile_units"])
        self.assertEqual(result["hands_at_exposure"], [100.0] * 12)

    def test_invalid_or_stale_feedback_and_actuation_are_rejected(self):
        for change, error in (
            ({"hand_time": 9.0}, "stale"),
            ({"tactile_time": 9.0}, "stale"),
            ({"tactile_time": float("nan")}, "stale"),
            ({"hands": [103.0] * 12}, "fingers_moved"),
            ({"command_writes": 1}, "command_write"),
            ({"fault": "CAN fault"}, "trace_fault"),
            ({"tactile": [float("nan")] * 10}, "capture_tactile"),
        ):
            with self.subTest(change=change):
                records = copy.deepcopy(self.records)
                records[10].update(change)
                with self.assertRaisesRegex(ValueError, error):
                    check_hand_trace(records, 10, 0.25)

    def test_missing_coverage_and_sampling_gaps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not_bracketed"):
            check_hand_trace(self.records[10:], 10, 0.25)
        with self.assertRaisesRegex(ValueError, "gap"):
            check_hand_trace(self.records[:7] + self.records[14:], 10, 0.25)


if __name__ == "__main__":
    unittest.main()
