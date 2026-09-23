"""Reject motion hidden between identical endpoints and missing exposure data."""

import copy
import unittest

from astrabot.robot.manual_capture import check_trace


class ManualCaptureTests(unittest.TestCase):
    def setUp(self):
        self.records = [{"joint_time": 10 + i * 0.01, "body": [0.0] * 29, "joint_fault": ""} for i in range(41)]

    def test_stationary_body_does_not_require_finger_feedback(self):
        result = check_trace(self.records, 10.2)
        self.assertTrue(result["passed"])
        self.assertEqual(result["body_at_exposure"], [0.0] * 29)

    def test_mid_capture_motion_rejected_despite_equal_endpoints(self):
        self.records[20]["body"][18] = 0.02
        with self.assertRaisesRegex(ValueError, "body_moved"):
            check_trace(self.records, 10.2)

    def test_missing_or_gapped_exposure_feedback_rejected(self):
        for records in (self.records[:20], self.records[25:], self.records[:14] + self.records[26:]):
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                check_trace(records, 10.2)

    def test_fault_and_nonfinite_and_duplicate_time_rejected(self):
        for field, value in (("joint_fault", "motor_fault"), ("body", [float("nan")] * 29), ("joint_time", 10.19)):
            records = copy.deepcopy(self.records)
            records[20][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_trace(records, 10.2)


if __name__ == "__main__":
    unittest.main()
