"""Adaptive torque regressions; no DDS and no claim of closed-loop robot validation."""

import json
import unittest
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np

from astrabot.robot.compensation import AdaptiveBias
from astrabot.robot.config import REST
from astrabot.robot.hardware import Gravity
from astrabot.robot.motion_checks import MotionChecks


class AdaptiveBiasTests(unittest.TestCase):
    def test_static_lag_retains_novus_integration_and_deadband(self):
        bias = AdaptiveBias(MotionChecks())
        error = np.zeros(14)
        error[:3] = [0.01, -0.01, 0.0009]
        np.testing.assert_allclose(bias.update(error, 0.02)[:3], [0.016, -0.016, 0])
        np.testing.assert_allclose(bias.update(error, 0.02)[:3], [0.0319936, -0.0319936, 0])
        self.assertFalse(bias.braking.any())
        bias.bias[2] = 0.5
        bias.update(error, 0.02)
        self.assertAlmostEqual(bias.bias[2], 0.4998)

    def test_fast_approach_releases_before_crossing_in_both_directions(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                bias = AdaptiveBias(MotionChecks())
                bias.bias[8] = sign * 1.1
                bias.previous_error = np.zeros(14)
                bias.previous_error[8] = sign * 0.004
                error = np.zeros(14)
                error[8] = sign * 0.002
                bias.update(error, 0.02)
                self.assertTrue(bias.braking[8])
                self.assertAlmostEqual(bias.bias[8], sign * 0.7)
                self.assertFalse(bias.braking[:8].any())

    def test_overshoot_unwinds_without_changing_sign_in_one_update(self):
        bias = AdaptiveBias(MotionChecks())
        bias.bias[8] = -1.1
        error = np.zeros(14)
        error[8] = 0.002
        for expected in (-0.7, -0.3, 0):
            bias.update(error, 0.02)
            self.assertTrue(bias.braking[8])
            self.assertAlmostEqual(bias.bias[8], expected)
        bias.update(error, 0.02)
        self.assertGreater(bias.bias[8], 0)  # Relearn only after old bias is removed.

    def test_slow_approach_and_receding_reference_do_not_drop_learned_bias(self):
        for old_error, error in ((0.0021, 0.002), (0.002, 0.004)):
            bias = AdaptiveBias(MotionChecks())
            bias.bias[:] = 1
            bias.previous_error = np.full(14, old_error)
            bias.update(np.full(14, error), 0.02)
            self.assertFalse(bias.braking.any())
            self.assertTrue(np.all(bias.bias > 1))

    def test_large_lag_obeys_torque_and_slew_limits(self):
        bias = AdaptiveBias(MotionChecks())
        for _ in range(100):
            previous = bias.bias.copy()
            bias.update(np.full(14, 1), 0.02)
            self.assertTrue(np.all(np.abs(bias.bias - previous) <= 0.4 + 1e-12))
            self.assertTrue(np.all(np.abs(bias.bias) <= 10))
        np.testing.assert_allclose(bias.bias, 10)

    def test_invalid_input_does_not_mutate_state(self):
        bias = AdaptiveBias(MotionChecks())
        bias.bias[:] = 0.5
        for error, dt in (
            (np.zeros(13), 0.02),
            (np.full(14, np.nan), 0.02),
            (np.zeros(14), 0),
            (np.zeros(14), -1),
            (np.zeros(14), np.inf),
        ):
            with self.subTest(dt=dt), self.assertRaisesRegex(ValueError, "invalid_adaptive_feedback"):
                bias.update(error, dt)
            np.testing.assert_array_equal(bias.bias, 0.5)
            self.assertIsNone(bias.previous_error)

    def test_unwinding_preserves_model_gravity_and_reset_clears_history(self):
        with patch("astrabot.robot.hardware.time.monotonic", return_value=10.0) as clock:
            gravity = Gravity()
            gravity.bias[8] = -0.3
            measured = REST.copy()
            measured[8] -= 0.002
            clock.return_value = 10.03
            torque = gravity.evaluate(REST, measured)
            self.assertTrue(gravity.adaptation.braking[8])
            self.assertEqual(gravity.bias[8], 0)
            self.assertGreater(np.max(np.abs(gravity.model_torque)), 0.1)
            np.testing.assert_allclose(torque, gravity.model_torque)
            gravity.adaptation.reset()
            self.assertIsNone(gravity.adaptation.previous_error)
            self.assertFalse(gravity.adaptation.braking.any())

    def test_recorded_v7_bias_releases_before_soft_limit(self):
        fixture = Path(__file__).parent / "fixtures/robot_tracking_bias_v7.json"
        rows = np.array(json.loads(fixture.read_text())["samples"])
        bias = AdaptiveBias(MotionChecks())
        bias.bias[[1, 8]] = rows[0, 3:5]
        bias.previous_error = np.zeros(14)
        bias.previous_error[[1, 8]] = rows[0, 1:3]
        old = rows[0, 3:5].copy()
        first_brake = soft_limit = None
        for previous, row in pairwise(rows):
            dt = min(row[0] - previous[0], 0.1)
            # Verify the recorded torque is explained by the original integrator.
            error = np.where(np.abs(row[1:3]) > 0.001, row[1:3], 0)
            old += np.clip((80 * error - 0.02 * old) * dt, -20 * dt, 20 * dt)
            np.testing.assert_allclose(old, row[3:5], atol=2e-7, rtol=0)
            feedback_error = np.zeros(14)
            feedback_error[[1, 8]] = row[1:3]
            bias.update(feedback_error, dt)
            if bias.braking[8] and first_brake is None:
                first_brake = row.copy()
            if row[2] >= 0.005 and soft_limit is None:
                soft_limit = row.copy()
                self.assertLess(row[4], -1.0)
                self.assertGreaterEqual(bias.bias[8], 0)
        self.assertIsNotNone(first_brake)
        self.assertIsNotNone(soft_limit)
        self.assertLess(first_brake[2], 0)  # Release starts before crossing target.
        self.assertLess(first_brake[0], soft_limit[0])
