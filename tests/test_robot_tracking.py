"""Novus-style limiting must preserve direction and bound feedback-relative steps."""

import json
import unittest
from itertools import pairwise
from pathlib import Path

import numpy as np

from astrabot.robot.motion_checks import MotionChecks
from astrabot.robot.tracking import TrackingPacer, feedback_limited_target


class FeedbackLimiterTests(unittest.TestCase):
    def test_all_joint_deltas_share_scale_without_overshooting_target(self):
        feedback = np.linspace(-1, 1, 14)
        delta = np.array([0.04, -0.02, 0.01] + [0.0] * 11)
        limited = feedback_limited_target(feedback + delta, feedback, 0.002)
        np.testing.assert_allclose(limited - feedback, delta * 0.05, atol=1e-12)
        self.assertLessEqual(np.max(np.abs(limited - feedback)), 0.002 + 1e-12)

    def test_already_close_target_and_zero_step(self):
        feedback = np.linspace(-1, 1, 14)
        target = feedback + 0.0001
        np.testing.assert_array_equal(feedback_limited_target(target, feedback, 0.002), target)
        np.testing.assert_array_equal(feedback_limited_target(target, feedback, 0), feedback)

    def test_bad_feedback_or_limit_rejected(self):
        for target, feedback, step in (
            (np.zeros(14), np.zeros(13), 0.001),
            (np.full(14, np.nan), np.zeros(14), 0.001),
            (np.zeros(14), np.zeros(14), -0.001),
            (np.zeros(14), np.zeros(14), np.inf),
        ):
            with self.assertRaisesRegex(ValueError, "invalid_tracking_limiter_input"):
                feedback_limited_target(target, feedback, step)


class TrackingPacerTests(unittest.TestCase):
    def setUp(self):
        self.pacer = TrackingPacer(MotionChecks())
        self.now = 10.0
        self.position = np.zeros(14)

    def step(self, lag, speed=0.0, stamp=None):
        self.now += 0.004
        return self.pacer.advance(
            lag,
            0.005,
            self.now,
            0.004,
            self.now if stamp is None else stamp,
            self.position,
            np.full(14, speed),
        )

    def test_recorded_threshold_chatter_stays_one_wait_and_cannot_reset_timeout(self):
        # Recorded pattern: right shoulder repeatedly crosses the 0.002 gate.
        self.assertEqual(self.step(0.00213), 0)
        started = self.pacer.waiting_since
        for i in range(1875):  # 7.5 seconds, longer than the five-second budget.
            self.assertEqual(self.step(0.0019 if i % 2 else 0.00213), 0)
            self.assertEqual(self.pacer.waiting_since, started)
        for _ in range(40):
            self.assertEqual(self.step(0.0005), 0)  # A late good sample cannot revive it.
        self.assertGreater(self.now - started, 5)

    def test_position_crossing_during_overshoot_does_not_resume_until_velocity_settles(self):
        self.step(0.0021)
        for _ in range(40):
            self.assertEqual(self.step(0.001, speed=0.1), 0)
        for _ in range(25):
            self.assertEqual(self.step(0.001), 0)
        self.assertIsNotNone(self.pacer.waiting_since)
        increments = [self.step(0.001) for _ in range(100)]
        self.assertIsNone(self.pacer.waiting_since)
        self.assertLess(increments[1], 0.00001)
        self.assertAlmostEqual(increments[-1], 0.004)
        self.assertTrue(all(a <= b for a, b in pairwise(increments)))

    def test_duplicate_feedback_cannot_finish_stability_window(self):
        self.step(0.0021)
        self.step(0.001)
        stamp = self.now
        for _ in range(40):
            self.assertEqual(self.step(0.001, stamp=stamp), 0)
        self.assertIsNotNone(self.pacer.waiting_since)

    def test_excursion_resets_continuous_stability_and_repause_resets_ramp(self):
        self.step(0.0021)
        for _ in range(24):
            self.step(0.001)
        self.step(0.0016)
        for _ in range(25):
            self.assertEqual(self.step(0.001), 0)
        self.step(0.001)
        self.step(0.001)
        self.assertIsNone(self.pacer.waiting_since)
        self.assertGreater(self.pacer.scale, 0)
        self.assertEqual(self.step(0.0021), 0)
        self.assertEqual(self.pacer.ramp_elapsed, 0)

    def test_mock_velocity_estimate_uses_feedback_time_not_control_time(self):
        p = self.pacer
        p.advance(0.0021, 0.005, 10, 0.004, 10, self.position)
        self.position[8] = 0.001
        p.advance(0.001, 0.005, 10.02, 0.004, 10.01, self.position)
        self.assertAlmostEqual(p.raw_speed, 0.1)
        self.assertIsNone(p.speed)  # Still filling the 50 ms observation window.
        self.assertIsNone(p.stable_since)

    def test_recorded_v6_wait_recovers_within_budget_without_relaxing_position_bounds(self):
        recorded = json.loads((Path(__file__).parent / "fixtures/robot_tracking_wait_v6.json").read_text())
        p = self.pacer
        p.waiting_since = 0.0
        p.ramp_elapsed = 0.0
        raw_stable_since = None
        raw_longest = 0.0
        resumed_at = None
        last_stamp = recorded["samples"][0][0] - 0.004
        for stamp, lag, velocity, position in recorded["samples"]:
            raw_quiet = lag < 0.0015 and max(abs(v) for v in velocity) <= 0.02
            if not raw_quiet:
                raw_stable_since = None
            elif raw_stable_since is None:
                raw_stable_since = stamp
            else:
                raw_longest = max(raw_longest, stamp - raw_stable_since)
            if resumed_at is None:
                progress = p.advance(
                    lag, 0.005, stamp, stamp - last_stamp, stamp, np.array(position), np.array(velocity)
                )
                if p.waiting_since is None:
                    resumed_at = stamp
                    self.assertLess(lag, 0.0015)
                    self.assertLessEqual(p.speed, 0.02)
                    self.assertLessEqual(p.raw_speed, 0.06)
                    self.assertLess(progress, 0.0001)
            last_stamp = stamp
        self.assertLess(raw_longest, 0.1)  # Reproduce why v6 could never resume.
        self.assertIsNotNone(resumed_at)
        self.assertLess(resumed_at, 4.0)  # Recovery before the original 5 s deadline.

    def test_fast_alternating_motion_cannot_cancel_in_speed_average(self):
        self.step(0.0021)
        for i in range(100):
            self.assertEqual(self.step(0.001, speed=0.05 if i % 2 else -0.05), 0)
        self.assertGreater(self.pacer.speed, 0.02)

    def test_large_instantaneous_speed_still_interrupts_stability(self):
        self.step(0.0021)
        for _ in range(32):
            self.step(0.001)
        self.assertIsNotNone(self.pacer.stable_since)
        self.assertEqual(self.step(0.001, speed=0.07), 0)
        self.assertEqual(self.pacer.resume_blocker, "speed_peak")
        self.assertIsNone(self.pacer.stable_since)

    def test_position_movement_blocks_recovery_even_with_quiet_sdk_velocity(self):
        self.step(0.0021)
        for i in range(80):
            self.position[8] = 0.001 if i % 2 else -0.001
            self.assertEqual(self.step(0.001, speed=0), 0)
        self.assertGreater(self.pacer.speed, 0.02)
