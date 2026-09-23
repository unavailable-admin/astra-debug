"""Ownership handoff regressions; no robot SDK publication."""

import unittest
from unittest.mock import patch

from astrabot.robot.config import Config
from astrabot.robot.executor import Executor
from astrabot.robot.hardware import MockHardware


class DampingHandoffTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.hardware = MockHardware(lambda: self.now)
        snapshot = self.hardware.snapshot

        def damping_snapshot():
            s = snapshot()
            s.mode = self.mode
            return s

        self.mode = 1
        self.hardware.snapshot = damping_snapshot
        self.executor = Executor(self.hardware, None, Config(), lambda: self.now)
        self.addCleanup(self.executor.close)

    def tick(self, duration=0.01):
        self.now += duration
        self.executor.tick()

    def test_zero_frames_complete_without_opening_hands_or_replaying_targets(self):
        self.executor.attached = True
        self.executor.weight = 1.0
        self.executor.request({"op": "disarm"})
        for _ in range(51):
            self.tick()
        self.assertEqual(self.executor.state, "DISARMED")
        self.assertFalse(self.executor.attached)
        self.assertEqual(self.executor.weight, 0)
        self.assertGreaterEqual(self.executor.disarm_writes, 10)
        self.assertTrue(all(hand is None and weight == 0 for _, _, hand, weight in self.hardware.writes))
        count = len(self.hardware.writes)
        self.mode = 4
        self.tick()
        self.assertEqual(len(self.hardware.writes), count)

    def test_standing_rejected_and_mode_change_interrupts_before_next_write(self):
        self.mode = 4
        with self.assertRaisesRegex(ValueError, "damping_mode_required"):
            self.executor.request({"op": "disarm"})
        self.assertFalse(self.hardware.writes)
        self.mode = 1
        self.executor.request({"op": "disarm"})
        self.tick()
        count = len(self.hardware.writes)
        self.mode = 4
        self.tick()
        self.assertEqual(len(self.hardware.writes), count)
        self.assertEqual(self.executor.state, "FAULT")

    def test_failed_publication_does_not_clear_attached_or_record_success(self):
        self.executor.attached = True
        self.executor.weight = 1
        self.executor.request({"op": "disarm"})
        with patch.object(self.hardware, "clear_arm_control", side_effect=RuntimeError("dds_write_failed")):
            self.tick()
        self.assertTrue(self.executor.attached)
        self.assertEqual(self.executor.weight, 1)
        self.assertEqual(self.executor.disarm_writes, 0)
        self.assertEqual(self.executor.state, "FAULT")

    def test_stale_feedback_or_mode_query_cannot_relinquish(self):
        self.hardware.frozen = True
        self.now += 1
        with self.assertRaisesRegex(ValueError, "joint_feedback_stale"):
            self.executor.request({"op": "disarm"})
        self.hardware.frozen = False
        self.hardware.fault = "locomotion_mode_stale"
        with self.assertRaisesRegex(ValueError, "locomotion_mode_stale"):
            self.executor.request({"op": "disarm"})
        self.assertFalse(self.hardware.writes)

    def test_pause_cancels_handoff_and_begin_cannot_interrupt_it(self):
        self.executor.request({"op": "disarm"})
        with self.assertRaisesRegex(ValueError, "handoff_in_progress"):
            self.executor.request({"op": "begin"})
        self.executor.request({"op": "pause"})
        self.tick()
        self.assertFalse(self.hardware.writes)
