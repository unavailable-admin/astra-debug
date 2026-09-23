"""Fault/race tests use a controllable clock; they never load robot SDKs."""

import itertools
import threading
import time
import unittest
from concurrent.futures import Future
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from astrabot.robot.config import REST, Config
from astrabot.robot.executor import Executor
from astrabot.robot.hardware import MockHardware, from_sdk, to_sdk
from astrabot.robot.trajectory import Segment


class Clock:
    def __init__(self):
        self.time = 10.0

    def __call__(self):
        return self.time


class Planner:
    def ready(self, snapshot):
        q = snapshot.body[15:]
        return [Segment(q.copy(), snapshot.hands.copy(), q.copy(), np.full(12, 255.0), 0.05)]

    park = ready

    def move(self, snapshot, request):
        q = snapshot.body[15:]
        return [Segment(q.copy(), snapshot.hands.copy(), np.array(request["arms"]), snapshot.hands.copy(), 0.2)]


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.hardware = MockHardware(self.clock)
        self.executor = Executor(self.hardware, Planner(), Config(), self.clock)
        self.addCleanup(self.executor.close)

    def tick(self, seconds=0.01):
        self.clock.time += seconds
        self.executor.tick()

    def begin(self):
        result = self.executor.request({"op": "begin"})
        self.executor.weight = 1.0
        return {k: result[k] for k in ("session", "generation")}

    def test_preserve_right_rejects_changing_the_held_command(self):
        token = self.begin()
        arms = self.executor.arms.copy()
        hands = self.executor.hands.copy()
        for key in ("arms", "hands"):
            request = {
                "op": "move",
                **token,
                "action": "trial_joint_step",
                "preserve_right": True,
                "arms": arms.copy(),
                "hands": hands.copy(),
            }
            request[key][-1] += 0.01
            with self.assertRaisesRegex(ValueError, "right_hold_command_changed"):
                self.executor.request(request)
        self.assertIsNone(self.executor.future)

    def test_fixed_grip_rejects_retargeting_during_carry(self):
        self.executor.config.trial_grip_mode = "supervised_position"
        token = self.begin()
        self.hardware.hands[4] = 126.0
        held = self.hardware.hands.copy()
        held[4] = 124.7767781746445
        self.executor.hands = held.copy()
        for thumb in (126.0, 123.0):
            changed = held.copy()
            changed[4] = thumb
            with self.assertRaisesRegex(ValueError, "held_grip_command_changed"):
                self.executor.request(
                    {
                        "op": "move",
                        **token,
                        "action": "pinch_step",
                        "holding": True,
                        "hands": changed.tolist(),
                        "center": [0, 0, 0],
                    }
                )
        future = Future()
        with patch.object(self.executor.pool, "submit", return_value=future):
            self.executor.request(
                {
                    "op": "move",
                    **token,
                    "action": "pinch_step",
                    "holding": True,
                    "hands": held.tolist(),
                    "center": [0, 0, 0],
                }
            )
        np.testing.assert_array_equal(self.executor.hands, held)

    def test_fixed_grip_commands_do_not_relax_during_or_after_motion(self):
        self.executor.config.trial_grip_mode = "supervised_position"
        self.begin()
        self.hardware.hands[4] = 126.0
        measured = self.hardware.hands.copy()
        held = measured.copy()
        held[4] = 124.7767781746445
        self.executor.hands = held.copy()
        command = self.hardware.command

        def blocked_command(arms, hands, weight):
            command(arms, hands, weight)
            self.hardware.hands = measured.copy()

        self.hardware.command = blocked_command
        arms = self.hardware.body[15:].copy()
        self.executor.queue.append(Segment(arms, held.copy(), arms + 0.001, held.copy(), 0.1))
        for _ in range(30):
            self.tick()
        self.assertFalse(self.executor.queue)
        self.assertEqual(self.executor.state, "RUNNING")
        self.assertTrue(self.hardware.writes)
        for _, _, hands, _ in self.hardware.writes:
            np.testing.assert_array_equal(hands, held)
        self.assertEqual(self.hardware.hands[4], 126.0)

    def test_carry_waits_for_cartesian_arrival_and_keeps_command(self):
        self.begin()
        arms = self.hardware.body[15:].copy()
        hands = self.hardware.hands.copy()
        center = np.zeros(3)
        target_center = np.array([0, 0, 0.003])

        def poses(*args):
            from astrabot.pinch import INDEX_TIP, THUMB_TIP

            result = {}
            for link, tip in (("left_index_distal", INDEX_TIP), ("left_thumb_distal", THUMB_TIP)):
                pose = np.eye(4)
                pose[:3, 3] = center - tip[:3]
                result[link] = pose
            return result

        self.executor.planner.model = Mock(pinch_poses=poses)
        segment = Segment(arms, hands, arms + 0.001, hands, 0.1, pinch_center=target_center)
        self.executor.queue.append(segment)
        for _ in range(30):
            self.tick()
        self.assertTrue(self.executor.queue)  # Joint tolerance alone would pass.
        np.testing.assert_allclose(self.executor.arms, segment.target)
        center[:] = [0, 0, 0.0025]
        self.tick()
        self.assertFalse(self.executor.queue)

    def test_continuous_carry_skips_internal_arrival_but_waits_at_endpoint(self):
        self.begin()
        arms = self.hardware.body[15:].copy()
        hands = self.hardware.hands.copy()
        self.executor.planner.model = Mock(
            pinch_poses=lambda *args: {"left_index_distal": np.eye(4), "left_thumb_distal": np.eye(4)}
        )
        for index in range(2):
            self.executor.queue.append(
                Segment(
                    arms + index * 0.001,
                    hands,
                    arms + (index + 1) * 0.001,
                    hands,
                    0.1,
                    pinch_center=np.zeros(3),
                    continuous_carry=True,
                )
            )
        for _ in range(30):
            self.tick()
        self.assertEqual(len(self.executor.queue), 1)
        self.assertIsNotNone(self.executor.arrival_started)
        self.assertFalse(self.executor.error)
        self.executor.config.settle_timeout = 0.1
        for _ in range(20):
            self.tick()
        self.assertEqual(self.executor.state, "FAULT")
        self.assertIn("pinch_arrival_timeout", self.executor.error)

    def test_carry_arrival_config_accepts_recorded_error_but_caps_at_half_step(self):
        self.begin()
        self.executor.config.motion_checks = replace(self.executor.config.motion_checks, pinch_arrival_m=0.0015)
        arms = self.hardware.body[15:].copy()
        hands = self.hardware.hands.copy()
        error = 0.0016

        def poses(*args):
            from astrabot.pinch import INDEX_TIP, THUMB_TIP

            result = {}
            for link, tip in (("left_index_distal", INDEX_TIP), ("left_thumb_distal", THUMB_TIP)):
                pose = np.eye(4)
                pose[:3, 3] = np.array([0, 0, error]) - tip[:3]
                result[link] = pose
            return result

        self.executor.planner.model = Mock(pinch_poses=poses)
        segment = Segment(arms, hands, arms + 0.001, hands, 0.1, pinch_center=np.zeros(3))
        self.executor.queue.append(segment)
        for _ in range(30):
            self.tick()
        self.assertTrue(self.executor.queue)
        error = 0.001258
        self.executor.config.motion_checks = replace(self.executor.config.motion_checks, pinch_substep_m=0.002)
        self.tick()
        self.assertTrue(self.executor.queue)  # Smaller step caps tolerance at 1 mm.
        self.executor.config.motion_checks = replace(self.executor.config.motion_checks, pinch_substep_m=0.003)
        self.tick()
        self.assertFalse(self.executor.queue)
        np.testing.assert_array_equal(segment.target, arms + 0.001)

    def test_carry_stall_times_out_without_replanning_or_increasing_target(self):
        self.begin()
        self.executor.config.settle_timeout = 0.1
        arms = self.hardware.body[15:].copy()
        hands = self.hardware.hands.copy()
        # Both tips remain 35 mm above the goal even though joints arrive.
        self.executor.planner.model = Mock(
            pinch_poses=lambda *args: {"left_index_distal": np.eye(4), "left_thumb_distal": np.eye(4)}
        )
        self.executor.queue.append(Segment(arms, hands, arms + 0.001, hands, 0.1, pinch_center=np.zeros(3)))
        for _ in range(40):
            self.tick()
        self.assertEqual(self.executor.state, "FAULT")
        self.assertIn("pinch_arrival_timeout:remaining_mm=", self.executor.error)
        self.assertFalse(self.executor.queue)

    def test_transfer_arrival_uses_four_mm_without_changing_microsteps(self):
        self.begin()
        arms = self.hardware.body[15:].copy()
        hands = self.hardware.hands.copy()
        error = 0.0041

        def poses(*args):
            from astrabot.pinch import INDEX_TIP, THUMB_TIP

            result = {}
            for link, tip in (("left_index_distal", INDEX_TIP), ("left_thumb_distal", THUMB_TIP)):
                pose = np.eye(4)
                pose[:3, 3] = np.array([0, 0, error]) - tip[:3]
                result[link] = pose
            return result

        self.executor.planner.model = Mock(pinch_poses=poses)
        segment = Segment(arms, hands, arms + 0.001, hands, 0.1, name="transfer_path", pinch_center=np.zeros(3))
        self.executor.queue.append(segment)
        for _ in range(30):
            self.tick()
        self.assertTrue(self.executor.queue)
        error = 0.003608
        self.tick()
        self.assertFalse(self.executor.queue)
        self.assertEqual(self.executor.config.motion_checks.pinch_arrival_m, 0.001)

    def test_bound_scene_mismatch_rejected_before_planning(self):
        token = self.begin()
        self.executor.scene_id = "current"
        with self.assertRaisesRegex(ValueError, "motion_scene_changed"):
            self.executor.request({"op": "move", **token, "expected_scene_id": "old"})
        self.assertIsNone(self.executor.future)

    def test_default_is_read_only(self):
        for _ in range(10):
            self.tick()
        self.assertEqual(self.hardware.writes, [])

    def test_wrist_recovery_rejects_planning_drift(self):
        self.begin()
        snapshot = self.hardware.snapshot()
        q = snapshot.body[15:].copy()
        target = q.copy()
        target[12] += 0.05
        future = Future()
        future.set_result([Segment(q, snapshot.hands.copy(), target, snapshot.hands.copy(), 4, "recover_wrists")])
        self.executor.future = (self.executor.generation, future, snapshot, self.clock())
        self.hardware.body[27] -= 0.0015
        self.tick()
        self.assertEqual(self.executor.error, "preflight:state_changed_during_planning")
        self.assertFalse(self.executor.queue)

    def test_wrist_recovery_stops_outward_or_unrelated_motion(self):
        for joint, delta in ((27, -0.0021), (15, 0.011), (12, 0.011)):
            self.begin()
            snapshot = self.hardware.snapshot()
            q = snapshot.body[15:].copy()
            target = q.copy()
            target[12] += 0.05
            self.executor.queue.append(
                Segment(q, snapshot.hands.copy(), target, snapshot.hands.copy(), 4, "recover_wrists")
            )
            self.hardware.body[joint] += delta
            self.tick()
            self.assertEqual(self.executor.error, "wrist_recovery_feedback_outside_envelope")
            self.assertFalse(self.executor.queue)

    def test_wrist_recovery_requires_tighter_arrival_than_normal_move(self):
        self.begin()
        snapshot = self.hardware.snapshot()
        q = snapshot.body[15:].copy()
        target = q.copy()
        target[12] += 0.05
        self.executor.queue.append(
            Segment(q, snapshot.hands.copy(), target, snapshot.hands.copy(), 4, "recover_wrists")
        )
        self.executor.segment_elapsed = 4
        self.executor.arms = target.copy()
        self.hardware.body[15:] = target
        self.hardware.body[27] -= 0.02
        self.tick()
        self.assertEqual(self.executor.error, "arm_tracking_error")

    def test_thigh_escape_refreshes_allowed_drift_then_accepts_checked_start(self):
        self.begin()
        self.executor.lease = None
        snapshot = self.hardware.snapshot()
        q = snapshot.body[15:].copy()
        future = Future()
        future.set_result([Segment(q, snapshot.hands.copy(), q.copy(), snapshot.hands.copy(), 1, name="clear_thighs")])
        self.executor.future = (self.executor.generation, future, snapshot, self.clock())
        self.executor.planning_request = ("prepare", {})
        self.executor.planning_origin = snapshot
        self.executor.state = "PLANNING"
        self.hardware.body[14] += 0.0093
        self.hardware.body[16] += 0.09
        refreshed = Future()
        with patch.object(self.executor.pool, "submit", return_value=refreshed) as submit:
            self.tick()
        self.assertEqual(self.executor.error, "")
        self.assertFalse(self.executor.queue)
        self.assertEqual(self.executor.planning_refreshes, 1)
        current = submit.call_args.args[1]
        self.assertAlmostEqual(current.body[16] - snapshot.body[16], 0.09)
        np.testing.assert_array_equal(self.executor.arms, current.body[15:])
        q = current.body[15:].copy()
        refreshed.set_result([Segment(q, current.hands.copy(), q.copy(), current.hands.copy(), 1, name="clear_thighs")])
        self.tick()
        self.assertEqual(self.executor.error, "")
        self.assertTrue(self.executor.queue)
        np.testing.assert_array_equal(self.executor.baseline, current.body[:15])

    def test_drift_above_point_one_still_rejects_before_dispatch(self):
        self.begin()
        snapshot = self.hardware.snapshot()
        q = snapshot.body[15:].copy()
        future = Future()
        future.set_result([Segment(q, snapshot.hands.copy(), q.copy(), snapshot.hands.copy(), 1, name="clear_thighs")])
        self.executor.future = (self.executor.generation, future, snapshot, self.clock())
        self.hardware.body[16] += 0.101
        self.tick()
        self.assertEqual(self.executor.error, "preflight:state_changed_during_planning")
        self.assertFalse(self.executor.queue)

    def test_pause_invalidates_refreshed_plan(self):
        self.begin()
        snapshot = self.hardware.snapshot()
        q = snapshot.body[15:].copy()
        future = Future()
        future.set_result([Segment(q, snapshot.hands.copy(), q.copy(), snapshot.hands.copy(), 1, name="clear_thighs")])
        self.executor.future = (self.executor.generation, future, snapshot, self.clock())
        self.executor.planning_request = ("prepare", {})
        self.hardware.body[16] += 0.01
        refreshed = Future()
        with patch.object(self.executor.pool, "submit", return_value=refreshed):
            self.tick()
        self.executor.request({"op": "pause"})
        self.assertTrue(refreshed.cancelled())
        self.assertIsNone(self.executor.future)
        self.assertFalse(self.executor.queue)

    def install_escape(self):
        self.begin()
        self.executor.lease = None
        self.executor.planning_request = ("prepare", {})
        state = self.hardware.snapshot()
        self.executor.planning_origin = state
        q = state.body[15:].copy()
        self.executor.queue.append(Segment(q, state.hands.copy(), q.copy(), state.hands.copy(), 1, "clear_thighs"))

    def test_escape_body_change_holds_then_accepts_new_certificate(self):
        self.install_escape()
        original = self.executor.baseline.copy()
        self.hardware.body[14] += 0.049
        future = Future()
        with patch.object(self.executor.pool, "submit", return_value=future) as submit:
            self.tick()
        self.assertEqual(self.executor.state, "PLANNING")
        self.assertEqual(self.executor.error, "")
        self.assertFalse(self.executor.queue)
        self.assertEqual(self.executor.segment_elapsed, 0)
        np.testing.assert_array_equal(self.executor.escape_body_origin, original)
        state = submit.call_args.args[1]
        q = state.body[15:].copy()
        future.set_result([Segment(q, state.hands.copy(), q.copy(), state.hands.copy(), 1, "clear_thighs")])
        self.tick()
        self.assertEqual(self.executor.state, "RESETTING")
        self.assertEqual(self.executor.error, "")
        np.testing.assert_array_equal(self.executor.baseline, state.body[:15])
        # Recertification must not reset the cumulative 0.05 rad budget.
        self.hardware.body[14] += 0.002
        self.tick()
        self.assertEqual(self.executor.error, "base_or_waist_moved")

    def test_escape_body_recheck_cannot_resume_after_pause(self):
        self.install_escape()
        self.hardware.body[14] += 0.01
        future = Future()
        future.set_running_or_notify_cancel()
        with patch.object(self.executor.pool, "submit", return_value=future):
            self.tick()
        self.executor.request({"op": "pause"})
        future.set_result(Planner().ready(self.hardware.snapshot()))
        self.tick()
        self.assertFalse(self.executor.queue)
        self.assertEqual(self.executor.state, "HOLD")

    def test_escape_body_recheck_keeps_limit_during_planning(self):
        self.install_escape()
        self.hardware.body[14] += 0.01
        with patch.object(self.executor.pool, "submit", return_value=Future()):
            self.tick()
        self.hardware.body[14] += 0.041
        self.tick()
        self.assertEqual(self.executor.error, "base_or_waist_moved")
        self.assertIsNone(self.executor.future)

    def test_escape_refresh_is_bounded_and_does_not_replace_trial_intent(self):
        for leased in (True, False):
            self.install_escape()
            if leased:
                self.executor.lease = self.clock()
            else:
                self.executor.escape_refreshes = 2
            self.hardware.body[14] += 0.01
            with patch.object(self.executor.pool, "submit") as submit:
                self.tick()
            submit.assert_not_called()
            self.assertEqual(self.executor.state, "HOLD")
            self.assertEqual(self.executor.error, "thigh_escape_body_recheck_required")
            self.assertFalse(self.executor.queue)

    def test_escape_body_over_limit_faults_without_replan(self):
        self.install_escape()
        self.hardware.body[14] += 0.051
        with patch.object(self.executor.pool, "submit") as submit:
            self.tick()
        submit.assert_not_called()
        self.assertEqual(self.executor.error, "base_or_waist_moved")

    def test_thigh_escape_stops_on_finger_or_tracking_drift(self):
        for finger_drift in (True, False):
            with self.subTest(finger_drift=finger_drift):
                self.begin()
                snapshot = self.hardware.snapshot()
                q = snapshot.body[15:].copy()
                segment = Segment(q, snapshot.hands.copy(), q.copy(), snapshot.hands.copy(), 1, name="clear_thighs")
                self.executor.queue.append(segment)
                if finger_drift:
                    self.hardware.hands[0] -= 2
                else:
                    self.hardware.body[16] += 0.011
                self.tick()
                self.assertEqual(self.executor.state, "FAULT")
                self.assertFalse(self.executor.queue)
                self.assertEqual(
                    self.executor.error, "thigh_escape_body_or_fingers_moved" if finger_drift else "arm_tracking_error"
                )

    def test_thigh_escape_waits_for_lag_then_resumes_without_raising_limit(self):
        self.begin()
        snapshot = self.hardware.snapshot()
        start = snapshot.body[15:].copy()
        target = start.copy()
        target[1] += 0.1
        segment = Segment(start, snapshot.hands.copy(), target, snapshot.hands.copy(), 2, name="clear_thighs")
        self.executor.queue.append(segment)
        self.executor.segment_elapsed = 0.3
        self.executor.arms, _ = segment.sample(0.3)
        self.hardware.body[15:] = self.executor.arms
        self.hardware.body[16] -= 0.003
        self.tick()
        self.assertEqual(self.executor.segment_elapsed, 0.3)
        self.assertNotEqual(self.executor.state, "FAULT")
        # Catching up alone is insufficient: wait for quiet, sustained feedback.
        for _ in range(20):
            self.tick()
        self.assertGreater(self.executor.segment_elapsed, 0.3)
        self.assertIsNone(self.executor.tracking_wait_started)

    def test_contact_completion_does_not_leak_wait_state_into_next_segment(self):
        self.begin()
        snapshot = self.hardware.snapshot()
        q, hand = snapshot.body[15:].copy(), snapshot.hands.copy()
        self.executor.queue.append(Segment(q, hand, q.copy(), hand.copy(), 1, contact_guard=True))
        self.executor.tracking.waiting_since = self.clock() - 1
        snapshot.tactile[:2] = [10, 10]
        with patch.object(self.hardware, "snapshot", return_value=snapshot):
            self.tick()
        self.assertTrue(self.executor.contact_detected)
        self.assertIsNone(self.executor.tracking_wait_started)
        self.assertEqual(self.executor.tracking.scale, 1)

    def test_recorded_escape_overshoot_waits_and_recovers_without_advancing(self):
        self.begin()
        snapshot = self.hardware.snapshot()
        q = snapshot.body[15:].copy()
        target = q.copy()
        target[8] -= 0.1
        segment = Segment(q, snapshot.hands.copy(), target, snapshot.hands.copy(), 2, "clear_thighs")
        self.executor.queue.append(segment)
        self.executor.segment_elapsed = 0.3
        self.executor.arms, _ = segment.sample(0.3)
        self.hardware.body[15:] = self.executor.arms
        # Exact peak from executor-8a6f393..., right shoulder roll, 2026-09-18.
        self.hardware.body[23] -= 0.005182123955750323
        before = self.executor.arms.copy()
        self.tick()
        self.assertEqual(self.executor.error, "")
        self.assertEqual(self.executor.segment_elapsed, 0.3)
        self.assertLess(np.max(abs(self.hardware.writes[-1][1] - before)), 0.005182123955750323)
        self.assertIsNotNone(self.executor.tracking_violation_started)
        # Commands converge to the frozen reference under the speed limit.
        for _ in range(20):
            self.tick()
        self.assertEqual(self.executor.error, "")
        self.assertGreater(self.executor.segment_elapsed, 0.3)
        self.assertIsNone(self.executor.tracking_violation_started)

    def test_persistent_small_escape_overshoot_faults_instead_of_waiting_forever(self):
        self.begin()
        snapshot = self.hardware.snapshot()
        q = snapshot.body[15:].copy()
        self.executor.queue.append(
            Segment(q, snapshot.hands.copy(), q.copy(), snapshot.hands.copy(), 1, "clear_thighs")
        )
        self.hardware.body[23] -= 0.006
        commands = []
        self.hardware.command = lambda arms, hands, weight: commands.append(np.array(arms).copy())
        for _ in range(28):
            self.tick()
        self.assertEqual(self.executor.error, "arm_tracking_error")
        self.assertFalse(self.executor.queue)
        for command in commands[:25]:
            self.assertLessEqual(
                np.max(abs(command - self.hardware.body[15:])),
                self.executor.config.motion_checks.escape_tracking_hard_rad,
            )
        self.assertEqual(self.executor.segment_elapsed, 0.0)

    def test_limited_command_reaches_frozen_reference_instead_of_deadlocking(self):
        self.begin()
        snapshot = self.hardware.snapshot()
        q = snapshot.body[15:].copy()
        target = q.copy()
        target[8] -= 0.1
        segment = Segment(q, snapshot.hands.copy(), target, snapshot.hands.copy(), 2, "clear_thighs")
        self.executor.queue.append(segment)
        self.executor.segment_elapsed = 0.3
        reference, _ = segment.sample(0.3)
        # Recorded wait-trigger magnitude from executor-a1d0678..., 2026-09-18.
        measured = reference.copy()
        measured[8] += 0.0021323830861123827
        self.hardware.body[15:] = measured
        self.executor.arms = measured.copy()  # Last limited command already reached.
        self.tick(0.004)
        self.assertEqual(self.executor.segment_elapsed, 0.3)
        self.assertGreater(np.max(abs(self.executor.arms - measured)), 0)
        self.assertLessEqual(np.max(abs(self.executor.arms - measured)), 0.0012 + 1e-10)
        for _ in range(40):
            self.tick(0.004)
        self.assertGreater(self.executor.segment_elapsed, 0.3)
        self.assertEqual(self.executor.error, "")

    def test_slew_limit_accumulates_through_static_deadband_without_raising_feedback_limits(self):
        self.begin()
        self.executor.lease = None
        self.executor.config.joint_speed = 0.15
        snapshot = self.hardware.snapshot()
        q = snapshot.body[15:].copy()
        target = q.copy()
        target[8] -= 0.1
        segment = Segment(q, snapshot.hands.copy(), target, snapshot.hands.copy(), 2, "clear_thighs")
        self.executor.queue.append(segment)
        self.executor.segment_elapsed = 0.3
        reference, _ = segment.sample(0.3)
        self.hardware.body[15:] = reference
        self.hardware.body[23] += 0.00203310399725512
        self.executor.arms = self.hardware.body[15:].copy()
        commands, leads = [], []

        def static_friction(arms, hands, weight):
            commands.append(np.array(arms).copy())
            error = arms - self.hardware.body[15:]
            leads.append(float(np.max(abs(error))))
            moving = abs(error) > 0.001  # Same scale as the compensator deadband.
            self.hardware.body[15:][moving] += 0.5 * error[moving]

        self.hardware.command = static_friction
        previous = self.executor.arms.copy()
        for _ in range(1500):
            self.tick(0.004)
            if self.executor.error or self.executor.segment_elapsed > 0.4:
                break
        self.assertEqual(self.executor.error, "")
        self.assertGreater(self.executor.segment_elapsed, 0.4)
        self.assertGreater(max(leads), 0.001)
        self.assertLessEqual(max(leads), self.executor.config.motion_checks.escape_tracking_hard_rad)
        self.assertLessEqual(np.max(abs(np.diff(np.vstack((previous, commands)), axis=0))), 0.0006 + 1e-10)

    def test_prepare_waits_for_takeover_and_uses_settled_body(self):
        self.executor.config.scene_measured = True
        with patch.object(self.executor.pool, "submit", return_value=Future()) as submit:
            state = self.executor.request({"op": "prepare"})
            self.assertTrue(state["planning_active"])
            self.assertTrue(state["planning_waiting_for_stability"])
            self.hardware.body[14] += 0.007949512917548418
            for _ in range(199):
                self.tick()
            submit.assert_not_called()
            for _ in range(40):
                self.tick()
            submit.assert_called_once()
            self.assertAlmostEqual(submit.call_args.args[1].body[14], 0.007949512917548418)
            self.assertEqual(self.executor.planning_refreshes, 0)
            self.assertIsNone(self.executor.pending_plan)

    def test_pause_cancels_wait_for_settled_preparation(self):
        self.executor.config.scene_measured = True
        with patch.object(self.executor.pool, "submit") as submit:
            self.executor.request({"op": "prepare"})
            self.executor.request({"op": "pause"})
            for _ in range(300):
                self.tick()
            submit.assert_not_called()
        self.assertIsNone(self.executor.pending_plan)
        self.assertFalse(self.executor.queue)
        self.assertEqual(self.executor.state, "HOLD")

    def test_unstable_preparation_start_times_out_without_planning(self):
        self.executor.config = replace(
            self.executor.config,
            scene_measured=True,
            motion_checks=replace(
                self.executor.config.motion_checks, planning_stable_window_s=0.03, planning_settle_timeout_s=0.1
            ),
        )
        with patch.object(self.executor.pool, "submit") as submit:
            self.executor.request({"op": "prepare"})
            self.executor.weight, self.executor.takeover_active = 1.0, False
            for i in range(15):
                self.hardware.body[22] = 0.002 if i % 2 else 0
                self.tick()
            submit.assert_not_called()
        self.assertEqual(self.executor.error, "preparation_start_did_not_settle")
        self.assertIsNone(self.executor.pending_plan)

    def test_small_planning_drift_does_not_shift_existing_hold_target(self):
        self.install_escape()
        self.executor.queue.clear()
        snapshot = self.hardware.snapshot()
        original = self.executor.arms.copy()
        result = Future()
        result.set_result(
            [Segment(original.copy(), snapshot.hands.copy(), original.copy(), snapshot.hands.copy(), 1, "clear_thighs")]
        )
        self.executor.future = (self.executor.generation, result, snapshot, self.clock())
        self.executor.state = "PLANNING"
        self.hardware.body[22] += 0.0010066628456115723
        with patch.object(self.executor.pool, "submit", return_value=Future()):
            self.tick()
        np.testing.assert_array_equal(self.executor.arms, original)
        self.assertEqual(self.executor.planning_refreshes, 1)

    def test_normal_stalled_actuator_cannot_be_hidden_by_command_limiter(self):
        self.begin()
        snapshot = self.hardware.snapshot()
        q = snapshot.body[15:].copy()
        self.executor.queue.append(Segment(q, snapshot.hands.copy(), q + 0.1, snapshot.hands.copy(), 1))
        commands = []
        self.hardware.command = lambda arms, hands, weight: commands.append(np.array(arms).copy())
        for _ in range(650):
            self.executor.lease = self.clock()
            self.tick()
            if self.executor.error:
                break
        self.assertEqual(self.executor.error, "arrival_timeout")
        self.assertFalse(self.executor.queue)
        # Continuous time cannot hide failure to reach the final target.
        self.assertLessEqual(np.max(np.abs(np.asarray(commands) - q)), self.executor.config.tracking_error + 1e-12)
        self.assertTrue(np.isfinite(commands).all())

    def test_novus_ready_tolerance_applies_only_after_final_preparation(self):
        self.begin()
        self.executor.lease = None
        target = np.array(self.executor.config.ready_arms)
        hands = np.array(self.executor.config.ready_hands)
        self.hardware.body[15:] = target
        self.hardware.body[16] += 0.04
        self.hardware.hands[:] = hands
        self.executor.arms, self.executor.hands = target.copy(), hands.copy()
        self.executor.queue.append(Segment(target.copy(), hands.copy(), target.copy(), hands.copy(), 1, "ready_pinch"))
        self.executor.segment_elapsed = 1
        self.hardware.command = lambda arms, hands, weight: None
        self.tick()
        self.assertFalse(self.executor.status()["grasp_ready"])
        self.tick()
        self.assertTrue(self.executor.status()["grasp_ready"])
        self.hardware.body[16] = target[1] + 0.051
        self.assertFalse(self.executor.status()["grasp_ready"])

    def test_thigh_escape_stall_times_out_with_no_further_target_advance(self):
        self.begin()
        snapshot = self.hardware.snapshot()
        start = snapshot.body[15:].copy()
        target = start.copy()
        target[1] += 0.1
        segment = Segment(start, snapshot.hands.copy(), target, snapshot.hands.copy(), 2, name="clear_thighs")
        self.executor.queue.append(segment)
        self.executor.segment_elapsed = 0.3
        self.executor.arms, _ = segment.sample(0.3)
        self.hardware.body[15:] = self.executor.arms
        self.hardware.body[16] -= 0.003
        commands = []
        self.hardware.command = lambda arms, hands, weight: commands.append(np.array(arms).copy())
        for _ in range(502):
            self.executor.lease = self.clock()
            self.tick()
        self.assertEqual(self.executor.error, "thigh_escape_tracking_wait_timeout")
        self.assertFalse(self.executor.queue)
        for command in commands[:-1]:
            np.testing.assert_array_equal(command, commands[0])

    def test_canonical_fingers_round_trip_and_thumb_positions(self):
        raw = np.array([11, 22, 33, 44, 55, 66.0])
        self.assertTrue(np.allclose(from_sdk(to_sdk(raw)), raw))
        self.assertTrue(np.allclose(to_sdk(raw)[:2], raw[4:] * 100 / 255))

    def test_pause_cancels_inflight_planner_and_expired_api(self):
        token = self.begin()
        gate = threading.Event()
        started = threading.Event()

        def slow(s):
            started.set()
            gate.wait(2)
            return Planner().ready(s)

        future = self.executor.pool.submit(slow, self.hardware.snapshot())
        started.wait(1)
        self.executor.future = (self.executor.generation, future, self.hardware.snapshot(), self.clock())
        self.executor.state = "PLANNING"
        try:
            before = time.monotonic()
            self.executor.request({"op": "pause"})
            self.assertLess(time.monotonic() - before, 0.1)
            with self.assertRaisesRegex(ValueError, "expired_task"):
                self.executor.request({"op": "move", **token, "arms": REST.tolist()})
            gate.set()
            future.result(1)
            self.tick()
            self.assertFalse(self.executor.queue)
            self.assertEqual(self.executor.state, "HOLD")
        finally:
            gate.set()

    def test_release_can_open_when_arm_feedback_stale_and_will_not_reset(self):
        self.begin()
        original = self.hardware.snapshot

        def stale():
            s = original()
            s.joint_time -= 1
            return s

        self.hardware.snapshot = stale
        self.hardware.hands[:] = 40
        result = self.executor.request({"op": "reset"})
        self.assertEqual(result["error"], "joint_feedback_stale")
        self.tick()
        self.tick()
        self.assertTrue(np.all(self.hardware.hands == 255))
        self.assertIsNone(self.executor.future)
        self.assertFalse(self.executor.queue)

    def test_reset_waits_for_new_open_feedback(self):
        self.begin()
        self.hardware.hands[:] = 0
        self.executor.request({"op": "reset"})
        self.executor.tick()
        self.assertIsNone(self.executor.future)
        self.assertEqual(self.executor.state, "RELEASING")
        self.tick()
        self.assertIn(self.executor.state, ("PLANNING", "RESETTING"))

    def test_stale_feedback_latches_fault(self):
        self.begin()
        self.hardware.frozen = True
        for _ in range(30):
            self.tick()
        self.assertEqual(self.executor.state, "FAULT")
        self.assertEqual(self.executor.error, "joint_feedback_stale")

    def test_deadline_fault_and_lease_pause_invalidate_tokens(self):
        old = self.begin()
        self.tick(0.12)
        self.assertEqual(self.executor.error, "control_deadline_missed")
        details = self.executor.status()["deadline_miss"]
        self.assertAlmostEqual(details["gap_seconds"], 0.12)
        self.assertEqual(details["limit_seconds"], 0.1)
        self.assertEqual(details["lock_wait_seconds"], 0.0)
        event = next(e for e in self.executor.events if e["kind"] == "control_deadline_missed")
        self.assertAlmostEqual(event["gap_seconds"], 0.12)
        self.tick()
        self.assertEqual(self.executor.status()["deadline_miss"], details)
        with self.assertRaises(ValueError):
            self.executor.request({"op": "heartbeat", **old})
        self.begin()
        for _ in range(210):
            self.tick()
        self.assertEqual(self.executor.error, "task_lease_expired")
        self.assertEqual(self.executor.state, "HOLD")

    def test_failed_reset_path_stays_open_and_held(self):
        self.begin()

        def reject(s):
            raise ValueError("table_collision")

        self.executor.planner.ready = reject
        self.executor.request({"op": "reset"})
        for _ in range(40):
            self.tick()
            time.sleep(0.001)
        self.assertIn("table_collision", self.executor.error)
        self.assertEqual(self.executor.state, "HOLD")
        self.assertTrue(np.all(self.hardware.hands == 255))

    def test_rate_limit_and_feedback_completion(self):
        token = self.begin()
        target = REST.copy()
        target[0] += 0.025
        self.executor.request({"op": "move", **token, "arms": target.tolist()})
        for _ in range(50):
            self.tick()
            time.sleep(0.001)
        self.assertFalse(self.executor.queue)
        self.assertTrue(np.allclose(self.hardware.body[15:], target))
        values = self.hardware.writes
        for previous, current in itertools.pairwise(values):
            self.assertLessEqual(np.max(np.abs(current[1] - previous[1])), 0.3 * (current[0] - previous[0]) + 1e-8)

    def test_shutdown_handoff_never_abruptly_drops_weight(self):
        self.begin()
        self.executor.request({"op": "shutdown"})
        for _ in range(300):
            self.tick()
            time.sleep(0.001)
        self.assertEqual(self.executor.state, "STOPPED")
        self.assertFalse(self.executor.attached)
        self.assertEqual(self.hardware.writes[-1][3], 0)

    def test_pause_during_takeover_freezes_weight(self):
        self.executor.request({"op": "begin"})
        for _ in range(50):
            self.tick()
        weight = self.executor.weight
        self.assertGreater(weight, 0)
        self.assertLess(weight, 1)
        self.executor.request({"op": "pause"})
        for _ in range(100):
            self.tick()
        self.assertEqual(self.executor.weight, weight)
        self.assertEqual(self.executor.state, "HOLD")

    def test_handoff_fault_holds_weight_and_requires_explicit_retry(self):
        self.begin()
        self.executor.request({"op": "shutdown"})
        for _ in range(100):
            self.tick()
        weight = self.executor.weight
        self.assertEqual(self.executor.state, "HANDOFF")
        self.hardware.fault = "locomotion_mode_stale"
        for _ in range(20):
            self.tick()
        self.assertEqual(self.executor.weight, weight)
        self.assertEqual(self.executor.state, "FAULT")
        self.assertTrue(self.executor.shutdown_requested)
        self.assertEqual(self.executor.arm_output, "hold")
        self.assertEqual(self.hardware.writes[-1][0], self.clock())
        self.assertEqual(self.hardware.writes[-1][3], weight)
        self.hardware.fault = ""
        for _ in range(120):
            self.tick()
        self.assertEqual(self.executor.state, "FAULT")
        self.assertEqual(self.executor.weight, weight)
        self.assertIsNone(self.hardware.writes[-1][2])
        with self.assertRaisesRegex(ValueError, "handoff_pending"):
            self.executor.request({"op": "begin"})
        start = len(self.hardware.writes)
        self.executor.request({"op": "shutdown"})
        for _ in range(200):
            self.tick()
        self.assertEqual(self.executor.state, "STOPPED")
        weights = [w[3] for w in self.hardware.writes[start:]]
        self.assertTrue(all(b <= a for a, b in zip([weight] + weights, weights)))
        self.assertEqual(weights[-1], 0)

    def test_mode_stale_does_not_mask_joint_staleness_or_changed_mode(self):
        self.begin()
        original = self.hardware.snapshot
        for cause in ("joint", "mode", "temperature", "base"):
            with self.subTest(cause=cause):

                def unhealthy(cause=cause):
                    s = original()
                    s.fault = "locomotion_mode_stale"
                    if cause == "joint":
                        s.joint_time -= 1
                    elif cause == "mode":
                        s.mode = 1
                    elif cause == "temperature":
                        s.temperature = self.executor.config.motor_temperature_limit
                    else:
                        s.body[0] += 0.1
                    return s

                self.hardware.snapshot = unhealthy
                count = len(self.hardware.writes)
                self.tick()
                self.assertEqual(len(self.hardware.writes), count)
                self.assertTrue(self.executor.arm_output.startswith("blocked:"))

    def test_fault_drops_queued_motion_and_old_token_after_feedback_recovers(self):
        token = self.begin()
        q = self.hardware.body[15:].copy()
        self.executor.queue.append(Segment(q, self.hardware.hands.copy(), q + 0.01, np.zeros(12), 1))
        self.hardware.fault = "hand_worker_exited"
        self.tick()
        held = self.executor.arms.copy()
        self.hardware.fault = ""
        for _ in range(20):
            self.tick()
        self.assertEqual(self.executor.state, "FAULT")
        self.assertFalse(self.executor.queue)
        self.assertTrue(np.array_equal(self.executor.arms, held))
        with self.assertRaisesRegex(ValueError, "expired_task_generation"):
            self.executor.request({"op": "heartbeat", **token})

    def test_partial_handoff_pose_change_requires_new_explicit_takeover(self):
        self.begin()
        self.executor.request({"op": "shutdown"})
        for _ in range(100):
            self.tick()
        self.hardware.fault = "locomotion_mode_stale"
        self.tick()
        self.hardware.body[15] += 0.05
        self.hardware.fault = ""
        with self.assertRaisesRegex(ValueError, "handoff_pose_changed"):
            self.executor.request({"op": "shutdown"})
        self.assertEqual(self.executor.state, "FAULT")

    def test_joint_recovery_holds_new_measured_pose_without_weight_ramp(self):
        self.begin()
        self.hardware.frozen = True
        for _ in range(30):
            self.tick()
        weight = self.executor.weight
        self.hardware.body[15] += 0.2
        self.hardware.frozen = False
        measured = self.hardware.body[15:].copy()
        self.tick()
        self.assertEqual(self.executor.state, "FAULT")
        self.assertEqual(self.executor.weight, weight)
        self.assertTrue(np.array_equal(self.hardware.writes[-1][1], measured))

    def test_failed_handoff_write_retains_last_published_weight(self):
        self.begin()
        self.executor.request({"op": "shutdown"})
        for _ in range(100):
            self.tick()
        weight = self.executor.weight

        def fail(*args):
            raise RuntimeError("dds_write_failed")

        self.hardware.command = fail
        with self.assertRaisesRegex(RuntimeError, "dds_write_failed"):
            self.tick()
        self.assertEqual(self.executor.weight, weight)

    def test_read_only_hardware_rejects_all_actuator_entrypoints(self):
        from astrabot.robot.hardware import RealHardware

        hardware = object.__new__(RealHardware)
        hardware.read_only = True
        with self.assertRaisesRegex(ValueError, "read_only"):
            hardware.command(REST, np.zeros(12), 1)
        with self.assertRaisesRegex(ValueError, "read_only"):
            hardware.release_hands()


if __name__ == "__main__":
    unittest.main()
