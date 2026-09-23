"""Novus preparation semantics, with geometry admission kept separate from time lag."""

import unittest
from unittest.mock import Mock

import numpy as np

from astrabot.robot.config import OPEN_RAISE, RAISE, Config
from astrabot.robot.executor import Executor
from astrabot.robot.hardware import MockHardware
from astrabot.robot.tracking import novus_command_target, path_progress_interval
from astrabot.robot.trajectory import Planner, Segment


class NovusCommandTests(unittest.TestCase):
    def test_feedback_clip_matches_novus_cpp_component_bound_then_vector_scale(self):
        feedback = np.linspace(-1, 1, 14)
        delta = np.array([0.1, -0.2, 0.03] + [0.0] * 11)
        command = novus_command_target(feedback + delta, feedback)
        np.testing.assert_allclose(command - feedback, [0.08, -0.08, 0.03] + [0.0] * 11)
        np.testing.assert_allclose(novus_command_target(feedback + 0.07, feedback), feedback + 0.07)

    def test_bad_input_cannot_be_clipped_into_a_valid_target(self):
        for q, lead in ((np.full(14, np.nan), 0.08), (np.zeros(13), 0.08), (np.zeros(14), 0.09)):
            with self.assertRaises(ValueError):
                novus_command_target(q, np.zeros(14), lead)

    def test_path_envelope_allows_common_lag_but_not_independent_joint_progress(self):
        start, target = np.zeros(14), np.zeros(14)
        target[:2] = [1, -1]
        self.assertIsNotNone(path_progress_interval(target * 0.2, start, target, 0.005))
        wrong = target * 0.2
        wrong[1] = -0.3
        self.assertIsNone(path_progress_interval(wrong, start, target, 0.005))
        wrong = target * 0.2
        wrong[8] = 0.006
        self.assertIsNone(path_progress_interval(wrong, start, target, 0.005))
        self.assertIsNone(path_progress_interval(target * 1.02, start, target, 0.005))

    def test_raise_envelope_uses_actual_shoulder_leading_curve(self):
        segment = Segment(np.zeros(14), np.zeros(12), np.linspace(-1, 1, 14), np.zeros(12), 2, profile="raise")
        for elapsed in np.linspace(0, 2, 41):
            q, _ = segment.sample(elapsed)
            interval = path_progress_interval(q, segment.arms, segment.target, 0.001, "raise")
            self.assertIsNotNone(interval)
            self.assertLessEqual(interval[0], elapsed / 2 + 1e-8)
            self.assertGreaterEqual(interval[1], elapsed / 2 - 1e-8)
        self.assertIsNone(path_progress_interval(segment.target * 0.5, segment.arms, segment.target, 0.001, "raise"))

    def test_open_raise_envelope_includes_early_shoulder_roll_and_its_plateau(self):
        segment = Segment(np.zeros(14), np.full(12, 254), OPEN_RAISE, np.full(12, 254), 2, profile="raise_open")
        for elapsed in np.linspace(0, 2, 81):
            q, hands = segment.sample(elapsed)
            interval = path_progress_interval(q, segment.arms, segment.target, 0.001, segment.profile)
            self.assertIsNotNone(interval)
            self.assertLessEqual(interval[0], elapsed / 2 + 1e-8)
            self.assertGreaterEqual(interval[1], elapsed / 2 - 1e-8)
            np.testing.assert_array_equal(hands, segment.hands)

    def test_ready_plan_does_not_insert_an_outward_escape(self):
        model = Mock()
        config = Config(scene_measured=True)
        hardware = MockHardware()
        segments = Planner(model, config).ready(hardware.snapshot())
        self.assertEqual([segment.name for segment in segments], ["raise", "prepare", "ready_pinch"])
        for segment in segments[:-1]:
            np.testing.assert_array_equal(segment.hands, hardware.hands)
            np.testing.assert_array_equal(segment.hand_target, hardware.hands)
        np.testing.assert_array_equal(segments[-1].arms, config.ready_arms)
        np.testing.assert_array_equal(segments[-1].target, config.ready_arms)
        model.check_thigh_escape.assert_not_called()

    def test_all_entry_branches_preserve_every_measured_finger_until_final_ready(self):
        config = Config(scene_measured=True)
        for start in (RAISE, np.array(config.ready_arms)):
            hardware = MockHardware()
            hardware.body[15:] = start
            hardware.hands = np.arange(12) * 17.0 + 10
            segments = Planner(Mock(), config).ready(hardware.snapshot())
            for segment in segments[:-1]:
                for elapsed in (0, segment.duration / 2, segment.duration):
                    np.testing.assert_array_equal(segment.sample(elapsed)[1], hardware.hands)
            np.testing.assert_array_equal(segments[-1].arms, config.ready_arms)
            np.testing.assert_array_equal(segments[-1].target, config.ready_arms)
            np.testing.assert_array_equal(segments[-1].hand_target, config.ready_hands)

    def test_threefold_speed_preserves_paths_and_bounds_peak_velocity(self):
        slow = Config(joint_speed=0.15, cartesian_speed=0.01, hand_speed=40)
        fast = Config(joint_speed=0.45, cartesian_speed=0.03, hand_speed=120)
        hardware = MockHardware()
        for profile in ("uniform", "raise", "raise_open"):
            segments = [
                Planner(Mock(), cfg).segment(
                    hardware.body, hardware.hands, OPEN_RAISE, hardware.hands - 20, "raise", profile
                )
                for cfg in (slow, fast)
            ]
            self.assertAlmostEqual(segments[0].duration / segments[1].duration, 3)
            for fraction in np.linspace(0, 1, 21):
                for old, new in zip(*(s.sample(fraction * s.duration) for s in segments)):
                    np.testing.assert_allclose(old, new, atol=1e-12)
            samples = [segments[1].sample(t) for t in np.linspace(0, segments[1].duration, 2001)]
            dt = segments[1].duration / 2000
            self.assertLessEqual(np.max(np.abs(np.diff([s[0] for s in samples], axis=0))) / dt, fast.joint_speed)
            self.assertLessEqual(np.max(np.abs(np.diff([s[1] for s in samples], axis=0))) / dt, fast.hand_speed)
        for values in ({"joint_speed": 0.4501}, {"cartesian_speed": 0.0301}):
            with self.assertRaises(ValueError):
                Config(**values)

    def test_open_raise_profile_respects_speed_for_each_joint(self):
        config = Config()
        hardware = MockHardware()
        segment = Planner(Mock(), config).segment(
            hardware.body, hardware.hands, OPEN_RAISE, hardware.hands, "raise", "raise_open"
        )
        samples = np.array([segment.sample(t)[0] for t in np.linspace(0, segment.duration, 2001)])
        self.assertLessEqual(np.max(np.abs(np.diff(samples, axis=0))) / (segment.duration / 2000), config.joint_speed)


class PreparationExecutionTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.hardware = MockHardware(lambda: self.now)
        self.executor = Executor(self.hardware, None, Config(), lambda: self.now)
        self.addCleanup(self.executor.close)
        self.executor.request({"op": "begin"})
        self.executor.lease = None
        self.executor.weight = 1
        self.executor.takeover_active = False
        self.commands = []
        self.hand_commands = []

        def capture(arms, hands, weight):
            self.commands.append(np.array(arms))
            self.hand_commands.append(None if hands is None else np.array(hands))

        self.hardware.command = capture

    def tick(self):
        self.now += 0.01
        self.executor.tick()

    def install(self):
        q, hands = self.hardware.body[15:].copy(), self.hardware.hands.copy()
        target = q.copy()
        target[0] += 0.3
        self.executor.queue.append(Segment(q, hands, target, hands, 1, "prepare"))

    def install_finger_stage(self):
        self.install()
        target = self.executor.queue[-1].target.copy()
        hands = self.hardware.hands.copy()
        self.executor.queue.append(Segment(target, hands, target, np.arange(12) * 10.0, 1, "ready_pinch"))

    def assert_no_finger_motion(self):
        for hands in self.hand_commands:
            if hands is not None:
                np.testing.assert_array_equal(hands, np.full(12, 255.0))

    def test_fingers_wait_for_fresh_final_arm_arrival_not_just_elapsed_time(self):
        self.install_finger_stage()
        self.executor.segment_elapsed = 1
        for _ in range(10):
            self.tick()
        self.assertEqual(len(self.executor.queue), 2)
        self.assert_no_finger_motion()
        self.hardware.body[15:] = self.executor.queue[0].target
        self.hardware.frozen = True
        self.hardware.stamp = self.executor.arrival_started
        self.tick()
        self.assertEqual(len(self.executor.queue), 2)
        self.assert_no_finger_motion()
        self.hardware.frozen = False
        self.tick()
        self.assertEqual(self.executor.queue[0].name, "ready_pinch")
        self.assert_no_finger_motion()
        self.tick()
        self.assertTrue(np.all(self.hand_commands[-1] < 255))

    def test_timeout_clears_pending_finger_stage_without_closing_any_finger(self):
        self.install_finger_stage()
        self.executor.segment_elapsed = 1
        for _ in range(305):
            self.tick()
        self.assertEqual(self.executor.error, "arrival_timeout")
        self.assertFalse(self.executor.queue)
        self.assert_no_finger_motion()

    def test_cancel_at_arm_arrival_cannot_run_queued_finger_stage(self):
        self.install_finger_stage()
        self.executor.segment_elapsed = 1
        self.hardware.body[15:] = self.executor.queue[0].target
        self.tick()
        self.tick()
        self.assertEqual(self.executor.queue[0].name, "ready_pinch")
        self.executor.request({"op": "pause"})
        self.tick()
        self.assertFalse(self.executor.queue)
        self.assert_no_finger_motion()

    def test_arm_drift_before_finger_stage_prevents_hand_command(self):
        self.install_finger_stage()
        self.executor.segment_elapsed = 1
        self.hardware.body[15:] = self.executor.queue[0].target
        self.tick()
        self.tick()
        self.hardware.body[15] -= 0.051
        self.tick()
        self.assertEqual(self.executor.error, "prepare_arm_left_ready_pose")
        self.assertIsNone(self.hand_commands[-1])
        self.assert_no_finger_motion()

    def test_non_thumb_finger_drift_during_raise_stops(self):
        self.install_finger_stage()
        self.hardware.hands[2] -= 2
        self.tick()
        self.assertEqual(self.executor.error, "prepare_fingers_moved_before_ready")
        self.assertIsNone(self.hand_commands[-1])

    def test_stalled_on_path_keeps_time_and_clips_then_fails_endpoint(self):
        self.install()
        for _ in range(90):
            self.tick()
        self.assertEqual(self.executor.error, "")
        self.assertGreater(self.executor.segment_elapsed, 0.89)
        self.assertIsNone(self.executor.tracking_wait_started)
        self.assertAlmostEqual(self.commands[-1][0] - self.hardware.body[15], 0.08)
        for _ in range(325):
            self.tick()
            if self.executor.error:
                break
        self.assertEqual(self.executor.error, "arrival_timeout")
        self.assertFalse(self.executor.queue)

    def test_off_path_joint_motion_still_stops(self):
        self.install()
        self.hardware.body[16] += 0.081
        self.tick()
        self.assertEqual(self.executor.error, "prepare_feedback_outside_path_envelope")

    def test_task_execution_still_rejects_geometric_path_deviation(self):
        self.install()
        self.executor.lease = self.now
        self.hardware.body[16] += 0.081
        self.tick()
        self.assertEqual(self.executor.error, "task_feedback_outside_path_envelope")

    def test_task_lag_uses_continuous_reference_until_endpoint_timeout(self):
        self.install()
        self.executor.lease = self.now
        self.executor.config.settle_timeout = 0.2
        for _ in range(90):
            self.executor.lease = self.now
            self.tick()
        self.assertEqual(self.executor.error, "")
        self.assertGreater(self.executor.segment_elapsed, 0.89)
        self.assertIsNone(self.executor.tracking_wait_started)
        self.assertAlmostEqual(self.commands[-1][0] - self.hardware.body[15], 0.08)
        self.assertEqual(self.executor.cycle_reference["tracking_mode"], "novus_task")
        for _ in range(35):
            self.executor.lease = self.now
            self.tick()
        self.assertEqual(self.executor.error, "arrival_timeout")
        self.assertFalse(self.executor.queue)
