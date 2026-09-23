"""Feedback/thermal commissioning regressions, with no physical SDK writes."""

import json
import time
import unittest
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from astrabot.robot.config import Config
from astrabot.robot.executor import Executor
from astrabot.robot.hardware import MockHardware
from astrabot.robot.model import RobotModel
from astrabot.robot.trajectory import Planner, Segment
from astrabot.robot.trial_runtime import (
    SingleTrial,
    contact_is_fresh,
    pinch_pose,
    pinch_step,
    trial_joint_step,
)


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.mode = 4
        self.temperature = 30.0
        self.tactile = [0.0] * 10
        self.tactile_offset = 0.0
        self.hardware = MockHardware(lambda: self.now)
        original = self.hardware.snapshot

        def snapshot():
            s = original()
            s.mode = self.mode
            s.temperature = self.temperature
            s.tactile = self.tactile.copy()
            s.tactile_time -= self.tactile_offset
            return s

        self.hardware.snapshot = snapshot
        self.executor = Executor(self.hardware, None, Config(), lambda: self.now)
        self.addCleanup(self.executor.close)
        self.executor.request({"op": "begin"})
        self.executor.weight = 1.0

    def tick(self):
        self.now += 0.01
        self.executor.tick()

    def guarded(self, holding=False):
        s = self.hardware.snapshot()
        seg = Segment(
            s.body[15:].copy(),
            s.hands.copy(),
            s.body[15:] + 0.01,
            s.hands - 1,
            1,
            contact_guard=not holding,
            require_contact=holding,
        )
        self.executor.queue.append(seg)
        return seg

    def test_contact_stops_before_any_further_closure(self):
        self.guarded()
        self.tactile[:2] = [10, 11]
        old = self.hardware.hands.copy()
        token = self.executor.generation
        self.tick()
        self.assertTrue(self.executor.contact_detected)
        self.assertFalse(self.executor.queue)
        self.assertEqual(self.executor.generation, token)
        np.testing.assert_array_equal(self.hardware.hands, old)
        self.assertIsNone(self.hardware.writes[-1][2])

    def test_stale_or_one_sided_contact_never_completes_grasp(self):
        self.guarded()
        self.tactile[:2] = [10, 0]
        self.tick()
        self.assertFalse(self.executor.contact_detected)
        self.tactile_offset = 1
        self.tick()
        self.assertEqual(self.executor.error, "pinch_tactile_stale_or_invalid")
        self.assertFalse(self.executor.queue)

    def test_lift_lost_contact_stops_and_invalidates_task(self):
        self.guarded(holding=True)
        token = self.executor.generation
        self.tick()
        self.assertEqual(self.executor.error, "trial_lost_contact")
        self.assertGreater(self.executor.generation, token)
        self.assertFalse(self.executor.queue)

    def test_heat_latches_until_verified_damping_and_zero_frames(self):
        self.temperature = self.executor.config.motor_temperature_limit
        self.tick()
        self.assertTrue(self.executor.thermal_handoff_required)
        self.assertIn("retained SDK hold", self.executor.status()["operator_action"])
        count = len(self.hardware.writes)
        self.temperature = 30
        for _ in range(10):
            self.tick()
        self.assertEqual(len(self.hardware.writes), count)
        with self.assertRaisesRegex(ValueError, "thermal_handoff_required"):
            self.executor.request({"op": "begin"})
        self.mode = 0
        self.tick()
        self.assertTrue(self.executor.thermal_handoff_required)
        self.mode = 1
        for _ in range(55):
            self.tick()
        self.assertFalse(self.executor.attached)
        self.assertEqual(self.executor.state, "DISARMED")
        self.assertFalse(self.executor.thermal_handoff_required)
        self.assertEqual(self.executor.weight, 0)
        self.assertTrue(all(h is None and w == 0 for _, _, h, w in self.hardware.writes[count:]))

    def test_failed_automatic_handoff_requires_explicit_retry(self):
        self.temperature = self.executor.config.motor_temperature_limit
        self.tick()
        self.mode = 1
        with patch.object(self.hardware, "clear_arm_control", side_effect=RuntimeError("write_failed")) as write:
            self.tick()
            self.tick()
            self.assertEqual(write.call_count, 1)
        self.assertTrue(self.executor.attached)
        self.assertTrue(self.executor.thermal_handoff_required)
        self.executor.request({"op": "disarm"})
        for _ in range(55):
            self.tick()
        self.assertFalse(self.executor.attached)

    def test_default_motor_120_keeps_shell_and_hand_90(self):
        config = self.executor.config
        self.assertEqual(config.motor_temperature_limit, 120)
        self.assertEqual(config.shell_temperature_limit, 90)
        self.assertEqual(config.hand_temperature_limit, 90)
        sample = self.hardware.snapshot()
        sample.temperature, sample.shell_temperature, sample.hand_temperature = 119.9, 69, 45
        self.assertEqual(self.executor.health(sample), "")
        for field, limit in (("temperature", 120), ("shell_temperature", 90), ("hand_temperature", 90)):
            candidate = SimpleNamespace(**vars(sample))
            setattr(candidate, field, limit)
            self.assertEqual(self.executor.health(candidate), "motor_temperature_limit")

    def test_motor_limit_change_preserves_shell_and_hand_limits(self):
        self.executor.config.shell_temperature_limit = 70
        self.executor.config.hand_temperature_limit = 70
        self.executor.config.motor_temperature_limit = 90
        self.temperature = 89
        s = self.hardware.snapshot()
        s.shell_temperature = 59
        s.hand_temperature = 50
        self.assertEqual(self.executor.health(s), "")
        for channel, value in (("temperature", 90), ("shell_temperature", 70), ("hand_temperature", 70)):
            with self.subTest(channel=channel):
                sample = self.hardware.snapshot()
                setattr(sample, channel, value)
                self.assertEqual(self.executor.health(sample), "motor_temperature_limit")

    def test_shell_heat_still_latches_when_motor_limit_is_raised(self):
        self.executor.config.motor_temperature_limit = 90
        self.executor.config.shell_temperature_limit = 70
        sample = self.hardware.snapshot()
        sample.temperature = sample.shell_temperature = 70
        with patch.object(self.hardware, "snapshot", return_value=sample):
            self.tick()
        self.assertTrue(self.executor.thermal_handoff_required)
        self.assertEqual(self.executor.state, "FAULT")

    def test_default_limits_reject_each_channel_at_its_limit(self):
        sample = self.hardware.snapshot()
        sample.temperature = sample.shell_temperature = sample.hand_temperature = 89.9
        self.assertEqual(self.executor.health(sample), "")
        for channel, limit in (("temperature", 120), ("shell_temperature", 90), ("hand_temperature", 90)):
            with self.subTest(channel=channel):
                setattr(sample, channel, limit)
                self.assertEqual(self.executor.health(sample), "motor_temperature_limit")
                setattr(sample, channel, 89.9)


class PinchPlanningTests(unittest.TestCase):
    def setUp(self):
        source = json.loads((Path(__file__).parent / "fixtures/robot_aligned_trial.json").read_text())
        self.config = Config(scene_measured=True, table_min=source["table_min"], table_max=source["table_max"])
        self.model = RobotModel(self.config)
        self.planner = Planner(self.model, self.config)
        self.snapshot = SimpleNamespace(body=np.array(source["body"]), hands=np.array(source["hands"]))

    def test_full_carry_path_checks_all_segments_before_execution(self):
        center, rotation = pinch_pose(self.model, self.snapshot.body, self.snapshot.hands)
        segments = self.planner.move(
            self.snapshot,
            {
                "action": "carry_path",
                "holding": True,
                "goal_center": center + [0, 0, 0.12],
                "rotation": rotation,
                "hands": self.snapshot.hands.copy(),
            },
        )
        self.assertGreater(len(segments), 1)
        for before, after in pairwise(segments):
            np.testing.assert_array_equal(before.target, after.arms)
        for segment in segments:
            self.assertTrue(segment.continuous_carry)
            np.testing.assert_array_equal(segment.hand_target, self.snapshot.hands)
        self.assertEqual(segments[0].carry_blend, "start")
        self.assertEqual(segments[-1].carry_blend, "end")
        for segment in segments[1:-1]:
            self.assertEqual(segment.carry_blend, "cruise")
            q1, _ = segment.sample(segment.duration * 0.98)
            q2, _ = segment.sample(segment.duration * 0.99)
            np.testing.assert_allclose(q2 - q1, (segment.target - segment.arms) * 0.01, atol=1e-12)
        final = self.snapshot.body.copy()
        final[15:] = segments[-1].target
        measured, _ = pinch_pose(self.model, final, self.snapshot.hands)
        self.assertLess(np.linalg.norm(measured - center - [0, 0, 0.12]), 0.001)

    def test_carry_rejects_distance_beyond_120mm_plus_start_margin(self):
        center, _ = pinch_pose(self.model, self.snapshot.body, self.snapshot.hands)
        with self.assertRaisesRegex(ValueError, "carry_path_exceeds_130mm"):
            self.planner.move(
                self.snapshot,
                {
                    "action": "carry_path",
                    "holding": True,
                    "goal_center": center + [0, 0, 0.131],
                },
            )

    def test_joint_path_certifies_bounded_pieces_before_single_ramp(self):
        from astrabot.robot.trial_runtime import trial_joint_path

        target = self.snapshot.body[15:].copy()
        target[0] += 0.29
        pieces = []

        def piece(planner, snapshot, request):
            end = np.asarray(request["arms"])
            self.assertLessEqual(np.max(np.abs(end - snapshot.body[15:])), 0.1 + 1e-9)
            pieces.append(end.copy())
            return Segment(snapshot.body[15:].copy(), snapshot.hands.copy(), end, np.asarray(request["hands"]), 1.0)

        request = {"arms": target, "hands": self.snapshot.hands, "duration": 3.0}
        with patch("astrabot.robot.trial_runtime.trial_joint_step", side_effect=piece):
            result = trial_joint_path(self.planner, self.snapshot, request)
        self.assertEqual(len(pieces), 3)
        self.assertAlmostEqual(result.duration, 3.0)
        # No artificial velocity zero at the old one-third boundary.
        a, _ = result.sample(1.0 - 0.001)
        b, _ = result.sample(1.0 + 0.001)
        self.assertGreater(b[0] - a[0], 0.0001)

    def test_held_right_hand_survives_feedback_offset_without_changing_target(self):
        hands = self.snapshot.hands.copy()
        hands[-1] += 4
        request = {"arms": self.snapshot.body[15:].copy(), "hands": hands, "preserve_right": True}
        segment = trial_joint_step(self.planner, self.snapshot, request)
        np.testing.assert_array_equal(segment.hand_target[6:], hands[6:])
        with self.assertRaisesRegex(ValueError, "preserve_right_arm_and_hand"):
            trial_joint_step(self.planner, self.snapshot, {**request, "preserve_right": False})
        request["hands"][-1] += 2
        with self.assertRaisesRegex(ValueError, "preserve_right_arm_and_hand"):
            trial_joint_step(self.planner, self.snapshot, request)

    def test_carry_and_release_preserve_retained_right_hand(self):
        center, rotation = pinch_pose(self.model, self.snapshot.body, self.snapshot.hands)
        hands = self.snapshot.hands.copy()
        hands[-1] += 4
        for mode in ("holding", "opening"):
            with self.subTest(mode=mode):
                segment = pinch_step(
                    self.planner,
                    self.snapshot,
                    {
                        "center": center,
                        "rotation": rotation,
                        "hands": hands,
                        "preserve_right": True,
                        "right_arms": self.snapshot.body[22:],
                        mode: True,
                    },
                )
                np.testing.assert_array_equal(segment.hand_target[6:], hands[6:])
                np.testing.assert_array_equal(segment.target[7:], self.snapshot.body[22:])

    def test_carry_rebases_stale_microstep_from_latest_feedback(self):
        center, rotation = pinch_pose(self.model, self.snapshot.body, self.snapshot.hands)
        request = {
            "center": center + [0, 0, 0.006085],
            "goal_center": center + [0, 0, 0.07],
            "rotation": rotation,
            "hands": self.snapshot.hands.copy(),
            "holding": True,
        }
        from dataclasses import replace

        self.config.motion_checks = replace(self.config.motion_checks, pinch_substep_m=0.005)
        old_request = {key: value for key, value in request.items() if key != "goal_center"}
        with self.assertRaisesRegex(ValueError, "pinch_step_too_large"):
            pinch_step(self.planner, self.snapshot, old_request)
        segment = pinch_step(self.planner, self.snapshot, request)
        np.testing.assert_allclose(segment.pinch_center, center + [0, 0, 0.005], atol=1e-12)
        final = pinch_step(self.planner, self.snapshot, {**request, "goal_center": center + [0, 0, 0.0005]})
        np.testing.assert_allclose(final.pinch_center, center + [0, 0, 0.0005], atol=1e-12)
        with self.assertRaisesRegex(ValueError, "requires_carry_motion"):
            pinch_step(self.planner, self.snapshot, {**request, "holding": False})

    def test_compensated_step_preserves_other_arm_and_unmeasured_flag(self):
        center, rotation = pinch_pose(self.model, self.snapshot.body, self.snapshot.hands)
        hand = self.snapshot.hands.copy()
        hand[3] += 1
        seg = pinch_step(
            self.planner, self.snapshot, {"center": center, "rotation": rotation, "hands": hand, "closing": True}
        )
        q = self.snapshot.body.copy()
        q[15:] = seg.target
        actual, _ = pinch_pose(self.model, q, seg.hand_target)
        self.assertLess(np.linalg.norm(actual - center), 0.003)
        np.testing.assert_array_equal(q[22:], self.snapshot.body[22:])
        self.assertFalse(self.config.tcp_measured)
        self.assertTrue(seg.contact_guard)

    def test_closure_rebases_stale_target_without_enlarging_step(self):
        center, rotation = pinch_pose(self.model, self.snapshot.body, self.snapshot.hands)
        for drift in (-3.0, 3.0):
            hands = self.snapshot.hands.copy()
            hands[4] += drift
            request = {"center": center, "rotation": rotation, "hands": hands, "closing": True}
            segment = pinch_step(self.planner, self.snapshot, request)
            expected = self.snapshot.hands.copy()
            expected[4] += np.clip(drift, -2, 2)
            np.testing.assert_array_equal(segment.hand_target, expected)
            np.testing.assert_array_equal(segment.hand_target[6:], self.snapshot.hands[6:])
            with self.assertRaisesRegex(ValueError, "pinch_finger_increment_exceeds_2"):
                pinch_step(self.planner, self.snapshot, {**request, "closing": False})
            with self.assertRaisesRegex(ValueError, "closing_conflicts"):
                pinch_step(self.planner, self.snapshot, {**request, "holding": True})

    def test_open_and_close_position_steps_rebase_from_executor_feedback(self):
        center, rotation = pinch_pose(self.model, self.snapshot.body, self.snapshot.hands)
        goal = center + [0, 0, 0.006413]
        for mode in ("opening", "closing"):
            request = {"center": goal, "rotation": rotation, "hands": self.snapshot.hands.copy(), mode: True}
            with self.assertRaisesRegex(ValueError, "pinch_step_too_large"):
                pinch_step(self.planner, self.snapshot, request)
            solve = self.model.solve
            with patch.object(self.model, "solve", wraps=solve) as checked:
                pinch_step(self.planner, self.snapshot, {**request, "goal_center": goal})
            target = checked.call_args.args[2]
            self.assertAlmostEqual(np.linalg.norm(target - center), self.config.motion_checks.pinch_substep_m)
            np.testing.assert_allclose(target[:2], center[:2])

    def test_release_rebases_stale_fingers_and_keeps_two_raw_bound(self):
        center, rotation = pinch_pose(self.model, self.snapshot.body, self.snapshot.hands)
        stale = self.snapshot.hands.copy()
        stale[3] += 3
        request = {"center": center, "rotation": rotation, "hands": stale, "opening": True}
        seg = pinch_step(self.planner, self.snapshot, request)
        expected = self.snapshot.hands.copy()
        expected[:6] += np.clip(self.config.trial_hand("open") - expected[:6], -2, 2)
        np.testing.assert_array_equal(seg.hand_target, expected)
        self.assertLessEqual(np.max(abs(seg.hand_target - self.snapshot.hands)), 2 + 1e-10)
        with self.assertRaisesRegex(ValueError, "opening_conflicts"):
            pinch_step(self.planner, self.snapshot, {**request, "holding": True})
        for scale in (2, 4):
            with self.subTest(scale=scale):
                self.config.trial_speed_scale = scale
                self.config.__post_init__()
                fast = pinch_step(self.planner, self.snapshot, request)
                np.testing.assert_array_equal(fast.target, seg.target)
                np.testing.assert_array_equal(fast.hand_target, seg.hand_target)
                self.assertAlmostEqual(fast.duration, max(0.02, seg.duration / scale))

    def test_trial_speed_scale_bounds(self):
        for scale in (0.9, 4.01, float("nan"), float("inf"), True):
            with self.subTest(scale=scale):
                self.config.trial_speed_scale = scale
                with self.assertRaisesRegex(ValueError, "trial_speed_scale_outside_1_to_4"):
                    self.config.__post_init__()

    def test_oversized_steps_rejected(self):
        center, rotation = pinch_pose(self.model, self.snapshot.body, self.snapshot.hands)
        with self.assertRaisesRegex(ValueError, "pinch_step_too_large"):
            pinch_step(self.planner, self.snapshot, {"center": center + [0.01, 0, 0], "rotation": rotation})
        hand = self.snapshot.hands.copy()
        hand[3] += 3
        with self.assertRaisesRegex(ValueError, "increment_exceeds_2"):
            pinch_step(self.planner, self.snapshot, {"center": center, "hands": hand})

    def test_invalid_contact_never_passes(self):
        for values, age in (([10, float("nan")], 0), ([10, 10], -1), ([10, 10], 1), ([10], 0)):
            self.assertFalse(contact_is_fresh({"tactile": values, "tactile_age": age}, self.config))


class TrialGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_release_within_eight_mm_advances_hand_without_chasing_old_goal(self):
        state = {"body": [0.0] * 29, "hands": [100.0] * 12}
        for distance, opening in ((0.006021, True), (0.008, True), (0.0081, True), (0.006021, False)):
            task = SimpleNamespace(status=lambda: state, move=AsyncMock(return_value={}))
            trial = SingleTrial(task, None, Config(), AsyncMock())
            with patch("astrabot.robot.trial_runtime.pinch_pose", return_value=(np.zeros(3), np.eye(3))):
                if opening and distance <= 0.008:
                    await trial.move_center([0, 0, -distance], opening=True)
                    task.move.assert_awaited_once()
                    np.testing.assert_array_equal(task.move.call_args.kwargs["goal_center"], [0, 0, 0])
                    self.assertTrue(trial.events[-1]["release_in_position_tolerance"])
                else:
                    with self.assertRaisesRegex(ValueError, "pinch_progress_limit"):
                        await trial.move_center([0, 0, -distance], opening=opening)
                    self.assertTrue(all(not e["release_in_position_tolerance"] for e in trial.events))

    async def test_cube_observation_alone_cannot_verify_hover_fingertips(self):
        async def observe(stage):
            return {"center": [0, 0, 0], "acquired_monotonic": time.monotonic(), "scene_clear": True}

        trial = SingleTrial(SimpleNamespace(status=dict), None, Config(), observe)
        with self.assertRaisesRegex(ValueError, "hover_pinch_alignment_not_verified"):
            await trial.verified_center("before_descent", [0, 0, 0])

    async def test_explicit_fresh_hover_verification_accepted(self):
        async def observe(stage):
            return {
                "center": [0, 0, 0],
                "acquired_monotonic": time.monotonic(),
                "scene_clear": True,
                "pinch_alignment_verified": True,
            }

        trial = SingleTrial(SimpleNamespace(status=dict), None, Config(), observe)
        await trial.verified_center("before_descent", [0, 0, 0])

    async def test_incomplete_scene_rejected_before_motion(self):
        task = SimpleNamespace(move=AsyncMock())
        trial = SingleTrial(task, None, Config(), AsyncMock())
        with self.assertRaisesRegex(ValueError, "complete_current_scene"):
            await trial.run({"geometry_passed": True, "scene_complete": False})
        task.move.assert_not_awaited()

    async def test_saved_large_joint_phase_is_dispatched_once(self):
        class Task:
            def __init__(self):
                self.body = np.zeros(29)
                self.hands = np.full(12, 255.0)
                self.moves = []

            def status(self):
                return {"body": self.body.tolist(), "hands": self.hands.tolist()}

            async def move(self, **request):
                target = np.asarray(request["arms"])
                assert request["action"] == "trial_joint_path"
                self.body[15:] = target
                self.hands[:] = request["hands"]
                self.moves.append(request)

        task = Task()
        trial = SingleTrial(task, None, Config(), AsyncMock())
        target = np.zeros(14)
        target[0] = 0.29
        await trial.joint_phase({"arms": target, "hands": task.hands.copy(), "duration": 3.0}, [])
        self.assertEqual(len(task.moves), 1)
        self.assertAlmostEqual(task.body[15], 0.29)

    async def test_saved_waypoints_are_sent_as_one_phase(self):
        state = {"body": [0.0] * 29, "hands": [255.0] * 12}
        task = SimpleNamespace(status=lambda: state, move=AsyncMock())
        trial = SingleTrial(task, None, Config(), AsyncMock())
        points = [{"arms": [x] + [0.0] * 13, "hands": state["hands"], "duration": 0.1} for x in (0.03, 0.06, 0.09)]
        await trial.joint_phase(points, [])
        task.move.assert_awaited_once()
        request = task.move.call_args.kwargs
        self.assertEqual(request["action"], "trial_joint_sequence")
        self.assertEqual(len(request["waypoints"]), 3)

    async def test_joint_phase_keeps_right_command_instead_of_relatching_feedback(self):
        state = {
            "body": np.zeros(29).tolist(),
            "hands": [60.0] * 12,
            "arms_command": np.zeros(14).tolist(),
            "hands_command": [60.0] * 11 + [66.0],
        }
        state["hands"][-1] = 62.0
        task = SimpleNamespace(status=lambda: state, move=AsyncMock())
        trial = SingleTrial(task, None, Config(), AsyncMock())
        phase = {"arms": np.zeros(14), "hands": [60.0] * 12, "duration": 0.1}
        for measured in (62.0, 63.0, 64.0):
            state["hands"][-1] = measured
            await trial.joint_phase(phase, [])
        for call in task.move.await_args_list:
            self.assertEqual(call.kwargs["hands"][-1], 66.0)
            self.assertIs(call.kwargs["preserve_right"], True)

    async def test_stale_or_wrong_visual_observation_rejected(self):
        task = SimpleNamespace(status=dict)
        observe = AsyncMock(return_value={"center": [0, 0, 0], "acquired_monotonic": 0, "scene_clear": True})
        trial = SingleTrial(task, None, Config(), observe)
        with self.assertRaisesRegex(ValueError, "not_fresh"):
            await trial.verified_center("lift", [0, 0, 0])


class CarrySubdivisionTests(unittest.TestCase):
    def plan_with_residual(self, residual=0.0012, fail=None):
        from astrabot.robot.trial_runtime import carry_path

        config = Config(motion_checks={"pinch_substep_m": 0.005})
        snapshot = SimpleNamespace(body=np.zeros(29), hands=np.full(12, 128.0))
        planner = SimpleNamespace(model=None, config=config)
        targets = []

        def step(planner, current, request, **kwargs):
            center = np.array(request["center"])
            distance = np.linalg.norm(center - current.body[15:18])
            if distance > config.motion_checks.pinch_step_m:
                raise ValueError("pinch_step_too_large")
            if fail:
                raise ValueError(fail)
            targets.append(center.copy())
            target = current.body[15:].copy()
            target[:3] = center - [residual, 0, 0] if residual is not None else 0
            return Segment(current.body[15:].copy(), current.hands.copy(), target, current.hands.copy(), 0.1)

        with (
            patch("astrabot.robot.trial_runtime.pinch_pose", side_effect=lambda m, b, h: (b[15:18].copy(), np.eye(3))),
            patch("astrabot.robot.trial_runtime.pinch_step", side_effect=step) as checked,
        ):
            request = {"action": "transfer_path", "holding": True, "goal_center": [0.02, 0, 0]}
            if residual is None or fail:
                with self.assertRaisesRegex(ValueError, fail or "pinch_step_too_large"):
                    carry_path(planner, snapshot, request)
                return targets, checked.call_count
            return targets, carry_path(planner, snapshot, request)

    def test_ik_residual_subdivides_same_line_without_raising_step_limit(self):
        targets, segments = self.plan_with_residual()
        self.assertGreater(len(segments), 4)
        np.testing.assert_allclose(np.array(targets)[:, 1:], 0)
        np.testing.assert_allclose(targets[-1], [0.02, 0, 0])
        self.assertTrue(np.all(np.diff(np.array(targets)[:, 0]) > 0))
        self.assertTrue(all(segment.continuous_carry for segment in segments))
        self.assertTrue(all(segment.carry_blend == "cruise" for segment in segments[1:-1]))

    def test_unreachable_target_has_bounded_refinement(self):
        _, calls = self.plan_with_residual(residual=None)
        self.assertLess(calls, 100)

    def test_collision_failure_is_not_retried_or_ignored(self):
        _, calls = self.plan_with_residual(fail="arm_torso:left")
        self.assertEqual(calls, 1)
