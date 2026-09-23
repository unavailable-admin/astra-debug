"""Bounded supervised grip, tactile dropout and consistent object/hand offsets."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from astrabot.pinch import raw_hand
from astrabot.robot.calibration import binding
from astrabot.robot.config import Config
from astrabot.robot.model import RobotModel
from astrabot.robot.trajectory import Segment
from astrabot.robot.trial_plan import TrialPlanner
from astrabot.robot.trial_runtime import SingleTrial, pinch_step


class PositionGripTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = Config(trial_grip_mode="supervised_position", trial_grasp_offset=[0, -0.01, 0])
        self.model = RobotModel(self.config)
        self.state = {
            "grip_hold_policy_version": 1,
            "carry_path_version": 1,
            "trial_joint_path_version": 1,
            "body": np.zeros(29),
            "hands": np.tile(raw_hand("open"), 2),
            "tactile": [251, 0] + [0] * 8,
            "tactile_age": 1.0,
        }

    def test_target_is_bounded_38mm_with_unchanged_other_fingers(self):
        planner = TrialPlanner(self.model, self.config)
        target = planner.position_grip(self.state["body"], self.state["hands"])
        hands = self.state["hands"].copy()
        hands[:6] = target
        a, b = planner.tips(self.state["body"], hands)
        self.assertAlmostEqual(np.linalg.norm(a - b), 0.038, places=9)
        self.assertTrue(np.all(target >= raw_hand("closed")))
        self.assertTrue(np.all(target <= raw_hand("open")))
        np.testing.assert_array_equal(target[[0, 1, 2, 5]], raw_hand("open")[[0, 1, 2, 5]])

    def test_invalid_offsets_and_grip_targets_are_rejected(self):
        for kwargs in [
            {"trial_grasp_offset": [0, -0.011, 0]},
            {"trial_grasp_offset": [0, 0, 0.001]},
            {"trial_grip_gap_m": 0.03},
            {"trial_grip_mode": "force"},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)

    def test_thumb_yaw_preserves_grasp_anchor_and_other_finger_presets(self):
        from astrabot.robot.trial_runtime import pinch_pose

        before, rotation = pinch_pose(self.model, self.state["body"], self.state["hands"])
        self.config.trial_thumb_yaw_offset_raw = 16.6
        self.config.__post_init__()
        moved = self.state["hands"].copy()
        moved[:6] = self.config.trial_hand("open")
        actual_before = self.model.poses(self.state["body"], self.state["hands"])
        actual_after = self.model.poses(self.state["body"], moved)
        np.testing.assert_array_equal(actual_before["left_index_distal"], actual_after["left_index_distal"])
        self.assertGreater(np.linalg.norm(actual_before["left_thumb_distal"] - actual_after["left_thumb_distal"]), 0.01)
        after, new_rotation = pinch_pose(self.model, self.state["body"], moved)
        np.testing.assert_allclose(before, after, atol=1e-12)
        np.testing.assert_array_equal(rotation, new_rotation)
        for phase in ("open", "closed"):
            np.testing.assert_array_equal(self.config.trial_hand(phase)[:5], raw_hand(phase)[:5])
        target = TrialPlanner(self.model, self.config).position_grip(self.state["body"], moved)
        moved[:6] = target
        a, b = TrialPlanner(self.model, self.config).tips(self.state["body"], moved)
        self.assertAlmostEqual(np.linalg.norm(a - b), 0.038, places=9)
        self.assertAlmostEqual(target[5], raw_hand("open")[5] + 16.6)
        for offset in (31, -31, float("nan")):
            with self.assertRaises(ValueError):
                Config(trial_thumb_yaw_offset_raw=offset)

    async def run_trial(
        self, confirmed, result_confirmed=True, blocked_thumb=None, release_yaw_error=0.0, placement=False
    ):
        task = SimpleNamespace(status=lambda: self.state)
        observe = AsyncMock(
            side_effect=lambda stage: (
                {"grip_verified": confirmed} if stage == "grip_confirmation" else {"result_verified": result_confirmed}
            )
        )
        trial = SingleTrial(task, self.model, self.config, observe)
        trial.verified_center = AsyncMock(side_effect=lambda stage, expected: np.array(expected))
        trial.joint_phase = AsyncMock()
        moves = []
        self.held_commands = []

        async def move(center, hands=None, **kwargs):
            moves.append((np.array(center), kwargs))
            if kwargs.get("holding") and hands is not None:
                self.held_commands.append(np.asarray(hands).copy())
            if hands is not None:
                previous = self.state["hands"].copy()
                self.state["hands"] = np.array(hands)
                if blocked_thumb is not None and (kwargs.get("closing") or kwargs.get("holding")):
                    self.state["hands"][4] = max(blocked_thumb, self.state["hands"][4])
                if kwargs.get("opening"):
                    self.state["hands"][5] -= release_yaw_error
                self.assertLessEqual(np.max(np.abs(previous - hands)), 2 + 1e-9)
            return {}

        trial.move_center = move
        plan = {
            "geometry_passed": True,
            "lift_height_m": self.config.trial_lift_height_m,
            "scene_complete": True,
            "binding": binding(self.config),
            "start_body": self.state["body"].copy(),
            "start_hands": self.state["hands"].copy(),
            "grasp_center_m": [0.4, 0.1, 0.1],
            "obstacles": [],
            "phases": [{"phase": "descend"}, {"phase": "close"}, {"phase": "return_ready"}],
        }
        if placement:
            plan["placement"] = {"xy": [0.34, 0.16]}
        with patch("astrabot.robot.trial_runtime.pinch_pose", return_value=(np.array([0.4, 0.09, 0.1]), np.eye(3))):
            if confirmed and result_confirmed:
                await trial.run(plan)
            elif confirmed:
                with self.assertRaisesRegex(ValueError, "trial_result_not_operator_verified"):
                    await trial.run(plan)
            else:
                with self.assertRaisesRegex(ValueError, "position_grip_not_operator_verified"):
                    await trial.run(plan)
        return trial, moves, observe

    async def test_recorded_release_yaw_error_completes(self):
        trial, _, _ = await self.run_trial(True, release_yaw_error=1.1)
        release = next(e for e in trial.events if e.get("release_position_confirmed"))
        self.assertLessEqual(max(abs(x) for x in release["error_raw"]), 2.0)

    async def test_release_error_above_tolerance_still_times_out(self):
        with self.assertRaisesRegex(TimeoutError, "trial_release_timeout"):
            await self.run_trial(True, release_yaw_error=3.0)

    async def test_placement_never_requests_grip_or_result_confirmation(self):
        self.config.trial_require_grip_confirmation = True
        with patch("astrabot.robot.transfer.execute_transfer", new=AsyncMock(return_value={})) as transfer:
            _, _, observe = await self.run_trial(True, placement=True)
        transfer.assert_awaited_once()
        observe.assert_not_awaited()

    async def test_normal_transfer_lift_uses_120mm_then_returns_to_contact(self):
        self.config.trial_lift_height_m = Config(trial_lift_height_m=0.12).trial_lift_height_m
        trial, moves, _ = await self.run_trial(True)
        carried = [center for center, values in moves if values.get("holding")]
        np.testing.assert_allclose(carried, [[0.4, 0.09, 0.22], [0.4, 0.09, 0.1]])
        self.assertTrue(any(event.get("lift_height_m") == 0.12 for event in trial.events))
        for hands in self.held_commands[1:]:
            np.testing.assert_array_equal(hands, self.held_commands[0])

    def test_lift_height_bounds(self):
        for height in (0, -0.02, 0.121, float("nan"), float("inf"), True):
            with self.subTest(height=height), self.assertRaisesRegex(ValueError, "trial_lift_height"):
                Config(trial_lift_height_m=height)

    async def test_auto_grip_skips_input_without_claiming_operator_verification(self):
        self.config.trial_require_grip_confirmation = False
        trial, moves, observe = await self.run_trial(True)
        self.assertEqual([call.args[0] for call in observe.await_args_list], ["result_confirmation"])
        self.assertTrue(any(values.get("holding") for _, values in moves))
        self.assertTrue(any(event.get("position_grip_auto_advance") for event in trial.events))
        self.assertFalse(any(event.get("operator_grip_verified") for event in trial.events))
        from astrabot.robot.pick_a import outcome_checks

        checks = outcome_checks(trial.events, self.state, {}, self.config, None)
        self.assertTrue(checks["position_closure_completed"])
        self.assertTrue(checks["lift_verified"])
        self.assertFalse(all(checks.values()))  # No final HOLD in this isolated fixture.

    async def test_zero_tactile_allows_supervised_lift_and_preserves_right_offset(self):
        trial, moves, observe = await self.run_trial(True)
        np.testing.assert_allclose(moves[0][0], [0.4, 0.09, 0.1])
        self.assertTrue(any(k.get("holding") for _, k in moves))
        self.assertEqual([c.args[0] for c in observe.await_args_list], ["grip_confirmation", "result_confirmation"])
        self.assertTrue(any(e.get("operator_grip_verified") for e in trial.events))
        self.assertFalse(any(e.get("contact_verified") for e in trial.events))
        calls = trial.verified_center.await_args_list
        stages = [c.args[0] for c in calls]
        self.assertNotIn("before_descent", stages)
        self.assertIn("before_approach", stages)
        self.assertNotIn("before_close", stages)
        self.assertTrue(moves[0][1].get("closing"))
        self.assertTrue(any(e.get("visual_skipped") == "before_close" for e in trial.events))
        self.assertFalse(any("visual_grasp_correction_m" in e for e in trial.events))
        self.assertTrue(any(e.get("visual_skipped") == "before_descent" for e in trial.events))
        self.assertEqual(stages, ["before_approach"])
        for stage in ("lifted", "replaced", "released", "returned"):
            self.assertTrue(any(e.get("visual_skipped") == stage for e in trial.events))
        self.assertTrue(any(e.get("operator_result_verified") for e in trial.events))

    async def test_unconfirmed_result_is_not_reported_as_verified(self):
        trial, _, _ = await self.run_trial(True, False)
        self.assertTrue(any(e.get("motion_completed") for e in trial.events))
        self.assertFalse(any(e.get("operator_result_verified") for e in trial.events))

    async def test_blocked_finger_can_be_manually_accepted_without_reaching_target(self):
        for blocked in (126.0, 135.0):
            self.setUp()
            trial, moves, _ = await self.run_trial(True, blocked_thumb=blocked)
            closure = next(e for e in trial.events if e.get("position_grip_closure_finished"))
            self.assertEqual(closure["reason"], "feedback_plateau")
            self.assertFalse(closure["target_reached"])
            self.assertLess(closure["closure_steps"], 60)
            self.assertLess(closure["held_command_left_raw"][4], blocked)
            self.assertEqual(len(self.held_commands), 2)
            for command in self.held_commands:
                np.testing.assert_array_equal(command[:6], closure["held_command_left_raw"])
            self.assertTrue(any(k.get("holding") for _, k in moves))
            self.assertFalse(any(e.get("contact_verified") for e in trial.events))

    async def test_blocked_finger_without_operator_confirmation_never_lifts(self):
        trial, moves, _ = await self.run_trial(False, blocked_thumb=126.0)
        self.assertTrue(any(e.get("position_grip_closure_finished") for e in trial.events))
        self.assertFalse(any(k.get("holding") for _, k in moves))

    async def test_old_executor_is_rejected_before_trial_motion(self):
        self.state.pop("grip_hold_policy_version")
        with self.assertRaisesRegex(ValueError, "executor_restart_required_for_fixed_grip_hold"):
            await self.run_trial(True)

    async def test_old_carry_executor_rejected_before_approach(self):
        self.state.pop("carry_path_version")
        with self.assertRaisesRegex(ValueError, "executor_restart_required_for_continuous_carry"):
            await self.run_trial(True)

    async def test_unconfirmed_grip_never_lifts(self):
        _, moves, _ = await self.run_trial(False)
        self.assertFalse(any(k.get("holding") for _, k in moves))

    async def test_carry_dispatches_one_full_path_and_preserves_grip(self):
        current = np.zeros(3)
        requests = []
        held = self.state["hands"].copy()
        held[4] -= 1.2

        async def move(**values):
            target = np.asarray(values["goal_center"])
            self.assertEqual(values["action"], "carry_path")
            np.testing.assert_array_equal(values["hands"][:6], held[:6])
            requests.append(target)
            current[:] = target
            return {}

        task = SimpleNamespace(status=lambda: self.state, move=move)
        trial = SingleTrial(task, self.model, self.config, AsyncMock())
        with patch("astrabot.robot.trial_runtime.pinch_pose", side_effect=lambda *a: (current.copy(), np.eye(3))):
            await trial.move_center([0, 0, 0.02], held, holding=True)
        self.assertEqual(len(requests), 1)
        np.testing.assert_allclose(requests[-1], [0, 0, 0.02])
        self.assertLess(np.linalg.norm(current - [0, 0, 0.02]), 0.003)

    async def test_executor_arrival_failure_stops_without_replaying_motion(self):
        task = SimpleNamespace(
            status=lambda: self.state,
            move=AsyncMock(side_effect=RuntimeError("pinch_arrival_timeout:remaining_mm=3.100")),
        )
        trial = SingleTrial(task, self.model, self.config, AsyncMock())
        with (
            patch("astrabot.robot.trial_runtime.pinch_pose", return_value=(np.zeros(3), np.eye(3))),
            self.assertRaisesRegex(RuntimeError, "pinch_arrival_timeout"),
        ):
            await trial.move_center([0, 0, 0.02], holding=True)
        self.assertEqual(task.move.await_count, 1)
        self.assertFalse(any(e.get("carry_path_completed") for e in trial.events))

    async def test_later_feedback_jitter_does_not_overrule_executor_arrival(self):
        task = SimpleNamespace(status=lambda: self.state, move=AsyncMock(return_value={}))
        trial = SingleTrial(task, self.model, self.config, AsyncMock())
        with patch(
            "astrabot.robot.trial_runtime.pinch_pose",
            side_effect=[(np.zeros(3), np.eye(3)), (np.array([0, 0, 0.0184]), np.eye(3))],
        ):
            await trial.move_center([0, 0, 0.02], holding=True)
        self.assertEqual(task.move.await_count, 1)
        event = trial.events[-1]
        self.assertTrue(event["carry_path_completed"])
        self.assertEqual(event["arrival_checked_by"], "executor")
        self.assertAlmostEqual(event["post_completion_error_m"], 0.0016)

    async def test_client_move_does_not_reject_zero_tactile_in_supervised_mode(self):
        task = SimpleNamespace(status=lambda: self.state, move=AsyncMock(return_value={}))
        trial = SingleTrial(task, self.model, self.config, AsyncMock())
        with patch(
            "astrabot.robot.trial_runtime.pinch_pose",
            side_effect=[
                (np.zeros(3), np.eye(3)),
                (np.array([0, 0, 0.002]), np.eye(3)),
                (np.zeros(3), np.eye(3)),
            ],
        ):
            await trial.move_center([0, 0, 0.002], holding=True)
            self.config.trial_grip_mode = "tactile"
            with self.assertRaisesRegex(ValueError, "trial_lost_contact"):
                await trial.move_center([0, 0, 0.002], holding=True)
        self.assertEqual(task.move.await_count, 1)

    def test_server_preserves_sweep_but_only_tactile_mode_sets_contact_guard(self):
        snap = SimpleNamespace(body=self.state["body"], hands=self.state["hands"])
        from unittest.mock import Mock

        model = Mock()
        model.solve.return_value = snap.body.copy()
        planner = SimpleNamespace(model=model, config=self.config, segment=Mock())
        segment = Segment(snap.body[15:], snap.hands, snap.body[15:], snap.hands, 1)
        planner.segment.return_value = segment
        with (
            patch("astrabot.robot.trial_runtime.pinch_pose", return_value=(np.zeros(3), np.eye(3))),
            patch.object(TrialPlanner, "tips", return_value=(np.zeros(3), np.zeros(3))),
            patch("astrabot.robot.trial_runtime.check_sweep", return_value=segment) as sweep,
        ):
            for mode in ["supervised_position", "tactile"]:
                self.config.trial_grip_mode = mode
                result = pinch_step(planner, snap, {"center": [0, 0, 0.002], "holding": True})
                self.assertEqual(result.require_contact, mode == "tactile")
                self.assertTrue(sweep.call_args.kwargs.get("holding", sweep.call_args.args[-1]))


class GripConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmation_accepts_only_explicit_token(self):
        from unittest.mock import Mock

        from astrabot.robot.pick_a import confirm_grip

        for token in ("GRIP_OK", "VERIFIED", ""):
            task = SimpleNamespace(status=Mock(return_value={}))
            stream = Mock()
            stream.isatty.return_value = True
            stream.readline.return_value = token + "\n"
            with (
                patch("astrabot.robot.pick_a.sys.stdin", stream),
                patch("astrabot.robot.pick_a.select.select", return_value=([stream], [], [])),
            ):
                if token == "GRIP_OK":
                    self.assertEqual(await confirm_grip(task, 1), {"grip_verified": True})
                else:
                    with self.assertRaisesRegex(ValueError, "grip_review_not_verified"):
                        await confirm_grip(task, 1)

    async def test_result_confirmation_requires_its_own_token_and_checks_cancellation(self):
        from unittest.mock import Mock

        from astrabot.robot.pick_a import confirm_result

        for token in ("RESULT_OK", "GRIP_OK", ""):
            task = SimpleNamespace(status=Mock(return_value={}))
            stream = Mock()
            stream.isatty.return_value = True
            stream.readline.return_value = token + "\n"
            with (
                patch("astrabot.robot.pick_a.sys.stdin", stream),
                patch("astrabot.robot.pick_a.select.select", return_value=([stream], [], [])),
            ):
                if token == "RESULT_OK":
                    self.assertEqual(await confirm_result(task, 1), {"result_verified": True})
                else:
                    with self.assertRaisesRegex(ValueError, "result_review_not_verified"):
                        await confirm_result(task, 1)
        task = SimpleNamespace(status=Mock(side_effect=ValueError("cancelled")))
        with (
            patch("astrabot.robot.pick_a.sys.stdin.isatty", return_value=True),
            self.assertRaisesRegex(ValueError, "cancelled"),
        ):
            await confirm_result(task, 1)

    async def test_operator_cancel_interrupts_confirmation(self):
        from unittest.mock import Mock

        from astrabot.robot.pick_a import confirm_grip

        task = SimpleNamespace(status=Mock(side_effect=ValueError("cancelled")))
        with (
            patch("astrabot.robot.pick_a.sys.stdin.isatty", return_value=True),
            self.assertRaisesRegex(ValueError, "cancelled"),
        ):
            await confirm_grip(task, 1)
