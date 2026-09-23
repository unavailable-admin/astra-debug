"""Startup reports must fail closed without publishing robot commands."""

import copy
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from astrabot.robot.config import Config
from astrabot.robot.executor import Executor
from astrabot.robot.hardware import MockHardware
from astrabot.robot.preflight import evaluate, failure_summary, inspect
from astrabot.robot.trajectory import Segment


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.hardware = MockHardware(lambda: self.now)
        self.executor = Executor(self.hardware, None, Config(), lambda: self.now)
        self.addCleanup(self.executor.close)
        self.executor.request({"op": "console_heartbeat", "console_id": "terminal"})
        self.executor.request({"op": "pause", "console_id": "terminal"})
        self.before = self.executor.status()
        self.now += 0.15
        self.executor.tick()
        self.state = self.executor.status()
        # Only the pure report evaluator sees a synthetic real-backend label.
        # IPC tests below retain MockHardware and can never report a real pass.
        self.state["hardware_backend"] = "RealHardware"

    def check(self, state, name):
        return next(item for item in evaluate(self.before, state)["checks"] if item["name"] == name)["passed"]

    def test_valid_startup_does_not_claim_pick_ready(self):
        report = evaluate(self.before, self.state)
        self.assertTrue(report["startup_checks_passed"])
        self.assertFalse(report["pick_ready"])
        self.assertEqual(self.hardware.writes, [])

    def test_check_only_issues_status_and_mock_never_passes(self):
        client = Mock()
        client.request.side_effect = [self.before, self.executor.status()]
        with patch("astrabot.robot.preflight.time.sleep"):
            report = inspect(client)
        self.assertEqual([call.args for call in client.request.call_args_list], [("status",), ("status",)])
        self.assertFalse(report["startup_checks_passed"])

    def test_temperature_checked_even_when_mode_masks_executor_health(self):
        state = copy.deepcopy(self.state)
        state.update(mode=1, health="standing_mode_required:1", temperature=120)
        self.assertFalse(self.check(state, "当前模式"))
        self.assertFalse(self.check(state, "温度 motor"))

    def test_damping_explains_mode_instead_of_requesting_another_space(self):
        state = copy.deepcopy(self.state)
        state.update(mode=1, health="standing_mode_required:1")
        text = failure_summary(evaluate(self.before, state))
        self.assertIn("阻尼模式(mode=1)", text)
        self.assertIn("站立模式(mode=4)", text)
        self.assertNotIn("按空格", text)

    def test_stale_console_or_generation_cannot_pass(self):
        for changes in ({"connected": False}, {"pause_generation": 0}, {"pause_age": 301}):
            state = copy.deepcopy(self.state)
            state["console_channel"].update(changes)
            self.assertFalse(self.check(state, "暂停控制台与空格回执"))

    def test_feedback_must_advance_and_loop_must_be_alive(self):
        state = copy.deepcopy(self.state)
        state["joint_time"] = self.before["joint_time"]
        state["tick_age"] = 1
        self.assertFalse(self.check(state, "joint 反馈"))
        self.assertFalse(self.check(state, "控制循环"))

    def test_missing_invalid_old_or_active_status_cannot_pass(self):
        for field, value in (
            ("feedback_timeout", None),
            ("temperature", float("nan")),
            ("standing_mode", None),
            ("feedback_fault", None),
            ("task_active", True),
            ("planning_active", True),
            ("weight", 0.5),
            ("thermal_handoff_required", True),
            ("session", "another-executor"),
        ):
            with self.subTest(field=field):
                state = copy.deepcopy(self.state)
                state[field] = value
                self.assertFalse(evaluate(self.before, state)["startup_checks_passed"])

    def test_telemetry_does_not_extend_task_or_move_robot(self):
        token = self.executor.request({"op": "begin"})
        lease = self.executor.lease
        self.now += 0.2
        self.executor.request({"op": "console_heartbeat", "console_id": "terminal"})
        self.assertEqual(self.executor.generation, token["generation"])
        self.assertEqual(self.executor.lease, lease)
        self.assertEqual(self.hardware.writes, [])

    def test_console_expiry_detach_and_new_terminal_require_pause(self):
        self.now += 4
        self.assertFalse(self.executor.status()["console_channel"]["connected"])
        status = self.executor.request({"op": "console_heartbeat", "console_id": "new"})
        self.assertIsNone(status["console_channel"]["pause_age"])
        status = self.executor.request({"op": "console_detach", "console_id": "new"})
        self.assertFalse(status["console_channel"]["connected"])

    def test_ready_requires_actual_pinch_fingers_and_healthy_full_control(self):
        self.executor.state = "READY"
        self.executor.attached, self.executor.weight = True, 1
        self.hardware.body[15:] = self.executor.config.ready_arms
        self.hardware.hands[:] = self.executor.config.ready_hands
        self.assertTrue(self.executor.status()["grasp_ready"])
        self.hardware.hands[:] = 255
        self.assertFalse(self.executor.status()["grasp_ready"])
        self.hardware.hands[:] = self.executor.config.ready_hands
        self.executor.request({"op": "pause", "console_id": "terminal"})
        self.assertFalse(self.executor.status()["grasp_ready"])

    def test_ready_hand_config_is_validated(self):
        for hands in ([0] * 6, [float("nan")] * 12, [256] * 12):
            with self.assertRaises(ValueError):
                Config(ready_hands=hands)
        self.assertTrue(np.all(np.array(Config().ready_hands).reshape(2, 6)[:, :3] < 60))

    def test_reset_cancels_old_task_opens_then_reaches_pinch_ready(self):
        # Deterministic planner completion; geometry is exercised separately by
        # test_robot_preparation with the production Planner and RobotModel.
        def submit(function, snapshot):
            future = Future()
            future.set_result(function(snapshot))
            return future

        def ready(snapshot):
            arms = snapshot.body[15:]
            return [Segment(arms, snapshot.hands, arms, np.array(self.executor.config.ready_hands), 0.1)]

        self.hardware.body[15:] = self.executor.config.ready_arms
        self.hardware.hands[:] = 0
        token = self.executor.request({"op": "begin"})
        self.executor.weight = 1.0
        self.executor.planner = SimpleNamespace(ready=ready)
        with patch.object(self.executor.pool, "submit", side_effect=submit):
            result = self.executor.request({"op": "reset"})
            self.assertEqual(result["state"], "RELEASING")
            np.testing.assert_array_equal(result["hands_command"], np.full(12, 255.0))
            with self.assertRaisesRegex(ValueError, "expired_task_generation"):
                self.executor.request(
                    {"op": "heartbeat", "session": token["session"], "generation": token["generation"]}
                )
            for _ in range(500):
                self.now += 0.01
                self.executor.tick()
        self.assertTrue(self.executor.status()["grasp_ready"])
        np.testing.assert_allclose(
            self.hardware.hands, self.executor.config.ready_hands, atol=self.executor.config.hand_arrival_error
        )


if __name__ == "__main__":
    unittest.main()
