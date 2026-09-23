"""Preparation-only protocol; no camera, API, planning or hardware connection."""

import copy
import unittest
from dataclasses import asdict
from unittest.mock import AsyncMock, Mock, patch

from astrabot.robot.motion_checks import MotionChecks
from astrabot.robot.prepare_motion import run
from astrabot.robot.tracking import TRACKING_POLICY_VERSION


class PrepareMotionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.state = {
            "session": "test",
            "generation": 3,
            "tracking_policy_version": TRACKING_POLICY_VERSION,
            "scene_id": "scene",
            "recovery_scene_valid": True,
            "health": "",
            "error": "",
            "motion_checks": asdict(MotionChecks()),
            "grasp_ready": False,
            "console_channel": {"connected": True},
        }
        self.client = Mock()
        self.client.request.side_effect = self.request
        self.cancel = self.disconnect = False
        self.preflight = {"startup_checks_passed": True, "snapshot": copy.deepcopy(self.state)}

    def request(self, op, **values):
        if op != "status":
            if values != {"session": "test", "generation": self.state["generation"]}:
                raise ValueError("expired_task_generation")
            self.state["generation"] += 1
        if op == "prepare":
            acknowledgement = copy.deepcopy(self.state)
            self.state["grasp_ready"] = True
            if self.cancel:
                self.state["generation"] += 1
            if self.disconnect:
                self.state["console_channel"]["connected"] = False
            return acknowledgement
        return copy.deepcopy(self.state)

    async def test_only_prepares_and_waits_for_measured_ready(self):
        report = {"success_verified": False, "hardware_ready": False}
        with (
            patch("astrabot.robot.prepare_motion.inspect", return_value=self.preflight),
            patch("astrabot.robot.pick_a.plan_trial", side_effect=AssertionError("full trial planner")),
            patch("astrabot.robot.camera.capture", side_effect=AssertionError("camera")),
        ):
            await run(self.client, report)
        self.assertTrue(report["hardware_ready"])
        self.assertFalse(report["success_verified"])
        self.assertEqual([c.args[0] for c in self.client.request.call_args_list], ["status", "prepare", "status"])

    async def test_invalid_scene_or_startup_prevents_motion(self):
        for changes in ({"scene_id": None}, {"recovery_scene_valid": False}, {"generation": 4}):
            with self.subTest(changes=changes):
                preflight = copy.deepcopy(self.preflight)
                preflight["snapshot"].update(changes)
                self.client.reset_mock()
                with (
                    patch("astrabot.robot.prepare_motion.inspect", return_value=preflight),
                    self.assertRaises(ValueError),
                ):
                    await run(self.client, {})
                self.client.request.assert_called_once_with("status")
        with (
            patch("astrabot.robot.prepare_motion.inspect", return_value={"startup_checks_passed": False}),
            self.assertRaisesRegex(ValueError, "startup_preflight_blocked"),
        ):
            await run(self.client, {})

    async def test_operator_cancel_never_pauses_new_generation(self):
        self.cancel = True
        report = {}
        with (
            patch("astrabot.robot.prepare_motion.inspect", return_value=self.preflight),
            self.assertRaisesRegex(ValueError, "cancelled"),
        ):
            await run(self.client, report)
        self.assertEqual(self.state["generation"], 5)
        self.assertIn("expired_task_generation", report["cleanup_error"])

    async def test_console_loss_pauses_owned_action(self):
        self.disconnect = True
        report = {}
        with (
            patch("astrabot.robot.prepare_motion.inspect", return_value=self.preflight),
            self.assertRaisesRegex(ValueError, "connected_console"),
        ):
            await run(self.client, report)
        self.assertEqual(self.client.request.call_args.args, ("pause",))
        self.assertFalse(report.get("hardware_ready"))

    async def test_timeout_pauses_only_our_request(self):
        with (
            patch("astrabot.robot.prepare_motion.inspect", return_value=self.preflight),
            patch("astrabot.robot.prepare_motion.wait_ready", new=AsyncMock(side_effect=TimeoutError("not_ready"))),
            self.assertRaises(TimeoutError),
        ):
            await run(self.client, {})
        self.assertEqual(self.client.request.call_args.args, ("pause",))
