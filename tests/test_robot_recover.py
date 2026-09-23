"""Recovery protocol regressions; no SDK, camera, API or robot connection."""

import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from astrabot.robot.motion_checks import MotionChecks
from astrabot.robot.recover import refresh_scene, run, wait_state
from astrabot.robot.tracking import TRACKING_POLICY_VERSION


class Client:
    def __init__(self, valid=True):
        self.state = {
            "session": "test",
            "generation": 7,
            "state": "RUNNING",
            "health": "",
            "error": "arm_tracking_error",
            "console_channel": {"connected": True},
            "tracking_policy_version": TRACKING_POLICY_VERSION,
            "motion_checks": asdict(MotionChecks()),
            "scene_id": "scene",
            "recovery_scene_valid": valid,
            "grasp_ready": False,
        }
        self.calls = []
        self.cancel_on_prepare = False

    def request(self, op, **values):
        self.calls.append((op, values))
        if op != "status":
            if (values.get("session"), values.get("generation")) != ("test", self.state["generation"]):
                raise ValueError("expired_task_generation")
            self.state["generation"] += 1
            self.state["error"] = ""
            self.state["state"] = "HOLD"
        if op == "prepare":
            if self.cancel_on_prepare:
                self.state["generation"] += 1
            else:
                self.state.update(grasp_ready=True, state="READY")
        if op == "load_scene":
            self.state.update(scene_id="fresh", recovery_scene_valid=True)
        return dict(self.state)


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = SimpleNamespace(config=Path("config.json"), output=Path("unused"), socket=None, api_timeout=120)

    def inspect(self, client):
        return {
            "snapshot": client.request("status"),
            "checks": [{"name": "故障", "passed": True}, {"name": "暂停控制台与空格回执", "passed": False}],
        }

    async def test_valid_scene_release_and_ready_without_full_prepare(self):
        client, report = Client(), {}
        with (
            patch("astrabot.robot.recover.inspect", side_effect=self.inspect),
            patch("astrabot.robot.recover.refresh_scene", new_callable=AsyncMock) as refresh,
        ):
            await run(self.args, report, client)
        refresh.assert_not_called()
        self.assertEqual([op for op, _ in client.calls if op != "status"], ["release", "prepare"])
        self.assertTrue(report["recovery_verified"])

    async def test_invalid_scene_is_refreshed_before_prepare(self):
        client, report = Client(valid=False), {}

        async def refresh(args, current, expected):
            previous = dict(expected)
            expected["generation"] += 1
            return current.request("load_scene", **previous)

        with (
            patch("astrabot.robot.recover.inspect", side_effect=self.inspect),
            patch("astrabot.robot.recover.refresh_scene", side_effect=refresh),
        ):
            await run(self.args, report, client)
        self.assertEqual([op for op, _ in client.calls if op != "status"], ["release", "load_scene", "prepare"])
        self.assertTrue(report["scene_refreshed"])

    async def test_scene_failure_holds_and_never_requests_motion(self):
        client, report = Client(valid=False), {}
        with (
            patch("astrabot.robot.recover.inspect", side_effect=self.inspect),
            patch("astrabot.robot.recover.refresh_scene", side_effect=ValueError("uncertain_scene")),
            self.assertRaisesRegex(ValueError, "uncertain_scene"),
        ):
            await run(self.args, report, client)
        self.assertEqual([op for op, _ in client.calls if op != "status"], ["release", "pause"])

    async def test_operator_cancel_is_not_adopted_or_overwritten(self):
        client, report = Client(), {}
        client.cancel_on_prepare = True
        with (
            patch("astrabot.robot.recover.inspect", side_effect=self.inspect),
            self.assertRaisesRegex(ValueError, "cancelled"),
        ):
            await run(self.args, report, client)
        self.assertEqual(client.state["generation"], 10)
        self.assertIn("expired_task_generation", report["cleanup_error"])
        self.assertFalse(report.get("recovery_verified"))

    async def test_missing_console_or_unhealthy_state_has_no_side_effect(self):
        for changes in (
            {"console_channel": {"connected": False}},
            {"health": "joint_feedback_stale"},
            {"tracking_policy_version": 2},
            {"tracking_policy_version": 5},
            {"tracking_policy_version": 9},
        ):
            client = Client()
            client.state.update(changes)
            with self.assertRaises(ValueError):
                await run(self.args, {}, client)
            self.assertEqual([op for op, _ in client.calls], ["status"])

    async def test_post_release_preflight_failure_prevents_return(self):
        client = Client()

        def inspect(current):
            return {"snapshot": current.request("status"), "checks": [{"name": "joint 反馈", "passed": False}]}

        with (
            patch("astrabot.robot.recover.inspect", side_effect=inspect),
            self.assertRaisesRegex(ValueError, "preflight_blocked"),
        ):
            await run(self.args, {}, client)
        self.assertNotIn("prepare", [op for op, _ in client.calls])

    async def test_wait_rejects_lost_console_and_times_out(self):
        client = Client()
        client.state["error"] = ""
        expected = {"session": "test", "generation": 7}
        with self.assertRaises(TimeoutError):
            await wait_state(client, expected, lambda s: False, 0)
        client.state["console_channel"] = {"connected": False}
        with self.assertRaisesRegex(ValueError, "connected_console"):
            await wait_state(client, expected, lambda s: True, 1)

    async def test_fresh_install_advances_owned_generation_even_on_lost_reply(self):
        client = Client(valid=False)
        client.state["error"] = ""
        expected = {"session": "test", "generation": 7}
        bundle = Mock(digest="fresh", directory=Path("scene"))
        remote = Mock()

        def install(op, **kwargs):
            client.request(op, **kwargs)
            raise OSError("reply_lost")

        remote.request.side_effect = install
        with (
            patch("astrabot.robot.recover.Config.load"),
            patch("astrabot.robot.camera.capture"),
            patch("astrabot.robot.scene_review.automatic_scene"),
            patch("astrabot.robot.scene_builder.SceneBuilder"),
            patch("astrabot.robot.recover.SceneBundle.load", return_value=bundle),
            patch("astrabot.robot.recover.Client", return_value=remote),
            self.assertRaisesRegex(OSError, "reply_lost"),
        ):
            await refresh_scene(self.args, client, expected)
        self.assertEqual(expected["generation"], 8)
        client.request("pause", **expected)
        self.assertEqual(client.state["state"], "HOLD")
