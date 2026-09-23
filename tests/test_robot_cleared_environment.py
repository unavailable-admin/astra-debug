"""Empty workspace guarantees never stand in for a measured grasp scene."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from astrabot.robot.cleared_environment import install_cleared_environment, run, stopped_receipt
from astrabot.robot.config import GRIP, READY, Config
from astrabot.robot.executor import Executor
from astrabot.robot.hardware import MockHardware
from astrabot.robot.model import RobotModel
from astrabot.robot.pick_a import token
from astrabot.robot.trajectory import Planner


class ClearedEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(scene_measured=True, table_min=[-2, -2, -2], table_max=[2, 2, 2])
        self.hardware = MockHardware(lambda: 10.0)
        self.executor = Executor(
            self.hardware, Planner(RobotModel(self.config), self.config), self.config, lambda: 10.0
        )
        self.addCleanup(self.executor.close)
        self.executor.request({"op": "console_heartbeat", "console_id": "operator"})

    def clear(self, **changes):
        return install_cleared_environment(
            self.executor,
            {**token(self.executor.status()), "operator_confirmed_empty": True, **changes},
        )

    def test_clear_keeps_self_checks_and_invalidates_visual_scene_without_motion(self):
        body = np.r_[np.zeros(15), READY]
        with self.assertRaisesRegex(ValueError, "obstacle"):
            self.executor.planner.model.check(body, GRIP)
        result = self.clear()
        self.assertIsNone(result["scene_id"])
        self.assertFalse(result["recovery_scene_valid"])
        self.assertFalse(self.executor.config.scene_measured)
        self.assertEqual(self.hardware.writes, [])
        model = self.executor.planner.model
        model.check(body, GRIP)
        with (
            patch.object(model, "_check_head", side_effect=ValueError("head_collision")),
            self.assertRaisesRegex(ValueError, "head_collision"),
        ):
            model.check(body, GRIP)
        invalid = body.copy()
        invalid[15] = 100
        with self.assertRaisesRegex(ValueError, "joint_limit"):
            model.check(invalid, GRIP)
        with self.assertRaisesRegex(ValueError, "empty_hand_preparation"):
            model.check(body, GRIP, holding=True)
        with self.assertRaisesRegex(ValueError, "fresh_visual_scene"):
            self.executor.request({"op": "begin"})
        # Preparation is allowed, but is still asynchronously collision checked.
        with self.assertRaisesRegex(ValueError, "confirmation_required"):
            self.executor.request({"op": "prepare"})
        self.assertEqual(
            self.executor.request({"op": "prepare", "operator_confirmed_empty": True})["state"], "PLANNING"
        )

    def test_missing_confirmation_stale_token_and_disconnected_console_reject(self):
        for changes in ({"operator_confirmed_empty": False}, {"generation": -1}):
            with self.assertRaises(ValueError):
                self.clear(**changes)
        self.executor.request({"op": "console_detach", "console_id": "operator"})
        with self.assertRaisesRegex(ValueError, "connected_console"):
            self.clear()
        self.assertIs(self.executor.config, self.config)

    def test_latched_scene_error_can_be_cleared_after_pause_but_not_live_heat(self):
        self.executor.error = "preflight:hand_obstacle:left:1"
        self.executor.request({"op": "pause"})
        self.assertEqual(self.clear()["error"], "")
        self.executor.thermal_handoff_required = True
        with self.assertRaisesRegex(ValueError, "healthy_idle_executor"):
            self.clear()

    def test_install_race_does_not_overwrite_operator_pause(self):
        real_model = RobotModel

        def raced(*args, **kwargs):
            self.executor.request({"op": "pause"})
            return real_model(*args, **kwargs)

        with (
            patch("astrabot.robot.cleared_environment.RobotModel", side_effect=raced),
            self.assertRaisesRegex(ValueError, "cancelled"),
        ):
            self.clear()
        self.assertIs(self.executor.config, self.config)

    def test_new_visual_model_restores_table_check(self):
        self.clear()
        model = RobotModel(replace(self.executor.config, scene_measured=True))
        with self.assertRaisesRegex(ValueError, "obstacle"):
            model.check(np.r_[np.zeros(15), READY], GRIP)

    def test_shutdown_receipt_requires_final_owned_zero_weight_record(self):
        expected = {"session": "current", "generation": 3}
        state = {**expected, "state": "STOPPED", "attached": False, "weight": 0, "error": ""}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            for change in ({"generation": 2}, {"weight": 1}, {"attached": True}, {"error": "fault"}):
                path.write_text(json.dumps({"final": True, "status": {**state, **change}}) + "\n")
                self.assertIsNone(stopped_receipt(path, expected))
            path.write_text(json.dumps({"final": True, "status": state}) + "\n")
            self.assertEqual(stopped_receipt(path, expected), state)


class ClearedCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_actions_use_no_camera_and_confirm_completion(self):
        for action, attached in (("prepare", False), ("recover", True), ("shutdown", True), ("shutdown", False)):
            with self.subTest(action=action, attached=attached):
                state = {
                    "session": "s",
                    "generation": 0,
                    "state": "HOLD",
                    "attached": attached,
                    "cleared_environment_version": 1,
                    "console_channel": {"connected": True},
                    "health": "",
                    "error": "old_planning_error" if action != "prepare" else "",
                    "motion_checks": asdict(Config().motion_checks),
                    "weight": int(attached),
                }
                operations = []

                def request(op, *, state=state, operations=operations, **values):
                    operations.append(op)
                    if op != "status":
                        self.assertEqual(token(values), token(state))
                        if op not in ("pause",):
                            self.assertIs(values["operator_confirmed_empty"], True)
                        if op != "shutdown" or state["attached"]:
                            state["generation"] += 1
                    if op == "clear_environment":
                        state.update(error="", scene_id=None)
                    elif op in ("prepare", "reset"):
                        state.update(grasp_ready=True, state="READY", attached=True, weight=1)
                    elif op == "shutdown":
                        state.update(state="STOPPED", attached=False, weight=0)
                    return dict(state)

                client = Mock(path="unused", request=Mock(side_effect=request))
                report = {}
                with (
                    patch("astrabot.robot.cleared_environment.Client", return_value=client),
                    patch(
                        "astrabot.robot.cleared_environment.inspect",
                        return_value={"startup_checks_passed": True, "snapshot": dict(state)},
                    ),
                    patch("astrabot.robot.camera.capture", side_effect=AssertionError("camera called")),
                    patch("astrabot.robot.scene_review.automatic_scene", side_effect=AssertionError("API called")),
                ):
                    await run(client, action, report)
                self.assertTrue(report["completed"])
                self.assertFalse(report["visual_verified"])
                self.assertEqual(operations[:3], ["status", "pause", "clear_environment"])
                self.assertEqual(
                    operations[3], {"prepare": "prepare", "recover": "reset", "shutdown": "shutdown"}[action]
                )
