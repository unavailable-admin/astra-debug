"""Task-side scene updates must reject collisions and moving starts before dispatch."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np

from astrabot.robot.config import Config
from astrabot.robot.model import RobotModel
from astrabot.robot.scene_task import SceneCheckedTask


class SceneTaskTests(unittest.IsolatedAsyncioTestCase):
    async def test_updated_obstacle_blocks_dispatch(self):
        source = json.loads((Path(__file__).parent / "fixtures/robot_aligned_trial.json").read_text())
        config = Config(scene_measured=True, table_min=source["table_min"], table_max=source["table_max"])
        model = RobotModel(config)
        body, hands = np.array(source["body"]), np.array(source["hands"])
        box = model.boxes(model.poses(body, hands), "left")[0]
        task = SimpleNamespace(status=lambda: {"body": body.tolist(), "hands": hands.tolist()}, move=AsyncMock())
        checked = SceneCheckedTask(task, model, config)
        self.addCleanup(checked.close)
        with self.assertRaisesRegex(ValueError, "obstacle"):
            await checked.move(arms=body[15:].tolist(), hands=hands.tolist(), obstacles=[box])
        task.move.assert_not_awaited()

    async def test_certified_carry_batch_dispatches_only_after_full_check(self):
        task = SimpleNamespace(status=lambda: {"body": [0.0] * 29, "hands": [255.0] * 12}, move=AsyncMock())
        checked = SceneCheckedTask(task, None, Config(), isolated=False)
        checked.planner = SimpleNamespace(move=Mock(return_value=[SimpleNamespace(continuous_carry=True)] * 2))
        await checked.move(action="carry_path")
        task.move.assert_awaited_once_with(action="carry_path")
        task.move.reset_mock()
        checked.planner.move.side_effect = ValueError("late_path_collision")
        with self.assertRaisesRegex(ValueError, "late_path_collision"):
            await checked.move(action="carry_path")
        task.move.assert_not_awaited()

    async def test_5mrad_start_threshold_accepts_small_drift_and_rejects_larger(self):
        for drift, accepted in ((0.00114, True), (0.0049, True), (0.0051, False)):
            body = np.zeros(29)

            def status(body=body, drift=drift):
                body[24] += drift
                return {"body": body.tolist(), "hands": [255.0] * 12}

            task = SimpleNamespace(status=status, move=AsyncMock())
            checked = SceneCheckedTask(
                task, None, Config(settle_timeout=0.01, motion_checks={"scene_task_rad": 0.005}), isolated=False
            )
            checked.planner = SimpleNamespace(move=Mock(return_value=[object()]))
            if accepted:
                await checked.move(arms=[0.0] * 14)
                task.move.assert_awaited_once()
            else:
                with self.assertRaisesRegex(ValueError, "start_did_not_settle"):
                    await checked.move(arms=[0.0] * 14)
                task.move.assert_not_awaited()

    async def test_one_raw_roundoff_is_accepted_by_local_fallback(self):
        hands = np.full(12, 128.0)

        def status():
            hands[0] -= 1.0000000000000284
            return {"body": [0.0] * 29, "hands": hands.tolist()}

        task = SimpleNamespace(status=status, move=AsyncMock())
        checked = SceneCheckedTask(task, None, Config(), isolated=False)
        checked.planner = SimpleNamespace(move=Mock(return_value=[object()]))
        await checked.move(action="pinch_step")
        task.move.assert_awaited_once()
        self.assertEqual(checked.planner.move.call_count, 1)

    async def test_unsettled_start_cannot_dispatch_old_plan(self):
        body = np.zeros(29)

        def status():
            body[15] += 0.002
            return {"body": body.tolist(), "hands": [255.0] * 12}

        task = SimpleNamespace(status=status, move=AsyncMock())
        checked = SceneCheckedTask(task, None, Config(settle_timeout=0.02), isolated=False)
        checked.planner = SimpleNamespace(move=Mock(return_value=[object()]))
        with self.assertRaisesRegex(ValueError, "start_did_not_settle"):
            await checked.move(arms=[0.0] * 14)
        self.assertGreaterEqual(checked.planner.move.call_count, 1)
        task.move.assert_not_awaited()


class BlockingPlanner:
    """Simulate a native planning call holding its process's Python GIL."""

    def __init__(self):
        import os

        self.owner_pid = os.getpid()

    def move(self, snapshot, values):
        import ctypes
        import os

        assert os.getpid() != self.owner_pid
        ctypes.PyDLL(None).usleep(1_200_000)
        return [SimpleNamespace(continuous_carry=True)]


class IsolatedPlanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_planning_cannot_starve_task_heartbeat(self):
        from astrabot.robot.task import Task

        heartbeats = []

        class Client:
            def request(self, operation, **kwargs):
                if operation == "heartbeat":
                    heartbeats.append(operation)
                return {
                    "session": "test",
                    "generation": 1,
                    "health": "",
                    "error": "",
                    "state": "RUNNING",
                    "weight": 1,
                    "remaining_segments": 0,
                    "body": [0.0] * 29,
                    "hands": [255.0] * 12,
                }

        task = Task(Client())
        checked = SceneCheckedTask(task, None, Config())
        checked.planner = BlockingPlanner()
        try:
            await checked.move(action="carry_path")
            self.assertGreaterEqual(len(heartbeats), 4)
            self.assertIsNone(task.error)
        finally:
            task.close()
            checked.close()


class InstalledSceneTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_spelling_phase_uses_one_executor_check_for_identical_scene(self):
        state = {"scene_id": "current", "transfer_path_version": 2}
        task = SimpleNamespace(status=lambda: state, move=AsyncMock())
        checked = SceneCheckedTask(task, None, Config(), isolated=False)
        checked.executor_scene_id = "current"
        checked.planner = SimpleNamespace(move=Mock(side_effect=AssertionError("duplicate planning")))
        for action in (
            "trial_joint_path",
            "trial_joint_sequence",
            "pinch_step",
            "carry_path",
            "transfer_path",
            "empty_pinch_path",
        ):
            await checked.move(action=action, obstacles=[[[0, 0, 0], [0.1, 0.1, 0.1]]])
            self.assertEqual(task.move.await_args.kwargs["expected_scene_id"], "current")
            self.assertIn("obstacles", task.move.await_args.kwargs)
        self.assertEqual(len(checked.checks), 6)
        self.assertTrue(all("elapsed_s" in item for item in checked.checks))
        state["scene_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "motion_scene_changed"):
            await checked.move(action="transfer_path")
        self.assertEqual(task.move.await_count, 6)
