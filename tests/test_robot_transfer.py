"""Transfer endpoints, retained table height and interruption; no robot SDK."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from astrabot.robot.config import Config
from astrabot.robot.transfer import execute_transfer, front_slots, placement_geometry


class PlacementTests(unittest.TestCase):
    def test_robot_left_to_right_and_common_height_on_tilted_plane(self):
        config = Config(
            table_min=[0.1, -0.7, -0.8],
            table_max=[0.9, 0.7, 0.1],
            table_plane=[0.12, 0.01, -0.02],
            table_footprint=[[0.16, -0.6], [0.8, -0.6], [0.8, 0.6], [0.16, 0.6]],
        )
        slots = front_slots(config, 3)
        np.testing.assert_allclose(np.diff(np.array(slots)[:, 1]), [-0.12, -0.12])
        np.testing.assert_allclose(np.array(slots)[:, 0], [0.22] * 3)
        first = placement_geometry(config, np.array([0.4, 0.1, 0.065]), {"xy": slots[0], "height_above_table": None})
        second = placement_geometry(
            config, np.array([0.5, 0.2, 0.09]), {"xy": slots[1], "height_above_table": first[3]}
        )
        self.assertEqual(first[3], second[3])
        self.assertAlmostEqual(second[0][2] - first[0][2], -0.12 * 0.01)
        self.assertGreaterEqual(first[1][2] - first[0][2], config.trial_lift_height_m)


class TransferTests(unittest.IsolatedAsyncioTestCase):
    async def setup_transfer(self, fail_translate=False, measured_grip=False, release_rotation=None):
        config = Config(trial_grip_mode="supervised_position", trial_lift_height_m=0.12)
        hand = np.tile(config.trial_hand("open"), 2)
        hand[3:5] -= 4
        state = {"body": np.zeros(29), "hands": hand.copy(), "hands_command": hand.copy(), "arms_command": np.zeros(14)}
        requests = []

        async def move(**values):
            requests.append(values)
            if fail_translate and len(requests) == 2:
                raise RuntimeError("task_generation_changed")
            state["hands"] = np.array(values["hands"])

        current_rotation = [np.eye(3)]

        async def open_hand(goal, hands, **values):
            state["hands"] = hands.copy()
            if release_rotation is not None:
                current_rotation[0] = release_rotation

        trial = SimpleNamespace(
            config=config,
            model=None,
            task=SimpleNamespace(status=lambda: state, move=move),
            move_center=open_hand,
            joint_phase=AsyncMock(),
            events=[],
        )
        plan = {
            "placement": {"xy": [0.34, 0.16], "height_above_table": None},
            "phases": [{"phase": "retract", "pinch_center_m": [0.4, 0.05, 0.12]}, {"phase": "return_ready"}],
        }
        with patch(
            "astrabot.robot.transfer.pinch_pose",
            side_effect=lambda *_: (np.array([0.4, 0.05, 0.06]), current_rotation[0]),
        ):
            if fail_translate:
                with self.assertRaisesRegex(RuntimeError, "generation_changed"):
                    await execute_transfer(trial, plan, np.array([0.4, 0.05, 0.06]), hand, [])
                return trial, requests, None
            result = await execute_transfer(
                trial, plan, np.array([0.4, 0.05, 0.06]), None if measured_grip else hand, []
            )
        return trial, requests, result

    async def test_transfer_releases_at_destination_and_records_assumption(self):
        trial, requests, result = await self.setup_transfer()
        self.assertEqual([r["holding"] for r in requests], [True, True, True, False, False])
        np.testing.assert_allclose(requests[2]["goal_center"][:2], [0.34, 0.16])
        self.assertFalse(requests[3]["obstacles"])  # Existing contact exclusion until clear.
        self.assertEqual(len(requests[4]["obstacles"]), 1)
        self.assertTrue(result["motion_completed"])
        self.assertFalse(result["success_verified"])
        trial.joint_phase.assert_awaited_once()

    async def test_interrupt_during_translation_never_releases_or_continues(self):
        trial, requests, _ = await self.setup_transfer(fail_translate=True)
        self.assertEqual(len(requests), 2)
        trial.joint_phase.assert_not_awaited()

    async def test_feedback_grip_without_position_command_uses_measured_hands(self):
        _, requests, result = await self.setup_transfer(measured_grip=True)
        self.assertEqual(len(requests[0]["hands"]), 12)
        self.assertTrue(result["motion_completed"])

    async def test_empty_retreat_preserves_post_release_measured_orientation(self):
        rotation = Rotation.from_rotvec([0, 0.064108, 0]).as_matrix()
        _, requests, result = await self.setup_transfer(release_rotation=rotation)
        for request in requests[:3]:
            np.testing.assert_allclose(request["rotation"], np.eye(3))
        for request in requests[3:]:
            np.testing.assert_allclose(request["rotation"], rotation)
        self.assertTrue(result["motion_completed"])
