"""Regression checks for the retained trajectory and contact behavior."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np

from astrabot.motion import Grasp
from astrabot.simulation import Sim


class Motion(unittest.IsolatedAsyncioTestCase):
    async def actions(self, continuous, delta=0.28, speed=1.0):
        sim = Sim(None, None)
        sim.speed = speed
        sim.names = ["joint"]
        sim.frame = {}
        sim.q = lambda: np.array([0.0])
        sim.kin = SimpleNamespace(by_name={"joint": {"lower": -1, "upper": 1}})
        sim.validate_frame = Mock()
        sim.record = Mock()
        sim.observation = lambda: {}
        sim.send = AsyncMock()

        async def receive(kind, **kwargs):
            if kind == "submit_actions_response":
                return {"ok": True, "accepted_count": len(sim.send.call_args.args[0]["actions"])}
            return {"frames": [{}]}

        sim.receive = receive
        await sim.move({"joint": delta}, continuous=continuous)
        return np.array(sim.send.call_args.args[0]["actions"])[:, 0], sim.motion_stats

    async def test_speed_preserves_endpoint_and_contact_settling(self):
        base, _ = await self.actions(False)
        for speed in (1.2, 1.5, 2.0, 2.5, 3.0):
            fast, stats = await self.actions(False, speed=speed)
            self.assertLess(len(fast), len(base))
            self.assertEqual(fast[-1], base[-1])
            self.assertGreaterEqual(stats["hold_frames"], 4)
        slow, _ = await self.actions(False, speed=0.5)
        self.assertGreater(len(slow), len(base))

    async def test_invalid_speed_and_large_joint_changes_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf"), True, "1.5", 4):
            with self.assertRaises(ValueError):
                await self.actions(True, speed=value)
        with self.assertRaises(ValueError):
            await self.actions(True, delta=0.36)

    async def test_transit_and_contact_keep_distinct_rates(self):
        transit, ts = await self.actions(True)
        contact, cs = await self.actions(False)
        self.assertLess(len(transit), len(contact))
        self.assertLessEqual(np.max(np.abs(np.diff(np.r_[0, transit]))), 0.0225 + 1e-12)
        self.assertLessEqual(np.max(np.abs(np.diff(np.r_[0, contact]))), 0.015 + 1e-12)
        self.assertEqual(ts["hold_frames"], 0)
        self.assertEqual(cs["hold_frames"], 10)

    async def test_release_finishes_opening_after_35mm_clearance(self):
        grasp = Grasp.__new__(Grasp)
        grasp.sim = SimpleNamespace(frame_count=1, move=AsyncMock())
        grasp.held_hand = {}
        grasp.center = AsyncMock()
        grasp.hand = AsyncMock()
        await grasp.release(np.array([-0.17, 1.66, 0.792]))
        self.assertEqual(grasp.sim.move.await_count, 3)
        np.testing.assert_allclose(grasp.center.call_args.args[0], [-0.17, 1.66, 0.827])
        grasp.hand.assert_awaited_once()

    async def test_reorientation_settles_only_final_waypoint(self):
        grasp = Grasp.__new__(Grasp)
        grasp.yaw = 30
        grasp.set_yaw = lambda angle: setattr(grasp, "yaw", angle)
        grasp.center = AsyncMock()
        await grasp.reorient([0, 0, 0], 90, [])
        self.assertEqual(
            [c.kwargs["settle"] for c in grasp.center.await_args_list], [False] * 5 + [True]
        )
