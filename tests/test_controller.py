"""Mocked closed-loop orchestration: no API calls or robot connection."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from astrabot import controller
from astrabot.reset import change, reset


class Controller(unittest.IsolatedAsyncioTestCase):
    async def test_ace_reobserves_after_each_placement_without_historical_files(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new_parent" / "episode"
            targets = controller.targets_for("ACE")

            def estimates(placed):
                return {
                    letter: {
                        "estimated_xyz": [
                            t["target_x"],
                            t["target_y"] if i < placed else 1.85,
                            0.78,
                        ],
                        "grasp_center_xyz": [
                            t["target_x"],
                            t["target_y"] if i < placed else 1.85,
                            0.795,
                        ],
                    }
                    for i, (letter, t) in enumerate(targets.items())
                }

            observations = [
                {"next_action": {"action": "pick_place", "letter": letter}} for letter in "ACE"
            ] + [{"next_action": {"action": "finish"}, "row_complete": True}]
            api = SimpleNamespace(scene_candidates=AsyncMock(side_effect=observations))
            pose = np.eye(4)
            pose[:3, 3] = [0.0, 1.6, 1.1]
            sim = SimpleNamespace(
                names=[],
                frame_count=1,
                save_observation=Mock(side_effect=lambda: out / f"observation_{sim.frame_count - 1:04d}.jpg"),
                actions_sent=0,
                initial_sim_time=0,
                frame={"sim_time": 0, "dt": 1 / 30},
                status={"action_layout": "full36"},
                full_q=lambda: {},
                kin=SimpleNamespace(fk=lambda *a: pose, by_child={"lh_thumb_distal": {}}),
                start=AsyncMock(),
                send=AsyncMock(),
                receive=AsyncMock(return_value={}),
                check_live_status=AsyncMock(),
                move_wrist=AsyncMock(return_value={"position_error_m": 0}),
            )
            tracker = SimpleNamespace(
                locate=Mock(
                    side_effect=[
                        {"ok": True, "letters": estimates(i), "table_z": 0.76} for i in range(4)
                    ]
                )
            )

            async def execute(*args):
                sim.actions_sent += 30
                sim.frame_count += 30
                sim.frame["sim_time"] += 1

            skill = AsyncMock(side_effect=execute)
            socket = AsyncMock()
            args = SimpleNamespace(
                word="ACE",
                start_x=-0.26,
                speed=1.2,
                output=out,
                api_timeout=120,
                uri="ws://unused",
                max_skills=3,
                api_max_attempts=10,
                inspect_only=False,
            )
            with (
                patch.object(controller, "Sim", return_value=sim),
                patch.object(controller, "AstraVision", return_value=api),
                patch.object(controller, "StereoTracker", return_value=tracker),
                patch.object(controller, "Grasp", return_value=SimpleNamespace()),
                patch.object(controller, "validate_status"),
                patch.object(controller, "identified_pixels", return_value={"A": [10, 10]}),
                patch.object(controller.cv2, "imread", return_value=np.zeros((480, 640, 3))),
                patch.object(controller.websockets, "connect", return_value=socket),
                patch.object(controller, "execute_pick_place", skill),
            ):
                result = await controller.run(args)
            self.assertTrue(result["success_verified"])
            self.assertFalse(result["recording_enabled"])
            self.assertNotIn("video", result)
            self.assertEqual(sim.save_observation.call_count, 4)
            self.assertEqual(result["timing"]["stages"]["api_observation"]["count"], 4)
            self.assertGreater(result["wall_seconds"], 0)
            self.assertFalse(list(out.glob("*.mp4")))
            self.assertEqual(api.scene_candidates.await_count, 4)
            self.assertEqual([c.args[2] for c in skill.await_args_list], list("ACE"))
            self.assertEqual(result["simulation_seconds"], 3)
            self.assertEqual(json.loads((out / "report.json").read_text())["skills_completed"], 3)
            self.assertFalse((Path(directory) / "runs").exists())

    async def test_invalid_word_or_speed_does_not_create_output_or_connect(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "episode"
            with patch.object(controller.websockets, "connect") as connect:
                for word, speed in [("BOOK", 1), ("ACE", float("nan")), ("ACE", 4)]:
                    args = SimpleNamespace(word=word, start_x=-0.26, speed=speed, output=out)
                    with self.assertRaises(ValueError):
                        await controller.run(args)
                connect.assert_not_called()
                self.assertFalse(out.exists())


class Reset(unittest.IsolatedAsyncioTestCase):
    async def test_reset_uses_requested_uri_and_writes_new_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new" / "reset.json"
            status = {"action_layout": "full36"}
            with patch("astrabot.reset.change", new=AsyncMock(side_effect=[{}, status])) as change:
                await reset("ws://unused:8082", output)
            self.assertEqual(
                [c.args for c in change.await_args_list],
                [("arm26", "ws://unused:8082"), ("full36", "ws://unused:8082")],
            )
            self.assertEqual(json.loads(output.read_text()), status)

    async def test_occupied_worker_does_not_switch_layout(self):
        ws = AsyncMock()
        ws.recv.return_value = json.dumps({"type": "status_response", "is_executing": True})
        connection = AsyncMock()
        connection.__aenter__.return_value = ws
        with patch("astrabot.reset.websockets.connect", return_value=connection):
            with self.assertRaises(RuntimeError):
                await change("arm26", "ws://unused")
        self.assertEqual(ws.send.await_count, 1)
