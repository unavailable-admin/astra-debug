"""Offline decision-gate checks; these do not claim robot/API success."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError

import numpy as np
from scipy.spatial.transform import Rotation

from astrabot.decisions import observe_scene, select_action
from astrabot.kinematics import Kinematics
from astrabot.simulation import Sim, SimulationResetError
from astrabot.vision import identified_pixels, parse_json

TARGETS = {"H": {"target_x": -0.15, "target_y": 1.66}, "I": {"target_x": -0.074, "target_y": 1.66}}
AT_TARGET = {
    letter: {"estimated_xyz": [p["target_x"], p["target_y"], 0.777]}
    for (letter, p) in TARGETS.items()
}


class Gates(unittest.TestCase):
    def test_api_pause_checks_live_step_without_a_second_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(None, Path(directory))
            sim.last_step = 55
            status = {
                "scene_id": "showroom_scene_11",
                "action_layout": "full36",
                "action_joint_names": [],
                "showroom_scene_11_robot_pose": {
                    "actual_pos": [0, 1.5, 0.76],
                    "actual_quat_wxyz": [1, 0, 0, 0],
                },
                "step": 55,
                "is_executing": False,
                "queue_length": 0,
            }
            sim.status = status
            sim.send = AsyncMock()
            sim.receive = AsyncMock(return_value=status)
            asyncio.run(sim.check_live_status())
            for step in (0, 56):
                sim.receive = AsyncMock(return_value={**status, "step": step})
                with self.assertRaises(SimulationResetError):
                    asyncio.run(sim.check_live_status())

    def test_live_feedback_detects_root_translation_and_rotation(self):
        sim = Sim(None, None)
        sim.kin = Kinematics([0, 0, 0], [1, 0, 0, 0])
        sim.status = {"full_body_joint_names": list(sim.kin.by_name)}
        state = {"joint_position": [0.0] * len(sim.status["full_body_joint_names"])}
        for side in ("left", "right"):
            pose = sim.kin.fk(
                dict.fromkeys(sim.status["full_body_joint_names"], 0.0), side + "_wrist_yaw_link"
            )
            state[side + "_eef_pos"] = pose[:3, 3].tolist()
            state[side + "_eef_quat"] = (
                Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist()
            )
        frame = {"state": state, "latest_step": 55}
        sim.validate_frame(frame)
        state["left_eef_pos"][1] -= 0.1
        with self.assertRaises(SimulationResetError):
            sim.validate_frame(frame)
        state["left_eef_pos"][1] += 0.1
        old = Rotation.from_quat(np.array(state["left_eef_quat"])[[1, 2, 3, 0]])
        state["left_eef_quat"] = (
            (Rotation.from_euler("z", 0.1) * old).as_quat()[[3, 0, 1, 2]].tolist()
        )
        with self.assertRaises(SimulationResetError):
            sim.validate_frame(frame)

    def test_finish_needs_all_measured_letters(self):
        decision = {"next_action": {"action": "finish"}, "row_complete": True}
        self.assertEqual(select_action(decision, AT_TARGET, TARGETS)[0], "finish")
        self.assertEqual(select_action(decision, {"H": AT_TARGET["H"]}, TARGETS)[0], "stop")
        displaced = {**AT_TARGET, "I": {"estimated_xyz": [-0.074, 1.76, 0.777]}}
        self.assertEqual(select_action(decision, displaced, TARGETS)[0], "stop")

    def test_model_completion_does_not_override_numeric_failure(self):
        bad = {**AT_TARGET, "H": {"estimated_xyz": [float("nan"), 1.66, 0.777]}}
        self.assertEqual(
            select_action(
                {"next_action": {"action": "finish"}, "row_complete": True}, bad, TARGETS
            )[0],
            "stop",
        )

    def test_pick_requires_valid_workspace_and_unplaced_letter(self):
        decision = {"next_action": {"action": "pick_place", "letter": "I"}}
        self.assertEqual(select_action(decision, AT_TARGET, TARGETS)[0], "stop")
        unplaced = {"I": {"estimated_xyz": [-0.074, 1.76, 0.777]}}
        self.assertEqual(select_action(decision, unplaced, TARGETS)[:2], ("pick_place", "I"))
        for xyz in [[-0.074, 2.5, 0.777], [float("nan"), 1.76, 0.777]]:
            self.assertEqual(
                select_action(decision, {"I": {"estimated_xyz": xyz}}, TARGETS)[0], "stop"
            )

    def test_no_arbitrary_action_execution(self):
        for action in [
            {"action": "shell", "letter": "I"},
            {"action": "pick_place", "letter": "C"},
            "pick_place",
        ]:
            self.assertEqual(select_action({"next_action": action}, AT_TARGET, TARGETS)[0], "stop")

    def test_response_is_json_not_executable_text(self):
        self.assertEqual(parse_json('```json\n{"state":"uncertain"}\n```'), {"state": "uncertain"})
        for text in ["[]", 'run_shell("echo wrong")', '{"state":']:
            with self.assertRaises(ValueError):
                parse_json(text)

    def test_candidate_ids_use_measured_pixels_without_axis_conversion(self):
        decision = {
            "coordinate_mode": "candidate_id",
            "candidate_map": {"3": {"pixel": [636.0, 342.5]}},
            "letters": [
                {
                    "letter": "A",
                    "candidate_id": 3,
                    "upright": True,
                    "confidence": 0.99,
                    "u": 0.5,
                    "v": 0.4,
                }
            ],
        }
        self.assertEqual(
            identified_pixels(decision, (720, 1280, 3), word="ACE"), {"A": [636.0, 342.5]}
        )
        for value in [999, "3", True]:
            decision["letters"][0]["candidate_id"] = value
            self.assertEqual(identified_pixels(decision, (720, 1280, 3), word="ACE"), {})

    def test_ace_source_workspace_includes_back_row_but_not_off_table(self):
        targets = {
            letter: {"target_x": x, "target_y": 1.66}
            for (letter, x) in zip("ACE", [-0.26, -0.17, -0.08])
        }
        decision = {"next_action": {"action": "pick_place", "letter": "A"}}
        self.assertEqual(
            select_action(decision, {"A": {"estimated_xyz": [0, 1.903, 0.777]}}, targets)[0],
            "pick_place",
        )
        self.assertEqual(
            select_action(decision, {"A": {"estimated_xyz": [0.5, 1.903, 0.777]}}, targets)[0],
            "stop",
        )


class ApiRetries(unittest.IsolatedAsyncioTestCase):
    async def test_recovers_on_tenth_attempt_with_capped_delays(self):
        failure = HTTPError("internal", 502, "Bad Gateway", {}, None)
        api = SimpleNamespace(
            scene_candidates=AsyncMock(side_effect=[failure] * 9 + [{"letters": []}])
        )
        sim = SimpleNamespace(check_live_status=AsyncMock())
        with patch("astrabot.decisions.asyncio.sleep", new=AsyncMock()) as sleep:
            self.assertEqual(await observe_scene(api, sim, "frame", "ACE"), {"letters": []})
        self.assertEqual(api.scene_candidates.await_count, 10)
        self.assertEqual(sim.check_live_status.await_count, 10)
        self.assertEqual(
            [call.args[0] for call in sleep.await_args_list], [2, 4, 8, 16, 30, 30, 30, 30, 30]
        )

    async def test_configured_limit_stops_without_final_sleep(self):
        api = SimpleNamespace(scene_candidates=AsyncMock(side_effect=TimeoutError()))
        sim = SimpleNamespace(check_live_status=AsyncMock())
        with patch("astrabot.decisions.asyncio.sleep", new=AsyncMock()) as sleep:
            with self.assertRaises(TimeoutError):
                await observe_scene(api, sim, "frame", "ACE", max_attempts=20)
        self.assertEqual(api.scene_candidates.await_count, 20)
        self.assertEqual(sleep.await_count, 19)

    async def test_reset_and_nonretryable_errors_stop_immediately(self):
        api = SimpleNamespace(
            scene_candidates=AsyncMock(
                side_effect=HTTPError("internal", 401, "Unauthorized", {}, None)
            )
        )
        sim = SimpleNamespace(check_live_status=AsyncMock())
        with self.assertRaises(HTTPError):
            await observe_scene(api, sim, "frame", "ACE")
        self.assertEqual(api.scene_candidates.await_count, 1)
        api.scene_candidates.reset_mock()
        sim.check_live_status = AsyncMock(side_effect=SimulationResetError("root moved"))
        with self.assertRaises(SimulationResetError):
            await observe_scene(api, sim, "frame", "ACE")
        api.scene_candidates.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
