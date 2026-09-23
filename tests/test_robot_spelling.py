"""Queue order and cancellation, independent of APIs and hardware."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from astrabot.robot.config import Config
from astrabot.robot.spelling import run


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def run_queue(self, interrupt=False):
        config = Config()
        state = {
            "session": "test",
            "generation": 1,
            "attached": True,
            "weight": 1,
            "body": [0.0] * 15 + list(config.ready_arms),
            "hands": list(config.ready_hands),
            "transfer_path_version": 1,
        }
        record = {"estimated_xyz": [0.4, 0.1, 0.02], "grasp_center_xyz": [0.4, 0.1, 0.035]}
        bundle = SimpleNamespace(
            config=config,
            directory=Path("first-scene"),
            digest="test-scene",
            letters={"A": record},
            identity_image=Path("left.jpg"),
            source={"grasp_center_m": record["grasp_center_xyz"]},
        )
        identified, planned = [], []

        class Client:
            def __init__(self, *args):
                pass

            def request(self, *args, **kwargs):
                return state.copy()

        class Task:
            def __init__(self, client, expected):
                state["generation"] += 1

            def status(self):
                return state.copy()

            def close(self):
                return state.copy()

        class Checked:
            def __init__(self, *args):
                self.pool = ThreadPoolExecutor(max_workers=1)
                self.checks = []

            def close(self):
                self.pool.shutdown()

        async def refresh(args, client, owned):
            owned["generation"] += 1
            state["generation"] += 1

        def locate(config, directory, letter, placed, timeout):
            identified.append((letter, [p["letter"] for p in placed]))
            return record, Path("left.jpg")

        def plan(config, source, state, placement, placed):
            planned.append((placement, [p["letter"] for p in placed]))
            return {"placement": placement}

        class Trial:
            def __init__(self, *args):
                self.events = []

            async def run(self, plan):
                if interrupt:
                    raise RuntimeError("task_generation_changed")
                index = len(planned) - 1
                return {
                    "height_above_table": 0.04,
                    "cube_center": [0.34, 0.16 - index * 0.08, 0.02],
                    "obstacle": [[0.3, 0, 0], [0.4, 0.2, 0.04]],
                    "motion_completed": True,
                }

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                word="ACE", execute=True, config="unused", socket=None, output=Path(directory), api_timeout=120
            )
            report = {}
            with (
                patch("astrabot.robot.spelling.Config.load", return_value=config),
                patch("astrabot.robot.spelling.Client", Client),
                patch(
                    "astrabot.robot.spelling.inspect",
                    return_value={"startup_checks_passed": True, "snapshot": state.copy()},
                ),
                patch("astrabot.robot.spelling.refresh_scene", refresh),
                patch("astrabot.robot.spelling.SceneBundle.load", return_value=bundle),
                patch("astrabot.robot.spelling.Task", Task),
                patch("astrabot.robot.spelling.SceneCheckedTask", Checked),
                patch("astrabot.robot.spelling.capture"),
                patch("astrabot.robot.spelling.current_target", side_effect=locate),
                patch("astrabot.robot.spelling.make_plan", side_effect=plan),
                patch("astrabot.robot.spelling.TrialObserver", return_value=AsyncMock()),
                patch("astrabot.robot.spelling.SingleTrial", Trial),
            ):
                if interrupt:
                    with self.assertRaisesRegex(RuntimeError, "generation_changed"):
                        await run(args, report)
                else:
                    await run(args, report)
        return report, identified, planned

    async def test_ace_observes_only_next_letter_and_retains_placements(self):
        report, identified, planned = await self.run_queue()
        self.assertEqual(identified, [("C", ["A"]), ("E", ["A", "C"])])
        self.assertEqual([p[1] for p in planned], [[], ["A"], ["A", "C"]])
        self.assertIsNone(planned[0][0]["height_above_table"])
        self.assertEqual(planned[1][0]["height_above_table"], 0.04)
        self.assertTrue(report["motion_completed"])
        self.assertFalse(report["success_verified"])
        self.assertEqual([p["letter"] for p in report["placements"]], list("ACE"))
        np.testing.assert_allclose(np.diff(np.array(report["slots"])[:, 1]), [-0.12, -0.12])

    async def test_interruption_stops_before_next_identification(self):
        report, identified, planned = await self.run_queue(interrupt=True)
        self.assertFalse(identified)
        self.assertEqual(len(planned), 1)
        self.assertFalse(report["motion_completed"])
        self.assertEqual(report["placements"], [])


class PublicInterfaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_word_argument_is_normalized_and_persisted(self):
        import json

        from astrabot.robot.spelling import spell

        for word in ("ACE", " cat ", "AA"):
            with self.subTest(word=word), tempfile.TemporaryDirectory() as directory:
                with patch("astrabot.robot.spelling.run", new_callable=AsyncMock) as runner:
                    report = await spell(word, config="config.json", scene="scene", output=Path(directory) / "run")
                args = runner.await_args.args[0]
                self.assertEqual(args.word, word.strip().upper())
                self.assertFalse(args.execute)
                self.assertEqual(json.loads(Path(report["report"]).read_text())["word"], args.word)
                self.assertFalse(report["success_verified"])

    async def test_invalid_input_never_starts_queue(self):
        from astrabot.robot.spelling import spell

        with patch("astrabot.robot.spelling.run", new_callable=AsyncMock) as runner:
            for word in ("", "A C", "A1", "字母"):
                with self.assertRaises(ValueError):
                    await spell(word, config="config.json", execute=True)
            with self.assertRaisesRegex(ValueError, "requires scene"):
                await spell("ACE", config="config.json")
            runner.assert_not_awaited()

    async def test_failure_is_reported_without_claiming_success(self):
        from astrabot.robot.spelling import spell

        with tempfile.TemporaryDirectory() as directory:
            with patch("astrabot.robot.spelling.run", new_callable=AsyncMock, side_effect=RuntimeError("interrupted")):
                report = await spell("CAT", config="config.json", execute=True, output=Path(directory) / "run")
            self.assertEqual(report["error"], "RuntimeError:interrupted")
            self.assertFalse(report["motion_completed"])
            self.assertFalse(report["success_verified"])

    async def test_caller_cancellation_propagates_and_persists(self):
        import asyncio
        import json

        from astrabot.robot.spelling import spell

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch("astrabot.robot.spelling.run", new_callable=AsyncMock, side_effect=asyncio.CancelledError),
                self.assertRaises(asyncio.CancelledError),
            ):
                await spell("ACE", config="config.json", execute=True, output=output)
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["motion_completed"])
            self.assertEqual(report["error"], "CancelledError:caller_cancelled")


class TransferPlanningTests(unittest.TestCase):
    def test_orientation_selection_includes_transfer_and_keeps_placed_obstacles(self):
        from astrabot.robot.spelling import make_plan

        placed = [{"obstacle": [[0.3, 0.2, 0.0], [0.4, 0.3, 0.05]]}]
        with (
            patch("astrabot.robot.spelling.RobotModel"),
            patch("astrabot.robot.spelling.plan_trial", side_effect=[{}, {}]) as grasp,
            patch("astrabot.robot.spelling.preflight_transfer", side_effect=[ValueError("arm_torso:left"), []]),
        ):
            plan = make_plan(Config(), {"parameters": {}}, {}, {"xy": [0.38, 0.16]}, placed)
        self.assertEqual(grasp.call_count, 2)
        self.assertEqual(grasp.call_args.args[0].source["parameters"]["contact_yaw"], -15)
        self.assertEqual(grasp.call_args.args[0].source["parameters"]["obstacles"], [placed[0]["obstacle"]])
        self.assertTrue(plan["transfer_orientation_attempts"][-1]["geometry_passed"])


class CurrentTargetTests(unittest.TestCase):
    def test_repeat_letter_excludes_placed_cube_and_measures_only_new_target(self):
        from astrabot.robot.spelling import current_target

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        decision = {
            "target_letter": "A",
            "target_id": 2,
            "target_confidence": 0.99,
            "target_verified": True,
            "target_upright": True,
        }
        candidates = [{"id": 1, "pixel": [10, 10]}, {"id": 2, "pixel": [80, 80]}]
        measured = {"letters": {"A": {"grasp_center_xyz": [0.4, 0.1, 0.05]}}}
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "astrabot.robot.spelling.read_capture", return_value=({"body_after": {"body": []}}, image, image)
                ),
                patch("astrabot.robot.spelling.load", return_value=(None, None, None)),
                patch("astrabot.robot.spelling.discover", return_value=candidates),
                patch("astrabot.robot.spelling.project", return_value=np.array([[0, 0], [20, 20]])),
                patch("astrabot.robot.spelling.AstraVision") as vision,
                patch("astrabot.robot.spelling.candidate_images", return_value=[]) as images,
                patch("astrabot.robot.spelling.StereoTracker") as tracker,
                patch("astrabot.robot.spelling.known_glyph_mask", return_value=None),
            ):
                vision.return_value._call.return_value = decision
                tracker.return_value.locate_images.return_value = measured
                record, _ = current_target(Config(), Path(directory), "A", [{"cube_center": [0.38, 0.4, 0.02]}], 120)
            self.assertEqual(images.call_args.args[0].candidates, [candidates[1]])
            self.assertEqual(tracker.return_value.locate_images.call_args.args[2], {"A": [80, 80]})
            self.assertEqual(record["workspace_policy"], "operator_cleared")
            self.assertIn("Inspect only this target", vision.return_value._call.call_args.args[1])
