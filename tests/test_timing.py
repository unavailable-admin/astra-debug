"""No-recording feedback and timing boundaries, without a live robot."""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from PIL import Image

from astrabot.kinematics import Kinematics
from astrabot.simulation import Sim
from astrabot.timing import Timings, server_frame_span


class Feedback(unittest.TestCase):
    def test_motion_feedback_never_decodes_or_writes_images(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(None, Path(directory))
            sim.validate_frame = Mock()
            with patch("astrabot.simulation.image_bytes", side_effect=AssertionError("Decoded")):
                for i in range(30):
                    frame = {"image": {"data": "not an image"}, "sim_time": i / 30}
                    sim.record(frame)
            self.assertEqual(sim.frame_count, 30)
            self.assertIs(sim.frame, frame)
            self.assertEqual(sim.validate_frame.call_count, 30)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_observation_saves_only_the_latest_synchronized_pair(self):
        def image_spec(color):
            data = io.BytesIO()
            Image.new("RGB", (16, 16), color).save(data, format="PNG")
            return {"encoding": "png_base64", "data": base64.b64encode(data.getvalue()).decode()}

        with tempfile.TemporaryDirectory() as directory:
            sim = Sim(None, Path(directory))
            sim.validate_frame = Mock()
            for i in range(3):
                sim.record({"sim_time": i, "image": image_spec("red"),
                            "images": {"right": image_spec("blue")}})
            left = sim.save_observation()
            right = left.with_name(left.stem + "_right.jpg")
            self.assertEqual(left.name, "observation_0002.jpg")
            self.assertEqual(set(Path(directory).iterdir()), {left, right})
            with Image.open(left) as im:
                self.assertGreater(im.getpixel((0, 0))[0], 240)
            with Image.open(right) as im:
                self.assertGreater(im.getpixel((0, 0))[2], 240)
            self.assertEqual(sim.timings.stages["observation_write"]["count"], 1)


class Measurement(unittest.IsolatedAsyncioTestCase):
    async def test_batch_timing_keeps_server_clock_and_simulation_time_separate(self):
        clock = SimpleNamespace(now=5000.0)
        queue = []

        async def send(raw):
            clock.now += 1
            count = len(json.loads(raw)["actions"])
            queue.extend([
                {"type": "submit_actions_response", "ok": True, "accepted_count": count},
                {"type": "step_result", "frames": [
                    {"sim_time": (i + 1) / 30, "wall_time": 100 + i * 0.1}
                    for i in range(count)
                ]},
            ])

        async def receive():
            response = queue.pop(0)
            clock.now += 2 if response["type"] == "submit_actions_response" else 5
            return json.dumps(response)

        with tempfile.TemporaryDirectory() as directory, patch(
            "astrabot.timing.time.perf_counter", side_effect=lambda: clock.now
        ):
            ws = SimpleNamespace(send=send, recv=receive)
            sim = Sim(ws, Path(directory))
            sim.names = ["joint"]
            sim.kin = SimpleNamespace(by_name={"joint": {"lower": -1, "upper": 1}})
            sim.q = lambda: np.array([0.0])
            sim.validate_frame = Mock()
            sim.observation = lambda: {}
            sim.record({"sim_time": 0})
            await sim.move({"joint": 0.28}, continuous=True)
            entry = json.loads((Path(directory) / "motion.jsonl").read_text())
            t = entry["timing"]
            self.assertEqual(t["round_trip_seconds"], 8)
            self.assertEqual(t["send_seconds"], 1)
            self.assertEqual(t["ack_wait_seconds"], 2)
            self.assertEqual(t["result_wait_seconds"], 5)
            self.assertAlmostEqual(t["simulation_seconds"], entry["frames"] / 30)
            self.assertAlmostEqual(t["server_frame_span_seconds"], (entry["frames"] - 1) * .1)
            self.assertIsNone(t["network_transfer_seconds"])
            self.assertIsNone(t["server_execution_seconds"])
            self.assertGreater(t["result_message"]["message_bytes"], 0)
            self.assertEqual(sim.timings.stages["ws_receive_wait"]["seconds"], 7)
            self.assertEqual(sim.validate_frame.call_count, entry["frames"] + 2)
            self.assertEqual([f.name for f in Path(directory).iterdir()], ["motion.jsonl"])

    async def test_receive_failure_is_timed_and_propagated(self):
        sim = Sim(SimpleNamespace(recv=AsyncMock(side_effect=TimeoutError)), None)
        with self.assertRaises(TimeoutError):
            await sim.receive("step_result")
        self.assertEqual(sim.timings.stages["ws_receive_wait"]["failures"], 1)

    def test_missing_or_invalid_server_timestamps_remain_unknown(self):
        for frames in ([], [{}], [{}, {}], [{"wall_time": 1}, {"wall_time": 0}],
                       [{"wall_time": 1}, {"wall_time": float("nan")}],
                       [{"wall_time": True}, {"wall_time": 1}]):
            self.assertIsNone(server_frame_span(frames))

    def test_ik_failure_is_counted_without_suppressing_the_error(self):
        kin = Kinematics([0, 0, 0], [1, 0, 0, 0])
        q = {n: 0.0 for n in kin.by_name}
        with patch("astrabot.kinematics.least_squares", side_effect=ValueError("IK failure")):
            with self.assertRaisesRegex(ValueError, "IK failure"):
                kin.solve(q, "left", np.zeros(3))
        self.assertEqual(kin.timings.stages["ik_solve"]["count"], 1)
        self.assertEqual(kin.timings.stages["ik_solve"]["failures"], 1)

    def test_snapshots_do_not_mutate_as_more_samples_arrive(self):
        timings = Timings()
        timings.add("ik", 1)
        before = timings.snapshot()
        timings.add("ik", 2)
        self.assertEqual(before["stages"]["ik"]["seconds"], 1)
        self.assertEqual(timings.snapshot()["stages"]["ik"]["seconds"], 3)
