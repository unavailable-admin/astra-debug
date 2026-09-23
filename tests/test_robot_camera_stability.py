"""Bound camera settling and retain rejected exposures without using hardware."""

import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from astrabot.robot.camera import _wait_stable, capture
from astrabot.robot.config import ASSETS, Config, file_digest
from astrabot.robot.scene_builder import read_capture


def stationary_state():
    return {
        "session": "session",
        "generation": 4,
        "health": "",
        "error": "",
        "state": "RUNNING",
        "mode": 4,
        "planning_active": False,
        "remaining_segments": 0,
        "body": [0.0] * 29,
        "hands": [0.0] * 12,
    }


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def sleep(self, duration):
        self.now += duration


class CameraStabilityTests(unittest.TestCase):
    def test_drift_resets_the_quiet_window_then_settles(self):
        clock = FakeClock()
        initial = stationary_state()

        def status(_):
            result = copy.deepcopy(initial)
            result["body"][21] = min(clock.now - 100, 0.2) * 0.1
            return result

        client = Mock()
        client.request.side_effect = status
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("astrabot.robot.camera.time.monotonic", side_effect=lambda: clock.now),
                patch("astrabot.robot.camera.time.sleep", side_effect=clock.sleep),
            ):
                result = _wait_stable(Config(), client, initial, Path(directory))
            self.assertGreaterEqual(clock.now, 100.5)
            self.assertLess(clock.now, 100.7)
            self.assertAlmostEqual(result["body"][21], 0.02)
            self.assertGreater(len(json.loads((Path(directory) / "capture-settle.json").read_text())["states"]), 6)

    def test_continued_motion_times_out(self):
        clock = FakeClock()
        initial = stationary_state()

        def status(_):
            result = copy.deepcopy(initial)
            result["body"][21] = clock.now - 100
            return result

        client = Mock()
        client.request.side_effect = status
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("astrabot.robot.camera.time.monotonic", side_effect=lambda: clock.now),
                patch("astrabot.robot.camera.time.sleep", side_effect=clock.sleep),
                self.assertRaisesRegex(TimeoutError, "did_not_settle"),
            ):
                _wait_stable(Config(settle_timeout=0.5), client, initial, Path(directory))
            self.assertAlmostEqual(clock.now, 100.5)
            self.assertTrue((Path(directory) / "capture-settle.json").exists())

    def test_interruption_fault_or_motion_aborts_without_adopting_new_state(self):
        for change, error in (
            ({"generation": 5}, "invalidated_by_operator"),
            ({"session": "new"}, "invalidated_by_operator"),
            ({"health": "stale"}, "feedback_unhealthy"),
            ({"error": "fault"}, "feedback_unhealthy"),
            ({"remaining_segments": 1}, "stationary_capture_required"),
            ({"planning_active": True}, "stationary_capture_required"),
            ({"state": "RESETTING"}, "stationary_capture_required"),
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                client = Mock()
                client.request.return_value = stationary_state() | change
                with patch("astrabot.robot.camera.time.sleep"), self.assertRaisesRegex(ValueError, error):
                    _wait_stable(Config(), client, stationary_state(), Path(directory))
                client.request.assert_called_once_with("status")

    def test_exposure_rejection_preserves_snapshots_and_cannot_build_scene(self):
        for change, error in (
            ({"body": [0.0051] + [0.0] * 28}, "robot_moved_during_capture"),
            ({"body": [0.0] * 21 + [0.0032117730006575584] + [0.0] * 7}, None),
            ({"body": [0.005] + [0.0] * 28}, None),
            ({"hands": [3.0] + [0.0] * 11}, "robot_moved_during_capture"),
            ({"generation": 5}, "capture_invalidated_by_operator"),
            ({"state": "PLANNING"}, "stationary_capture_required"),
            ({"health": "stale"}, "capture_robot_feedback_unhealthy"),
            ({}, None),
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / f"capture-{file_digest(ASSETS / 'capture.cpp')[:16]}").touch()
                output = root / "capture"
                config = Config(camera_width=128, camera_height=64)
                before, after = stationary_state(), stationary_state() | change
                client = Mock()
                client.request.side_effect = [before, after]

                def helper(command, **kwargs):
                    cv2.imwrite(command[-2], np.zeros((64, 128, 3), np.uint8))
                    Path(command[-1]).write_text(json.dumps({"acquired_monotonic": 100.0, "width": 128, "height": 64}))

                with (
                    patch("astrabot.robot.camera.runtime_dir", return_value=root),
                    patch("astrabot.robot.camera.time.monotonic", return_value=100.0),
                    patch("astrabot.robot.camera._wait_stable", return_value=before),
                    patch("astrabot.robot.camera.subprocess.run", side_effect=helper),
                    self.assertRaisesRegex(ValueError, error) if error else nullcontext(),
                ):
                    capture(config, output, client)
                report = json.loads((output / "capture.json").read_text())
                if error is None:
                    self.assertTrue(report["capture_valid"])
                    loaded, _, _ = read_capture(output, config)
                    self.assertEqual(loaded["body_after"]["state"], "RUNNING")
                    for changed in ({"planning_active": True}, {"remaining_segments": 1}):
                        invalid = copy.deepcopy(report)
                        invalid["body_after"].update(changed)
                        (output / "capture.json").write_text(json.dumps(invalid))
                        with self.assertRaisesRegex(ValueError, "stationary_capture_required"):
                            read_capture(output, config)
                    (output / "capture.json").write_text(json.dumps(report))
                    for name in ("stereo.jpg", "left.jpg", "right.jpg"):
                        self.assertEqual(report["image_sha256"][name], file_digest(output / name))
                    continue
                self.assertFalse(report["capture_valid"])
                self.assertIn(error, report["validation_error"])
                self.assertEqual(report["body_before"], before)
                self.assertEqual(report["body_after"], after)
                self.assertTrue((output / "stereo.jpg").exists())
                with self.assertRaisesRegex(ValueError, "capture_rejected"):
                    read_capture(output, config)


if __name__ == "__main__":
    unittest.main()
