"""Preserve actionable capture failures without changing freshness checks."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from astrabot.robot.camera import capture
from astrabot.robot.config import ASSETS, Config, file_digest


class CaptureErrorTests(unittest.TestCase):
    def test_helper_failure_and_timeout_are_saved_and_surfaced(self):
        cases = (
            (subprocess.CalledProcessError(1, ["helper"], stderr="fresh frame timeout\n"), "fresh frame timeout"),
            (subprocess.CalledProcessError(1, ["helper"], stderr="Permission denied\n"), "Permission denied"),
            (subprocess.CalledProcessError(1, ["helper"]), "exited with code 1"),
            (subprocess.TimeoutExpired(["helper"], 10, stderr=b"USB timeout"), "USB timeout"),
            (subprocess.TimeoutExpired(["helper"], 10), "timed out without an error message"),
        )
        for error, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                executable = root / f"capture-{file_digest(ASSETS / 'capture.cpp')[:16]}"
                executable.touch()
                output = root / "capture"
                client = Mock()
                with (
                    patch("astrabot.robot.camera.runtime_dir", return_value=root),
                    patch("astrabot.robot.camera._wait_stable", side_effect=lambda cfg, client, state, path: state),
                    patch("astrabot.robot.camera.subprocess.run", side_effect=error),
                    self.assertRaisesRegex(RuntimeError, expected),
                ):
                    capture(Config(), output, client)
                client.request.assert_called_once_with("status")
                report = json.loads((output / "capture-error.json").read_text())
                self.assertIn(expected, report["error"])
                self.assertEqual(report["device"], "/dev/video0")
                self.assertFalse((output / "capture.json").exists())
                # A retry must not overwrite the failed attempt's evidence.
                with self.assertRaises(FileExistsError):
                    capture(Config(), output, client)


if __name__ == "__main__":
    unittest.main()
