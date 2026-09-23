"""Host launcher dispatch tests using a fake docker binary; no devices or containers."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


class OperatorLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "docker.jsonl"
        docker = self.root / "docker"
        docker.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            'with open(os.environ["DOCKER_TEST_LOG"], "a") as stream:\n'
            '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
            'if sys.argv[-1] == "status":\n'
            '    sys.exit(0 if os.environ.get("DOCKER_TEST_RUNNING") == "1" else 1)\n'
        )
        docker.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.root}:{os.environ['PATH']}", DOCKER_TEST_LOG=str(self.log))
        self.env["ASTRA_CONFIG"] = "/tmp/operator config with spaces.json"
        self.env["ASTRA_OPERATOR_CLEARED_WORKSPACE"] = "1"
        self.script = Path(__file__).resolve().parents[1] / "scripts/robot_operator.sh"

    def invoke(self, *arguments):
        return subprocess.run(
            ["bash", str(self.script), *arguments], env=self.env, capture_output=True, text=True, check=False
        )

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_start_configures_access_and_uses_one_config_argument(self):
        self.assertEqual(self.invoke("start").returncode, 0)
        commands = self.commands()
        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[0][-1], "status")
        self.assertIn("root", commands[1])
        self.assertTrue(any(value.endswith("robot_device_access.sh") for value in commands[1]))
        self.assertEqual(commands[2][-3:], ["serve", "--config", self.env["ASTRA_CONFIG"]])

    def test_live_executor_is_not_restarted_or_reconfigured(self):
        self.env["DOCKER_TEST_RUNNING"] = "1"
        self.assertEqual(self.invoke("start").returncode, 0)
        self.assertEqual(len(self.commands()), 1)
        self.assertEqual(self.commands()[0][-1], "status")

    def test_shutdown_requests_checked_executor_handoff_without_killing_container(self):
        self.assertEqual(self.invoke("shutdown").returncode, 0)
        self.assertEqual(len(self.commands()), 1)
        self.assertEqual(self.commands()[0][-3:], ["-m", "astrabot.robot.cleared_environment", "shutdown"])
        self.assertEqual(self.commands()[0][0], "exec")

    def test_disarm_and_status_dispatch_to_guarded_server_without_restart(self):
        self.env["DOCKER_TEST_RUNNING"] = "1"
        for command in ("disarm", "status"):
            self.assertEqual(self.invoke(command).returncode, 0)
            self.assertEqual(self.commands()[-1][-3:], ["-m", "astrabot.robot", command])
        self.assertEqual(len(self.commands()), 2)

    def test_disarm_cli_propagates_server_rejection(self):
        from astrabot.robot.__main__ import main

        with patch("astrabot.robot.__main__.Client") as client, patch("sys.stderr"):
            client.return_value.request.side_effect = RuntimeError("fresh_damping_required")
            with self.assertRaises(SystemExit) as raised:
                main(["disarm"])
            self.assertEqual(raised.exception.code, 1)
            client.return_value.request.assert_called_once_with("disarm")

    def test_prepare_executes_empty_workspace_raise_and_trial_starts_from_ready(self):
        self.assertEqual(self.invoke("prepare").returncode, 0)
        self.assertEqual(self.commands()[-1][-2:], ["astrabot.robot.cleared_environment", "prepare"])
        self.assertNotIn("astrabot.robot.pick_a", self.commands()[-1])
        self.assertEqual(self.invoke("trial").returncode, 0)
        self.assertEqual(self.commands()[-1][-2:], ["--execute", "--from-ready"])
        self.assertIn("--operator-cleared-workspace", self.commands()[-1])
        count = len(self.commands())
        self.assertEqual(self.invoke("trial", "--help").returncode, 2)
        self.assertEqual(self.invoke("typo").returncode, 2)
        self.assertEqual(self.invoke("--help").returncode, 0)
        self.assertEqual(len(self.commands()), count)

    def test_scene_and_raise_are_separate_from_full_grasp_planning(self):
        self.assertEqual(self.invoke("scene").returncode, 0)
        self.assertEqual(self.commands()[-1][-1], "--load-scene-only")
        self.assertNotIn("--execute", self.commands()[-1])
        self.assertEqual(self.invoke("raise").returncode, 0)
        self.assertEqual(self.commands()[-1][-2:], ["-m", "astrabot.robot.prepare_motion"])

    def test_scene_cli_accepts_new_capture_without_old_export_or_motion_flag(self):
        from astrabot.robot.pick_a import main

        with patch("astrabot.robot.pick_a.run", new_callable=AsyncMock) as run, patch("sys.stdout"):
            result = main(["--config", "unused.json", "--load-scene-only", "--output", str(self.root / "scene")])
        self.assertEqual(result, 0)
        args = run.call_args.args[0]
        self.assertIsNone(args.scene)
        self.assertTrue(args.load_scene_only)
        self.assertFalse(args.execute)

    def test_recover_dispatch_is_explicit_and_does_not_start_trial(self):
        self.assertEqual(self.invoke("recover").returncode, 0)
        self.assertEqual(
            self.commands()[-1][-2:],
            ["astrabot.robot.cleared_environment", "recover"],
        )
        self.assertNotIn("astrabot.robot.pick_a", self.commands()[-1])
        count = len(self.commands())
        self.assertEqual(self.invoke("recover", "--extra").returncode, 2)
        self.assertEqual(len(self.commands()), count)

    def test_strict_override_and_invalid_experiment_setting_do_not_disable_shutdown(self):
        self.env["ASTRA_OPERATOR_CLEARED_WORKSPACE"] = "0"
        self.assertEqual(self.invoke("prepare").returncode, 0)
        self.assertNotIn("--operator-cleared-workspace", self.commands()[-1])
        self.env["ASTRA_OPERATOR_CLEARED_WORKSPACE"] = "invalid"
        self.assertEqual(self.invoke("trial").returncode, 2)
        self.assertEqual(self.invoke("shutdown").returncode, 0)
        self.assertEqual(self.commands()[-1][-1], "shutdown")
