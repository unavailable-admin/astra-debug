"""Heartbeat cancellation preserves the originating executor failure."""

import unittest
from unittest.mock import Mock

from astrabot.robot.task import Task


class TaskCauseTests(unittest.TestCase):
    def test_heartbeat_expiry_reports_cause_without_resuming(self):
        task = object.__new__(Task)
        task.token = {"session": "old", "generation": 11}
        task.error = RuntimeError("expired_task_generation")
        task.client = Mock()
        task.client.request.return_value = {
            "session": "old",
            "generation": 12,
            "error": "preflight:pinch_step_too_large",
            "health": "",
        }
        with self.assertRaisesRegex(RuntimeError, "task_cancelled:preflight:pinch_step_too_large"):
            task.status()
        task.client.request.assert_called_once_with("status")
        self.assertEqual(task.token["generation"], 11)

    def test_unavailable_or_restarted_executor_preserves_cancellation(self):
        for reply in (
            OSError("offline"),
            {"session": "new", "error": "unrelated_fault"},
            {"session": "old", "error": ""},
        ):
            task = object.__new__(Task)
            task.token = {"session": "old", "generation": 11}
            task.error = RuntimeError("expired_task_generation")
            task.client = Mock()
            if isinstance(reply, Exception):
                task.client.request.side_effect = reply
            else:
                task.client.request.return_value = reply
            with self.assertRaisesRegex(RuntimeError, "task_cancelled:expired_task_generation"):
                task.status()
