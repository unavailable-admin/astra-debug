"""Lost acknowledgements must never cause motion replay."""

import unittest
from unittest.mock import patch

from astrabot.robot.service import Client


class ClientRetryTests(unittest.TestCase):
    def test_heartbeat_and_status_reconnect_once(self):
        for operation in ("heartbeat", "status"):
            client = Client("/unused")
            with patch.object(client, "_request_once", side_effect=[BrokenPipeError(32, "pipe"), {}]) as send:
                client.request(operation, session="same", generation=7)
            self.assertEqual(send.call_count, 2)
            self.assertEqual(send.call_args_list[0], send.call_args_list[1])

    def test_move_is_never_replayed(self):
        client = Client("/unused")
        with (
            patch.object(client, "_request_once", side_effect=BrokenPipeError(32, "pipe")) as send,
            self.assertRaises(BrokenPipeError),
        ):
            client.request("move", session="same", generation=7)
        self.assertEqual(send.call_count, 1)

    def test_persistent_disconnect_and_expired_lease_still_fail(self):
        client = Client("/unused")
        for error, attempts in ((BrokenPipeError(32, "pipe"), 2), (RuntimeError("expired_task_generation"), 1)):
            with (
                patch.object(client, "_request_once", side_effect=error) as send,
                self.assertRaises(type(error)),
            ):
                client.request("heartbeat", session="same", generation=7)
            self.assertEqual(send.call_count, attempts)
