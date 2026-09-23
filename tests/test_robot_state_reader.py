"""Verify freshness and worker failure without importing a hardware SDK."""

import time
import unittest
from types import SimpleNamespace

from astrabot.robot.state_reader import LatestStateReader


class Mailbox:
    """Fake a DDS KeepLast(1) cache whose take operation consumes data."""

    def __init__(self):
        self.value = None

    def publish(self, tick, valid=True):
        self.value = SimpleNamespace(tick=tick, sample_info=SimpleNamespace(valid_data=valid))

    def take(self, count):
        value, self.value = self.value, None
        return [] if value is None else [value]


class StateReaderTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.received = []
        self.mailbox = Mailbox()
        self.reader = LatestStateReader(
            lambda sample, stamp: self.received.append((sample.tick, stamp)),
            reader=self.mailbox,
            clock=lambda: self.now,
            start=False,
        )
        self.addCleanup(self.reader.close)

    def test_empty_duplicate_and_invalid_samples_do_not_refresh_feedback(self):
        self.mailbox.publish(10)
        self.reader.poll_once()
        self.now += 1
        self.reader.poll_once()
        self.mailbox.publish(10)
        self.reader.poll_once()
        self.mailbox.publish(11, valid=False)
        self.reader.poll_once()
        self.assertEqual(self.received, [(10, 10.0)])
        self.assertEqual(self.reader.diagnostics()["duplicates"], 1)

    def test_consumes_latest_frame_and_accepts_tick_wrap(self):
        for tick in range(100):
            self.mailbox.publish(tick)
        self.reader.poll_once()
        self.assertEqual(self.received, [(99, 10.0)])
        self.reader._last_tick = 0xFFFFFFFE
        self.mailbox.publish(2)
        self.reader.poll_once()
        self.assertEqual(self.received[-1][0], 2)

    def test_tick_regression_cannot_be_presented_as_new_feedback(self):
        self.mailbox.publish(100)
        self.reader.poll_once()
        self.now += 1
        self.mailbox.publish(90)
        with self.assertRaisesRegex(ValueError, "joint_tick_regressed"):
            self.reader.poll_once()
        self.assertEqual(self.received, [(100, 10.0)])

    def test_exception_stops_worker_and_latches_fault(self):
        class BrokenReader:
            def take(self, count):
                raise OSError("reader disconnected")

        worker = LatestStateReader(lambda *_: self.fail("unexpected callback"), reader=BrokenReader())
        self.addCleanup(worker.close)
        until = time.monotonic() + 1
        while worker.diagnostics()["alive"] and time.monotonic() < until:
            time.sleep(0.001)
        self.assertFalse(worker.diagnostics()["alive"])
        self.assertIn("reader disconnected", worker.diagnostics()["fault"])

    def test_empty_worker_closes_promptly(self):
        worker = LatestStateReader(lambda *_: None, reader=Mailbox())
        started = time.monotonic()
        worker.close()
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertFalse(worker.diagnostics()["alive"])
