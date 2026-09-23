"""Real spawn/IPC tests with a fake CAN driver; no device SDKs are loaded."""

import os
import sys
import time
import unittest

import numpy as np

from astrabot.robot.config import Config
from astrabot.robot.hand_process import HandProcess, TargetLease


class FakeHandDriver:
    """Record command effects while deliberately retaining an old sensor stamp."""

    def __init__(self, config, read_only=False):
        self.read_only = read_only
        self.command_writes = 0
        self.hands = [255.0] * 12
        self.stamp = time.monotonic()

    def snapshot(self):
        return {
            "hands": self.hands,
            "hand_time": self.stamp,
            "fault": "",
            "command_writes": self.command_writes,
            "dds_loaded": any(n.startswith("unitree_sdk2py") for n in sys.modules),
        }

    def command(self, target):
        if self.read_only:
            raise AssertionError("read-only child attempted a write")
        self.command_writes += 1
        self.hands = target.tolist()

    def close(self):
        pass


class HandProcessTests(unittest.TestCase):
    def worker(self, read_only=False):
        worker = HandProcess(Config(), read_only=read_only, driver_factory=FakeHandDriver)
        self.addCleanup(worker.close)
        self.wait_for(lambda: "hands" in worker.snapshot())
        return worker

    def wait_for(self, predicate, timeout=4):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("timed out waiting for child process")

    def test_spawn_is_read_only_and_keeps_acquisition_timestamp(self):
        worker = self.worker(read_only=True)
        first = worker.snapshot()
        time.sleep(0.32)
        last = worker.snapshot()
        self.assertNotEqual(first["pid"], os.getpid())
        self.assertFalse(last["dds_loaded"])
        self.assertEqual(last["hand_time"], first["hand_time"])
        self.assertGreater(last["sent"], first["sent"])
        self.assertEqual(last["command_writes"], 0)
        with self.assertRaisesRegex(ValueError, "read_only"):
            worker.set_target(np.zeros(12))

    def test_latest_target_and_cancel_reach_child(self):
        worker = self.worker()
        # Hold the mailbox lock to simulate a burst between IPC sends.
        with worker.lock:
            worker.target = [40.0] * 12
            worker.created = time.monotonic()
            worker.sequence += 1
            worker.target = [90.0] * 12
            worker.sequence += 1
        self.wait_for(lambda: worker.snapshot()["hands"] == [90.0] * 12)
        worker.cancel()
        revision = worker.sequence
        self.wait_for(lambda: worker.snapshot().get("sequence") == revision)
        count = worker.snapshot()["command_writes"]
        time.sleep(0.12)
        self.assertEqual(worker.snapshot()["command_writes"], count)

    def test_worker_exit_is_latched_without_restart(self):
        worker = self.worker()
        pid = worker.process.pid
        worker.process.terminate()
        worker.process.join(2)
        self.assertTrue(worker.snapshot()["fault"])
        time.sleep(0.05)
        self.assertFalse(worker.process.is_alive())
        self.assertEqual(worker.process.pid, pid)

    def test_lost_parent_heartbeat_stops_worker(self):
        worker = self.worker()
        worker.stop.set()
        worker.thread.join(1)
        self.wait_for(lambda: not worker.process.is_alive(), timeout=2)
        self.assertTrue(worker.snapshot()["fault"])

    def test_old_target_cannot_replay_after_cancel_or_expiry(self):
        lease = TargetLease(0.25, False)

        def message(sequence, created, target):
            return {"sequence": sequence, "created": created, "target": target}

        old = message(1, 10, [40.0] * 12)
        lease.accept(old, 10)
        self.assertIsNotNone(lease.current(10.1))
        lease.accept(message(2, 10.1, None), 10.1)
        lease.accept(old, 10.11)
        self.assertIsNone(lease.current(10.11))
        lease.accept(message(3, 10, [90.0] * 12), 11)
        self.assertIsNone(lease.current(11))
        lease.accept(message(4, 11, [90.0] * 12), 11)
        self.assertIsNone(lease.current(11.3))
        lease.accept(message(4, 11.3, [90.0] * 12), 11.3)
        self.assertIsNone(lease.current(11.3))

    def test_child_read_only_gate_rejects_injected_target(self):
        worker = self.worker(read_only=True)
        with worker.lock:
            worker.sequence += 1
            worker.target, worker.created = [0.0] * 12, time.monotonic()
        self.wait_for(lambda: not worker.process.is_alive())
        self.assertTrue(worker.snapshot()["fault"])
        self.assertEqual(worker.snapshot().get("command_writes", 0), 0)


class StalledHandDriver(FakeHandDriver):
    """Pause feedback production without exiting the worker."""

    def snapshot(self):
        if time.monotonic() - self.stamp > 0.1:
            time.sleep(0.5)
        return super().snapshot()


class IPCStaleTests(unittest.TestCase):
    def test_live_but_stalled_worker_is_not_fresh(self):
        worker = HandProcess(Config(), read_only=True, driver_factory=StalledHandDriver)
        self.addCleanup(worker.close)
        until = time.monotonic() + 4
        while "hands" not in worker.snapshot() and time.monotonic() < until:
            time.sleep(0.01)
        self.assertIn("hands", worker.snapshot())
        time.sleep(0.4)
        self.assertTrue(worker.process.is_alive())
        self.assertEqual(worker.snapshot()["fault"], "hand_ipc_stale")
