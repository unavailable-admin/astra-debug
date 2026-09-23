"""Exercise final handoff logging through a real mock service process."""

import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from astrabot.robot.config import Config
from astrabot.robot.executor import Executor
from astrabot.robot.hardware import MockHardware, RealHardware
from astrabot.robot.service import Client, write_audit_record
from astrabot.robot.trajectory import Segment


class AuditTests(unittest.TestCase):
    def test_audit_yields_with_control_lock_available(self):
        executor = Executor(MockHardware(), None, Config())
        self.addCleanup(executor.close)
        executor.control_samples.extend({"sequence": i} for i in range(24))
        acquired = []

        def yield_to_control(_seconds):
            def control():
                locked = executor.lock.acquire(timeout=0.1)
                acquired.append(locked)
                if locked:
                    executor.lock.release()

            worker = threading.Thread(target=control)
            worker.start()
            worker.join(timeout=1)

        log = io.StringIO()
        with patch("astrabot.robot.service.time.sleep", side_effect=yield_to_control):
            write_audit_record(executor, log, True)
        self.assertEqual(acquired, [True] * 3)
        self.assertEqual(len(json.loads(log.getvalue())["control_samples"]), 24)

    def test_diagnostics_does_not_hold_feedback_lock_during_statistics(self):
        hardware = object.__new__(RealHardware)
        hardware.lock = threading.Lock()
        hardware.mode_stats = {}
        hardware.mode_time = time.monotonic()
        hardware.mode_latencies = deque([0.001, 0.003])
        hardware.hand_process = hardware.state_reader = None
        hardware.arm_writes, hardware.read_only = 0, True

        hardware.mode_stats["latency_p95"] = 0.0029
        with patch("astrabot.robot.hardware.np.percentile", side_effect=AssertionError("statistics on status thread")):
            self.assertEqual(hardware.diagnostics()["mode_query"]["latency_p95"], 0.0029)

    def test_failed_write_keeps_pending_events(self):
        class FailedLog:
            def write(self, data):
                raise OSError("disk full")

        executor = Executor(MockHardware(), None, Config())
        self.addCleanup(executor.close)
        executor.event("test_event")
        executor.control_samples.append({"sequence": 1})
        with self.assertRaisesRegex(OSError, "disk full"):
            write_audit_record(executor, FailedLog(), True)
        self.assertEqual(executor.events[-1]["kind"], "test_event")
        self.assertEqual(list(executor.control_samples), [{"sequence": 1}])

    def test_motion_samples_preserve_reference_before_fault_hold_and_drain_once(self):
        now = [10.0]
        hardware = MockHardware(lambda: now[0])
        executor = Executor(hardware, None, Config(), lambda: now[0])
        self.addCleanup(executor.close)
        executor.request({"op": "begin"})
        executor.weight = 1
        q, hand = hardware.body[15:].copy(), hardware.hands.copy()
        executor.queue.append(Segment(q.copy(), hand, q.copy(), hand, 1, "clear_thighs"))
        original_generation = executor.generation
        hardware.body[23] -= 0.010070030877882857  # Recorded hard-limit crossing.
        measured = hardware.body[15:].copy()
        now[0] += 0.004
        executor.tick()
        self.assertEqual(executor.error, "arm_tracking_error")
        log = io.StringIO()
        write_audit_record(executor, log, True)
        sample = json.loads(log.getvalue())["control_samples"][0]
        np.testing.assert_array_equal(sample["reference"]["q_reference"], q)
        np.testing.assert_array_equal(sample["reference"]["q_command_candidate"], q)
        np.testing.assert_array_equal(sample["q_command"], measured)
        self.assertEqual(sample["reference"]["generation"], original_generation)
        self.assertEqual(sample["generation"], original_generation + 1)
        self.assertIsNone(sample["published"])  # Never fabricate motor torques for mocks.
        self.assertFalse(executor.control_samples)
        write_audit_record(executor, log, True)
        self.assertEqual(json.loads(log.getvalue().splitlines()[-1])["control_samples"], [])

    def test_slow_logger_has_bounded_buffer_and_visible_sample_loss(self):
        now = [10.0]
        hardware = MockHardware(lambda: now[0])
        executor = Executor(hardware, None, Config(), lambda: now[0])
        self.addCleanup(executor.close)
        executor.request({"op": "begin"})
        executor.weight = 1
        executor.control_samples = deque(maxlen=3)
        executor.trace_until = 11
        for _ in range(8):
            now[0] += 0.004
            executor.tick()
        self.assertEqual(len(executor.control_samples), 3)
        self.assertEqual(executor.control_samples_dropped, 5)

    def test_samples_arriving_during_write_survive_acknowledgement(self):
        executor = Executor(MockHardware(), None, Config())
        self.addCleanup(executor.close)
        executor.control_samples.append({"sequence": 1})

        class ConcurrentLog:
            def write(self, _data):
                executor.control_samples.append({"sequence": 2})

        write_audit_record(executor, ConcurrentLog(), True)
        self.assertEqual(list(executor.control_samples), [{"sequence": 2}])

    def test_invalid_velocity_fault_remains_serializable(self):
        hardware = MockHardware()
        executor = Executor(hardware, None, Config())
        self.addCleanup(executor.close)
        executor.request({"op": "begin"})
        snapshot = hardware.snapshot()
        snapshot.body_velocity = np.full(29, np.nan)
        with patch.object(hardware, "snapshot", return_value=snapshot):
            executor.tick()
            self.assertEqual(executor.error, "invalid_velocity_feedback")
            log = io.StringIO()
            write_audit_record(executor, log, True)
        self.assertIsNone(json.loads(log.getvalue())["control_samples"][0]["dq_feedback"])

    def test_short_handoff_flushes_terminal_state_before_service_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"output": directory}))
            sock = root / "executor.sock"
            with (root / "console.log").open("w+") as console:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "astrabot.robot",
                        "serve",
                        "--mock",
                        "--config",
                        str(config),
                        "--socket",
                        str(sock),
                    ],
                    stdout=console,
                    stderr=subprocess.STDOUT,
                )
                try:
                    client = Client(sock)
                    until = time.monotonic() + 8
                    while not sock.exists() and process.poll() is None and time.monotonic() < until:
                        time.sleep(0.01)
                    self.assertTrue(sock.exists())
                    begun = client.request("begin")
                    time.sleep(0.03)
                    client.request("shutdown", **{k: begun[k] for k in ("session", "generation")})
                    self.assertEqual(process.wait(timeout=5), 0)
                    rows = [json.loads(line) for line in next(root.glob("executor-*.jsonl")).read_text().splitlines()]
                    self.assertTrue(rows[-1]["final"])
                    self.assertEqual(rows[-1]["status"]["state"], "STOPPED")
                    self.assertEqual(rows[-1]["status"]["weight"], 0)
                    self.assertEqual(rows[-1]["status"]["arm_output"], "disabled")
                    self.assertFalse(rows[-1]["status"]["attached"])
                    events = [e for row in rows for e in row["events"]]
                    self.assertEqual(sum(e["kind"] == "handoff_complete" for e in events), 1)
                    self.assertFalse(sock.exists())
                finally:
                    if process.poll() is None:
                        process.kill()  # Only the mock process owned by this test.
                        process.wait(timeout=2)
