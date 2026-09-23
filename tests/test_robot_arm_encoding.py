"""Exercise production frame construction with synthetic feedback and no DDS."""

import importlib.util
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from astrabot.robot.config import REST


@unittest.skipUnless(importlib.util.find_spec("unitree_sdk2py"), "requires the installed Unitree IDL definitions")
class ArmEncodingTests(unittest.TestCase):
    def test_published_diagnostics_match_frame_and_only_acknowledge_successful_write(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_

        from astrabot.robot.hardware import ArmCommandEncoder, RealHardware

        hardware = object.__new__(RealHardware)  # No device initialization.
        hardware.lock = threading.Lock()
        hardware.low = unitree_hg_msg_dds__LowState_()
        hardware.joint_time = time.monotonic()
        hardware.config = SimpleNamespace(feedback_timeout=1.0)
        hardware.read_only, hardware.hand_process = False, None
        hardware.arm_writes, hardware.last_arm_command = 0, None
        hardware.publisher = Mock()
        hardware.publisher.Write.return_value = True
        for i, value in enumerate(REST, 15):
            hardware.low.motor_state[i].q = float(value)
            hardware.low.motor_state[i].dq = 0.012
        with patch("unitree_sdk2py.core.channel.ChannelPublisher", side_effect=AssertionError("DDS publication")):
            hardware.encoder = ArmCommandEncoder()
            hardware.encoder.gravity.last -= 0.03
            hardware.command(REST + 0.002, None, 1)
            record = hardware.last_arm_command
            message = hardware.publisher.Write.call_args.args[0]
            self.assertEqual(record["sequence"], 1)
            np.testing.assert_allclose(record["q_feedback"], REST)
            np.testing.assert_allclose(record["dq_feedback"], 0.012)
            np.testing.assert_allclose(record["q_command"], [m.q for m in message.motor_cmd[15:29]])
            np.testing.assert_allclose(record["tau_command"], [m.tau for m in message.motor_cmd[15:29]])
            self.assertTrue(np.isfinite(record["compensation_time"]))
            self.assertGreater(np.max(np.abs(record["tau_adaptive"])), 0)
            np.testing.assert_allclose(record["compensation_error"], 0.002)
            np.testing.assert_array_equal(record["adaptive_braking"], False)
            np.testing.assert_array_equal(record["tracking_closing_speed"], 0)
            hardware.publisher.Write.return_value = False
            with self.assertRaisesRegex(RuntimeError, "dds_write_failed"):
                hardware.command(REST, None, 1)
            self.assertIs(hardware.last_arm_command, record)
            self.assertEqual(hardware.arm_writes, 1)

    def test_relinquish_frame_has_zero_weight_gains_and_feedforward(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_

        from astrabot.robot.hardware import ArmCommandEncoder

        low = unitree_hg_msg_dds__LowState_()
        encoder = ArmCommandEncoder()
        encoder.encode(low, REST, 1)
        frame = encoder.relinquish(low)
        self.assertEqual(frame.motor_cmd[29].q, 0)
        self.assertTrue(all(m.kp == m.kd == m.tau == m.dq == 0 for m in frame.motor_cmd))
        self.assertGreater(len(frame.serialize()), 0)

    def test_memory_encoding_latches_waist_and_never_creates_a_publisher(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_

        from astrabot.robot.hardware import ArmCommandEncoder

        low = unitree_hg_msg_dds__LowState_()
        low.mode_machine = 7
        for i, value in enumerate(REST, 15):
            low.motor_state[i].q = float(value)
        low.motor_state[12].q = 0.02
        with patch("unitree_sdk2py.core.channel.ChannelPublisher", side_effect=AssertionError("DDS publication")):
            encoder = ArmCommandEncoder()
            first = encoder.encode(low, REST, 0.6)
            self.assertGreater(len(first.serialize()), 0)
            low.motor_state[12].q = 0.025
            target = REST.copy()
            target[0] += 0.01
            last = encoder.encode(low, target, 0.3)
            self.assertAlmostEqual(last.motor_cmd[12].q, 0.02)
            self.assertAlmostEqual(last.motor_cmd[29].q, 0.3)
            self.assertEqual(last.mode_machine, 7)
            self.assertTrue(np.allclose([m.q for m in last.motor_cmd[15:29]], target))
            self.assertTrue(all(m.mode == 0 for m in last.motor_cmd[:12]))
            # Match the Novus waist-hold contract, independently of arm gains.
            for motor in last.motor_cmd[12:15]:
                self.assertEqual((motor.kp, motor.kd), (500.0, 5.0))
                self.assertEqual((motor.mode, motor.dq, motor.tau), (1, 0.0, 0.0))
            self.assertEqual(last.motor_cmd[15].kp, 80)
            self.assertEqual(last.motor_cmd[19].kp, 40)
