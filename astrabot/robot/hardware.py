"""Direct Unitree DDS + Linker O6 adapters, without ROS or a Novus process."""

import threading
import time

import numpy as np

from .compensation import AdaptiveBias
from .config import ARM_NAMES, ASSETS, REST
from .executor import Snapshot
from .hand_driver import from_sdk, probe_hand, to_sdk
from .hand_process import HandProcess
from .motion_checks import MotionChecks
from .state_reader import LatestStateReader

__all__ = ["ArmCommandEncoder", "Gravity", "MockHardware", "RealHardware", "from_sdk", "probe_hand", "to_sdk"]


class MockHardware:
    """Explicit deterministic fake for protocol tests; never loads device SDKs."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.body = np.r_[np.zeros(15), REST]
        self.hands = np.full(12, 255.0)
        self.frozen = False
        self.stamp = clock()
        self.fault = ""
        self.writes = []

    def snapshot(self):
        """Expose controllable freshness and faults for tests."""
        if not self.frozen:
            self.stamp = self.clock()
        return Snapshot(self.body.copy(), self.hands.copy(), self.stamp, self.stamp, [0.0] * 10, self.stamp, self.fault)

    def command(self, arms, hands, weight):
        """Idealized tracking only; tests must not present this as robot validation."""
        self.writes.append((self.clock(), np.array(arms), np.array(hands) if hands is not None else None, weight))
        self.writes[:] = self.writes[-2000:]
        self.body[15:] = arms
        if hands is not None:
            self.hands = np.array(hands)

    def cancel_hand_commands(self):
        """Mock commands are immediate, so there is no pending CAN target."""

    def release_hands(self):
        """Open while detached."""
        self.hands[:] = 255

    def close(self):
        """No resources to release."""

    def clear_arm_control(self):
        """Record relinquishing SDK control without a finger command."""
        self.writes.append((self.clock(), self.body[15:].copy(), None, 0.0))


class Gravity:
    """Fixed-base feedforward and bounded slow tracking compensation from Novus."""

    def __init__(self, checks=None):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(ASSETS / "gravity.xml"))
        self.data = mujoco.MjData(self.model)
        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_NAMES]
        if min(joints) < 0:
            raise ValueError("gravity_joint_layout")
        self.qidx = self.model.jnt_qposadr[joints]
        self.vidx = self.model.jnt_dofadr[joints]
        self.adaptation = AdaptiveBias(checks or MotionChecks())
        self.bias = self.adaptation.bias
        self.last = time.monotonic()
        self.torque = np.zeros(14)
        self.model_torque = np.zeros(14)
        self.updated_at = None
        self.limit = np.tile([45.0, 45.0, 45.0, 45.0, 12.0, 12.0, 12.0], 2)

    def evaluate(self, command, measured):
        """Update at 50 Hz; never integrate while targets/feedback are stale."""
        now = time.monotonic()
        dt = now - self.last
        if dt < 0.02:
            return self.torque
        dt = min(dt, 0.1)
        self.last = now
        self.data.qpos[self.qidx] = command
        self.mj.mj_forward(self.model, self.data)
        self.adaptation.update(command - measured, dt)
        self.model_torque = self.data.qfrc_bias[self.vidx].copy()
        self.torque = np.clip(self.model_torque + self.bias, -self.limit, self.limit)
        self.updated_at = now
        return self.torque


class ArmCommandEncoder:
    """Build production arm frames in memory; this class cannot publish to DDS."""

    def __init__(self, checks=None):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        self.msg, self.crc = unitree_hg_msg_dds__LowCmd_(), CRC()
        self.gravity = Gravity(checks)
        self.waist = None

    def encode(self, low, arms, weight):
        """Apply the same waist latch, PD gains, gravity and CRC as live control."""
        if self.waist is None:
            self.waist = np.array([low.motor_state[i].q for i in range(12, 15)])
        measured = np.array([m.q for m in low.motor_state[15:29]])
        torque = self.gravity.evaluate(np.asarray(arms), measured)
        self.msg.mode_pr, self.msg.mode_machine = 0, low.mode_machine
        for i in range(12, 29):
            motor = self.msg.motor_cmd[i]
            motor.mode = 1
            motor.q = float(self.waist[i - 12] if i < 15 else arms[i - 15])
            motor.dq = 0.0
            wrist = i >= 15 and (i - 15) % 7 >= 4
            if i < 15:
                # Match Novus ArmController's waist gains independently of
                # the softer arm/wrist gains; keep the takeover pose latched.
                motor.kp, motor.kd = 500.0, 5.0
            else:
                motor.kp, motor.kd = (40.0, 1.5) if wrist else (80.0, 3.0)
            motor.tau = float(torque[i - 15]) if i >= 15 else 0.0
        self.msg.motor_cmd[29].q = float(weight)
        self.msg.crc = self.crc.Crc(self.msg)
        return self.msg

    def relinquish(self, low):
        """Encode zero SDK weight and zero effort, without latching a pose."""
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

        message = unitree_hg_msg_dds__LowCmd_()
        message.mode_pr, message.mode_machine = 0, low.mode_machine
        for i in range(12, 29):
            message.motor_cmd[i].mode = 1
            message.motor_cmd[i].q = float(low.motor_state[i].q)
            message.motor_cmd[i].kp = message.motor_cmd[i].kd = 0.0
            message.motor_cmd[i].dq = message.motor_cmd[i].tau = 0.0
        message.motor_cmd[29].q = 0.0
        message.crc = self.crc.Crc(message)
        return message


class RealHardware:
    """Own Unitree DDS; a spawned process exclusively owns Linker O6 CAN.

    Args:
        config: Robot device and feedback settings.
        hands: Enable the separate CAN worker.
        read_only: Disable all actuator writes, including publisher creation.
    """

    def __init__(self, config, hands=True, read_only=False):
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
        )
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

        self.config, self.read_only = config, read_only
        self.lock = threading.Lock()
        self.low = None
        self.joint_time = 0.0
        self.mode, self.mode_time = -1, 0.0
        self.mode_stats = {
            "attempts": 0,
            "successes": 0,
            "errors": 0,
            "last_code": None,
            "last_error": "",
            "last_latency": 0.0,
            "max_latency": 0.0,
        }
        self.mode_latencies = []
        self.stop = threading.Event()
        self.hand_process = None
        self.publisher = self.state_reader = None
        self.threads = []
        self.arm_writes = 0
        self.encoder = None if read_only else ArmCommandEncoder(config.motion_checks)
        self.last_arm_command = None
        from unitree_sdk2py.core import channel

        channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
            "/tmp/cdds.LOG", "stderr"
        ).replace("<Verbosity>config</Verbosity>", "<Verbosity>warning</Verbosity>")
        try:
            ChannelFactoryInitialize(0, config.interface)
            if not read_only:
                self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
                self.publisher.Init()
            self.state_reader = LatestStateReader(self._on_state)
            self.loco = LocoClient()
            self.loco.SetTimeout(0.5)
            self.loco.Init()
            if hands:
                self.hand_process = HandProcess(config, read_only=read_only)
            thread = threading.Thread(target=self._mode_loop, name="locomotion-mode", daemon=True)
            thread.start()
            self.threads.append(thread)
        except BaseException:
            self.close()
            raise

    def _on_state(self, message, acquired):
        with self.lock:
            self.low, self.joint_time = message, acquired

    def _mode_loop(self):
        while not self.stop.is_set():
            started = time.monotonic()
            code, failure, mode = None, "", None
            try:
                code, value = self.loco.GetFsmId()
                if code == 0:
                    mode = int(value["data"] if isinstance(value, dict) else value)
            except Exception as exc:  # noqa: BLE001 -- isolate SDK/planner failures and latch diagnostics
                failure = f"{type(exc).__name__}:{exc}"
            finished = time.monotonic()
            with self.lock:
                stats = self.mode_stats
                stats["attempts"] += 1
                stats["successes" if mode is not None else "errors"] += 1
                stats.update(last_code=code, last_error=failure, last_latency=finished - started)
                stats["max_latency"] = max(stats["max_latency"], finished - started)
                self.mode_latencies.append(finished - started)
                self.mode_latencies[:] = self.mode_latencies[-2000:]
                if mode is not None:
                    self.mode, self.mode_time = mode, finished
                latencies = tuple(self.mode_latencies)
            percentile = float(np.percentile(latencies, 95))
            with self.lock:
                self.mode_stats["latency_p95"] = percentile
            self.stop.wait(0.1)

    def diagnostics(self):
        """Expose real RPC outcomes and IPC health without polling either SDK."""
        with self.lock:
            mode = dict(self.mode_stats, age=time.monotonic() - self.mode_time)
        # Statistics are produced by the mode worker, never under the caller's
        # executor mutex. Status reads only copy the cached result.
        mode.setdefault("latency_p95", None)
        hand = self.hand_process.snapshot() if self.hand_process else {}
        return {
            "mode_query": mode,
            "joint_reader": self.state_reader.diagnostics() if self.state_reader else {},
            "hand_worker": hand,
            "arm_command_writes": self.arm_writes,
            "read_only": self.read_only,
        }

    def snapshot(self):
        """Merge raw joint and hand acquisition times without refreshing cached data."""
        with self.lock:
            low, joint_time, mode, mode_time = self.low, self.joint_time, self.mode, self.mode_time
        if low is None:
            return None
        body = np.array([m.q for m in low.motor_state[:29]], float)
        temperature = float(np.max([m.temperature for m in low.motor_state[:29]]))
        fault = "unitree_motor_fault" if any(m.motorstate for m in low.motor_state[12:29]) else ""
        if self.state_reader:
            fault = fault or self.state_reader.diagnostics()["fault"]
        if time.monotonic() - mode_time > 1.0:
            fault = fault or "locomotion_mode_stale"
        hand = self.hand_process.snapshot() if self.hand_process else {}
        return Snapshot(
            body,
            np.array(hand.get("hands", [0.0] * 12)),
            joint_time,
            hand.get("hand_time", 0.0),
            hand.get("tactile", []),
            hand.get("tactile_time", 0.0),
            fault or hand.get("fault", ""),
            mode,
            max(temperature, hand.get("temperature", 0.0)),
            shell_temperature=float(np.max([m.temperature[0] for m in low.motor_state[:29]])),
            hand_temperature=float(hand.get("temperature", 0.0)),
            body_velocity=np.array([m.dq for m in low.motor_state[:29]], float),
        )

    def command(self, arms, hands, weight):
        """Publish arm targets; optional hand targets enter the latest-value mailbox."""
        if self.read_only:
            raise ValueError("read_only_hardware")
        with self.lock:
            low, stamp = self.low, self.joint_time
        if low is None or not 0 <= time.monotonic() - stamp <= self.config.feedback_timeout:
            raise ValueError("joint_feedback_stale")
        encode_started = time.monotonic()
        message = self.encoder.encode(low, arms, weight)
        publish_started = time.monotonic()
        if self.publisher.Write(message) is False:
            raise RuntimeError("dds_write_failed")
        published_at = time.monotonic()
        self.arm_writes += 1
        gravity = self.encoder.gravity
        self.last_arm_command = {
            "sequence": self.arm_writes,
            "time": published_at,
            "encode_seconds": publish_started - encode_started,
            "publish_seconds": published_at - publish_started,
            "joint_time": stamp,
            "q_command": [m.q for m in message.motor_cmd[15:29]],
            "q_feedback": [m.q for m in low.motor_state[15:29]],
            "dq_feedback": [m.dq for m in low.motor_state[15:29]],
            "tau_model": gravity.model_torque.tolist(),
            "tau_adaptive": gravity.bias.tolist(),
            "adaptive_braking": gravity.adaptation.braking.tolist(),
            "compensation_error": gravity.adaptation.error.tolist(),
            "tracking_closing_speed": gravity.adaptation.closing_speed.tolist(),
            "tau_command": [m.tau for m in message.motor_cmd[15:29]],
            "compensation_time": gravity.updated_at,
            "weight": float(weight),
        }
        if hands is not None and self.hand_process:
            self.hand_process.set_target(hands)

    def clear_arm_control(self):
        """Publish zero ownership only with fresh, confirmed damping feedback."""
        if self.read_only:
            raise ValueError("read_only_hardware")
        with self.lock:
            low, stamp, mode, mode_stamp = self.low, self.joint_time, self.mode, self.mode_time
        now = time.monotonic()
        if low is None or not 0 <= now - stamp <= self.config.feedback_timeout:
            raise ValueError("joint_feedback_stale")
        if mode != 1 or not 0 <= now - mode_stamp <= 0.25:
            raise ValueError("fresh_damping_required")
        if self.publisher.Write(self.encoder.relinquish(low)) is False:
            raise RuntimeError("dds_write_failed")
        self.arm_writes += 1
        self.last_arm_command = {
            "sequence": self.arm_writes,
            "time": time.monotonic(),
            "joint_time": stamp,
            "relinquished": True,
            "weight": 0.0,
        }
        self.encoder.waist = None
        self.encoder.gravity.adaptation.reset()
        self.encoder.gravity.torque[:] = 0

    def cancel_hand_commands(self):
        """Invalidate pending hand targets without a CAN call on the control thread."""
        if self.hand_process:
            self.hand_process.cancel()

    def release_hands(self):
        """Explicit open without requiring valid arm feedback."""
        if self.read_only:
            raise ValueError("read_only_hardware")
        if self.hand_process:
            self.hand_process.set_target(np.full(12, 255.0))

    def close(self):
        """Close only after executor handoff; never send implicit actuator commands."""
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=1.0)
        if self.hand_process:
            self.hand_process.close()
        if self.state_reader:
            self.state_reader.close()
        if self.publisher:
            self.publisher.Close()
