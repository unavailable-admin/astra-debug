"""O6 driver loaded only by the hand worker; no Unitree DDS imports."""

import socket
import struct
import time

import numpy as np

from .config import vector

CANONICAL_FROM_SDK = np.array([5, 4, 3, 2, 0, 1])


def to_sdk(raw):
    """Convert pinky-first raw 0..255 into thumb-first SDK percent."""
    raw = vector(raw, 6, "hand")
    if np.any((raw < 0) | (raw > 255)):
        raise ValueError("hand_raw_range")
    result = np.empty(6)
    result[CANONICAL_FROM_SDK] = raw * (100 / 255)
    return result.tolist()


def from_sdk(percent):
    """Convert SDK feedback into the canonical hardware order."""
    return vector(percent, 6, "hand_feedback")[CANONICAL_FROM_SDK] * (255 / 100)


def probe_hand(interface):
    """Identify side using Novus's non-motion Linker query, never by port number."""
    with socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as sock:
        sock.bind((interface,))
        sock.settimeout(0.05)
        for value in (0xC0, 0x01):
            sock.send(struct.pack("=IB3x8s", 0xFF, 1, bytes([value]) + bytes(7)))
            until = time.monotonic() + 0.5
            while time.monotonic() < until:
                try:
                    data = sock.recv(16)
                except TimeoutError:
                    continue
                if len(data) == 16:
                    identifier, _, _ = struct.unpack("=IB3x8s", data)
                    side = {0x28: "left", 0x27: "right"}.get(identifier & 0x7FF)
                    if side:
                        return side
    raise ValueError(f"no_hand_identity:{interface}")


class HandDriver:
    """Own CAN connections and retain acquisition times from each sensor."""

    def __init__(self, config, read_only=False):
        from linkerbot import O6
        from linkerbot.hand.o6.events import SensorSource

        self.config = config
        self.read_only = read_only
        self.hands = []
        self.initialized = False
        self.command_writes = 0
        try:
            for side, interface in (("left", config.left_can), ("right", config.right_can)):
                found = probe_hand(interface)
                if found != side:
                    raise ValueError(f"CAN_side_mismatch:{interface}:expected_{side}:found_{found}")
                hand = O6(side=side, interface_name=interface, interface_type="socketcan")
                self.hands.append(hand)
                hand.start_polling(
                    {
                        SensorSource.ANGLE: 1 / 60,
                        SensorSource.FORCE_SENSOR: 1 / 30,
                        SensorSource.FAULT: 0.2,
                        SensorSource.TEMPERATURE: 0.5,
                    }
                )
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        """Report sensor timestamps, never replacing missing data with fresh data."""
        angles, stamps, tactile, tactile_stamps = [], [], [], []
        wall, monotonic = time.time(), time.monotonic()
        fault, temperature = "", 0.0
        for hand in self.hands:
            sample = hand.angle.get_snapshot()
            if sample is None:
                angles.extend([0.0] * 6)
                stamps.append(0.0)
            else:
                angles.extend(from_sdk(sample.angles.to_list()).tolist())
                stamps.append(monotonic - (wall - sample.timestamp))
            forces = hand.force_sensor.get_snapshot()
            for name in ("thumb", "index", "middle", "ring", "pinky"):
                item = getattr(forces, name, None)
                tactile.append(float(np.max(item.values)) if item is not None else 0.0)
                tactile_stamps.append(monotonic - (wall - item.timestamp) if item is not None else 0.0)
            temperatures = hand.temperature.get_snapshot()
            if temperatures is None or not 0 <= wall - temperatures.timestamp <= 2:
                fault = fault or "hand_temperature_stale"
            else:
                temperature = max(temperature, *temperatures.temperatures.to_list())
            faults = hand.fault.get_snapshot()
            if faults is None or not 0 <= wall - faults.timestamp <= 2:
                fault = fault or "hand_fault_feedback_stale"
            elif faults.faults.has_any_fault():
                fault = fault or "hand_motor_fault"
        return {
            "hands": angles,
            "hand_time": min(stamps, default=0.0),
            "tactile": tactile,
            "tactile_time": min(tactile_stamps, default=0.0),
            "fault": fault,
            "temperature": temperature,
            "command_writes": self.command_writes,
        }

    def command(self, target):
        """Write one current target; configuration writes require actuation too."""
        if self.read_only:
            raise ValueError("read_only_hardware")
        from linkerbot.hand.o6.torque import O6Torque

        if not self.initialized:
            for hand in self.hands:
                hand.speed.set_speeds([100.0] * 6)
                hand.torque.set_torques(O6Torque.from_raw([self.config.hand_torque_raw] * 6))
            self.initialized = True
        for i, hand in enumerate(self.hands):
            hand.angle.set_angles(to_sdk(target[6 * i : 6 * i + 6]))
            self.command_writes += 1

    def close(self):
        """Stop polling without sending an open or position command."""
        for hand in self.hands:
            hand.close()
