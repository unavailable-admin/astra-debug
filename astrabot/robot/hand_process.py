"""Spawned CAN worker with bounded, expiring latest-target IPC and no DDS."""

import json
import multiprocessing
import os
import socket
import threading
import time

from .config import vector


def _send(sock, message):
    try:
        sock.send(json.dumps(message, allow_nan=False).encode())
        return True
    except BlockingIOError:
        return False


def _receive(sock):
    # Bounded work even if the peer misbehaves. The sender runs at 50 Hz.
    messages = []
    for _ in range(512):
        try:
            messages.append(json.loads(sock.recv(16384)))
        except BlockingIOError:
            break
    return messages


class TargetLease:
    """Reject old revisions and expired targets even after IPC catches up."""

    def __init__(self, timeout, read_only):
        self.timeout, self.read_only = timeout, read_only
        self.sequence = -1
        self.target = None
        self.created = 0.0

    def accept(self, message, now):
        """Replace the mailbox only with a newer revision, including cancellation."""
        sequence = message["sequence"]
        if sequence <= self.sequence:
            return
        self.sequence = sequence
        self.target = None
        self.created = message["created"]
        if message["target"] is not None:
            if self.read_only:
                raise ValueError("read_only_hardware")
            target = vector(message["target"], 12, "hand_target")
            if ((target < 0) | (target > 255)).any():
                raise ValueError("hand_raw_range")
            if 0 <= now - self.created <= self.timeout:
                self.target = target

    def current(self, now):
        """Return no target after expiry; a heartbeat cannot extend this lease."""
        if not 0 <= now - self.created <= self.timeout:
            self.target = None
        return self.target


def _worker(sock, config, read_only, parent_pid, driver_factory):
    # spawn is deliberate: never inherit a live CycloneDDS participant or threads.
    from .hand_driver import HandDriver

    sock.setblocking(False)
    driver = None
    lease = TargetLease(config.feedback_timeout, read_only)
    heartbeat = time.monotonic()
    try:
        driver = (driver_factory or HandDriver)(config, read_only=read_only)
        heartbeat = time.monotonic()
        while os.getppid() == parent_pid:
            started = time.monotonic()
            for message in _receive(sock):
                if message.get("stop"):
                    return
                heartbeat = max(heartbeat, message["heartbeat"])
                lease.accept(message, started)
            if started - heartbeat > 1.0:
                return
            packet = driver.snapshot()
            now = time.monotonic()
            target = lease.current(now)
            if (
                target is not None
                and not packet.get("fault")
                and 0 <= now - packet.get("hand_time", 0) <= config.feedback_timeout
            ):
                driver.command(target)
            packet["command_writes"] = driver.command_writes
            packet.update(sent=time.monotonic(), sequence=lease.sequence, pid=os.getpid())
            _send(sock, packet)
            time.sleep(max(0, 0.02 - (time.monotonic() - started)))
    except Exception as exc:  # noqa: BLE001 -- isolate SDK/planner failures and latch diagnostics
        try:
            _send(sock, {"fault": f"hand_worker:{type(exc).__name__}:{exc}", "sent": time.monotonic()})
        except OSError:
            pass
    finally:
        if driver is not None:
            driver.close()
        sock.close()


class HandProcess:
    """Keep CAN work out of the DDS process; never restart a failed worker.

    Args:
        config: Robot configuration, also used for target and feedback deadlines.
        read_only: Reject position, speed and torque commands in both processes.
        driver_factory: Optional spawn-compatible fake used by offline tests.
    """

    def __init__(self, config, read_only=False, driver_factory=None):
        self.timeout, self.read_only = config.feedback_timeout, read_only
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.packet = {}
        self.received = 0.0
        self.ipc_error = ""
        self.sequence = 0
        self.target = None
        self.created = 0.0
        self.sock, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.process = multiprocessing.get_context("spawn").Process(
            target=_worker, args=(child, config, read_only, os.getpid(), driver_factory), name="astra-o6"
        )
        try:
            self.process.start()
        except BaseException:
            self.sock.close()
            child.close()
            raise
        child.close()
        self.thread = threading.Thread(target=self._exchange, name="hand-ipc", daemon=True)
        self.thread.start()

    def _exchange(self):
        try:
            while not self.stop.is_set():
                # Serializes send with cancel/set_target; only a small nonblocking
                # datagram write occurs under this lock, never CAN or a wait.
                with self.lock:
                    _send(
                        self.sock,
                        {
                            "sequence": self.sequence,
                            "target": self.target,
                            "created": self.created,
                            "heartbeat": time.monotonic(),
                        },
                    )
                packets = _receive(self.sock)
                if packets:
                    with self.lock:
                        self.packet, self.received = packets[-1], time.monotonic()
                self.stop.wait(0.02)
        except Exception as exc:  # noqa: BLE001 -- isolate SDK/planner failures and latch diagnostics
            with self.lock:
                self.ipc_error = f"hand_ipc:{type(exc).__name__}:{exc}"

    def set_target(self, target):
        """Replace unsent targets; never enqueue a sequence of motor commands."""
        if self.read_only:
            raise ValueError("read_only_hardware")
        values = vector(target, 12, "hand_target")
        if ((values < 0) | (values > 255)).any():
            raise ValueError("hand_raw_range")
        with self.lock:
            self.sequence += 1
            self.target, self.created = values.tolist(), time.monotonic()

    def cancel(self):
        """Invalidate the latest target; a worker must see a new explicit target."""
        with self.lock:
            self.sequence += 1
            self.target, self.created = None, time.monotonic()

    def snapshot(self):
        """Return original sensor ages; receipt of cached data does not freshen it."""
        with self.lock:
            packet = self.packet.copy()
            received, ipc_error = self.received, self.ipc_error
        now = time.monotonic()
        packet["ipc_age"] = now - received
        packet["pid"] = self.process.pid
        if not self.process.is_alive():
            packet["fault"] = packet.get("fault") or "hand_worker_exited"
        elif ipc_error:
            packet["fault"] = ipc_error
        elif not 0 <= now - packet.get("sent", 0) <= self.timeout:
            packet["fault"] = "hand_ipc_stale"
        return packet

    def close(self):
        """Close the owned process without opening hands or starting a replacement."""
        self.stop.set()
        self.thread.join(timeout=1)
        try:
            _send(self.sock, {"stop": True})
        except OSError:
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.sock.close()
