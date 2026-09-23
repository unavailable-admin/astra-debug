"""Private Unix IPC and independent executor process for local robot ownership."""

import fcntl
import json
import multiprocessing
import os
import signal
import socket
import socketserver
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .collision import enable_acceleration
from .control_gc import ControlGC
from .executor import Executor
from .hardware import MockHardware, RealHardware
from .model import RobotModel
from .trajectory import Planner


def runtime_dir():
    """Use a per-user private directory, independent of current working directory."""
    path = Path(f"/tmp/astra-robot-{os.getuid()}")
    path.mkdir(mode=0o700, exist_ok=True)
    if path.is_symlink() or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError("robot_runtime_directory_must_be_private")
    return path


def socket_path():
    """Default endpoint shared by task CLI and emergency console."""
    return runtime_dir() / "executor.sock"


class Client:
    """Short-lived local requests; a hung task cannot monopolize the server."""

    def __init__(self, path=None, timeout=0.5):
        self.path, self.timeout = str(path or socket_path()), timeout

    def request(self, op, **values):
        """Raise on rejection, unavailable service or missing acknowledgement."""
        started = time.monotonic()
        payload = json.dumps({"op": op, **values}, allow_nan=False).encode() + b"\n"
        for attempt in range(2):
            try:
                status = self._request_once(payload)
                status["ack_seconds"] = time.monotonic() - started
                return status
            except OSError:
                # Read-only status and a lease refresh are safe to repeat with
                # the same generation. Never replay an unacknowledged motion.
                if attempt or op not in ("status", "heartbeat"):
                    raise

    def _request_once(self, payload):
        """Exchange one pre-encoded message on a fresh connection."""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(self.path)
            sock.sendall(payload)
            with sock.makefile("rb") as reader:
                line = reader.readline(65537)
            if not line.endswith(b"\n") or len(line) > 65536:
                raise RuntimeError("invalid_executor_response")
        result = json.loads(line)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "request_failed"))
        return result["status"]


class Handler(socketserver.StreamRequestHandler):
    """One bounded message per connection."""

    def handle(self):
        self.request.settimeout(0.5)
        try:
            line = self.rfile.readline(65537)
            if len(line) > 65536 or not line.endswith(b"\n"):
                raise ValueError("invalid_command_length")
            message = json.loads(line)
            if not isinstance(message, dict):
                raise TypeError("command_must_be_object")
            if message.get("op") == "clear_environment":
                from .cleared_environment import install_cleared_environment

                status = install_cleared_environment(self.server.executor, message)
            elif message.get("op") == "load_scene":
                from .scene_bundle import install_scene

                status = install_scene(self.server.executor, message)
            else:
                status = self.server.executor.request(message)
            response = {"ok": True, "status": status}
        except Exception as exc:  # noqa: BLE001 -- report protocol or hardware failure at the service boundary
            response = {"ok": False, "error": str(exc)}
        try:
            self.wfile.write(json.dumps(response, allow_nan=False).encode() + b"\n")
        except (BrokenPipeError, OSError):
            pass


class Server(socketserver.ThreadingUnixStreamServer):
    """Concurrent handlers keep status/stop independent of a stalled caller."""

    daemon_threads = True
    request_queue_size = 32

    def __init__(self, path, executor):
        self.executor = executor
        super().__init__(str(path), Handler)
        os.chmod(path, 0o600)


def write_audit_record(executor, log, mock, *, final=False):
    """Write a consistent snapshot and acknowledge events only after successful I/O.

    The logger is the sole writer. A final record is flushed and fsynced while
    hardware feedback is still available, before closing DDS/CAN resources.
    """
    with executor.lock:
        events = list(executor.events)
        samples = list(executor.control_samples)
        record = {
            "time": time.monotonic(),
            "mock": mock,
            "status": executor.status(),
            "events": events,
            "final": final,
            "control_samples_dropped": executor.control_samples_dropped,
        }
    # Encode one sample at a time so the C JSON encoder cannot hold the GIL
    # across a whole high-rate batch and starve the 4 ms control loop.
    encoded_samples = []
    for index, sample in enumerate(samples, 1):
        encoded_samples.append(json.dumps(sample, allow_nan=False))
        if index % 8 == 0:
            # Explicitly yield between small batches, including on a busy host.
            # Encoding remains outside the executor lock.
            time.sleep(0)
    samples_json = ",".join(encoded_samples)
    payload = json.dumps(record, allow_nan=False)[:-1] + ',"control_samples":[' + samples_json + "]}\n"
    log.write(payload)
    if final:
        log.flush()
        os.fsync(log.fileno())
    with executor.lock:
        for event in events:
            if executor.events and executor.events[0] is event:
                executor.events.popleft()
        if samples:
            last_sequence = samples[-1]["sequence"]
            while executor.control_samples and executor.control_samples[0]["sequence"] <= last_sequence:
                executor.control_samples.popleft()


def _planning_pool():
    """Warm native geometry only in the isolated planning worker."""
    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=enable_acceleration,
    )


def serve(config, mock=False, path=None):
    """Subscribe first; transmit nothing until an explicit command is accepted."""
    path = Path(path or socket_path())
    lock_path = runtime_dir() / ("mock.lock" if mock else "hardware.lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path.unlink(missing_ok=True)
        hardware = MockHardware() if mock else RealHardware(config)
        pool = None if mock else _planning_pool()
        executor = Executor(hardware, Planner(RobotModel(config), config), config, pool=pool)
        server = Server(path, executor)
        control_gc = ControlGC()
        executor.control_gc = control_gc
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def on_signal(signum, frame):
            if executor.attached:
                executor.request({"op": "pause"})
                print("Paused; executor stays alive to hold. Use checked shutdown to hand off.", flush=True)
            else:
                executor.request({"op": "shutdown"})

        old_signals = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
        output = Path(config.output)
        output.mkdir(parents=True, exist_ok=True)
        log = (output / f"executor-{executor.session}.jsonl").open("a", buffering=1)
        print(
            json.dumps(
                {
                    "socket": str(path),
                    "mock": mock,
                    "session": executor.session,
                    "state": "DISARMED",
                    "motor_commands_sent": 0,
                }
            ),
            flush=True,
        )
        executor.audit_path = str((output / f"executor-{executor.session}.jsonl").resolve())
        logger_stop = threading.Event()

        def logger():
            try:
                while not logger_stop.is_set():
                    write_audit_record(executor, log, mock)
                    logger_stop.wait(1.0)
                write_audit_record(executor, log, mock, final=True)
            except Exception as exc:  # noqa: BLE001 -- isolate audit I/O failures from the control loop
                with executor.lock:
                    if executor.state != "STOPPED":
                        executor.invalidate(hardware.snapshot(), "FAULT" if executor.attached else "DISARMED")
                    executor.error = f"audit_log:{exc}"
                print(f"Audit logging failed: {exc}", flush=True)
            finally:
                log.close()

        logger_thread = threading.Thread(target=logger, daemon=True)
        logger_thread.start()
        try:
            while executor.state != "STOPPED":
                start = time.monotonic()
                try:
                    executor.tick()
                except Exception as exc:  # noqa: BLE001 -- report protocol or hardware failure at the service boundary
                    with executor.lock:
                        executor.invalidate(hardware.snapshot(), "FAULT")
                        executor.error = f"hardware:{exc}"
                time.sleep(max(0.0, 0.004 - (time.monotonic() - start)))
        finally:
            for s, previous in old_signals.items():
                signal.signal(s, previous)
            server.shutdown()
            server.server_close()
            logger_stop.set()
            logger_thread.join(timeout=2.0)
            try:
                if logger_thread.is_alive():
                    raise RuntimeError("audit_logger_did_not_stop")
            finally:
                executor.close()
                try:
                    hardware.close()
                finally:
                    control_gc.close()
                path.unlink(missing_ok=True)
