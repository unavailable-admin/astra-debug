"""Bound Python decoding work by polling the newest DDS joint state."""

import threading
import time


class LatestStateReader:
    """Read at most one new sample per control period, without DDS callbacks.

    Uses Novus XR's dedicated read-thread approach. DDS retains only its latest
    sample, so high-rate intermediate packets are discarded before Python decode.

    Args:
        callback: Receives the decoded message and local monotonic acquisition time.
        reader: Optional nonblocking reader for offline tests.
        period: Polling period in seconds.
        clock: Monotonic clock, injectable for offline tests.
        start: Start the worker immediately; tests may instead call poll_once.
    """

    def __init__(self, callback, *, reader=None, period=0.004, clock=time.monotonic, start=True):
        self._callback, self._clock, self._period = callback, clock, period
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._fault = ""
        self._count = self._duplicates = 0
        self._last_tick = None
        self._max_read_seconds = 0.0
        self._thread = None
        self._participant = self._topic = None
        if reader is None:
            from cyclonedds.domain import DomainParticipant
            from cyclonedds.qos import Policy, Qos
            from cyclonedds.sub import DataReader
            from cyclonedds.topic import Topic
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

            # ChannelFactoryInitialize has already configured domain 0. Use public
            # DDS APIs and our own participant; never access SDK private members.
            self._participant = DomainParticipant(0)
            self._topic = Topic(self._participant, "rt/lowstate", LowState_)
            reader = DataReader(
                self._participant,
                self._topic,
                qos=Qos(Policy.History.KeepLast(1), Policy.Reliability.BestEffort),
            )
        self._reader = reader
        if start:
            self._thread = threading.Thread(target=self._run, name="latest-joint-state", daemon=True)
            self._thread.start()

    def poll_once(self):
        """Consume a new sample once; empty reads and repeated ticks stay stale."""
        started = self._clock()
        samples = self._reader.take(1)
        acquired = self._clock()
        with self._lock:
            self._max_read_seconds = max(self._max_read_seconds, acquired - started)
        for sample in samples:
            if not sample.sample_info.valid_data:
                continue
            tick = int(sample.tick)
            if self._last_tick is not None:
                delta = (tick - self._last_tick) & 0xFFFFFFFF
                if delta == 0:
                    with self._lock:
                        self._duplicates += 1
                    continue
                if delta >= 0x80000000:
                    raise ValueError("joint_tick_regressed")
            self._callback(sample, acquired)
            self._last_tick = tick
            with self._lock:
                self._count += 1

    def _run(self):
        try:
            while not self._stop.is_set():
                started = self._clock()
                self.poll_once()
                self._stop.wait(max(0, self._period - (self._clock() - started)))
        except Exception as exc:  # noqa: BLE001 -- latch DDS/decode failure instead of restarting
            with self._lock:
                self._fault = f"joint_reader:{type(exc).__name__}:{exc}"

    def diagnostics(self):
        """Report failures and acquisition counts without refreshing feedback."""
        with self._lock:
            return {
                "fault": self._fault,
                "samples": self._count,
                "duplicates": self._duplicates,
                "last_tick": self._last_tick,
                "max_read_seconds": self._max_read_seconds,
                "period": self._period,
                "alive": self._thread.is_alive() if self._thread is not None else False,
            }

    def close(self):
        """Stop the reader before freeing its DDS entities; never restart it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            if self._thread.is_alive():
                raise RuntimeError("joint_reader_did_not_stop")
        self._reader = self._topic = self._participant = None
