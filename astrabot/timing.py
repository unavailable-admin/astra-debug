"""Monotonic wall-clock measurements, aggregated in memory until report saves."""

import math
import time
from contextlib import contextmanager
from functools import wraps


class Timings:
    def __init__(self):
        self.started = time.perf_counter()
        self.stages = {}

    def add(self, name, seconds, failed=False):
        stage = self.stages.setdefault(
            name, {"count": 0, "seconds": 0.0, "max_seconds": 0.0, "failures": 0}
        )
        stage["count"] += 1
        stage["seconds"] += seconds
        stage["max_seconds"] = max(stage["max_seconds"], seconds)
        stage["failures"] += int(failed)

    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.add(name, time.perf_counter() - start, failed)

    def snapshot(self):
        return {
            "wall_seconds": time.perf_counter() - self.started,
            "stages": {key: dict(value) for key, value in self.stages.items()},
        }


def timed(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            with self.timings.measure(name):
                return fn(self, *args, **kwargs)
        return wrapped
    return decorate


def server_frame_span(frames):
    """Same-server capture span, NOT full execution or client/server clock subtraction.

    Existing servers expose wall_time per frame, but no batch execution timer.
    This excludes the first action and may include rendering/encoding between
    captures. Unknown/malformed timestamps must never become a measured zero.
    """
    if len(frames) < 2:
        return None
    stamps = [frame.get("wall_time") for frame in frames]
    if any(type(t) not in (int, float) or not math.isfinite(t) for t in stamps):
        return None
    if any(b < a for a, b in zip(stamps, stamps[1:])):
        return None
    return stamps[-1] - stamps[0]
