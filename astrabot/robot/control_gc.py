"""Keep initialized runtime graphs out of recurring cyclic garbage scans."""

import gc
import time


class ControlGC:
    """Freeze long-lived initialization objects; new cyclic garbage stays collected."""

    def __init__(self):
        self.started = None
        self.last = None
        self.maximum = 0.0
        self.count = 0
        self.preexisting_freeze = gc.get_freeze_count()
        # Called before serving control requests, while no arm ownership exists.
        gc.collect()
        gc.freeze()
        gc.callbacks.append(self.observe)

    def observe(self, phase, info):
        """Record GC pauses without logging or acquiring the control mutex."""
        if phase == "start":
            self.started = time.monotonic()
        elif self.started is not None:
            elapsed = time.monotonic() - self.started
            self.maximum = max(self.maximum, elapsed)
            self.count += 1
            self.last = {"started_at": self.started, "seconds": elapsed, "generation": info["generation"]}
            self.started = None

    def snapshot(self):
        """Return small timing counters; never trigger collection."""
        return {"count": self.count, "max_seconds": self.maximum, "last": self.last}

    def close(self):
        """Restore collection after hardware control has closed."""
        gc.callbacks.remove(self.observe)
        if not self.preexisting_freeze:
            gc.unfreeze()
