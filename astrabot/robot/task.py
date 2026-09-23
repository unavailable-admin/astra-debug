"""Task leases, cancellation and feedback-confirmed local commands."""

import asyncio
import threading
import time


class Task:
    """An operator interruption permanently invalidates this task instance."""

    def __init__(self, client, expected=None):
        self.client = client
        status = client.request("begin", **(expected or {}))
        self.token = {k: status[k] for k in ("session", "generation")}
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _heartbeat(self):
        while not self.stop.wait(0.25):
            try:
                self.client.request("heartbeat", **self.token)
            except (OSError, RuntimeError, ValueError) as exc:
                self.error = exc
                self.stop.set()

    def status(self):
        """Check lease and operator cancellation before accepting observations."""
        if self.error:
            # A planning fault increments generation before heartbeat notices.
            # Preserve the executor cause instead of hiding it behind expiry.
            reason = str(self.error)
            try:
                failed = self.client.request("status")
            except (OSError, RuntimeError, ValueError):
                pass
            else:
                if failed.get("session") == self.token["session"]:
                    reason = failed.get("error") or failed.get("health") or reason
            raise RuntimeError(f"task_cancelled:{reason}") from self.error
        status = self.client.request("status")
        if any(status[k] != v for k, v in self.token.items()):
            raise RuntimeError(status.get("error") or status.get("health") or "task_generation_changed")
        if status["health"] or status["error"] or status["state"] in ("FAULT", "HOLD", "DISARMED"):
            raise RuntimeError(status["health"] or status["error"] or status["state"])
        return status

    async def wait(self, timeout=120):
        """Wait for measured completion while the independent heartbeat continues."""
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            state = self.status()
            if state["weight"] >= 1 and state["state"] == "RUNNING" and not state["remaining_segments"]:
                return state
            await asyncio.sleep(0.04)
        raise TimeoutError("motion_completion_timeout")

    async def move(self, **values):
        """Send one checked trajectory; never queue stale long-horizon actions."""
        self.status()
        self.client.request("move", **self.token, **values)
        return await self.wait()

    def close(self):
        """End only our own lease; do not override an operator's reset or release."""
        self.stop.set()
        self.thread.join(timeout=1.0)
        try:
            return self.client.request("end", **self.token)
        except (RuntimeError, OSError):
            pass
