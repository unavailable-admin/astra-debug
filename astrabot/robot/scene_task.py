"""Supplement executor checks with the freshly measured task-side scene."""

import asyncio
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

from .collision import enable_acceleration
from .trajectory import Planner


class SceneCheckedTask:
    """Validate each requested curve against updated geometry before dispatch.

    The executor retains ownership, pause, thermal, contact and its own geometry
    checks. No live executor settings are modified. A changed planning start is
    recomputed locally; failed geometry never sends a physical command.
    """

    def __init__(self, task, model, config, *, isolated=True):
        self.task = task
        self.planner = Planner(model, config)
        self.settle_timeout = config.settle_timeout
        self.motion_checks = config.motion_checks
        self.checks = []
        self.executor_scene_id = None
        self.pool = (
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=enable_acceleration,
            )
            if isolated
            else None
        )

    def close(self):
        """Release the planning worker after the owning task has stopped."""
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def status(self):
        """Preserve the underlying task's lease and fault checks."""
        return self.task.status()

    async def move(self, **values):
        """Check the actual curve with current scene and bounded start drift."""
        state = self.status()
        if (
            self.executor_scene_id is not None
            and state.get("transfer_path_version", 0) >= 2
            and values.get("action")
            in (
                "transfer_path",
                "carry_path",
                "empty_pinch_path",
                "pinch_step",
                "trial_joint_path",
                "trial_joint_sequence",
                "trial_joint_step",
            )
        ):
            if state.get("scene_id") != self.executor_scene_id:
                raise ValueError("motion_scene_changed")
            timing = {"action": values["action"], "checked_by": "executor", "scene_id": self.executor_scene_id}
            self.checks.append(timing)
            started = time.monotonic()
            try:
                return await self.task.move(**values, expected_scene_id=self.executor_scene_id)
            finally:
                timing["elapsed_s"] = time.monotonic() - started
        deadline = time.monotonic() + self.settle_timeout
        while time.monotonic() < deadline:
            before = self.status()
            snapshot = SimpleNamespace(body=np.array(before["body"]), hands=np.array(before["hands"]))
            if self.pool is None:
                segments = await asyncio.to_thread(self.planner.move, snapshot, values)
            else:
                segments = await asyncio.get_running_loop().run_in_executor(
                    self.pool, self.planner.move, snapshot, values
                )
            if len(segments) != 1 and not (
                segments
                and (
                    (
                        values.get("action") in ("carry_path", "transfer_path", "empty_pinch_path")
                        and all(s.continuous_carry for s in segments)
                    )
                    or (values.get("action") == "trial_joint_sequence" and all(s.continuous_path for s in segments))
                )
            ):
                raise ValueError("scene_task_requires_single_segment")
            after = self.status()
            body_drift = float(np.max(np.abs(np.asarray(after["body"]) - snapshot.body)))
            hand_drift = float(np.max(np.abs(np.asarray(after["hands"]) - snapshot.hands)))
            if (
                body_drift <= self.motion_checks.scene_task_rad
                and hand_drift <= self.motion_checks.scene_task_hand_raw + 1e-6
            ):
                self.checks.append(
                    {"body_drift": body_drift, "hand_drift": hand_drift, "action": values.get("action", "joint")}
                )
                return await self.task.move(**values)
            await asyncio.sleep(min(0.1, max(0, deadline - time.monotonic())))
        raise ValueError("scene_task_start_did_not_settle")
