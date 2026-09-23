"""Nonblocking command arbitration and feedback-confirmed execution."""

import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from .tracking import (
    TRACKING_POLICY_VERSION,
    TrackingPacer,
    feedback_limited_target,
    novus_command_target,
    path_progress_interval,
)


@dataclass
class Snapshot:
    """Acquisition times are monotonic, not the time a cached value was read."""

    body: np.ndarray
    hands: np.ndarray
    joint_time: float
    hand_time: float
    tactile: list = field(default_factory=list)
    tactile_time: float = 0.0
    fault: str = ""
    mode: int = 4
    temperature: float = 0.0
    shell_temperature: float = 0.0
    hand_temperature: float = 0.0
    body_velocity: np.ndarray | None = None


class Executor:
    """One owner; commands from an invalidated generation can never resume motion."""

    def __init__(self, hardware, planner, config, clock=time.monotonic, pool=None):
        self.hardware, self.planner, self.config, self.clock = hardware, planner, config, clock
        self.lock = threading.RLock()
        self.pool = pool or ThreadPoolExecutor(max_workers=1, thread_name_prefix="trajectory-preflight")
        self.session = uuid.uuid4().hex
        self.generation = 0
        self.state, self.error = "DISARMED", ""
        self.attached = False
        self.arms = self.hands = None
        self.baseline = None
        self.takeover_arms = None
        self.queue = deque()
        self.future = None
        self.planning_request = None
        self.planning_origin = None
        self.planning_refreshes = 0
        self.planning_last_refresh = None
        self.pending_plan = None
        self.planning_stability = None
        self.escape_body_origin = None
        self.escape_refreshes = 0
        self.segment_started = None
        self.segment_elapsed = 0.0
        self.tracking = TrackingPacer(config.motion_checks)
        self.tracking_violation_started = None
        self.arrival_started = None
        self.last_tick = self.clock()
        self.last_tick_timing = {}
        self.deadline_miss = None
        self.lease = None
        self.reset_requested = False
        self.shutdown_requested = False
        self.release_started = None
        self.disarm_writes = 0
        self.weight = 0.0
        self.takeover_active = False
        self.arm_output = "disabled"
        self.arm_hold_needs_feedback = False
        self.command_id = 0
        self.events = deque(maxlen=2000)
        self.control_samples = deque(maxlen=2500)
        self.control_samples_dropped = 0
        self.control_sequence = 0
        self.trace_until = 0.0
        self.last_trace = float("-inf")
        self.cycle_snapshot = None
        self.cycle_reference = None
        self.max_tick_gap = 0.0
        self.thermal_handoff_required = False
        self.thermal_auto_disarm_started = False
        self.contact_detected = False
        self.console_id = None
        self.console_seen = None
        self.console_pause = None
        self.scene_id = None
        self.recovery_scene_valid = False
        self.recovery_notice = ""

    def event(self, kind, **values):
        """Keep a bounded event buffer; disk I/O belongs to the service thread."""
        self.events.append({"time": self.clock(), "kind": kind, "generation": self.generation, **values})

    @property
    def tracking_wait_started(self):
        """Expose the latched wait start to status and timeout checks."""
        return self.tracking.waiting_since

    def health(self, s):
        """Evaluate raw sample ages, driver faults and commissioning limits."""
        now = self.clock()
        if s is None:
            return "no_feedback"
        if (
            not np.isfinite(s.body).all()
            or s.body.shape != (29,)
            or not np.isfinite(s.hands).all()
            or s.hands.shape != (12,)
        ):
            return "invalid_feedback"
        if s.fault:
            return s.fault
        if s.body_velocity is not None and (
            np.shape(s.body_velocity) != (29,) or not np.isfinite(s.body_velocity).all()
        ):
            return "invalid_velocity_feedback"
        if not 0 <= now - s.joint_time <= self.config.feedback_timeout:
            return "joint_feedback_stale"
        if not 0 <= now - s.hand_time <= self.config.feedback_timeout:
            return "hand_feedback_stale"
        if s.mode != self.config.standing_mode:
            return f"standing_mode_required:{s.mode}"
        if not np.isfinite([s.temperature, s.shell_temperature, s.hand_temperature]).all():
            return "invalid_temperature_feedback"
        if self.over_temperature(s):
            return "motor_temperature_limit"
        base_limit = (
            self.config.motion_checks.planning_body_drift_rad
            if self.state == "PLANNING"
            else self.config.motion_checks.motion_body_drift_rad
        )
        if (
            self.escape_body_origin is not None
            and np.max(np.abs(s.body[:15] - self.escape_body_origin)) > self.config.motion_checks.motion_body_drift_rad
        ):
            return "base_or_waist_moved"
        if self.baseline is not None and np.max(np.abs(s.body[:15] - self.baseline)) > base_limit:
            return "base_or_waist_moved"
        return ""

    def over_temperature(self, s):
        """Keep shell and O6 protection independent of the motor interior limit."""
        return (
            s.temperature >= self.config.motor_temperature_limit
            or s.shell_temperature >= self.config.shell_temperature_limit
            or s.hand_temperature >= self.config.hand_temperature_limit
        )

    def invalidate(self, s, state):
        """Drop queued/in-flight work atomically and hold the measured arm pose."""
        self.generation += 1
        self.queue.clear()
        if self.future is not None:
            self.future[1].cancel()
        self.future = None
        self.planning_request = None
        self.planning_origin = None
        self.planning_refreshes = 0
        self.planning_last_refresh = None
        self.pending_plan = None
        self.planning_stability = None
        self.escape_body_origin = None
        self.escape_refreshes = 0
        self.segment_started = None
        self.segment_elapsed = 0.0
        self.tracking.reset()
        self.tracking_violation_started = None
        self.arrival_started = None
        self.lease = None
        self.reset_requested = False
        if state != "FAULT":
            self.shutdown_requested = False
        self.takeover_active = False
        self.hardware.cancel_hand_commands()
        self.state = state
        if s is not None and np.shape(s.body) == (29,) and np.isfinite(s.body).all():
            self.arms = s.body[15:].copy()
        if s is not None and np.shape(s.hands) == (12,) and np.isfinite(s.hands).all():
            self.hands = s.hands.copy()
        self.event(state)

    def request(self, message):
        """Accept local commands without doing IK, CAN I/O or API calls."""
        with self.lock:
            op = message.get("op")
            s = self.hardware.snapshot()
            if op in ("console_heartbeat", "console_detach"):
                console_id = message.get("console_id")
                if not isinstance(console_id, str) or not 1 <= len(console_id) <= 64:
                    raise ValueError("invalid_console_id")
                if op == "console_heartbeat":
                    if console_id != self.console_id:
                        self.console_pause = None
                    self.console_id, self.console_seen = console_id, self.clock()
                elif console_id == self.console_id:
                    self.console_id = self.console_seen = self.console_pause = None
                return self.status(s)
            if (
                op in ("pause", "release", "reset", "shutdown", "disarm", "prepare", "begin")
                and "generation" in message
                and (message.get("session") != self.session or message["generation"] != self.generation)
            ):
                raise ValueError("expired_task_generation")
            if op == "status":
                return self.status(s)
            if op in ("heartbeat", "move", "end", "scene_restored"):
                if (
                    message.get("session") != self.session
                    or message.get("generation") != self.generation
                    or self.lease is None
                ):
                    raise ValueError("expired_task_generation")
                self.lease = self.clock()
                if op == "scene_restored":
                    if (
                        not self.scene_id
                        or message.get("scene_id") != self.scene_id
                        or self.health(s)
                        or self.error
                        or self.queue
                        or self.future is not None
                        or np.max(np.abs(s.body[15:] - self.config.ready_arms)) > self.config.arrival_error
                        or np.max(np.abs(s.hands - self.config.ready_hands)) > self.config.hand_arrival_error
                    ):
                        raise ValueError("scene_restore_requires_verified_trial_return")
                    self.recovery_scene_valid = True
                    self.recovery_notice = ""
                    self.event("trial_scene_restored", scene_id=self.scene_id)
                    return self.status(s)
                if op == "end":
                    self.invalidate(s, "HOLD")
                    return self.status(s)
                if op == "heartbeat":
                    return self.status(s)
            if op == "pause":
                self.invalidate(s, "HOLD" if self.attached else "DISARMED")
                if (
                    self.console_id is not None
                    and message.get("console_id") == self.console_id
                    and self.clock() - self.console_seen <= 3.0
                ):
                    self.console_pause = (self.clock(), self.generation)
                return self.status(s)
            if op == "disarm":
                self.require_damping(s)
                self.invalidate(s, "DISARMING")
                self.release_started = self.clock()
                self.disarm_writes = 0
                self.error = ""
                return self.status(s)
            if self.state == "DISARMING":
                raise ValueError("damping_handoff_in_progress")
            if self.thermal_handoff_required and op != "shutdown":
                raise ValueError("thermal_handoff_required:enter_remote_damping")
            error = self.health(s)
            if op == "shutdown" and not self.attached:
                self.state = "STOPPED"
                return self.status(s)
            if (
                op in ("prepare", "reset", "shutdown")
                and getattr(getattr(self.planner, "model", None), "environment_cleared", False) is True
                and message.get("operator_confirmed_empty") is not True
            ):
                raise ValueError("operator_empty_workspace_confirmation_required")
            if op in ("release", "reset", "shutdown"):
                if (
                    op == "shutdown"
                    and not error
                    and self.weight < 1
                    and self.takeover_arms is not None
                    and np.max(np.abs(s.body[15:] - self.takeover_arms)) > self.config.arrival_error
                ):
                    raise ValueError("handoff_pose_changed:pause_and_explicitly_prepare")
                # A stale arm prevents retraction; fresh hands may still be opened.
                if s is None or not 0 <= self.clock() - s.hand_time <= self.config.feedback_timeout:
                    raise ValueError("cannot_release_without_fresh_hand_feedback")
                self.invalidate(s, "RELEASING")
                self.hands = np.full(12, 255.0)
                self.release_started = self.clock()
                stale_scene = self.scene_id is not None and not self.recovery_scene_valid
                self.reset_requested = op in ("reset", "shutdown") and not error and not stale_scene
                if stale_scene and op in ("reset", "shutdown"):
                    self.recovery_notice = "Scene may have changed during interrupted trial: released/hold; refresh scene before P/R return."
                self.shutdown_requested = op == "shutdown"
                self.error = error
                self.event("release_requested", return_ready=self.reset_requested)
                return self.status(s)
            if error:
                raise ValueError(error)
            if op not in ("begin", "prepare", "move"):
                raise ValueError("unknown_operation")
            if op in ("begin", "prepare"):
                if (
                    op == "begin"
                    and getattr(getattr(self.planner, "model", None), "environment_cleared", False) is True
                ):
                    raise ValueError("trial_requires_fresh_visual_scene")
                if (
                    op == "prepare"
                    and not self.config.scene_measured
                    and getattr(getattr(self.planner, "model", None), "environment_cleared", False) is not True
                ):
                    raise ValueError("prepare_requires_measured_scene:先加载本次有效场景及可观察工作区")
                if op == "prepare" and self.scene_id is not None and not self.recovery_scene_valid:
                    raise ValueError("prepare_requires_refreshed_scene_after_trial_interruption")
                if op == "prepare" and self.lease is not None:
                    raise ValueError("prepare_requires_idle:pause_or_release_current_task_first")
                if self.shutdown_requested:
                    raise ValueError("handoff_pending:retry_shutdown_or_explicitly_pause")
                if self.state in ("RELEASING", "RESETTING", "PLANNING"):
                    raise ValueError("recovery_in_progress")
                self.invalidate(s, "HOLD")
                self.arms, self.hands = s.body[15:].copy(), s.hands.copy()
                self.baseline = s.body[:15].copy()
                if self.takeover_arms is None:
                    self.takeover_arms = s.body[15:].copy()
                self.attached = True
                self.takeover_active = True
                self.error = ""
                if op == "begin":
                    if self.scene_id is not None:
                        self.recovery_scene_valid = False
                    self.lease = self.clock()
                    self.state = "RUNNING"
                    return self.status(s)
            if op == "move" and "expected_scene_id" in message:
                message = dict(message)
                expected_scene = message.get("expected_scene_id")
                if not self.scene_id or expected_scene != self.scene_id:
                    raise ValueError("motion_scene_changed")
            if self.future is not None or self.queue:
                raise ValueError("motion_in_progress")
            if (
                op == "move"
                and message.get("action") in ("pinch_step", "carry_path", "transfer_path")
                and message.get("holding")
                and self.config.trial_grip_mode == "supervised_position"
            ):
                requested = np.asarray(message.get("hands", s.hands), float)
                if (
                    self.hands is None
                    or requested.shape != (12,)
                    or not np.isfinite(requested).all()
                    or not np.allclose(requested[:6], self.hands[:6], atol=1e-8, rtol=0)
                ):
                    raise ValueError("held_grip_command_changed")
            if op == "move" and message.get("preserve_right"):
                requested_hands = np.asarray(message.get("hands"), float)
                requested_arms = np.asarray(
                    (
                        message.get("arms", [])[7:]
                        if message.get("action") in ("trial_joint_step", "trial_joint_path", "trial_joint_sequence")
                        else message.get("right_arms")
                    ),
                    float,
                )
                if (
                    requested_hands.shape != (12,)
                    or requested_arms.shape != (7,)
                    or self.hands is None
                    or self.arms is None
                    or not np.allclose(requested_hands[6:], self.hands[6:], atol=1e-8, rtol=0)
                    or not np.allclose(requested_arms, self.arms[7:], atol=1e-8, rtol=0)
                ):
                    raise ValueError("right_hold_command_changed")
            self.command_id += 1
            self.contact_detected = False
            self.state = "PLANNING"
            self.planning_request = (op, dict(message))
            self.planning_origin = s
            self.planning_refreshes = 0
            gen = self.generation
            if op == "prepare":
                self.pending_plan = (op, self.clock())
                self.planning_stability = None
            else:
                future = self.pool.submit(self.planner.move, s, message)
                self.future = (gen, future, s, self.clock())
            self.event("planning", command_id=self.command_id)
            return self.status(s)

    def advance_pending_planning(self, s, now):
        """Start expensive preparation only after takeover and observed settling."""
        op, started = self.pending_plan
        checks = self.config.motion_checks
        if now - started > checks.planning_settle_timeout_s:
            self.invalidate(s, "FAULT" if self.shutdown_requested else "HOLD")
            self.error = "preparation_start_did_not_settle"
            return
        if self.takeover_active or self.weight < 1.0:
            self.planning_stability = None
            return
        if self.planning_stability is None:
            self.planning_stability = (s, now)
            return
        previous, since = self.planning_stability
        if (
            np.max(np.abs(s.body - previous.body)) > checks.escape_recheck_rad
            or np.max(np.abs(s.hands - previous.hands)) > checks.fixed_hand_drift_raw
        ):
            self.planning_stability = (s, now)
            return
        if now - since < checks.planning_stable_window_s:
            return
        try:
            future = (
                self.pool.submit(self.planner.park, s, self.takeover_arms)
                if op == "park"
                else self.pool.submit(self.planner.ready, s)
            )
            self.future = (self.generation, future, s, now)
            self.pending_plan = self.planning_stability = None
            self.event("preparation_start_settled", waited_s=now - started)
        except Exception as exc:  # noqa: BLE001 -- preserve hold if the worker failed
            self.invalidate(s, "FAULT" if self.shutdown_requested else "HOLD")
            self.error = f"preflight:{exc}"

    def refresh_escape(self, s, now):
        """Recertify standalone preparation off-loop; never change trial intent."""
        op = self.planning_request[0] if self.planning_request else None
        if (
            self.lease is not None
            or op not in ("prepare", "park")
            or self.escape_refreshes >= self.config.motion_checks.escape_refreshes
        ):
            self.invalidate(s, "HOLD")
            self.error = "thigh_escape_body_recheck_required"
            return
        self.escape_refreshes += 1
        self.queue.clear()
        self.hardware.cancel_hand_commands()
        self.arms, self.hands = s.body[15:].copy(), s.hands.copy()
        self.segment_started = self.arrival_started = None
        self.tracking.reset()
        self.tracking_violation_started = None
        self.segment_elapsed = 0.0
        self.state = "PLANNING"
        try:
            future = (
                self.pool.submit(self.planner.ready, s)
                if op == "prepare"
                else self.pool.submit(self.planner.park, s, self.takeover_arms)
            )
            self.future = (self.generation, future, s, now)
            self.event("thigh_escape_body_recheck", attempt=self.escape_refreshes)
        except Exception as exc:  # noqa: BLE001 -- hold if the planning worker is unavailable
            self.invalidate(s, "HOLD")
            self.error = f"preflight:{exc}"

    def status(self, s=None):
        """Return observed and commanded state without extending the task lease."""
        s = self.hardware.snapshot() if s is None else s
        return {
            "session": self.session,
            "audit_path": getattr(self, "audit_path", None),
            "scene_id": self.scene_id,
            "scene_protocol_version": 3,
            "cleared_environment_version": 1,
            "environment_cleared": bool(getattr(getattr(self.planner, "model", None), "environment_cleared", False)),
            "recovery_scene_valid": self.recovery_scene_valid,
            "generation": self.generation,
            "state": self.state,
            "hardware_backend": type(self.hardware).__name__,
            "standing_mode": self.config.standing_mode,
            "feedback_timeout": self.config.feedback_timeout,
            "feedback_fault": s.fault if s is not None else "no_feedback",
            "joint_age": self.clock() - s.joint_time if s is not None else None,
            "hand_age": self.clock() - s.hand_time if s is not None else None,
            "task_active": self.lease is not None,
            "planning_active": self.future is not None or self.pending_plan is not None,
            "planning_waiting_for_stability": self.pending_plan is not None,
            "planning_refreshes": self.planning_refreshes,
            "planning_last_refresh": self.planning_last_refresh,
            "planning_body_drift_limit": self.config.motion_checks.planning_body_drift_rad,
            "tracking_policy_version": TRACKING_POLICY_VERSION,
            "tracking_wait_started": self.tracking_wait_started,
            "tracking_progress_scale": self.tracking.scale,
            "control_samples_dropped": self.control_samples_dropped,
            "motion_checks": asdict(self.config.motion_checks),
            "motion_body_drift_limit": self.config.motion_checks.motion_body_drift_rad,
            "escape_refreshes": self.escape_refreshes,
            "prepare_arrival_error": min(self.config.motion_checks.prepare_arrival_rad, self.config.tracking_error),
            "console_channel": {
                "connected": self.console_seen is not None and 0 <= self.clock() - self.console_seen <= 3.0,
                "age": self.clock() - self.console_seen if self.console_seen is not None else None,
                "pause_age": self.clock() - self.console_pause[0] if self.console_pause else None,
                "pause_generation": self.console_pause[1] if self.console_pause else None,
            },
            "grasp_ready": bool(
                s is not None
                and not self.health(s)
                and not self.error
                and self.state == "READY"
                and self.attached
                and self.weight == 1.0
                and not self.queue
                and self.future is None
                and np.max(np.abs(s.body[15:] - self.config.ready_arms))
                <= min(self.config.motion_checks.prepare_arrival_rad, self.config.tracking_error)
                and np.max(np.abs(s.hands - self.config.ready_hands)) <= self.config.hand_arrival_error
            ),
            "error": self.error,
            "health": self.health(s),
            "attached": self.attached,
            "weight": self.weight,
            "takeover_active": self.takeover_active,
            "handoff_pending": self.shutdown_requested,
            "disarm_zero_frames": self.disarm_writes,
            "arm_output": self.arm_output,
            "thermal_handoff_required": self.thermal_handoff_required,
            "contact_detected": self.contact_detected,
            "operator_action": (
                "THERMAL STOP: retained SDK hold may remain; support robot and use L2+B damping; "
                "zero-weight handoff follows verified damping"
                if self.thermal_handoff_required
                else self.recovery_notice
            ),
            "diagnostics": self.hardware.diagnostics() if hasattr(self.hardware, "diagnostics") else {},
            "command_id": self.command_id,
            "remaining_segments": len(self.queue),
            "max_tick_gap_seconds": self.max_tick_gap,
            "last_tick_timing": dict(self.last_tick_timing),
            "deadline_miss": self.deadline_miss,
            "control_gc": self.control_gc.snapshot() if hasattr(self, "control_gc") else None,
            "tick_age": self.clock() - self.last_tick,
            "body": s.body.tolist() if s is not None else None,
            "hands": s.hands.tolist() if s is not None else None,
            "tactile": s.tactile if s is not None else [],
            "tactile_age": self.clock() - s.tactile_time if s is not None else None,
            "joint_time": s.joint_time if s is not None else None,
            "hand_time": s.hand_time if s is not None else None,
            "temperature": s.temperature if s is not None else None,
            "shell_temperature": s.shell_temperature if s is not None else None,
            "hand_temperature": s.hand_temperature if s is not None else None,
            "temperature_limits": {
                "motor": self.config.motor_temperature_limit,
                "shell": self.config.shell_temperature_limit,
                "hand": self.config.hand_temperature_limit,
            },
            "mode": s.mode if s is not None else None,
            "arms_command": self.arms.tolist() if self.arms is not None else None,
            "hands_command": self.hands.tolist() if self.hands is not None else None,
            "grip_hold_policy_version": 1,
            "carry_path_version": 1,
            "trial_joint_path_version": 1,
            "transfer_path_version": 2,
        }

    def tick(self):
        """Capture every motion tick, including the reference before a fault hold.

        Samples are bounded in memory and drained by the audit thread. No JSON
        encoding or disk I/O occurs on the control thread.
        """
        entered = self.clock()
        with self.lock:
            now = self.clock()
            self.tick_lock_wait = max(0.0, now - entered)
            if self.queue:
                self.trace_until = now + 2.0
            self.cycle_snapshot = self.cycle_reference = None
            try:
                self._tick()
            finally:
                s = self.cycle_snapshot
                if (
                    s is not None
                    and (self.attached or now <= self.trace_until)
                    and (now <= self.trace_until or now - self.last_trace >= 0.05)
                ):
                    self.last_trace = now
                    self.control_sequence += 1
                    if len(self.control_samples) == self.control_samples.maxlen:
                        self.control_samples_dropped += 1
                    self.control_samples.append(
                        {
                            "sequence": self.control_sequence,
                            "time": now,
                            "joint_time": s.joint_time,
                            "state": self.state,
                            "generation": self.generation,
                            "error": self.error,
                            "q_feedback": s.body[15:].tolist(),
                            "dq_feedback": (
                                np.asarray(s.body_velocity)[15:].tolist()
                                if s.body_velocity is not None
                                and np.shape(s.body_velocity) == (29,)
                                and np.isfinite(s.body_velocity).all()
                                else None
                            ),
                            "q_command": self.arms.tolist() if self.arms is not None else None,
                            "reference": self.cycle_reference,
                            # Driver record includes the exact feedback used by
                            # the encoder and only successfully published frames.
                            "published": getattr(self.hardware, "last_arm_command", None),
                        }
                    )

                self.last_tick_timing = {
                    "lock_wait_seconds": self.tick_lock_wait,
                    "work_seconds": max(0.0, self.clock() - now),
                    "finished_at": self.clock(),
                }

    def _tick(self):
        """Advance bounded motion; segment completion requires newer physical feedback."""
        with self.lock:
            now = self.clock()
            elapsed = max(0, now - self.last_tick)
            self.max_tick_gap = max(self.max_tick_gap, elapsed)
            dt = min(elapsed, 0.02)
            self.last_tick = now
            s = self.hardware.snapshot()
            self.cycle_snapshot = s
            if (
                self.attached
                and s is not None
                and np.isfinite(s.temperature)
                and self.over_temperature(s)
                and not self.thermal_handoff_required
            ):
                self.invalidate(s, "FAULT")
                self.error = "motor_temperature_limit"
                self.thermal_handoff_required = True
                self.event("thermal_handoff_required", temperature=s.temperature)
            if (
                self.thermal_handoff_required
                and not self.thermal_auto_disarm_started
                and self.state != "DISARMING"
                and s is not None
                and s.mode == 1
            ):
                try:
                    self.require_damping(s)
                except ValueError:
                    pass
                else:
                    self.invalidate(s, "DISARMING")
                    self.thermal_auto_disarm_started = True
                    self.release_started = now
                    self.disarm_writes = 0
            if self.state == "DISARMING":
                self.arm_output = "disabled"
                try:
                    self.require_damping(s)
                    self.hardware.clear_arm_control()
                    self.disarm_writes += 1
                    self.weight = 0.0
                    self.arm_output = "relinquishing_zero_weight"
                    if now - self.release_started >= 0.5 and self.disarm_writes >= 10:
                        self.attached = False
                        self.baseline = self.takeover_arms = None
                        self.arm_hold_needs_feedback = False
                        self.state = "DISARMED"
                        self.thermal_handoff_required = False
                        self.thermal_auto_disarm_started = False
                        self.error = ""
                        self.arm_output = "disabled"
                        self.event("damping_handoff_complete", zero_frames=self.disarm_writes)
                except Exception as exc:  # noqa: BLE001 -- failed writes never count as a completed handoff
                    self.invalidate(s, "FAULT")
                    self.error = f"damping_handoff:{exc}"
                return
            error = "motor_temperature_limit" if self.thermal_handoff_required else self.health(s)
            if self.attached and error and self.state not in ("FAULT", "RELEASING"):
                self.invalidate(s, "FAULT")
                self.error = error
            if self.lease is not None and now - self.lease > self.config.lease_timeout:
                self.invalidate(s, "HOLD")
                self.error = "task_lease_expired"
            if self.attached and elapsed > 0.1 and self.state not in ("FAULT", "RELEASING"):
                self.invalidate(s, "FAULT")
                self.error = "control_deadline_missed"
                self.deadline_miss = {
                    "gap_seconds": elapsed,
                    "limit_seconds": 0.1,
                    "lock_wait_seconds": self.tick_lock_wait,
                    "previous_tick": dict(self.last_tick_timing),
                    "gc": self.control_gc.snapshot() if hasattr(self, "control_gc") else None,
                }
                self.event("control_deadline_missed", **self.deadline_miss)
            if self.state == "RELEASING":
                if (
                    s is not None
                    and s.hand_time > self.release_started
                    and np.max(np.abs(s.hands - 255)) <= self.config.hand_arrival_error
                ):
                    if self.reset_requested and not error:
                        self.attached = True
                        self.baseline = s.body[:15].copy()
                        if self.takeover_arms is None:
                            self.takeover_arms = s.body[15:].copy()
                        if (
                            self.shutdown_requested
                            and np.max(np.abs(s.body[15:] - self.takeover_arms)) <= self.config.arrival_error
                        ):
                            self.state = "HANDOFF"
                        elif self.shutdown_requested and self.weight < 1:
                            self.invalidate(s, "FAULT")
                            self.error = "handoff_pose_changed"
                        else:
                            self.takeover_active = True
                            self.pending_plan = ("park" if self.shutdown_requested else "prepare", now)
                            self.planning_stability = None
                            self.planning_request = ("park" if self.shutdown_requested else "prepare", {})
                            self.planning_origin = s
                            self.planning_refreshes = 0
                            self.state = "PLANNING"
                    else:
                        if self.shutdown_requested:
                            self.invalidate(s, "FAULT")
                            self.error = error or self.error or "handoff_interrupted"
                        else:
                            self.state = "HOLD" if self.attached else "DISARMED"
                elif now - self.release_started > self.config.settle_timeout:
                    self.invalidate(s, "FAULT")
                    self.error = "hand_open_timeout"
            if self.pending_plan is not None and not error:
                self.advance_pending_planning(s, now)
            if self.future is not None:
                gen, future, initial, _started = self.future
                if future.done():
                    self.future = None
                    try:
                        segments = future.result()
                        if gen != self.generation:
                            return
                        escaping = bool(segments and segments[0].name == "clear_thighs")
                        recovering = bool(segments and segments[0].name == "recover_wrists")
                        preparing = self.lease is None and not self.shutdown_requested
                        if (
                            error
                            or np.max(np.abs(s.body - initial.body))
                            > (
                                self.config.motion_checks.wrist_planning_drift_rad
                                if recovering
                                else self.config.motion_checks.planning_body_drift_rad
                            )
                            or (
                                self.planning_origin is not None
                                and np.max(np.abs(s.body - self.planning_origin.body))
                                > self.config.motion_checks.planning_body_drift_rad
                            )
                            or np.max(np.abs(s.hands - initial.hands))
                            > (
                                self.config.motion_checks.fixed_hand_drift_raw
                                if escaping or recovering or preparing
                                else self.config.motion_checks.planning_hand_drift_raw
                            )
                        ):
                            raise ValueError(error or "state_changed_during_planning")
                        drift = float(np.max(np.abs(s.body - initial.body)))
                        bound_scene = self.planning_request and self.planning_request[1].get("expected_scene_id")
                        if bound_scene and bound_scene != self.scene_id:
                            raise ValueError("motion_scene_changed")
                        recheck_limit = (
                            self.config.motion_checks.scene_task_rad
                            if bound_scene
                            else (
                                self.config.motion_checks.escape_recheck_rad
                                if escaping
                                else self.config.motion_checks.planning_recheck_rad
                            )
                        )
                        hand_recheck = bool(bound_scene) and (
                            np.max(np.abs(s.hands - initial.hands))
                            > self.config.motion_checks.scene_task_hand_raw + 1e-6
                        )
                        if not recovering and (drift > recheck_limit or hand_recheck):
                            # The larger admission limit is not permission to
                            # replay a close-contact curve from its old start.
                            # Recompute from fresh feedback outside the loop.
                            if (
                                self.planning_request is None
                                or self.planning_refreshes >= self.config.motion_checks.planning_refreshes
                            ):
                                raise ValueError(
                                    "planning_start_did_not_settle_after_refresh"
                                    f":body_rad={drift:.9f}:limit_rad={recheck_limit:.9f}"
                                    f":hand_raw={np.max(np.abs(s.hands - initial.hands)):.9f}"
                                    f":hand_limit_raw={self.config.motion_checks.scene_task_hand_raw if bound_scene else self.config.motion_checks.planning_hand_drift_raw:.9f}"
                                    f":refreshes={self.planning_refreshes}"
                                )
                            op, request = self.planning_request
                            self.planning_refreshes += 1
                            # Re-latching tiny steady tracking offsets shifts
                            # the hold target, then creates another planning drift.
                            # Preserve a close held target; re-anchor large deviations.
                            if np.max(np.abs(self.arms - s.body[15:])) > min(
                                self.config.tracking_error, self.config.motion_checks.escape_tracking_soft_rad
                            ):
                                right_arms = self.arms[7:].copy()
                                right_hands = self.hands[6:].copy()
                                self.arms = s.body[15:].copy()
                                if not (
                                    self.planning_request[1].get("holding")
                                    and self.config.trial_grip_mode == "supervised_position"
                                ):
                                    self.hands = s.hands.copy()
                                if self.planning_request[1].get("preserve_right"):
                                    self.arms[7:] = right_arms
                                    self.hands[6:] = right_hands
                            refreshed = (
                                self.pool.submit(self.planner.ready, s)
                                if op == "prepare"
                                else (
                                    self.pool.submit(self.planner.park, s, self.takeover_arms)
                                    if op == "park"
                                    else self.pool.submit(self.planner.move, s, request)
                                )
                            )
                            self.future = (gen, refreshed, s, now)
                            self.planning_last_refresh = {
                                "drift_rad": drift,
                                "attempt": self.planning_refreshes,
                                "threshold_rad": recheck_limit,
                                "hand_drift_raw": float(np.max(np.abs(s.hands - initial.hands))),
                                "joint_index": int(np.argmax(np.abs(s.body - initial.body))),
                            }
                            self.event("planning_start_refreshed", **self.planning_last_refresh)
                        else:
                            # Preparation now uses the base pose against which
                            # this accepted path was checked, not takeover's pose.
                            if self.lease is None:
                                self.baseline = initial.body[:15].copy()
                            self.event(
                                "path_checked",
                                command_id=self.command_id,
                                planning_s=now - _started,
                                segments=len(segments),
                            )
                            self.queue.extend(segments)
                            self.segment_started = None
                            self.segment_elapsed = 0.0
                            self.tracking.reset()
                            self.tracking_violation_started = None
                            self.arrival_started = None
                            self.state = "RESETTING" if self.lease is None else "RUNNING"
                    except Exception as exc:  # noqa: BLE001 -- isolate SDK/planner failures and latch diagnostics
                        self.invalidate(s, "FAULT" if self.shutdown_requested else "HOLD")
                        self.error = f"preflight:{exc}"
            if self.queue and not error and self.weight >= 1.0:
                segment = self.queue[0]
                if segment.contact_guard or segment.require_contact:
                    tactile = np.asarray(s.tactile, float)
                    if (
                        tactile.size < 2
                        or not np.isfinite(tactile[:2]).all()
                        or not 0 <= now - s.tactile_time <= self.config.feedback_timeout
                    ):
                        self.invalidate(s, "FAULT")
                        self.error = "pinch_tactile_stale_or_invalid"
                        return
                    if segment.require_contact and min(tactile[:2]) < self.config.tactile_threshold:
                        self.invalidate(s, "FAULT")
                        self.error = "trial_lost_contact"
                        return
                    if segment.contact_guard and min(tactile[:2]) >= self.config.tactile_threshold:
                        self.hardware.cancel_hand_commands()
                        self.arms, self.hands = s.body[15:].copy(), s.hands.copy()
                        self.queue.clear()
                        self.contact_detected = True
                        self.segment_started = None
                        self.segment_elapsed = 0.0
                        self.tracking.reset()
                        self.tracking_violation_started = None
                        self.arrival_started = None
                        self.event("pinch_contact_stop", command_id=self.command_id)
                        self.hardware.command(self.arms, None, self.weight)
                        self.arm_output = "hold"
                        return
                if segment.name == "clear_thighs":
                    if self.escape_body_origin is None:
                        self.escape_body_origin = self.baseline.copy()
                    if (
                        np.max(np.abs(s.body[:15] - self.baseline)) > self.config.motion_checks.escape_recheck_rad
                        and np.max(np.abs(s.hands - segment.hands)) <= self.config.motion_checks.fixed_hand_drift_raw
                        and np.max(np.abs(segment.sample(self.segment_elapsed)[0] - s.body[15:]))
                        <= min(self.config.tracking_error, self.config.motion_checks.escape_tracking_hard_rad)
                    ):
                        # The 0.05 admission bound does not extend the old mesh
                        # certificate. Hold, then recertify from measured pose.
                        self.refresh_escape(s, now)
                        self.hardware.command(self.arms, None, self.weight)
                        self.arm_output = "hold"
                        return
                if self.segment_started is None:
                    self.segment_started = now
                escaping = segment.name == "clear_thighs"
                recovering = segment.name == "recover_wrists"
                continuous_motion = not self.shutdown_requested and not (escaping or recovering)
                continuous_prepare = continuous_motion and self.lease is None
                fixed_prepare_hands = continuous_prepare and segment.name in ("raise", "prepare", "checked_return")
                ready_hand_stage = continuous_prepare and segment.name == "ready_pinch"
                tracking_limit = (
                    min(self.config.tracking_error, self.config.motion_checks.escape_tracking_soft_rad)
                    if escaping
                    else self.config.tracking_error
                )
                if recovering:
                    tracking_limit = min(tracking_limit, self.config.motion_checks.wrist_tracking_rad)
                # Preparation and tasks permit time lag. Explicit escape,
                # wrist recovery and shutdown retain reference-based pacing.
                reference, _ = segment.sample(self.segment_elapsed)
                lag = 0.0 if continuous_motion else float(np.max(np.abs(reference - s.body[15:])))
                was_waiting = self.tracking_wait_started is not None
                if continuous_motion:
                    # Novus advances the reference in time and clips the motor
                    # target around feedback. Do not turn ordinary lag into a
                    # stop/start cycle. Keep a separate geometric path envelope.
                    self.tracking.reset()
                    progress_dt = dt
                else:
                    progress_dt = self.tracking.advance(
                        lag,
                        tracking_limit,
                        now,
                        dt,
                        s.joint_time,
                        s.body[15:],
                        None if s.body_velocity is None else s.body_velocity[15:],
                    )
                waiting = self.tracking_wait_started is not None
                if waiting and not was_waiting:
                    self.event(
                        "tracking_waiting_for_feedback",
                        error_rad=lag,
                        segment=segment.name,
                        threshold_rad=self.config.motion_checks.tracking_wait_ratio * tracking_limit,
                        resume_threshold_rad=(
                            self.config.motion_checks.tracking_resume_ratio
                            * self.config.motion_checks.tracking_wait_ratio
                            * tracking_limit
                        ),
                        timeout_s=self.config.motion_checks.tracking_wait_timeout_s,
                        joint_index=15 + int(np.argmax(np.abs(reference - s.body[15:]))),
                    )
                elif was_waiting and not waiting:
                    self.event("tracking_resumed", error_rad=lag, segment=segment.name, speed_rad_s=self.tracking.speed)
                self.segment_elapsed += progress_dt
                age = self.segment_elapsed
                desired, hand = segment.sample(age)
                # Normal motion uses Novus's feedback-relative command bound;
                # explicit recovery and shutdown retain their slew limit.
                if continuous_motion:
                    self.arms = novus_command_target(
                        desired, s.body[15:], self.config.motion_checks.prepare_command_lead_rad
                    )
                else:
                    self.arms = feedback_limited_target(desired, self.arms, self.config.joint_speed * dt)
                self.cycle_reference = {
                    "generation": self.generation,
                    "segment": segment.name,
                    "elapsed": self.segment_elapsed,
                    "q_reference": desired.tolist(),
                    "q_command_candidate": self.arms.tolist(),
                    "waiting": waiting,
                    "progress_scale": self.tracking.scale,
                    "feedback_speed_rad_s": self.tracking.speed,
                    "feedback_raw_speed_rad_s": self.tracking.raw_speed,
                    "resume_blocker": self.tracking.resume_blocker,
                    "resume_stable_since": self.tracking.stable_since,
                    "tracking_mode": (
                        "novus_prepare"
                        if continuous_prepare
                        else "novus_task" if continuous_motion else "feedback_paced"
                    ),
                }
                if not waiting:
                    hand_speed = self.config.hand_speed * (
                        self.config.trial_speed_scale if self.lease is not None else 1.0
                    )
                    self.hands += np.clip(hand - self.hands, -hand_speed * dt, hand_speed * dt)
                errors = desired - s.body[15:]
                tracking_error = 0.0 if continuous_motion else float(np.max(np.abs(errors)))
                if tracking_error > tracking_limit:
                    if self.tracking_violation_started is None:
                        self.tracking_violation_started = now
                        self.event("tracking_tolerance_exceeded", errors_rad=errors.tolist(), segment=segment.name)
                else:
                    self.tracking_violation_started = None
                # A brief escape overshoot freezes the path rather than
                # latching a fault on one sample. Do not advance while waiting;
                # large excursions and wrist-limit recovery still stop at once.
                hard_limit = (
                    min(self.config.tracking_error, self.config.motion_checks.escape_tracking_hard_rad)
                    if escaping
                    else tracking_limit
                )
                tracking_failed = max(tracking_error, float(np.max(np.abs(self.arms - s.body[15:])))) > hard_limit or (
                    self.tracking_violation_started is not None
                    and now - self.tracking_violation_started >= self.config.motion_checks.tracking_transient_s
                )
                path_failed = False
                if continuous_motion:
                    tracking_failed = False
                    intervals = [
                        path_progress_interval(q, segment.arms, segment.target, tracking_limit, segment.profile)
                        for q in (s.body[15:], self.arms)
                    ]
                    self.cycle_reference["feedback_path_interval"], self.cycle_reference["command_path_interval"] = (
                        intervals
                    )
                    path_failed = any(interval is None for interval in intervals)
                recovery_bad = False
                if recovering:
                    moving = segment.target != segment.arms
                    displacement = s.body[15:] - segment.arms
                    inward = displacement[moving] * np.sign((segment.target - segment.arms)[moving])
                    recovery_bad = (
                        np.any(inward < -self.config.motion_checks.wrist_reverse_rad)
                        or np.any(
                            inward
                            > np.abs((segment.target - segment.arms)[moving])
                            + self.config.motion_checks.wrist_envelope_rad
                        )
                        or np.any(np.abs(displacement[~moving]) > self.config.motion_checks.wrist_envelope_rad)
                        or np.max(np.abs(s.body[:15] - self.baseline)) > self.config.motion_checks.wrist_envelope_rad
                        or np.max(np.abs(s.hands - segment.hands)) > self.config.motion_checks.fixed_hand_drift_raw
                    )
                if path_failed:
                    self.invalidate(s, "FAULT")
                    self.error = (
                        "prepare_feedback_outside_path_envelope"
                        if continuous_prepare
                        else "task_feedback_outside_path_envelope"
                    )
                elif fixed_prepare_hands and (
                    not np.array_equal(segment.hands, segment.hand_target)
                    or np.max(np.abs(s.hands - segment.hands)) > self.config.motion_checks.fixed_hand_drift_raw
                ):
                    self.invalidate(s, "FAULT")
                    self.error = "prepare_fingers_moved_before_ready"
                elif ready_hand_stage and (
                    np.max(np.abs(s.body[15:] - segment.target))
                    > min(self.config.motion_checks.prepare_arrival_rad, self.config.tracking_error)
                ):
                    self.invalidate(s, "FAULT")
                    self.error = "prepare_arm_left_ready_pose"
                elif recovery_bad:
                    self.invalidate(s, "FAULT")
                    self.error = "wrist_recovery_feedback_outside_envelope"
                elif escaping and (
                    np.max(np.abs(s.hands - segment.hands)) > self.config.motion_checks.fixed_hand_drift_raw
                ):
                    self.invalidate(s, "FAULT")
                    self.error = "thigh_escape_body_or_fingers_moved"
                elif tracking_failed:
                    self.event(
                        "tracking_limit_exceeded",
                        errors_rad=errors.tolist(),
                        limit_rad=tracking_limit,
                        hard_limit_rad=hard_limit,
                        segment=segment.name,
                    )
                    self.invalidate(s, "FAULT")
                    self.error = "arm_tracking_error"
                elif waiting and now - self.tracking_wait_started > self.config.motion_checks.tracking_wait_timeout_s:
                    self.event(
                        "tracking_wait_timeout",
                        resume_blocker=self.tracking.resume_blocker,
                        feedback_speed_rad_s=self.tracking.speed,
                        feedback_raw_speed_rad_s=self.tracking.raw_speed,
                        resume_stable_since=self.tracking.stable_since,
                        segment=segment.name,
                        reference_errors_rad=errors.tolist(),
                        command_errors_rad=(self.arms - s.body[15:]).tolist(),
                        threshold_rad=self.config.motion_checks.tracking_wait_ratio * tracking_limit,
                        timeout_s=self.config.motion_checks.tracking_wait_timeout_s,
                    )
                    self.invalidate(s, "FAULT")
                    self.error = (
                        "wrist_recovery_tracking_wait_timeout"
                        if recovering
                        else "thigh_escape_tracking_wait_timeout" if escaping else "arm_tracking_wait_timeout"
                    )
                elif age >= segment.duration:
                    if self.arrival_started is None:
                        self.arrival_started = now
                    fresh = min(s.joint_time, s.hand_time) > self.arrival_started
                    arm_arrival = (
                        min(
                            self.config.arrival_error,
                            (
                                self.config.motion_checks.escape_arrival_rad
                                if escaping
                                else self.config.motion_checks.wrist_arrival_rad
                            ),
                        )
                        if escaping or recovering
                        else self.config.arrival_error
                    )
                    if continuous_prepare and (
                        segment.name == "ready_pinch" or (len(self.queue) == 2 and self.queue[1].name == "ready_pinch")
                    ):
                        arm_arrival = min(self.config.motion_checks.prepare_arrival_rad, self.config.tracking_error)
                    reached = np.max(np.abs(s.body[15:] - segment.target)) <= arm_arrival and np.max(
                        np.abs(s.hands - segment.hand_target)
                    ) <= (
                        min(self.config.hand_arrival_error, self.config.motion_checks.fixed_hand_drift_raw)
                        if escaping or recovering
                        else self.config.hand_arrival_error
                    )
                    pinch_error = None
                    if fresh and segment.pinch_center is not None:
                        from ..pinch import INDEX_TIP, THUMB_TIP

                        poses = self.planner.model.pinch_poses(s.body, self.config.pinch_reference_hands(s.hands))
                        measured_center = (
                            (poses["left_index_distal"] @ INDEX_TIP)[:3] + (poses["left_thumb_distal"] @ THUMB_TIP)[:3]
                        ) / 2
                        pinch_error = float(np.linalg.norm(measured_center - segment.pinch_center))
                        # A 3 mm step must not complete under a 0.03 rad joint-only
                        # tolerance. Hold the same checked command while feedback
                        # catches up; never enlarge the requested motion or torque.
                        pinch_arrival = min(
                            self.config.motion_checks.pinch_arrival_m, self.config.motion_checks.pinch_substep_m / 2
                        )
                        if segment.name in ("transfer_path", "empty_pinch_path"):
                            pinch_arrival = self.config.motion_checks.transfer_arrival_m
                        reached = reached and pinch_error <= pinch_arrival
                    carry_through = len(self.queue) > 1 and (
                        (segment.continuous_carry and self.queue[1].continuous_carry)
                        or (segment.continuous_path and self.queue[1].continuous_path)
                    )
                    if carry_through or (fresh and reached):
                        self.queue.popleft()
                        self.segment_started = now
                        self.segment_elapsed = 0.0
                        self.tracking.reset()
                        self.tracking_violation_started = None
                        self.arrival_started = None
                        if not self.queue:
                            self.state = (
                                "HANDOFF"
                                if self.shutdown_requested
                                else ("RUNNING" if self.lease is not None else "READY")
                            )
                            self.event("motion_complete", command_id=self.command_id, pinch_error_m=pinch_error)
                    elif now - self.arrival_started > (
                        self.config.motion_checks.prepare_arrival_timeout_s
                        if continuous_prepare
                        else self.config.settle_timeout
                    ):
                        self.invalidate(s, "FAULT")
                        self.error = (
                            f"pinch_arrival_timeout:remaining_mm={pinch_error * 1000:.3f}"
                            f":limit_mm={pinch_arrival * 1000:.3f}"
                            if pinch_error is not None
                            else "arrival_timeout"
                        )
            self.arm_output = "disabled"
            if self.attached and self.arms is not None and (not error or self.can_hold_arms(s, error)):
                if self.arm_hold_needs_feedback:
                    self.arms = s.body[15:].copy()
                    if self.state in ("FAULT", "HOLD"):
                        self.hands = s.hands.copy()
                    self.arm_hold_needs_feedback = False
                next_weight = self.weight
                if not error and self.state == "HANDOFF":
                    next_weight = max(0.0, self.weight - dt / 2.0)
                elif not error and self.takeover_active:
                    next_weight = min(1.0, self.weight + dt / 2.0)
                # No hand target is refreshed during a feedback fault. The worker
                # invalidation/lease stops the previous target independently.
                self.hardware.command(
                    self.arms, self.hands if not error and self.state != "FAULT" else None, next_weight
                )
                self.weight = next_weight
                if self.weight == 1:
                    self.takeover_active = False
                self.arm_output = "hold" if error or self.state in ("FAULT", "HOLD") else "active"
                if self.state == "HANDOFF" and self.weight == 0:
                    self.attached = False
                    self.shutdown_requested = False
                    self.state = "STOPPED"
                    self.arm_output = "disabled"
                    self.event("handoff_complete")
            elif self.attached:
                self.arm_hold_needs_feedback = True
                self.arm_output = f"blocked:{error or self.error}"
            if self.state == "RELEASING" and self.hands is not None:
                self.hardware.release_hands()

    def require_damping(self, s):
        """Require fresh arm feedback and known damping before clearing ownership."""
        if s is None or s.body.shape != (29,) or not np.isfinite(s.body).all():
            raise ValueError("invalid_joint_feedback")
        if not 0 <= self.clock() - s.joint_time <= self.config.feedback_timeout:
            raise ValueError("joint_feedback_stale")
        if s.mode != 1:
            raise ValueError("damping_mode_required")
        if s.fault and not s.fault.startswith("hand_"):
            raise ValueError(s.fault)

    def can_hold_arms(self, s, error):
        """Allow constant-weight hold only for isolated mode-query/hand faults.

        Fresh joint data, the last known standing mode and physical limits are
        still required. An actual mode change or joint/motor fault blocks output.
        """
        if error != "locomotion_mode_stale" and not error.startswith("hand_"):
            return False
        if s is None or s.body.shape != (29,) or not np.isfinite(s.body).all():
            return False
        if not 0 <= self.clock() - s.joint_time <= self.config.feedback_timeout:
            return False
        if (
            s.mode != self.config.standing_mode
            or not np.isfinite([s.temperature, s.shell_temperature, s.hand_temperature]).all()
            or self.over_temperature(s)
        ):
            return False
        return (
            self.baseline is None
            or np.max(np.abs(s.body[:15] - self.baseline)) <= self.config.motion_checks.motion_body_drift_rad
        )

    def close(self):
        """Stop planner workers after the service has handed off hardware."""
        self.pool.shutdown(wait=False, cancel_futures=True)
