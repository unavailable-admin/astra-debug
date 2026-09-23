"""Command slew limiting and feedback-paced trajectory progress."""

from collections import deque

import numpy as np

TRACKING_POLICY_VERSION = 11


def novus_command_target(target, feedback, max_delta=0.08):
    """Match Novus arm clip (5 rad/s at 20 ms), independently of path speed.

    The 0.08 rad component bound is a command lead limit, not a tracking fault
    threshold. Never multiply the slow trajectory speed by our 4 ms tick here.
    """
    if not np.isfinite(max_delta) or not 0 < max_delta <= 0.08:
        raise ValueError("invalid_novus_command_delta")
    target, feedback = np.asarray(target, float), np.asarray(feedback, float)
    # Validate the complete vectors before clipping; NaN must not become a hold.
    feedback_limited_target(target, feedback, 0.1)
    bounded = np.clip(target, feedback - max_delta, feedback + max_delta)
    return feedback_limited_target(bounded, feedback, 5.0 * 0.02)


def path_progress_interval(position, start, target, limit, profile="uniform"):
    """Find common geometric progress whose joint errors all fit the envelope.

    A single common progress is essential: independent joint ranges would let
    a stalled shoulder and a moving elbow leave the checked curve. Fixed joints
    retain the same limit. This changes timing admission, not envelope width.
    """
    position, start, target = (np.asarray(value, float) for value in (position, start, target))
    if (
        any(value.shape != (14,) or not np.isfinite(value).all() for value in (position, start, target))
        or not np.isfinite(limit)
        or limit <= 0
        or profile not in ("uniform", "raise", "raise_open")
    ):
        raise ValueError("invalid_path_envelope")
    delta = target - start
    moving = np.abs(delta) > 1e-12
    if np.any(np.abs(position[~moving] - start[~moving]) > limit):
        return None
    if not moving.any():
        return (0.0, 1.0)
    a = (position[moving] - start[moving] - limit) / delta[moving]
    b = (position[moving] - start[moving] + limit) / delta[moving]
    lower, upper = np.minimum(a, b), np.maximum(a, b)
    if np.any(lower > 1) or np.any(upper < 0):
        return None
    lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
    if profile == "uniform":
        # Cubic smoothstep has the same geometric line as linear progress.
        lo, hi = float(np.max(lower)), float(np.min(upper))
    else:
        # Invert monotonic quintic progress, including the shoulder-leading delay.
        # Use inner bounds so finite bisection precision cannot enlarge admission.
        wanted = np.stack((lower, upper))
        left, right = np.zeros_like(wanted), np.ones_like(wanted)
        for _ in range(28):
            middle = (left + right) * 0.5
            value = middle**3 * (10 + middle * (-15 + 6 * middle))
            left = np.where(value < wanted, middle, left)
            right = np.where(value >= wanted, middle, right)
        delay = np.where(np.isin(np.flatnonzero(moving), [0, 7]), 0.0, 0.25)
        span = 1 - delay
        if profile == "raise_open":
            roll = np.isin(np.flatnonzero(moving), [1, 8])
            delay[roll], span[roll] = 0, 0.5
        low_time = np.where(lower == 0, 0, delay + span * right[0])
        high_time = np.where(upper == 1, 1, delay + span * left[1])
        lo, hi = float(np.max(low_time)), float(np.min(high_time))
    return (lo, hi) if lo <= hi else None


class TrackingPacer:
    """Latch a pause until fresh, quiet feedback settles below a lower threshold.

    Only trajectory time is scaled. The caller must keep commanding the frozen
    reference, enforce its original error limits and check the wait timeout.
    """

    def __init__(self, checks):
        self.checks = checks
        self.reset()

    def reset(self):
        """Discard timing and feedback history at a segment/generation boundary."""
        self.waiting_since = None
        self.stable_since = None
        self.ramp_elapsed = self.checks.tracking_resume_ramp_s
        self.scale = 1.0
        self.feedback_time = None
        self.feedback_position = None
        self.speed = None
        self.raw_speed = None
        self.speed_history = deque(maxlen=128)
        self.resume_blocker = ""

    def _update_speed(self, joint_time, position, velocity):
        """Average per-joint absolute speed by acquisition time, never signed dq.

        Keep one sample before the window boundary for partial interval weights.
        Position total variation also guards against misleadingly quiet SDK dq.
        """
        sample_dt = None if self.feedback_time is None else joint_time - self.feedback_time
        position_speed = None
        if sample_dt is not None and sample_dt > 0:
            position_speed = np.abs(position - self.feedback_position) / sample_dt
        absolute_speed = np.abs(velocity) if velocity is not None else position_speed
        self.raw_speed = None if absolute_speed is None else float(np.max(absolute_speed))
        self.feedback_time = joint_time
        self.feedback_position = np.array(position, copy=True)
        self.speed_history.append((joint_time, absolute_speed, position_speed))
        start = joint_time - self.checks.tracking_speed_window_s
        while len(self.speed_history) > 1 and self.speed_history[1][0] <= start:
            self.speed_history.popleft()
        self.speed = None
        if self.speed_history[0][0] > start + 1e-9:
            return  # Require a full observation window before recovery.
        speed_integral, position_integral = np.zeros(14), np.zeros(14)
        covered = 0.0
        previous_time = self.speed_history[0][0]
        for stamp, speed, observed in list(self.speed_history)[1:]:
            duration = stamp - max(start, previous_time)
            previous_time = stamp
            if speed is None or observed is None or duration <= 0:
                continue
            speed_integral += speed * duration
            position_integral += observed * duration
            covered += duration
        if covered >= self.checks.tracking_speed_window_s - 1e-9:
            self.speed = float(max(np.max(speed_integral), np.max(position_integral)) / covered)

    def advance(self, lag, limit, now, dt, joint_time, position, velocity=None):
        """Return trajectory-time increment; duplicates cannot complete recovery."""
        fresh = self.feedback_time is None or joint_time > self.feedback_time
        if fresh:
            self._update_speed(joint_time, position, velocity)

        pause_limit = self.checks.tracking_wait_ratio * limit
        resume_limit = self.checks.tracking_resume_ratio * pause_limit
        if self.waiting_since is None and lag >= pause_limit:
            self.waiting_since = now
            self.stable_since = None
            self.ramp_elapsed = 0.0
        if self.waiting_since is not None:
            if lag >= resume_limit:
                self.resume_blocker = "position"
            elif self.speed is None:
                self.resume_blocker = "speed_window_incomplete"
            elif self.speed > self.checks.tracking_resume_speed_rad_s:
                self.resume_blocker = "speed"
            elif self.raw_speed > self.checks.tracking_resume_peak_speed_rad_s:
                self.resume_blocker = "speed_peak"
            else:
                self.resume_blocker = "stable_window"
            quiet = self.resume_blocker == "stable_window"
            if not quiet:
                self.stable_since = None
            elif fresh:
                if self.stable_since is None:
                    self.stable_since = joint_time
                # Never reset an expired timeout, including on a late good sample.
                if (
                    joint_time - self.stable_since >= self.checks.tracking_resume_stable_s
                    and now - self.waiting_since <= self.checks.tracking_wait_timeout_s
                ):
                    self.waiting_since = None
                    self.stable_since = None
                    self.resume_blocker = ""
            if self.waiting_since is not None:
                self.scale = 0.0
                return 0.0

        self.ramp_elapsed = min(self.checks.tracking_resume_ramp_s, self.ramp_elapsed + dt)
        u = self.ramp_elapsed / self.checks.tracking_resume_ramp_s
        self.scale = u * u * (3 - 2 * u)
        return dt * self.scale


def feedback_limited_target(target, feedback, max_step):
    """Scale all joint deltas together, as Novus clipArmQTarget does.

    For command slew limiting pass the last command as the anchor. Passing
    feedback instead bounds position error, which is NOT a velocity bound.

    This limits a command; callers must measure lag against the original
    trajectory, never against this clipped result.
    """
    target, feedback = np.asarray(target, float), np.asarray(feedback, float)
    if (
        target.shape != (14,)
        or feedback.shape != (14,)
        or not np.isfinite(target).all()
        or not np.isfinite(feedback).all()
        or not np.isfinite(max_step)
        or max_step < 0
    ):
        raise ValueError("invalid_tracking_limiter_input")
    if max_step == 0:
        return feedback.copy()
    scale = max(1.0, float(np.max(np.abs(target - feedback))) / max_step)
    return feedback + (target - feedback) / scale
