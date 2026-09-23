"""Bounded adaptive torque with early release of bias that would drive overshoot."""

import numpy as np

from .motion_checks import MotionChecks


class AdaptiveBias:
    """Retain Novus gains and limits, but unwind transient bias during braking.

    Model gravity is deliberately outside this class and is never cleared here.
    Relative closing speed comes from successive command-minus-feedback errors,
    so motion of the reference is accounted for without using noisy raw SDK dq.
    """

    def __init__(self, checks: MotionChecks):
        self.checks = checks
        self.bias = np.zeros(14)
        self.reset()

    def reset(self):
        """Reset only after a verified zero-weight handoff, or at construction."""
        self.bias[:] = 0
        self.previous_error = None
        self.error = np.zeros(14)
        self.closing_speed = np.zeros(14)
        self.braking = np.zeros(14, dtype=bool)

    def update(self, error: np.ndarray, dt: float) -> np.ndarray:
        """Integrate steady lag; rate-limit release before or after overshoot."""
        error = np.asarray(error, float)
        if error.shape != (14,) or not np.isfinite(error).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError("invalid_adaptive_feedback")
        self.closing_speed = np.zeros(14) if self.previous_error is None else (self.previous_error - error) / dt
        active_error = np.abs(error) > 0.001
        wrong_direction = active_error & (self.bias * error < 0)
        approaching_fast = (
            (error * self.closing_speed > 0)
            & (self.bias * self.closing_speed > 0)
            & (np.abs(self.closing_speed) > self.checks.adaptive_brake_speed_rad_s)
            & (np.abs(error) <= np.abs(self.closing_speed) * self.checks.adaptive_brake_horizon_s)
        )
        self.braking = wrong_direction | approaching_fast
        integral_error = np.where(active_error, error, 0.0)
        target = self.bias + (80 * integral_error - 0.02 * self.bias) * dt
        # Do not switch the sign of compensation in one braking operation. Keep
        # the existing 20 Nm/s slew limit and 10 Nm bias cap, including on faults.
        target = np.where(self.braking, 0.0, target)
        self.bias[:] = np.clip(self.bias + np.clip(target - self.bias, -20 * dt, 20 * dt), -10, 10)
        self.previous_error = error.copy()
        self.error = error.copy()
        return self.bias
