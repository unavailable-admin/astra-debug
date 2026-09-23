"""Bounded trajectories and WMP-derived arm preparation, with offline preflight."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from .config import GRIP, OPEN_RAISE, RAISE, REST, vector


@dataclass
class Segment:
    """One feedback-confirmed trajectory segment."""

    arms: np.ndarray
    hands: np.ndarray
    target: np.ndarray
    hand_target: np.ndarray
    duration: float
    name: str = "move"
    profile: str = "uniform"
    contact_guard: bool = False
    require_contact: bool = False
    pinch_center: np.ndarray | None = None
    continuous_carry: bool = False
    carry_blend: str = ""
    continuous_path: bool = False

    def sample(self, elapsed):
        """Evaluate smoothstep with shoulder-leading raise ordering."""
        u = float(np.clip(elapsed / self.duration, 0, 1))
        if self.carry_blend:
            progress = {"start": lambda: u * u, "cruise": lambda: u, "end": lambda: 2 * u - u * u}[self.carry_blend]()
        elif self.profile in ("raise", "raise_open"):
            s = lambda x: x**3 * (10 + x * (-15 + 6 * x))
            trailing = s(float(np.clip((u - 0.25) / 0.75, 0, 1)))
            progress = np.full(14, trailing)
            progress[[0, 7]] = s(u)
            if self.profile == "raise_open":
                # Coordinate shoulder roll with pitch throughout the lift;
                # there is no separate outward segment or finger movement.
                progress[[1, 8]] = s(min(2 * u, 1))
        else:
            progress = u * u * (3 - 2 * u)
        return (
            self.arms + progress * (self.target - self.arms),
            self.hands + (progress if self.carry_blend else u * u * (3 - 2 * u)) * (self.hand_target - self.hands),
        )


class Planner:
    """Slow work stays outside the hardware loop and command handler."""

    def __init__(self, model, config):
        self.model, self.config = model, config
        self.recovery_obstacles = ()

    def _clear_thighs_target(self, body, hands):
        """Preserve the fingers and widen a low entry as in Novus preparation."""
        arms = np.asarray(body)[15:].copy()
        if arms[1] >= 0.28 and arms[8] <= -0.28:
            return arms
        poses = self.model.poses(body, hands)
        if np.max(np.abs(arms[[3, 10]] - 0.98)) > self.config.motion_checks.escape_elbow_entry_rad or any(
            poses[f"{side}_wrist_yaw_link"][2, 3] > 0 for side in ("left", "right")
        ):
            raise ValueError("thigh_clearance_requires_low_rest_pose")
        arms[1], arms[8] = max(arms[1], 0.28), min(arms[8], -0.28)
        return arms

    def segment(self, body, hands, target, hand_target, name="move", profile="uniform", **checks):
        """Build and preflight exactly the curve the sender will evaluate."""
        start = vector(body, 29, "body")[15:].copy()
        target, hands, hand_target = (
            vector(v, n, label)
            for v, n, label in ((target, 14, "target"), (hands, 12, "hands"), (hand_target, 12, "hand_target"))
        )
        factor = np.full(14, 2.5 if profile in ("raise", "raise_open") else 1.5)
        if profile == "raise_open":
            factor[[1, 8]] = 3.75  # Quintic peak 1.875 over half the duration.
        duration = max(
            0.02,
            np.max(factor * np.abs(target - start)) / self.config.joint_speed,
            1.5 * np.max(np.abs(hand_target - hands)) / self.config.hand_speed,
        )
        seg = Segment(start, hands.copy(), target.copy(), hand_target.copy(), float(duration), name, profile)
        if name == "clear_thighs":
            if profile != "uniform" or not np.array_equal(hands, hand_target) or checks.get("holding", False):
                raise ValueError("thigh_escape_requires_fixed_empty_hands")
            # Smoothstep is monotonic; its geometric curve equals this dense
            # joint-space sweep. Do not run the same full sweep a second time.
            self.model.check_thigh_escape(body, hands, target, checks.get("extra_obstacles", ()))
            seg.duration = max(
                seg.duration, 1.5 * np.max(np.abs(target - start)) / self.config.motion_checks.escape_speed_rad_s
            )
            return seg
        if profile == "uniform":
            # Arms and fingers share the same monotonic smoothstep parameter.
            # Its geometric sweep is exactly this one joint-space path; sampling
            # time first and resweeping each sample only repeats the same work.
            self.model.check_path(body, start, target, hands, hand_target, **checks)
            return seg
        previous, old_hand = seg.sample(0)
        for t in np.linspace(0, duration, max(2, int(np.ceil(duration * 50)) + 1)):
            current, hand = seg.sample(t)
            self.model.check_path(body, previous, current, old_hand, hand, **checks)
            previous, old_hand = current, hand
        return seg

    def _thigh_escape_segment(self, body, hands, extra_obstacles=(), require_compact=False):
        """Find a bounded outward escape, reserving room for the following hand closure.

        Each candidate passes the same complete escape check. No tolerance is
        relaxed if a candidate fails; caller cannot supply the corrections.
        """
        target = self._clear_thighs_target(body, hands)
        candidates = []
        for width in ((0.28, 0.30, 0.32) if require_compact else (0.28,)):
            for left_yaw, right_yaw in product((0.0, -0.02, 0.02, -0.04, 0.04), repeat=2):
                candidate = target.copy()
                candidate[[1, 8]] = [max(candidate[1], width), min(candidate[8], -width)]
                candidate[[2, 9]] += [left_yaw, right_yaw]
                if (
                    np.max(np.abs((candidate - body[15:])[[1, 8]]))
                    <= self.config.motion_checks.escape_shoulder_roll_rad
                    and max(abs(left_yaw), abs(right_yaw)) <= self.config.motion_checks.escape_shoulder_yaw_rad
                ):
                    candidates.append(candidate)
        candidates.sort(key=lambda q: float(np.sum(np.abs(q - body[15:]))))
        last_error = None
        for candidate in candidates:
            try:
                final = body.copy()
                final[15:] = candidate
                if require_compact:
                    # A safe empty-hand endpoint need not leave enough room
                    # for all fingers to curl. Check that endpoint first.
                    self.model.check(final, GRIP, extra_obstacles=extra_obstacles)
                segment = self.segment(
                    body, hands, candidate, hands, name="clear_thighs", extra_obstacles=extra_obstacles
                )
                if require_compact:
                    self.model.check_path(final, candidate, candidate, hands, GRIP, extra_obstacles=extra_obstacles)
                return segment
            except ValueError as exc:
                if str(exc) not in ("thigh_escape_clearance_decreased", "thigh_escape_wrist_moved_inward") and not (
                    require_compact and str(exc).startswith(("hand_thigh:", "hand_obstacle:", "arm_obstacle:"))
                ):
                    raise
                last_error = exc
        raise ValueError(f"no_monotonic_thigh_escape:{last_error}")

    def ready(self, snapshot):
        """Reach the final ready arm pose before changing any finger target."""
        body, hands = snapshot.body.copy(), snapshot.hands.copy()
        start = body[15:]
        near_rest = np.max(np.abs(start - REST)) <= self.config.motion_checks.rest_entry_rad
        near_raise = np.max(np.abs(start - RAISE)) <= self.config.motion_checks.raised_entry_rad
        targets = []
        if near_rest or near_raise:
            if not near_raise:
                targets.append((OPEN_RAISE.copy(), hands.copy(), "raise", "raise_open"))
            targets.append((np.array(self.config.ready_arms), hands.copy(), "prepare", "uniform"))
        else:
            targets.append((np.array(self.config.ready_arms), hands.copy(), "checked_return", "uniform"))
        targets.append((np.array(self.config.ready_arms), np.array(self.config.ready_hands), "ready_pinch", "uniform"))
        segments = []
        for target, hand, name, profile in targets:
            seg = self.segment(
                body,
                hands,
                target,
                hand,
                name,
                profile,
                extra_obstacles=self.recovery_obstacles,
                thigh_mesh=True,
            )
            segments.append(seg)
            body[15:], hands = seg.target, seg.hand_target
        return segments

    def park(self, snapshot, takeover_arms):
        """Return through checked compact/raise/rest poses before weight handoff."""
        body, hands = snapshot.body.copy(), snapshot.hands.copy()
        segments = []
        entry = body.copy()
        entry[15:] = takeover_arms
        clear = self._clear_thighs_target(entry, np.full(12, 255.0))
        for target, hand, name in (
            (body[15:].copy(), GRIP.copy(), "park_compact"),
            (RAISE.copy(), GRIP.copy(), "park_raise"),
            (clear, GRIP.copy(), "park_clear_thighs"),
            (clear, np.full(12, 255.0), "park_open"),
            (np.asarray(takeover_arms), np.full(12, 255.0), "park_takeover_pose"),
        ):
            segments.append(self.segment(body, hands, target, hand, name, extra_obstacles=self.recovery_obstacles))
            body[15:], hands = target, hand
        return segments

    def move(self, snapshot, request):
        """Plan a bounded joint or Cartesian command against the current snapshot."""
        request = {key: value for key, value in request.items() if key != "expected_scene_id"}
        body, hands = snapshot.body.copy(), snapshot.hands.copy()
        if "action" in request:
            if request["action"] in ("carry_path", "transfer_path", "empty_pinch_path"):
                from .trial_runtime import carry_path

                return carry_path(self, snapshot, request)
            if request["action"] == "pinch_step":
                from .trial_runtime import pinch_step

                return [pinch_step(self, snapshot, request)]
            if request["action"] == "trial_joint_sequence":
                from .trial_runtime import trial_joint_sequence

                return trial_joint_sequence(self, snapshot, request)
            if request["action"] == "trial_joint_path":
                from .trial_runtime import trial_joint_path

                return [trial_joint_path(self, snapshot, request)]
            if request["action"] == "trial_joint_step":
                from .trial_runtime import trial_joint_step

                return [trial_joint_step(self, snapshot, request)]
            if request["action"] == "recover_wrists":
                if set(request) - {"op", "session", "generation", "action"}:
                    raise ValueError("invalid_wrist_recovery_request")
                target = self.model.wrist_recovery_target(body)
                self.model.check_wrist_recovery(body, hands, target)
                duration = (
                    1.5
                    * np.max(np.abs(target - body[15:]))
                    / min(self.config.joint_speed, self.config.motion_checks.wrist_speed_rad_s)
                )
                return [Segment(body[15:].copy(), hands.copy(), target, hands.copy(), duration, "recover_wrists")]
            if request["action"] != "clear_thighs" or set(request) - {"op", "session", "generation", "action"}:
                raise ValueError("invalid_thigh_escape_request")
            return [self._thigh_escape_segment(body, hands)]
        final_hand = vector(request.get("hands", hands), 12, "hands")
        checks = {"holding": bool(request.get("holding", False)), "extra_obstacles": request.get("obstacles", [])}
        if "xyz" not in request:
            target = vector(request.get("arms", body[15:]), 14, "arms")
            return [self.segment(body, hands, target, final_hand, **checks)]
        if not self.config.tcp_measured:
            raise ValueError("tcp_not_measured")
        target = vector(request["xyz"], 3, "xyz")
        quat = vector(request.get("rotation", self.config.grasp_rotation_xyzw), 4, "rotation")
        if abs(np.linalg.norm(quat) - 1) > 1e-6:
            raise ValueError("nonunit_rotation")
        rotation = Rotation.from_quat(quat).as_matrix()
        start, initial_rotation = self.model.tcp(body, hands)
        relative = Rotation.from_matrix(initial_rotation.T @ rotation).as_rotvec()
        count = max(
            1,
            int(np.ceil(np.linalg.norm(target - start) / self.config.motion_checks.cartesian_step_m)),
            int(np.ceil(np.linalg.norm(relative) / self.config.motion_checks.rotation_step_rad)),
        )
        if np.linalg.norm(target - start) > 0.25:
            raise ValueError("split_cartesian_move_at_0.25m")
        segments = []
        original_hand = hands.copy()
        for f in np.linspace(1 / count, 1, count):
            hand = original_hand + f * (final_hand - original_hand)
            q = self.model.solve(
                body,
                hand,
                start + f * (target - start),
                initial_rotation @ Rotation.from_rotvec(relative * f).as_matrix(),
            )
            seg = self.segment(body, hands, q[15:], hand, **checks)
            # Smoothstep's peak speed is 1.5 times average speed.
            old_position, _ = self.model.tcp(body, hands)
            new_position, _ = self.model.tcp(q, hand)
            seg.duration = max(
                seg.duration, 1.5 * np.linalg.norm(new_position - old_position) / self.config.cartesian_speed
            )
            segments.append(seg)
            body, hands = q, hand
        return segments
