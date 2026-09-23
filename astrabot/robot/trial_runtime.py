"""Feedback-conditioned, supervised single-cube commissioning.

Uses the explicit model fingertip basis without changing measured-TCP flags.
The generic calibrated Cartesian controller remains gated independently.
"""

import asyncio
import time
from dataclasses import replace
from itertools import groupby

import numpy as np
from scipy.spatial.transform import Rotation

from ..pinch import INDEX_TIP, THUMB_TIP
from .collision import segment_hits_box
from .config import vector


def pinch_pose(model, body, hands):
    """Return the original pinch anchor and wrist rotation, preserving independent thumb yaw."""
    poses = model.poses(body, model.config.pinch_reference_hands(hands))
    center = ((poses["left_index_distal"] @ INDEX_TIP)[:3] + (poses["left_thumb_distal"] @ THUMB_TIP)[:3]) / 2
    return center, poses["left_wrist_yaw_link"][:3, :3]


def contact_is_fresh(state, config):
    """Require finite, independently acquired opposing left-finger contacts."""
    values = np.asarray(state.get("tactile", []), float)
    age = state.get("tactile_age")
    return bool(
        values.size >= 2
        and np.isfinite(values[:2]).all()
        and age is not None
        and 0 <= age <= config.feedback_timeout
        and min(values[:2]) >= config.tactile_threshold
    )


def check_sweep(planner, body, hands, segment, obstacles, holding=False, pose_cache=None):
    """Bound actual FK speed and check the carried cube along a joint sweep."""
    from .trial_plan import TrialPlanner

    trial = TrialPlanner(planner.model, planner.config)
    count = max(
        20,
        int(
            np.ceil(
                max(
                    np.max(np.abs(segment.target - segment.arms)),
                    np.max(np.abs(segment.hand_target - segment.hands)) / 255 * 1.6,
                )
                / planner.config.motion_checks.path_step_rad
            )
        ),
    )
    offset = np.asarray(planner.config.trial_grasp_offset)
    # A cache is local to this synchronous immutable-model sweep unless the
    # caller supplies its path-local cache. All original samples still run.
    pose_cache = {} if pose_cache is None else pose_cache
    fixed = np.array_equal(hands, segment.hand_target)
    local_points = None
    if fixed:
        key = ("tips", hands.tobytes())
        if key not in pose_cache:
            center, _ = pinch_pose(planner.model, body, hands)
            actual = planner.model.pinch_poses(body, hands)
            wrist = actual["left_wrist_yaw_link"]
            actual_center = (
                (actual["left_index_distal"] @ INDEX_TIP)[:3] + (actual["left_thumb_distal"] @ THUMB_TIP)[:3]
            ) / 2
            pose_cache[key] = (np.array([center, wrist[:3, 3], actual_center]) - wrist[:3, 3]) @ wrist[:3, :3]
        local_points = pose_cache[key]
    previous = None
    derivative = 0.0
    for alpha in np.linspace(0, 1, count + 1):
        q = body.copy()
        q[15:] = segment.arms + alpha * (segment.target - segment.arms)
        h = hands + alpha * (segment.hand_target - hands)
        if fixed:
            key = ("wrist", q.tobytes())
            if key not in pose_cache:
                pose_cache[key] = planner.model.tcp(q, h)
            position, rotation = pose_cache[key]
            points = local_points @ rotation.T + position
            center = points[0]
        else:
            center, _ = pinch_pose(planner.model, q, h)
            actual = planner.model.pinch_poses(q, h)
            wrist = actual["left_wrist_yaw_link"][:3, 3]
            actual_center = (
                (actual["left_index_distal"] @ INDEX_TIP)[:3] + (actual["left_thumb_distal"] @ THUMB_TIP)[:3]
            ) / 2
            points = np.array([center, wrist, actual_center])
        if previous is not None:
            derivative = max(derivative, float(np.max(np.linalg.norm(points - previous, axis=1))) * count)
        if holding:
            trial.check_cube(center - [0, 0, 0.015] - offset, obstacles)
            if previous is not None:
                for lower, upper in (*planner.config.obstacles, *obstacles):
                    if segment_hits_box(
                        previous[0] - [0, 0, 0.015] - offset,
                        center - [0, 0, 0.015] - offset,
                        lower,
                        upper,
                        0.024 + planner.config.clearance,
                    ):
                        raise ValueError("trial_cube_swept_obstacle")
        previous = points
    segment.duration = max(segment.duration, 1.5 * derivative * 1.02 / planner.config.cartesian_speed)
    return segment


def trial_joint_step(planner, snapshot, request):
    """Recheck a saved empty-hand joint segment from the current observed pose."""
    allowed = {"op", "session", "generation", "action", "arms", "hands", "obstacles", "duration", "preserve_right"}
    if set(request) - allowed:
        raise ValueError("invalid_trial_joint_request")
    if type(request.get("preserve_right", False)) is not bool:
        raise ValueError("invalid_preserve_right")
    arms = vector(request["arms"], 14, "arms")
    hands = vector(request["hands"], 12, "hands")
    if np.max(np.abs(arms - snapshot.body[15:])) > planner.config.motion_checks.trial_joint_step_rad:
        raise ValueError("trial_joint_step_too_large")
    if (
        np.max(np.abs(arms[7:] - snapshot.body[22:])) > planner.config.motion_checks.trial_right_arm_rad
        or np.max(np.abs(hands[6:] - snapshot.hands[6:]))
        > (
            min(planner.config.hand_arrival_error, 5.0)
            if request.get("preserve_right")
            else planner.config.motion_checks.fixed_hand_drift_raw
        )
        + 1e-6
    ):
        raise ValueError("trial_must_preserve_right_arm_and_hand")
    obstacles = request.get("obstacles", [])
    segment = planner.segment(snapshot.body, snapshot.hands, arms, hands, extra_obstacles=obstacles)
    duration = float(request.get("duration", 0.02))
    if not np.isfinite(duration) or not 0.02 <= duration <= 30:
        raise ValueError("invalid_trial_duration")
    segment = check_sweep(planner, snapshot.body, snapshot.hands, segment, obstacles)
    segment.duration = max(0.02, segment.duration / planner.config.trial_speed_scale, duration)
    return segment


def trial_joint_path(planner, snapshot, request):
    """Certify bounded pieces, then traverse their straight path with one ramp."""
    from types import SimpleNamespace

    target = vector(request["arms"], 14, "arms")
    hand = vector(request["hands"], 12, "hands")
    duration = float(request.get("duration", 0.02))
    if not np.isfinite(duration) or not 0.02 <= duration <= 120:
        raise ValueError("invalid_trial_duration")
    right_hand_limit = (
        min(planner.config.hand_arrival_error, 5.0)
        if request.get("preserve_right")
        else planner.config.motion_checks.fixed_hand_drift_raw
    )
    if (
        np.max(np.abs(target[7:] - snapshot.body[22:])) > planner.config.motion_checks.trial_right_arm_rad
        or np.max(np.abs(hand[6:] - snapshot.hands[6:])) > right_hand_limit + 1e-6
    ):
        raise ValueError("trial_must_preserve_right_arm_and_hand")
    count = max(
        1,
        int(
            np.ceil(np.max(np.abs(target - snapshot.body[15:])) / planner.config.motion_checks.trial_joint_substep_rad)
        ),
    )
    current = SimpleNamespace(body=snapshot.body.copy(), hands=snapshot.hands.copy())
    segments = []
    for fraction in np.linspace(1 / count, 1, count):
        step = dict(
            request,
            action="trial_joint_step",
            arms=snapshot.body[15:] + fraction * (target - snapshot.body[15:]),
            hands=snapshot.hands + fraction * (hand - snapshot.hands),
            duration=max(0.02, duration / count),
        )
        segment = trial_joint_step(planner, current, step)
        segments.append(segment)
        current.body[15:] = segment.target
        current.hands = segment.hand_target.copy()
    # Every piece lies on this same straight joint/hand curve. A single
    # smoothstep removes intermediate zero-velocity boundaries.
    return replace(
        segments[0],
        target=target.copy(),
        hand_target=hand.copy(),
        duration=sum(segment.duration for segment in segments),
        name="trial_joint_path",
    )


def trial_joint_sequence(planner, snapshot, request):
    """Check every saved waypoint before traversing a complete empty-hand phase."""
    from types import SimpleNamespace

    points = request["waypoints"]
    if not request.get("preserve_right") or not isinstance(points, list) or not 1 <= len(points) <= 512:
        raise ValueError("invalid_trial_joint_sequence")
    current = SimpleNamespace(body=snapshot.body.copy(), hands=snapshot.hands.copy())
    segments = []
    for point in points:
        arms = vector(point["arms"], 14, "arms").copy()
        hands = vector(point["hands"], 12, "hands").copy()
        arms[7:] = vector(request["arms"], 14, "arms")[7:]
        hands[6:] = vector(request["hands"], 12, "hands")[6:]
        step = {k: v for k, v in request.items() if k != "waypoints"}
        step.update(action="trial_joint_path", arms=arms, hands=hands, duration=point["duration"])
        segment = trial_joint_path(planner, current, step)
        segment.continuous_path = True
        segments.append(segment)
        current.body[15:] = segment.target
        current.hands = segment.hand_target.copy()
    if len(segments) > 1:
        for index, segment in enumerate(segments):
            segment.carry_blend = "start" if index == 0 else "end" if index == len(segments) - 1 else "cruise"
            if segment.carry_blend != "cruise":
                segment.duration *= 2
    return segments


def pinch_step(planner, snapshot, request, *, carry_cache=None):
    """Plan one <=6 mm model-pinch step, including closure compensation."""
    allowed = {
        "op",
        "session",
        "generation",
        "action",
        "center",
        "rotation",
        "hands",
        "obstacles",
        "closing",
        "holding",
        "opening",
        "preserve_right",
        "right_arms",
        "goal_center",
    }
    if set(request) - allowed or np.any(np.asarray(planner.config.tcp_offset) != 0):
        raise ValueError("invalid_model_pinch_request")
    for flag in ("closing", "holding", "opening", "preserve_right"):
        if type(request.get(flag, False)) is not bool:
            raise ValueError("invalid_pinch_flag")
    center = vector(request["center"], 3, "center")
    hand = vector(request.get("hands", snapshot.hands), 12, "hands").copy()
    if request.get("closing", False):
        if request.get("holding", False) or request.get("opening", False):
            raise ValueError("closing_conflicts_with_hold_or_open")
        if np.any((hand < 0) | (hand > 255)):
            raise ValueError("invalid_pinch_hand")
        # Preserve the requested destination but take at most one bounded step
        # from the executor's current feedback, not the older client sample.
        hand[:6] = snapshot.hands[:6] + np.clip(
            hand[:6] - snapshot.hands[:6],
            -planner.config.motion_checks.pinch_hand_step_raw,
            planner.config.motion_checks.pinch_hand_step_raw,
        )
    if request.get("opening", False):
        if request.get("holding", False) or request.get("closing", False):
            raise ValueError("opening_conflicts_with_hold_or_close")
        # Recompute release from this planning snapshot, not the client's older
        # finger feedback. The unchanged 2 raw bound is checked below.
        right_hand = hand[6:].copy()
        hand = np.asarray(snapshot.hands, float).copy()
        if request.get("preserve_right"):
            hand[6:] = right_hand
        hand[:6] += np.clip(
            planner.config.trial_hand("open") - hand[:6],
            -planner.config.motion_checks.pinch_hand_step_raw,
            planner.config.motion_checks.pinch_hand_step_raw,
        )

    if (
        np.any((hand < 0) | (hand > 255))
        or np.max(np.abs(hand[6:] - snapshot.hands[6:]))
        > (
            min(planner.config.hand_arrival_error, 5.0)
            if request.get("preserve_right")
            else planner.config.motion_checks.fixed_hand_drift_raw
        )
        + 1e-6
    ):
        raise ValueError("invalid_pinch_hand")
    old_center, old_rotation = pinch_pose(planner.model, snapshot.body, snapshot.hands)
    if "goal_center" in request:
        if not any(request.get(flag) for flag in ("holding", "closing", "opening")):
            raise ValueError("goal_center_requires_carry_motion")
        goal = vector(request["goal_center"], 3, "goal_center")
        remaining = goal - old_center
        distance = float(np.linalg.norm(remaining))
        step = planner.config.motion_checks.pinch_substep_m
        center = goal if distance <= step else old_center + remaining * (step / distance)
    rotation = np.asarray(request.get("rotation", old_rotation), float)
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        or np.linalg.det(rotation) < 0
    ):
        raise ValueError("invalid_pinch_rotation")
    if (
        np.linalg.norm(center - old_center) > planner.config.motion_checks.pinch_step_m
        or Rotation.from_matrix(old_rotation.T @ rotation).magnitude() > planner.config.motion_checks.pinch_rotation_rad
    ):
        raise ValueError(
            f"pinch_step_too_large:position_mm={np.linalg.norm(center - old_center) * 1000:.3f}"
            f":limit_mm={planner.config.motion_checks.pinch_step_m * 1000:.3f}"
            f":rotation_rad={Rotation.from_matrix(old_rotation.T @ rotation).magnitude():.6f}"
            f":limit_rad={planner.config.motion_checks.pinch_rotation_rad:.6f}"
        )
    hand_delta = hand - snapshot.hands
    if request.get("preserve_right"):
        hand_delta = hand_delta[:6]
    if np.max(np.abs(hand_delta)) > planner.config.motion_checks.pinch_hand_step_raw + 1e-6:
        raise ValueError(
            f"pinch_finger_increment_exceeds_2:delta_raw={np.max(np.abs(hand_delta)):.6f}"
            f":limit_raw={planner.config.motion_checks.pinch_hand_step_raw:.6f}"
        )
    holding = request.get("holding", False)
    position_hold = holding and planner.config.trial_grip_mode == "supervised_position"
    if holding and not position_hold and np.max(np.abs(hand_delta)) > 1e-8:
        raise ValueError("carried_object_requires_fixed_grip")
    if position_hold and request.get("closing", False):
        raise ValueError("cannot_close_during_position_hold")
    geometry_hand = snapshot.hands.copy() if position_hold else hand
    if position_hold and request.get("preserve_right"):
        geometry_hand[6:] = hand[6:]
    from .trial_plan import TrialPlanner

    tips = TrialPlanner(planner.model, planner.config).reference_tips(snapshot.body, geometry_hand)
    nxt = planner.model.solve(snapshot.body, geometry_hand, center, rotation, local_point=sum(tips) / 2)
    if request.get("preserve_right"):
        right_arms = vector(request["right_arms"], 7, "right_arms")
        if np.max(np.abs(right_arms - snapshot.body[22:])) > planner.config.motion_checks.trial_right_arm_rad:
            raise ValueError("trial_right_arm_tracking_error")
        nxt[22:] = right_arms
    elif "right_arms" in request:
        raise ValueError("right_arms_requires_preserve_right")
    obstacles = request.get("obstacles", [])
    segment = planner.segment(snapshot.body, snapshot.hands, nxt[15:], geometry_hand, extra_obstacles=obstacles)
    tactile_control = planner.config.trial_grip_mode == "tactile"
    segment.contact_guard = request.get("closing", False) and tactile_control
    segment.require_contact = holding and tactile_control
    segment = check_sweep(
        planner,
        snapshot.body,
        snapshot.hands,
        segment,
        obstacles,
        holding,
        pose_cache=carry_cache["poses"] if carry_cache is not None else None,
    )
    if position_hold and not np.array_equal(hand, snapshot.hands):
        # Keep the servo command fixed while using measured fingers for IK.
        # Cover the entire sampled measured-to-command grip interval at every
        # arm sample, not only a diagonal interpolation between two poses.
        count = max(
            1,
            int(
                np.ceil(np.max(np.abs(hand - snapshot.hands)) / 255 * 1.6 / planner.config.motion_checks.path_step_rad)
            ),
        )
        covered = False
        poses_cache = None
        if carry_cache is not None:
            from .grip_sweep import covers_grip

            covered = covers_grip(
                planner, snapshot.body, segment, snapshot.hands, hand, obstacles, carry_cache["envelopes"]
            )
            poses_cache = carry_cache["poses"]
        for alpha in np.linspace(1 / count, 1, count):
            fingers = snapshot.hands + alpha * (hand - snapshot.hands)
            if not covered:
                planner.model.check_path(
                    snapshot.body, segment.arms, segment.target, fingers, fingers, extra_obstacles=obstacles
                )
            envelope = replace(segment, hands=fingers.copy(), hand_target=fingers.copy())
            check_sweep(planner, snapshot.body, fingers, envelope, obstacles, holding=True, pose_cache=poses_cache)
            segment.duration = max(segment.duration, envelope.duration)
        segment.hands = hand.copy()
        segment.hand_target = hand.copy()
    if holding:
        segment.pinch_center = center.copy()
    segment.duration = max(0.02, segment.duration / planner.config.trial_speed_scale)
    return segment


def carry_path(planner, snapshot, request):
    """Certify a continuous carry or empty retreat; legacy carry is capped at 130 mm."""
    from types import SimpleNamespace

    empty = request.get("action") == "empty_pinch_path"
    if request.get("holding") is not (not empty) or request.get("closing") or request.get("opening"):
        raise ValueError("carry_path_requires_fixed_grip")
    goal = vector(request["goal_center"], 3, "goal_center")
    start, _ = pinch_pose(planner.model, snapshot.body, snapshot.hands)
    distance = float(np.linalg.norm(goal - start))
    if request.get("action") == "carry_path" and distance > 0.13:
        raise ValueError("carry_path_exceeds_130mm")
    count = max(1, int(np.ceil(distance / planner.config.motion_checks.pinch_substep_m)))
    current = SimpleNamespace(body=snapshot.body.copy(), hands=snapshot.hands.copy())
    segments = []
    carry_cache = {"envelopes": {}, "poses": {}}

    def append_target(previous_target, center, depth=0):
        actual, _ = pinch_pose(planner.model, current.body, current.hands)
        # Nominal samples are <=5 mm apart, but the preceding IK solution has
        # a residual. Subdivide on the same nominal line before the 6 mm guard;
        # never enlarge that guard or bend the path toward the residual.
        if np.linalg.norm(center - actual) > planner.config.motion_checks.pinch_step_m and depth < 8:
            midpoint = (previous_target + center) / 2
            append_target(previous_target, midpoint, depth + 1)
            append_target(midpoint, center, depth + 1)
            return
        step = {key: value for key, value in request.items() if key != "goal_center"}
        step.update(action="pinch_step", center=center.tolist())
        segment = pinch_step(planner, current, step, carry_cache=carry_cache)
        segment.continuous_carry = True
        segment.pinch_center = center.copy()
        segment.name = request.get("action", "carry_path")
        segments.append(segment)
        current.body[15:] = segment.target
        if empty:
            current.hands = segment.hand_target.copy()
        # Keep measured finger geometry and the retained grip envelope throughout
        # planning. Do not assume an obstructed finger reached its command.

    for index in range(1, count + 1):
        append_target(start + (goal - start) * ((index - 1) / count), start + (goal - start) * (index / count))
    if len(segments) > 1:
        for index, segment in enumerate(segments):
            segment.carry_blend = "start" if index == 0 else "end" if index == len(segments) - 1 else "cruise"
            if segment.carry_blend != "cruise":
                segment.duration *= 2
    return segments


class SingleTrial:
    """Run one preflighted A trial with fresh visual and contact gates.

    ``observe(stage)`` must return a freshly stereo-measured cube center and
    acquisition timestamp, after checking the current scene remains clear.
    Placement plans delegate to the full transfer after closure; other plans
    retain the original supervised lift/replace workflow. No automatic retries.
    """

    def __init__(self, task, model, config, observe):
        self.task, self.model, self.config, self.observe = task, model, config, observe
        self.events = []

    async def verified_center(self, stage, expected):
        started = time.monotonic()
        result = await self.observe(stage)
        self.task.status()
        center = vector(result["center"], 3, "observed_cube")
        if not started <= result["acquired_monotonic"] <= time.monotonic() or result.get("scene_clear") is not True:
            raise ValueError("trial_observation_not_fresh_and_clear")
        if np.linalg.norm(center - np.asarray(expected)) > self.config.motion_checks.target_observation_m:
            raise ValueError(f"trial_visual_mismatch:{stage}")
        if stage == "before_descent" and result.get("pinch_alignment_verified") is not True:
            raise ValueError("trial_hover_pinch_alignment_not_verified")
        self.events.append({"visual": stage, **result})
        return center

    async def confirmed_contact(self):
        first = self.task.status()
        if not contact_is_fresh(first, self.config):
            return False
        stamp = time.monotonic() - first["tactile_age"]
        await asyncio.sleep(0.08)
        second = self.task.status()
        return contact_is_fresh(second, self.config) and time.monotonic() - second["tactile_age"] > stamp + 0.02

    async def move_center(self, goal, hands=None, holding=False, closing=False, obstacles=(), opening=False):
        state = self.task.status()
        initial, rotation = pinch_pose(self.model, state["body"], state["hands"])
        count = max(
            1, int(np.ceil(np.linalg.norm(np.asarray(goal) - initial) / self.config.motion_checks.pinch_substep_m))
        )
        goal = vector(goal, 3, "pinch_goal")
        if holding:
            if self.config.trial_grip_mode == "tactile" and not contact_is_fresh(state, self.config):
                raise ValueError("trial_lost_contact")
            commanded_hands = np.asarray(state.get("hands_command", state["hands"]), float)
            requested_hands = np.array(state["hands"] if hands is None else hands, float)
            requested_hands[6:] = commanded_hands[6:]
            result = await self.task.move(
                action="carry_path",
                holding=True,
                preserve_right=True,
                center=initial.tolist(),
                goal_center=goal.tolist(),
                rotation=rotation.tolist(),
                hands=requested_hands.tolist(),
                right_arms=list(state.get("arms_command", state["body"][15:])[7:]),
                obstacles=obstacles,
            )
            measured = self.task.status()
            actual, _ = pinch_pose(self.model, measured["body"], measured["hands"])
            # Task.move returns only after the executor checks fresh endpoint
            # feedback. A later sample can cross the same threshold again;
            # record it without introducing a second arrival decision.
            self.events.append(
                {
                    "carry_path_completed": True,
                    "arrival_checked_by": "executor",
                    "post_completion_error_m": float(np.linalg.norm(goal - actual)),
                    "goal_center": goal.tolist(),
                    "measured_final_center": actual.tolist(),
                    "holding": True,
                }
            )
            return result
        # Advance from feedback, not from a schedule that can outrun the hand.
        # Bounded retries prevent a stalled arm from looping indefinitely.
        for _ in range(max(10, count * 4)):
            state = self.task.status()
            current, _ = pinch_pose(self.model, state["body"], state["hands"])
            remaining = goal - current
            distance = float(np.linalg.norm(remaining))
            release_in_range = opening and distance <= self.config.motion_checks.release_position_tolerance_m
            final_step = release_in_range or distance <= self.config.motion_checks.pinch_substep_m
            target = (
                current
                if release_in_range
                else (
                    goal if final_step else current + remaining * (self.config.motion_checks.pinch_substep_m / distance)
                )
            )
            commanded_hands = np.asarray(state.get("hands_command", state["hands"]), float)
            requested_hands = np.array(state["hands"] if hands is None else hands, float)
            requested_hands[6:] = commanded_hands[6:]
            result = await self.task.move(
                action="pinch_step",
                preserve_right=True,
                right_arms=list(state.get("arms_command", state["body"][15:])[7:]),
                center=target.tolist(),
                rotation=rotation.tolist(),
                hands=requested_hands.tolist(),
                holding=holding,
                closing=closing,
                obstacles=obstacles,
                **({"opening": True} if opening else {}),
                **({"goal_center": (current if release_in_range else goal).tolist()} if closing or opening else {}),
            )
            self.events.append(
                {
                    "center": target.tolist(),
                    "measured_start_center": current.tolist(),
                    "holding": holding,
                    "contact_stop": result.get("contact_detected", False),
                    "release_in_position_tolerance": release_in_range,
                    "distance_to_goal_m": distance,
                }
            )
            if result.get("contact_detected") or final_step:
                return result
        raise ValueError(
            f"pinch_progress_limit:target_not_reached_with_bounded_steps:remaining_mm={distance * 1000:.3f}"
        )

    async def joint_phase(self, phase, obstacles):
        """Subdivide the saved curve instead of increasing the execution step bound."""
        phases = phase if isinstance(phase, list) else [phase]
        state = self.task.status()
        target = np.array(phases[-1]["arms"])
        hand_target = np.array(phases[-1]["hands"])
        target[7:] = state.get("arms_command", state["body"][15:])[7:]
        hand_target[6:] = state.get("hands_command", state["hands"])[6:]
        values = {
            "action": "trial_joint_path",
            "preserve_right": True,
            "arms": target.tolist(),
            "hands": hand_target.tolist(),
            "duration": max(0.02, phases[-1]["duration"]),
            "obstacles": obstacles,
        }
        if len(phases) > 1:
            values.update(
                action="trial_joint_sequence",
                waypoints=[
                    {"arms": p["arms"], "hands": p["hands"], "duration": max(0.02, p["duration"])} for p in phases
                ],
            )
        await self.task.move(**values)

    async def run(self, plan, start_phase=0):
        """Approach from the current start, stop on contact, lift, replace, return."""
        if plan.get("geometry_passed") is not True or (
            plan.get("scene_complete") is not True and self.config.table_observed is None
        ):
            raise ValueError("complete_current_scene_plan_required")
        lift_height = plan.get("lift_height_m", self.config.trial_lift_height_m)
        if isinstance(lift_height, bool) or not np.isfinite(lift_height) or not 0 < lift_height <= 0.12:
            raise ValueError("trial_lift_height_outside_0_to_120mm")
        from .calibration import binding

        if plan.get("binding") != binding(self.config):
            raise ValueError("trial_model_or_config_changed")
        if type(start_phase) is not int or not 0 <= start_phase < len(plan["phases"]):
            raise ValueError("invalid_trial_resume_phase")
        if start_phase and any(
            p["phase"] not in ("approach_compact", "preshape", "clearance", "turn")
            for p in plan["phases"][:start_phase]
        ):
            raise ValueError("trial_resume_only_before_descent")
        expected_body, expected_hands = np.array(plan["start_body"]), np.array(plan["start_hands"])
        if start_phase:
            expected_body[15:] = plan["phases"][start_phase - 1]["arms"]
            expected_hands = np.array(plan["phases"][start_phase - 1]["hands"])
        state = self.task.status()
        if (
            np.max(np.abs(np.asarray(state["body"]) - expected_body)) > self.config.motion_checks.ready_plan_rad
            or np.max(np.abs(np.asarray(state["hands"]) - expected_hands)) > self.config.motion_checks.scene_hand_raw
        ):
            raise ValueError("trial_start_changed")
        if self.config.trial_grip_mode == "supervised_position" and state.get("grip_hold_policy_version", 0) < 1:
            raise ValueError("executor_restart_required_for_fixed_grip_hold")
        if state.get("carry_path_version", 0) < 1:
            raise ValueError("executor_restart_required_for_continuous_carry")
        if state.get("trial_joint_path_version", 0) < 1:
            raise ValueError("executor_restart_required_for_trial_joint_path")
        grasp = np.asarray(plan["grasp_center_m"])
        obstacles = plan["obstacles"]
        await self.verified_center("before_approach", grasp - [0, 0, 0.015])
        descent_started = False
        for _, entries in groupby(
            enumerate(plan["phases"][start_phase:], start_phase), key=lambda item: item[1]["phase"]
        ):
            entries = list(entries)
            index, phase = entries[-1]
            if phase["phase"] == "close":
                break
            if phase["phase"] == "descend" and not descent_started:
                # The operator disabled this view because the hand occludes A.
                # Continue the checked plan using the before-approach observation.
                self.events.append({"visual_skipped": "before_descent", "reason": "operator_requested_hand_occlusion"})
                descent_started = True
            await self.joint_phase([p for _, p in entries], obstacles)
            self.events.append({"completed_plan_phase": index, "phase": phase["phase"]})
        if not descent_started:
            raise ValueError("trial_missing_descent_phase")
        # Use the checked descent endpoint. The operator disabled the final
        # close-up view and its visual correction after confirming alignment.
        self.events.append(
            {
                "visual_skipped": "before_close",
                "reason": "operator_requested_planned_grasp",
                "grasp_center_m": grasp.tolist(),
                "trial_grasp_offset_m": list(self.config.trial_grasp_offset),
            }
        )
        position_mode = self.config.trial_grip_mode == "supervised_position"
        if position_mode:
            from .trial_plan import TrialPlanner

            state = self.task.status()
            grip_target = TrialPlanner(self.model, self.config).position_grip(state["body"], state["hands"])
        else:
            grip_target = self.config.trial_hand("closed")
        last_grip_command = None
        closure_window = []
        closure_reason = "target_reached"
        for closure_steps in range(120):
            if not position_mode and await self.confirmed_contact():
                self.events.append({"contact_verified": True, "state": self.task.status()})
                break
            state = self.task.status()
            if not position_mode and not 0 <= state["tactile_age"] <= self.config.feedback_timeout:
                raise ValueError("pinch_tactile_stale")
            hand = np.array(state["hands"])
            difference = grip_target - hand[:6]
            if np.max(np.abs(difference)) < 1:
                if position_mode:
                    break
                raise ValueError("closed_without_verified_opposing_contact")
            # Position feedback need not reach a free-space target when blocked
            # by a cube. A plateau only ends closure; it never certifies contact.
            if position_mode:
                closure_window.append(hand[:6].copy())
                closure_window = closure_window[-6:]
                if len(closure_window) == 6 and np.max(np.ptp(closure_window, axis=0)) <= 1.0:
                    closure_reason = "feedback_plateau"
                    break
                if closure_steps >= 60:
                    closure_reason = "step_budget_reached"
                    break
            hand[:6] += np.clip(
                difference,
                -self.config.motion_checks.pinch_hand_step_raw,
                self.config.motion_checks.pinch_hand_step_raw,
            )
            center, _ = pinch_pose(self.model, state["body"], state["hands"])
            closure_result = await self.move_center(center, hand, closing=True, obstacles=obstacles)
            last_grip_command = np.asarray(closure_result.get("hands_command", hand), float)[:6].copy()
        else:
            raise TimeoutError("trial_contact_timeout")
        if position_mode:
            state = self.task.status()
            measured = np.asarray(state["hands"][:6])
            if last_grip_command is None:
                last_grip_command = np.asarray(state.get("hands_command", state["hands"])[:6]).copy()
            self.events.append(
                {
                    "position_grip_closure_finished": True,
                    "reason": closure_reason,
                    "closure_steps": closure_steps,
                    "target_left_raw": grip_target.tolist(),
                    "held_command_left_raw": last_grip_command.tolist(),
                    "measured_left_raw": measured.tolist(),
                    "target_error_raw": (grip_target - measured).tolist(),
                    "target_reached": bool(np.max(np.abs(grip_target - measured)) < 1),
                    "operator_confirmation_required": self.config.trial_require_grip_confirmation
                    and "placement" not in plan,
                }
            )
            if self.config.trial_require_grip_confirmation and "placement" not in plan:
                confirmation = await self.observe("grip_confirmation")
                self.task.status()
                if confirmation.get("grip_verified") is not True:
                    raise ValueError("position_grip_not_operator_verified")
                self.events.append(
                    {
                        "operator_grip_verified": True,
                        "grip_mode": self.config.trial_grip_mode,
                        "target_gap_m": self.config.trial_grip_gap_m,
                        "state": self.task.status(),
                    }
                )
            else:
                self.events.append({"position_grip_auto_advance": True, "operator_grip_verified": False})
                action = "搬运到前排" if "placement" in plan else "提起并放回"
                print(f"闭指结束，按当前配置自动{action}；console 空格仍可暂停。", flush=True)
        state = self.task.status()
        contact_center, _ = pinch_pose(self.model, state["body"], state["hands"])
        # These post-contact moves are solved and swept from measured fingers,
        # not the nominal fully closed shape in the offline candidate.
        held_hands = None
        if position_mode:
            held_hands = np.array(state["hands"])
            held_hands[:6] = last_grip_command
        self.events.append({"lift_height_m": float(lift_height), "measured_contact_center_m": contact_center.tolist()})
        if "placement" in plan:
            from .transfer import execute_transfer

            return await execute_transfer(self, plan, contact_center, held_hands, obstacles)
        await self.move_center(contact_center + [0, 0, lift_height], held_hands, holding=True, obstacles=obstacles)
        self.events.append({"visual_skipped": "lifted", "reason": "operator_manual_review"})
        await self.move_center(contact_center, held_hands, holding=True, obstacles=obstacles)
        self.events.append({"visual_skipped": "replaced", "reason": "operator_manual_review"})
        for _ in range(120):
            state = self.task.status()
            hand = np.array(state["hands"])
            difference = self.config.trial_hand("open") - hand[:6]
            release_tolerance = min(2.0, self.config.hand_arrival_error)
            if np.max(np.abs(difference)) <= release_tolerance + 1e-6:
                self.events.append(
                    {
                        "release_position_confirmed": True,
                        "error_raw": difference.tolist(),
                        "tolerance_raw": release_tolerance,
                    }
                )
                break
            hand[:6] += np.clip(
                difference,
                -self.config.motion_checks.pinch_hand_step_raw,
                self.config.motion_checks.pinch_hand_step_raw,
            )
            await self.move_center(contact_center, hand, obstacles=obstacles, opening=True)
        else:
            raise TimeoutError(f"trial_release_timeout:step_limit=120:remaining_raw={np.max(np.abs(difference)):.3f}")
        self.events.append({"visual_skipped": "released", "reason": "operator_manual_review"})
        for name, entries in groupby(plan["phases"], key=lambda p: p["phase"]):
            if name not in ("retract", "return_orientation", "compact_for_return", "return", "return_ready"):
                continue
            await self.joint_phase(list(entries), obstacles)
        self.events.append({"visual_skipped": "returned", "reason": "operator_manual_review"})
        self.events.append({"motion_completed": True})
        confirmation = await self.observe("result_confirmation")
        self.task.status()
        if confirmation.get("result_verified") is not True:
            raise ValueError("trial_result_not_operator_verified")
        self.events.append({"operator_result_verified": True, "stages": ["lifted", "replaced", "released", "returned"]})
        return {"success_verified": True, "verification_source": "operator", "events": self.events}
