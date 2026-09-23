"""Full single-cube transfer using the existing checked trajectory executor."""

import time
from itertools import groupby
from types import SimpleNamespace

import numpy as np

from .config import vector
from .trajectory import Planner
from .trial_runtime import pinch_pose, trial_joint_sequence


def table_z(config, xy):
    """Evaluate the retained table plane in pelvis coordinates."""
    if config.table_plane is None:
        return float(config.table_max[2])
    return float(np.asarray(config.table_plane) @ [*xy, 1])


def front_slots(config, count):
    """Place a row near the robot, ordered from robot-left to robot-right."""
    footprint = np.asarray(config.table_footprint or config.table_observed or [])
    slots = []
    for index in range(count):
        y = 0.16 + (count - 1 - index) * 0.12
        intersections = []
        if footprint.size:
            for a, b in zip(footprint, np.roll(footprint, -1, axis=0)):
                if abs(b[1] - a[1]) > 1e-9 and min(a[1], b[1]) <= y <= max(a[1], b[1]):
                    slope = (b[0] - a[0]) / (b[1] - a[1])
                    intersections.append((a[0] + (y - a[1]) * slope, slope))
        near, slope = min(intersections) if intersections else (config.table_min[0], 0.0)
        # Cube center is 60 mm inside the physical near edge, even for a yawed table.
        slots.append([float(near + 0.06 * np.hypot(1, slope)), y])
    return slots


def placement_geometry(config, contact, placement):
    """Reuse the first cube's pinch-to-table height at every destination."""
    offset = np.asarray(config.trial_grasp_offset)
    anchor_xy = vector(placement["xy"], 2, "placement_xy") + offset[:2]
    height = placement.get("height_above_table")
    if height is None:
        height = float(contact[2] - table_z(config, contact[:2]))
    destination = np.array([*anchor_xy, table_z(config, anchor_xy) + height])
    travel_z = max(contact[2], destination[2]) + config.trial_lift_height_m
    lifted = np.array([contact[0], contact[1], travel_z])
    above = np.array([*destination[:2], travel_z])
    center = destination - offset - [0, 0, 0.015]
    # Reuse the existing nearby-cube obstacle envelope, including arbitrary yaw.
    half = np.array([0.04 / np.sqrt(2) + 0.005] * 2 + [0.025])
    obstacle = [(center - half).tolist(), (center + half).tolist()]
    return destination, lifted, above, float(height), center, obstacle


def path_request(state, goal, rotation, hands, obstacles, *, holding):
    """Keep grip and right-side commands fixed throughout a transfer segment."""
    commanded = state.get("hands_command", state["hands"])
    hand = np.asarray(hands, float).copy()
    hand[6:] = commanded[6:]
    return {
        "action": "transfer_path" if holding else "empty_pinch_path",
        "holding": holding,
        "preserve_right": True,
        "center": vector(goal, 3, "goal").tolist(),
        "goal_center": vector(goal, 3, "goal").tolist(),
        "rotation": np.asarray(rotation).tolist(),
        "hands": hand.tolist(),
        "right_arms": list(state.get("arms_command", state["body"][15:])[7:]),
        "obstacles": obstacles,
    }


def preflight_transfer(model, config, plan):
    """Check nominal transfer and retreat before approach; runtime rechecks feedback."""
    close = [p for p in plan["phases"] if p["phase"] == "close"][-1]
    body = np.array(plan["start_body"])
    body[15:] = close["arms"]
    hands = np.array(close["hands"])
    contact, rotation = pinch_pose(model, body, hands)
    dest, lift, above, _, _, obstacle = placement_geometry(config, contact, plan["placement"])
    planner = Planner(model, config)
    snapshot = SimpleNamespace(body=body, hands=hands)
    records = []
    obstacles = plan["obstacles"]

    def check(goal, holding, extra):
        state = {"body": snapshot.body, "hands": snapshot.hands}
        path_rotation = rotation if holding else pinch_pose(model, snapshot.body, snapshot.hands)[1]
        req = path_request(state, goal, path_rotation, snapshot.hands, extra, holding=holding)
        try:
            segments = planner.move(snapshot, req)
        except ValueError as exc:
            exc.transfer_goal = np.asarray(goal).tolist()
            raise
        snapshot.body[15:] = segments[-1].target
        snapshot.hands = segments[-1].hand_target.copy()
        records.append({"goal": np.asarray(goal).tolist(), "samples": len(segments), "holding": holding})

    for goal in (lift, above, dest):
        check(goal, True, obstacles)
    # Release uses exactly the same feedback-rebased, bounded finger increments
    # as the runtime. This also checks hand geometry while opening at the goal.
    opened = config.trial_hand("open")
    for _ in range(120):
        difference = opened - snapshot.hands[:6]
        if np.max(np.abs(difference)) <= min(2.0, config.hand_arrival_error):
            break
        hand = snapshot.hands.copy()
        hand[:6] += np.clip(
            difference, -config.motion_checks.pinch_hand_step_raw, config.motion_checks.pinch_hand_step_raw
        )
        request = path_request(
            {"body": snapshot.body, "hands": snapshot.hands}, dest, rotation, hand, obstacles, holding=False
        )
        request.update(action="pinch_step", opening=True)
        request.pop("goal_center")
        segment = planner.move(snapshot, request)[0]
        snapshot.body[15:] = segment.target
        snapshot.hands = segment.hand_target.copy()
    else:
        raise ValueError("transfer_preflight_release_step_limit")
    check(above, False, obstacles)
    extra = [*obstacles, obstacle]
    source_hover = np.asarray([p for p in plan["phases"] if p["phase"] == "retract"][-1]["pinch_center_m"])
    check(source_hover, False, extra)
    for name, entries in groupby(plan["phases"], key=lambda p: p["phase"]):
        if name not in ("return_orientation", "compact_for_return", "return", "return_ready"):
            continue
        points = list(entries)
        request = {
            "action": "trial_joint_sequence",
            "preserve_right": True,
            "arms": points[-1]["arms"],
            "hands": points[-1]["hands"],
            "waypoints": points,
            "obstacles": extra,
        }
        segments = trial_joint_sequence(planner, snapshot, request)
        snapshot.body[15:] = segments[-1].target
        snapshot.hands = segments[-1].hand_target.copy()
    return records


async def execute_transfer(trial, plan, contact, held_hands, obstacles):
    """Move, release and retreat without asking for object-success confirmation."""
    config = trial.config
    dest, lift, above, height, center, obstacle = placement_geometry(config, contact, plan["placement"])
    state = trial.task.status()
    _, rotation = pinch_pose(trial.model, state["body"], state["hands"])

    async def move(name, goal, holding, extra):
        event = {
            "transfer_stage": name,
            "goal_center": np.asarray(goal).tolist(),
            "started_monotonic": time.monotonic(),
        }
        trial.events.append(event)
        state = trial.task.status()
        hands = held_hands if holding and held_hands is not None else state["hands"]
        try:
            # Opening compensation and tracking can change wrist orientation.
            # Empty retreat preserves the current pose, not the old loaded pose.
            path_rotation = rotation if holding else pinch_pose(trial.model, state["body"], state["hands"])[1]
            await trial.task.move(**path_request(state, goal, path_rotation, hands, extra, holding=holding))
        finally:
            event["elapsed_s"] = time.monotonic() - event["started_monotonic"]

    await move("lift", lift, True, obstacles)
    await move("translate", above, True, obstacles)
    await move("lower_to_place", dest, True, obstacles)
    for _ in range(120):
        state = trial.task.status()
        hand = np.array(state["hands"])
        difference = config.trial_hand("open") - hand[:6]
        if np.max(np.abs(difference)) <= min(2.0, config.hand_arrival_error) + 1e-6:
            break
        hand[:6] += np.clip(
            difference, -config.motion_checks.pinch_hand_step_raw, config.motion_checks.pinch_hand_step_raw
        )
        await trial.move_center(dest, hand, obstacles=obstacles, opening=True)
    else:
        raise TimeoutError("transfer_release_timeout")
    await move("retract_from_placed_cube", above, False, obstacles)
    extra = [*obstacles, obstacle]
    source_hover = np.asarray([p for p in plan["phases"] if p["phase"] == "retract"][-1]["pinch_center_m"])
    await move("return_above_source", source_hover, False, extra)
    for name, entries in groupby(plan["phases"], key=lambda p: p["phase"]):
        if name in ("return_orientation", "compact_for_return", "return", "return_ready"):
            await trial.joint_phase(list(entries), extra)
    trial.task.status()
    trial.events.append(
        {"motion_completed": True, "placement_assumed": True, "cube_center": center.tolist(), "success_verified": False}
    )
    return {
        "motion_completed": True,
        "success_verified": False,
        "placement_assumed": True,
        "height_above_table": height,
        "cube_center": center.tolist(),
        "obstacle": obstacle,
        "events": trial.events,
    }
