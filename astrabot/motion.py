"""Stereo pinch, continuous Cartesian IK and checked pick/place trajectories."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from .timing import Timings, timed

NAMES = [
    "lh_index_mcp_pitch",
    "lh_index_dip",
    "lh_thumb_cmc_yaw",
    "lh_thumb_cmc_pitch",
    "lh_thumb_ip",
]
OPEN = np.array([0.5, 0.5 * 0.89, 1.0, 0.2, 0.2 * 2.29])
CLOSED = np.array([0.72, 0.72 * 0.89, 1.0, 0.32, 0.32 * 2.29])


def continuous_ik_step(kin, q, goal, goal_r, fraction):
    """Refine a Cartesian step rather than jumping to a remote IK branch."""
    current = kin.fk(q, "left_wrist_yaw_link")
    start_r = Rotation.from_matrix(current[:3, :3])
    rv = (start_r.inv() * goal_r).as_rotvec()
    failure = None
    for _ in range(8):
        subpos = current[:3, 3] + fraction * (goal - current[:3, 3])
        subr = start_r * Rotation.from_rotvec(fraction * rv)
        try:
            (target, error) = kin.solve(q, "left", subpos, subr.as_quat()[[3, 0, 1, 2]])
            actual = kin.fk(dict(q, **target), "left_wrist_yaw_link")
            angle = (subr.inv() * Rotation.from_matrix(actual[:3, :3])).magnitude()
            delta = max((abs(v - q[n]) for (n, v) in target.items()))
            if error <= 0.003 and angle <= 0.03 and (delta <= 0.35):
                return (target, error, fraction)
            failure = f"IK continuity/residual: joint={delta}, pos={error}, angle={angle}"
        except ValueError as exc:
            failure = str(exc)
        fraction *= 0.5
    raise ValueError("No continuous Cartesian step: " + str(failure))


class Grasp:
    def __init__(self, sim):
        self.sim = sim
        self.timings = getattr(sim, "timings", None)
        if self.timings is None:
            self.timings = Timings()
        self.rest = {
            n: v for (n, v) in sim.full_q().items() if n in sim.names and n.startswith("left_")
        }
        self.rotation = Rotation.from_euler("z", 150, degrees=True) * Rotation.from_euler(
            "y", 90, degrees=True
        )
        self.quat = self.rotation.as_quat()[[3, 0, 1, 2]]
        self.held_hand = {}
        original_move = sim.move

        async def move_with_grip(targets, **kwargs):
            merged = {**self.held_hand, **targets}
            q = sim.full_q()
            bounded = {n: float(q[n] + np.clip(v - q[n], -0.3, 0.3)) for (n, v) in merged.items()}
            return await original_move(bounded, **kwargs)

        sim.move = move_with_grip

    def tips(self, q):
        w = np.linalg.inv(self.sim.kin.fk(q, "left_wrist_yaw_link"))
        a = (w @ self.sim.kin.fk(q, "lh_index_distal") @ [0.006, 0, 0.03, 1])[:3]
        b = (w @ self.sim.kin.fk(q, "lh_thumb_distal") @ [-0.005, 0, 0.04, 1])[:3]
        return (a, b)

    async def hand(self, targets):
        self.held_hand = {}
        for _ in range(20):
            q = self.sim.full_q()
            if max((abs(v - q[n]) for (n, v) in targets.items())) < 0.04:
                self.held_hand = dict(targets)
                return
            await self.sim.move(
                {n: float(q[n] + np.clip(v - q[n], -0.3, 0.3)) for (n, v) in targets.items()}
            )
        raise RuntimeError("Hand did not reach free-space preshape")

    async def center(self, xyz, nominal=None, *, settle=True):
        q = self.sim.full_q()
        if nominal is not None:
            q.update(dict(zip(NAMES, nominal)))
        (a, b) = self.tips(q)
        await self.goto(np.array(xyz) - self.rotation.apply((a + b) / 2), settle=settle)

    async def release(self, xyz):
        opening = dict(zip(NAMES, OPEN.tolist()))
        self.held_hand.update(opening)
        for _ in range(3):
            await self.sim.move(opening)
        release_frame = self.sim.frame_count - 1
        rise = 0.035
        clear = np.array(xyz) + [0.0, 0.0, rise]
        await self.center(clear, OPEN)
        await self.hand(opening)
        return release_frame

    def set_yaw(self, yaw):
        self.yaw = float(yaw)
        self.rotation = Rotation.from_euler("z", yaw, degrees=True) * self.base_rotation
        self.quat = self.rotation.as_quat()[[3, 0, 1, 2]]

    async def reorient(self, xyz, yaw, nominal):
        start = self.yaw
        for angle in np.linspace(start, yaw, max(2, int(abs(yaw - start) / 10) + 1))[1:]:
            self.set_yaw(angle)
            await self.center(xyz, nominal, settle=abs(angle - yaw) < 1e-08)

    @timed("ik_preflight")
    def preflight(self, waypoints):
        """Check grasp-center waypoints, including rotations, before contact."""
        q = dict(self.sim.full_q())
        errors = []
        for xyz, yaw, hand in waypoints:
            q.update(dict(zip(NAMES, hand)))
            (a, b) = self.tips(q)
            rot = Rotation.from_euler("z", yaw, degrees=True) * self.base_rotation
            pos = np.array(xyz) - rot.apply((a + b) / 2)
            start = self.sim.kin.fk(q, "left_wrist_yaw_link")
            start_r = Rotation.from_matrix(start[:3, :3])
            rv = (start_r.inv() * rot).as_rotvec()
            count = max(
                1,
                int(np.ceil(np.linalg.norm(pos - start[:3, 3]) / 0.005)),
                int(np.ceil(np.linalg.norm(rv) / 0.02)),
            )
            max_delta = 0.0
            for fraction in np.linspace(1 / count, 1, count):
                subpos = start[:3, 3] + fraction * (pos - start[:3, 3])
                subrot = start_r * Rotation.from_rotvec(fraction * rv)
                (joints, pe) = self.sim.kin.solve(q, "left", subpos, subrot.as_quat()[[3, 0, 1, 2]])
                delta = max((abs(v - q[n]) for (n, v) in joints.items()))
                if delta > 0.35:
                    raise RuntimeError(
                        f"Dense preflight IK branch change: {delta:.3f} rad at {list(xyz)}, yaw={yaw}"
                    )
                max_delta = max(max_delta, delta)
                q.update(joints)
                actual = self.sim.kin.fk(q, "left_wrist_yaw_link")
                oe = (subrot.inv() * Rotation.from_matrix(actual[:3, :3])).magnitude()
                if pe > 0.003 or oe > 0.03:
                    raise RuntimeError(
                        f"Dense preflight residual at {list(xyz)}, yaw={yaw}: {pe}, {oe}"
                    )
            errors.append(
                {
                    "center": list(xyz),
                    "yaw": yaw,
                    "position_error": pe,
                    "orientation_error": oe,
                    "samples": count,
                    "max_joint_step_rad": max_delta,
                }
            )
        return errors

    async def goto(self, pos, quat=None, *, settle=True):
        goal = np.asarray(pos, float)
        goal_r = (
            self.rotation if quat is None else Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
        )
        bias = np.zeros(7)
        (previous_error, stalled) = (float("inf"), 0)
        for i in range(85):
            q = self.sim.full_q()
            current = self.sim.kin.fk(q, "left_wrist_yaw_link")
            r = Rotation.from_matrix(current[:3, :3])
            rotvec = (r.inv() * goal_r).as_rotvec()
            (pe, oe) = (float(np.linalg.norm(goal - current[:3, 3])), float(np.linalg.norm(rotvec)))
            if pe < 0.003 and oe < 0.03:
                print("GOTO_REACHED", np.round(goal, 4).tolist(), pe, oe, flush=True)
                return
            step = 0.036
            fraction = min(1.0, step / max(pe, 1e-09), 0.16 / max(oe, 1e-09))
            (target, err, fraction) = continuous_ik_step(self.sim.kin, q, goal, goal_r, fraction)
            names = list(target)
            nominal = np.array([target[n] for n in names])
            measured = np.array([q[n] for n in names])
            command = nominal + bias
            lo = np.array([self.sim.kin.by_name[n]["lower"] for n in names])
            hi = np.array([self.sim.kin.by_name[n]["upper"] for n in names])
            command = np.clip(command, lo + 1e-05, hi - 1e-05)
            delta = command - measured
            delta *= min(1.0, 0.28 / max(float(np.max(np.abs(delta))), 1e-09))
            await self.sim.move(
                dict(zip(names, (measured + delta).tolist())),
                continuous=fraction < 0.95 or not settle,
            )
            afterq = self.sim.full_q()
            if fraction > 0.95:
                bias = np.clip(
                    bias + 0.4 * (nominal - np.array([afterq[n] for n in names])), -0.08, 0.08
                )
            combined = pe + 0.12 * oe
            stalled = stalled + 1 if combined >= previous_error - 0.0001 else 0
            previous_error = combined
            if i % 5 == 0:
                print("GOTO_PROGRESS", i, round(pe, 4), round(oe, 4), flush=True)
            if stalled >= 14:
                raise RuntimeError(f"Cartesian motion stalled: pos error={pe}, orientation={oe}")
        raise RuntimeError("Cartesian motion step limit")

    async def park(self):
        print("PARK_FOR_VISION", flush=True)
        for _ in range(35):
            q = self.sim.full_q()
            if max((abs(v - q[n]) for (n, v) in self.rest.items())) < 0.04:
                return
            await self.sim.move(
                {n: float(q[n] + np.clip(v - q[n], -0.25, 0.25)) for (n, v) in self.rest.items()},
                continuous=True,
            )
        raise RuntimeError("Arm parking did not converge")


def plan_pick_place(c, xyz, target, transit_yaw):
    """Separate source turning clearance from destination release clearance.

    For stereo estimates, try bounded higher source clearances before any arm
    motion. Keep the measured pinch centered; adjust only free-space waypoints.
    """
    failures = []
    candidates = [
        (rise, destination_rise)
        for destination_rise in [0.035, 0.03]
        for rise in [0.065, 0.055, 0.045, 0.035]
    ]
    for rise, destination_rise in candidates:
        offset = np.zeros(3)
        grasp_xyz = xyz + offset
        hover = grasp_xyz.copy()
        hover[2] = xyz[2] + 0.148
        clearance = grasp_xyz.copy()
        clearance[2] = xyz[2] + rise
        lifted = grasp_xyz.copy()
        lifted[2] = target.get("carry_height", clearance[2])
        lifted[2] = max(lifted[2], clearance[2] + 0.005)
        placed = (
            np.array([target["target_x"], target["target_y"], target.get("contact_z", xyz[2])])
            + offset
        )
        above = placed.copy()
        above[2] = placed[2] + destination_rise
        contact_angles = list(range(transit_yaw + 10, 91, 10))
        transit_angles = list(range(80, transit_yaw - 1, -10))
        route = [above]
        if "front_corridor_y" in target:
            front_y = target["front_corridor_y"]
            staging_y = 1.78
            route = [
                np.array([grasp_xyz[0], staging_y, lifted[2]]),
                np.array([placed[0], staging_y, lifted[2]]),
                np.array([placed[0], front_y, lifted[2]]),
                np.array([placed[0], front_y, above[2]]),
                above,
            ]
        waypoints = [
            (hover, transit_yaw, OPEN),
            (clearance, transit_yaw, OPEN),
            *[(clearance, y, OPEN) for y in contact_angles],
            (grasp_xyz, 90, OPEN),
            (grasp_xyz, 90, CLOSED),
            (lifted, 90, CLOSED),
            *[(p, 90, CLOSED) for p in route],
            (placed, 90, CLOSED),
            (placed, 90, OPEN),
            *[(placed + [0, 0, 0.035], 90, OPEN)],
            (above, 90, OPEN),
            *[(above, y, OPEN) for y in transit_angles],
        ]
        try:
            preflight = c.preflight(waypoints)
        except (ValueError, RuntimeError) as exc:
            failures.append(
                {
                    "source_clearance_m": rise,
                    "destination_clearance_m": destination_rise,
                    "error": str(exc),
                }
            )
            continue
        return {
            "hover": hover,
            "clearance": clearance,
            "lifted": lifted,
            "placed": placed,
            "above": above,
            "route": route,
            "preflight": preflight,
            "grasp_xyz": grasp_xyz,
            "pinch_offset_m": offset.tolist(),
            "source_clearance_m": rise,
            "destination_clearance_m": destination_rise,
            "rejected_candidates": failures,
        }
    raise RuntimeError("No reachable pick/place clearance: " + json.dumps(failures))


async def execute_pick_place(sim, c, letter, target, report, estimate):
    """Execute a checked trajectory from a current stereo grasp estimate."""
    if not estimate or "grasp_center_xyz" not in estimate:
        raise ValueError("Current stereo grasp coordinates are required")
    xyz = np.array(estimate["grasp_center_xyz"], dtype=float)
    report["initial_letter_estimate"] = estimate

    def phase(name):
        sim.phase = f"{letter}:{name}"

    phase("hand_prepare")
    targets = dict(zip(NAMES, OPEN))
    targets.update(
        {
            f"lh_{finger}_{joint}": value
            for finger in ["middle", "ring", "pinky"]
            for (joint, value) in [("mcp_pitch", 1.3), ("dip", 1.3 * 0.89)]
        }
    )
    await c.hand(targets)
    (a, b) = c.tips(sim.full_q())
    d = a - b
    d /= np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    z -= d * z.dot(d)
    z /= np.linalg.norm(z)
    c.base_rotation = Rotation.from_matrix(
        np.diag([1.0, -1.0, -1.0]) @ np.column_stack([d, z, np.cross(d, z)]).T
    )
    transit_yaw = 60 if letter in ("H", "A") else 30
    c.set_yaw(transit_yaw)
    print("OPEN_TIPS", a.tolist(), b.tolist(), letter, xyz.tolist(), flush=True)
    plan = plan_pick_place(c, xyz, target, transit_yaw)
    xyz = plan["grasp_xyz"]
    (hover, clearance, lifted, placed, above, route) = [
        plan[k] for k in ("hover", "clearance", "lifted", "placed", "above", "route")
    ]
    place_clearance = above.copy()
    report["carry_route"] = [p.tolist() for p in route]
    report["preflight"] = plan["preflight"]
    report["source_clearance_m"] = plan["source_clearance_m"]
    report["destination_clearance_m"] = plan["destination_clearance_m"]
    report["pinch_offset_m"] = plan["pinch_offset_m"]
    report["preflight_rejected_candidates"] = plan["rejected_candidates"]
    print("PREFLIGHT_PASSED", len(report["preflight"]), flush=True)
    phase("approach_hover")
    await c.center(hover)
    phase("descend_clearance")
    await c.center(clearance, OPEN)
    phase("orient_grasp")
    await c.reorient(clearance, 90, OPEN)
    phase("descend_contact")
    await c.center(xyz)
    phase("close")
    for fraction in np.linspace(0.125, 1.0, 8):
        nominal = OPEN + fraction * (CLOSED - OPEN)
        c.held_hand.update(dict(zip(NAMES, nominal.tolist())))
        await sim.move(dict(zip(NAMES, nominal.tolist())))
        await c.center(xyz, nominal)
        (a, b) = c.tips(sim.full_q())
        print("CLOSE", float(fraction), "tip_gap", float(np.linalg.norm(a - b)), flush=True)
    report["before_lift_frame"] = sim.frame_count - 1
    phase("lift")
    await c.center(lifted, CLOSED)
    report["lift_frame"] = sim.frame_count - 1
    print("LIFT_CHECKPOINT", sim.frame_count - 1, flush=True)
    (a, b) = c.tips(sim.full_q())
    report["lift_decision"] = {
        "verification": "skipped",
        "source": "user_requested_trust_skill",
        "continue_transport": True,
        "grasp_verified": False,
    }
    report["grasp_success_verified"] = False
    print("LIFT_TRUST_SKILL", letter, flush=True)
    phase("carry")
    for point in route:
        await c.center(point, CLOSED)
    phase("lower_to_place")
    await c.center(placed, CLOSED)
    phase("release")
    report["release_frame"] = await c.release(placed)
    phase("retract")
    await c.center(place_clearance, OPEN)
    phase("orient_retreat")
    await c.reorient(place_clearance, transit_yaw, OPEN)
    await c.center(above, OPEN)
    phase("park")
    await c.park()
    report["final_frame"] = sim.frame_count - 1
