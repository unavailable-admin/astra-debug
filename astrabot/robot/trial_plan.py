"""Offline complete first-pinch planning, with explicit contact/return phases.

No IPC, hardware imports, calibration flag changes, or motion publication.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from ..pinch import CLOSED, INDEX_TIP, OPEN, THUMB_TIP, pinch_rotation, source_waypoints
from .collision import segment_hits_box
from .config import REST, Config, vector
from .scene_coverage import check_observed_box
from .table import cube_penetrates
from .trajectory import Planner


class TrialPlanningError(ValueError):
    """Preserve every rejected orientation instead of exposing only the last one."""

    def __init__(self, attempts):
        self.orientation_attempts = [dict(attempt) for attempt in attempts]
        reasons = list(dict.fromkeys(f"{a['phase']}:{a['error']}" for a in attempts))
        super().__init__("no_collision_free_trial_orientation:" + "; ".join(reasons))


class TrialPlanner:
    """Use the simulation's source sequence with real FK and collision checks."""

    def __init__(self, model, config):
        self.model, self.config = model, config
        self.planner = Planner(model, config)
        self.last_phase = "not_started"
        self.orientation_attempts = []

    def plan_adaptive(self, snapshot, grasp_center, **parameters):
        """Search bounded pinch orientations; preserve explicitly supplied yaw settings."""
        self.orientation_attempts = []
        candidates = (
            [None]
            if any(key in parameters for key in ("transit_yaw", "contact_yaw"))
            else [0, -15, 15, -30, 30, -45, 45, -60, 60, -90, 90]
        )
        for yaw in candidates:
            values = dict(parameters)
            if yaw is not None:
                values.update(transit_yaw=yaw, contact_yaw=yaw)
            try:
                result = self.plan(snapshot, grasp_center, **values)
            except ValueError as exc:
                self.orientation_attempts.append({"yaw_deg": yaw, "phase": self.last_phase, "error": str(exc)})
                if not str(exc).startswith(
                    ("hand_", "arm_", "head_", "joint_limit", "ik_", "IK_", "trial_cube_", "motion_enters_")
                ):
                    raise
                continue
            result["orientation_attempts"] = [*self.orientation_attempts, {"yaw_deg": yaw, "geometry_passed": True}]
            return result
        raise TrialPlanningError(self.orientation_attempts)

    def tips(self, body, hands):
        """Model-derived wrist-local tips; these are explicitly not a measured TCP."""
        poses = self.model.pinch_poses(body, hands)
        inverse = np.linalg.inv(poses["left_wrist_yaw_link"])
        return (
            (inverse @ poses["left_index_distal"] @ INDEX_TIP)[:3],
            (inverse @ poses["left_thumb_distal"] @ THUMB_TIP)[:3],
        )

    def reference_tips(self, body, hands):
        """Preserve the original grasp anchor instead of undoing thumb abduction with wrist motion."""
        return self.tips(body, self.config.pinch_reference_hands(hands))

    def position_grip(self, body, hands):
        """Bound model tip opening using only the existing open/closed joint range."""
        opened, closed = self.config.trial_hand("open"), self.config.trial_hand("closed")

        def gap(fraction):
            hand = np.array(hands, float).copy()
            hand[:6] = opened + fraction * (closed - opened)
            a, b = self.tips(body, hand)
            return float(np.linalg.norm(a - b))

        target = self.config.trial_grip_gap_m
        samples = np.array([gap(f) for f in np.linspace(0, 1, 21)])
        if np.any(np.diff(samples) >= 0) or not samples[-1] <= target <= samples[0]:
            raise ValueError("position_grip_gap_not_reachable_monotonically")
        lo, hi = 0.0, 1.0
        for _ in range(32):
            mid = (lo + hi) / 2
            if gap(mid) > target:
                lo = mid
            else:
                hi = mid
        return opened + hi * (closed - opened)

    def plan(
        self,
        snapshot,
        grasp_center,
        *,
        transit_yaw=0.0,
        contact_yaw=0.0,
        hover_height=0.06,
        clearance_height=0.035,
        lift_height=None,
        obstacles=(),
    ):
        """Plan approach through trial lift, replacement, release and return.

        Positions are in pelvis coordinates; yaw follows the measured table
        tangent plane (pelvis XY for a horizontal table). The saved plan
        requires a fresh observed start and opposing contact before lifting.
        Replacement lets the first trial finish without retaining a cube while
        returning. The runtime must stop/replan on contact mismatch or drift.
        """
        lift_height = self.config.trial_lift_height_m if lift_height is None else lift_height
        object_grasp = vector(grasp_center, 3, "grasp_center")
        offset = np.asarray(self.config.trial_grasp_offset)
        grasp = object_grasp + offset
        self.last_phase = "initial_pose"
        if np.any(np.asarray(self.config.tcp_offset) != 0):
            raise ValueError("trial_requires_wrist_origin_tcp")
        if not np.isfinite([transit_yaw, contact_yaw, hover_height, clearance_height, lift_height]).all():
            raise ValueError("invalid_trial_parameters")
        source_waypoints(grasp, transit_yaw, clearance_height, hover_height, contact_yaw)
        obstacles = tuple((vector(a, 3, "obstacle_lower"), vector(b, 3, "obstacle_upper")) for a, b in obstacles)
        if any(np.any(a >= b) for a, b in obstacles):
            raise ValueError("invalid_trial_obstacle")
        if isinstance(lift_height, bool) or not 0 < lift_height <= 0.12:
            raise ValueError("trial_lift_height_outside_0_to_120mm")
        body, hands = snapshot.body.copy(), snapshot.hands.copy()
        start_body, start_hands = body.copy(), hands.copy()
        grip = (
            self.position_grip(body, hands)
            if self.config.trial_grip_mode == "supervised_position"
            else self.config.trial_hand("closed")
        )
        self.model.check(body, hands, extra_obstacles=obstacles)
        opened = hands.copy()
        opened[:6] = self.config.trial_hand("open")
        index, thumb = self.reference_tips(body, opened)
        base = pinch_rotation(index, thumb)
        # Pinch yaw is measured in the tabletop tangent plane. The measured
        # plane is in pelvis coordinates and need not be pelvis-horizontal.
        table_rotation = np.eye(3)
        if self.config.table_plane is not None:
            a, b, _ = self.config.table_plane
            normal = np.array([-a, -b, 1.0])
            normal /= np.linalg.norm(normal)
            tangent = np.array([1.0, 0.0, a])
            tangent /= np.linalg.norm(tangent)
            table_rotation = np.column_stack((tangent, np.cross(normal, tangent), normal))

        def grasp_rotation(yaw):
            return table_rotation @ Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ base

        phases = []
        # Local to this immutable-scene plan. Reversing exactly the same joint
        # and hand endpoints traces the already checked geometric curve.
        approach_segments = {}
        reversed_segments_reused = 0

        def endpoint_key(q, h):
            return np.asarray(q, dtype=float).tobytes(), np.asarray(h, dtype=float).tobytes()

        def bound_cartesian_speed(segment):
            # Bound sampled wrist and pinch-centre speed on the same joint-space
            # interpolation, including fingertip motion while opening/closing.
            previous = None
            derivative = 0.0
            count = max(
                20,
                int(np.ceil(np.max(np.abs(segment.target - segment.arms)) / self.config.motion_checks.path_step_rad)),
            )
            for fraction in np.linspace(0, 1, count + 1):
                q = body.copy()
                q[15:] = segment.arms + fraction * (segment.target - segment.arms)
                h = segment.hands + fraction * (segment.hand_target - segment.hands)
                poses = self.model.pinch_poses(q, h)
                wrist = poses["left_wrist_yaw_link"][:3, 3]
                pinch = (
                    (poses["left_index_distal"] @ INDEX_TIP)[:3] + (poses["left_thumb_distal"] @ THUMB_TIP)[:3]
                ) / 2
                points = np.array([wrist, pinch])
                if previous is not None:
                    derivative = max(derivative, float(np.max(np.linalg.norm(points - previous, axis=1))) * count)
                previous = points
            segment.duration = max(segment.duration, 1.5 * derivative * 1.02 / self.config.cartesian_speed)

        def joint_phase(name, target, hand_target):
            nonlocal body, hands, reversed_segments_reused
            self.last_phase = name
            end_body = body.copy()
            end_body[15:] = target
            start_key, end_key = endpoint_key(body, hands), endpoint_key(end_body, hand_target)
            reverse = approach_segments.get((end_key, start_key)) if name in ("return", "return_ready") else None
            if reverse is None:
                segment = self.planner.segment(body, hands, target, hand_target, extra_obstacles=obstacles)
                bound_cartesian_speed(segment)
            else:
                segment = replace(
                    reverse,
                    arms=body[15:].copy(),
                    target=np.array(target, float),
                    hands=hands.copy(),
                    hand_target=np.array(hand_target, float),
                )
                reversed_segments_reused += 1
            if name == "approach_compact":
                approach_segments[(start_key, end_key)] = segment
            phases.append(
                {
                    "phase": name,
                    "arms": segment.target.tolist(),
                    "hands": segment.hand_target.tolist(),
                    "duration": segment.duration,
                }
            )
            body[15:], hands = segment.target.copy(), segment.hand_target.copy()

        def center_phase(name, center, yaw, preset, holding=False):
            nonlocal body, hands
            self.last_phase = name
            hand_target = hands.copy()
            hand_target[:6] = grip if np.array_equal(preset, CLOSED) else self.config.trial_hand("open")
            wanted_rotation = grasp_rotation(yaw)
            old_pos, old_rotation = self.model.tcp(body, hands)
            a, b = self.reference_tips(body, hands)
            old_center = old_pos + old_rotation @ ((a + b) / 2)
            rv = Rotation.from_matrix(old_rotation.T @ wanted_rotation).as_rotvec()
            count = max(
                1,
                int(np.ceil(np.linalg.norm(center - old_center) / self.config.motion_checks.cartesian_step_m)),
                int(np.ceil(np.linalg.norm(rv) / self.config.motion_checks.rotation_step_rad)),
                int(np.ceil(np.max(np.abs(hand_target - hands)) / self.config.motion_checks.pinch_hand_step_raw)),
            )
            original_hand = hands.copy()
            for fraction in np.linspace(1 / count, 1, count):
                hand = original_hand + fraction * (hand_target - original_hand)
                rotation = old_rotation @ Rotation.from_rotvec(fraction * rv).as_matrix()
                pinch = old_center + fraction * (center - old_center)
                a, b = self.reference_tips(body, hand)
                target = pinch - rotation @ ((a + b) / 2)
                nxt = self.model.solve(body, hand, target, rotation)
                self.last_candidate = {"body": nxt.tolist(), "hands": hand.tolist(), "pinch": pinch.tolist()}
                segment = self.planner.segment(body, hands, nxt[15:], hand, extra_obstacles=obstacles)
                segment.duration = max(
                    segment.duration,
                    1.5
                    * np.linalg.norm(
                        pinch
                        - (
                            self.model.tcp(body, hands)[0]
                            + self.model.tcp(body, hands)[1] @ (sum(self.reference_tips(body, hands)) / 2)
                        )
                    )
                    / self.config.cartesian_speed,
                )
                bound_cartesian_speed(segment)
                if holding:
                    # Follow actual FK through the complete interpolation, not
                    # just commanded Cartesian endpoints. Inflate the swept
                    # cube for interpolation and model tracking uncertainty.
                    previous = None
                    count_sweep = max(
                        2, int(np.ceil(np.max(np.abs(nxt[15:] - body[15:])) / self.config.motion_checks.path_step_rad))
                    )
                    for alpha in np.linspace(0, 1, count_sweep + 1):
                        q, h = body + alpha * (nxt - body), hands + alpha * (hand - hands)
                        pos, rot = self.model.tcp(q, h)
                        actual = pos + rot @ (sum(self.reference_tips(q, h)) / 2) - [0, 0, 0.015] - offset
                        self.check_cube(actual, obstacles)
                        if previous is not None:
                            for lower, upper in (*self.config.obstacles, *obstacles):
                                if segment_hits_box(previous, actual, lower, upper, 0.024 + self.config.clearance):
                                    raise ValueError("trial_cube_swept_obstacle")
                        previous = actual
                phases.append(
                    {
                        "phase": name,
                        "arms": segment.target.tolist(),
                        "hands": hand.tolist(),
                        "duration": segment.duration,
                        "pinch_center_m": pinch.tolist(),
                        "holding": holding,
                    }
                )
                body, hands = nxt, hand

        # Move with compact fingers first; opening near the body is a separate
        # checked sweep and never an implicit preparation side effect.
        first_rotation = grasp_rotation(transit_yaw)
        hover = grasp + [0, 0, hover_height]
        a, b = self.reference_tips(body, opened)
        wrist_goal = hover - first_rotation @ ((a + b) / 2)
        old_pos, old_rot = self.model.tcp(body, hands)
        rv = Rotation.from_matrix(old_rot.T @ first_rotation).as_rotvec()
        count = max(
            1,
            int(np.ceil(np.linalg.norm(wrist_goal - old_pos) / self.config.motion_checks.cartesian_step_m)),
            int(np.ceil(np.linalg.norm(rv) / self.config.motion_checks.rotation_step_rad)),
        )
        self.last_phase = "approach_compact"
        for fraction in np.linspace(1 / count, 1, count):
            target = old_pos + fraction * (wrist_goal - old_pos)
            rotation = old_rot @ Rotation.from_rotvec(fraction * rv).as_matrix()
            nxt = self.model.solve(body, hands, target, rotation)
            joint_phase("approach_compact", nxt[15:], hands)
        approach_end = len(phases)
        joint_phase("preshape", body[15:], opened)
        for name, center, yaw, preset in source_waypoints(
            grasp, transit_yaw, clearance_height, hover_height, contact_yaw
        )[1:]:
            center_phase(name, center, yaw, preset)
        contact_gate = len(phases)
        center_phase("trial_lift", grasp + [0, 0, lift_height], contact_yaw, CLOSED, holding=True)
        lift_gate = len(phases)
        center_phase("replace", grasp, contact_yaw, CLOSED, holding=True)
        center_phase("release", grasp, contact_yaw, OPEN)
        center_phase("retract", hover, contact_yaw, OPEN)
        center_phase("return_orientation", hover, transit_yaw, OPEN)
        joint_phase("compact_for_return", body[15:], start_hands)
        # Recheck any changed endpoint; only byte-identical reverse endpoints
        # may reuse an approach sweep from this same plan and scene.
        for previous in reversed(phases[: approach_end - 1]):
            joint_phase("return", np.array(previous["arms"]), start_hands)
        joint_phase("return_ready", start_body[15:], start_hands)
        for phase in phases:
            phase["duration"] = max(0.02, phase["duration"] / self.config.trial_speed_scale)
        return {
            "trial_speed_scale": self.config.trial_speed_scale,
            "lift_height_m": float(lift_height),
            "reversed_segments_reused": reversed_segments_reused,
            "schema": 1,
            "frame": "pelvis",
            "tcp_basis": "shared simulation tip geometry; unmeasured on hardware",
            "source_sequence": "astrabot.pinch.source_waypoints",
            "transit_yaw_deg": transit_yaw,
            "contact_yaw_deg": contact_yaw,
            "yaw_reference": "measured_table_tangent" if self.config.table_plane is not None else "pelvis",
            "table_rotation": table_rotation.tolist(),
            "grasp_center_m": object_grasp.tolist(),
            "command_grasp_center_m": grasp.tolist(),
            "trial_grasp_offset_m": offset.tolist(),
            "grip_target_left_raw": grip.tolist(),
            "grip_mode": self.config.trial_grip_mode,
            "phases": phases,
            "duration_seconds": sum(p["duration"] for p in phases),
            "gates": {"opposing_contact_before_phase": contact_gate, "visual_lift_check_before_phase": lift_gate},
            "start_body": start_body.tolist(),
            "start_hands": start_hands.tolist(),
            "obstacles": [[np.asarray(a).tolist(), np.asarray(b).tolist()] for a, b in obstacles],
            "execution_policy": {
                "offline_only": True,
                "contact": "check opposing tactile contact after every close increment (at most 2 raw units); "
                "on early contact stop and replan compensation/lift from measured fingers",
                "lift": "requires opposing contact and a fresh visual target check",
                "return": "requires verified replacement and released object",
                "scene": "fresh supplied obstacles and bounded observed table workspace required before hardware",
                "thermal": "offline model does not validate motor heating or retained-controller hold",
            },
            "geometry_passed": True,
            "hardware_passed": False,
        }

    def check_cube(self, center, obstacles):
        """Check the upright carried cube, allowing only intended table contact."""
        lo, hi = center - 0.02, center + 0.02
        check_observed_box(self.config, lo, hi, self.config.clearance)
        if cube_penetrates(self.config, center):
            raise ValueError("trial_cube_below_table")
        for lower, upper in (*self.config.obstacles, *obstacles):
            if np.all(hi + self.config.clearance >= lower) and np.all(lo - self.config.clearance <= upper):
                raise ValueError("trial_cube_obstacle")


def failure_diagnostics(model, start: dict, error: Exception, stage: str, scene_complete: bool) -> dict:
    """Explain a blocked plan using its recorded start, without asserting physical clearance."""
    distances = model._thigh_distances(model.poses(start["body"], start["hands"]))
    return {
        "geometry_passed": False,
        "hardware_ready": False,
        "error": str(error),
        "stage": stage,
        "scene_complete": scene_complete,
        "capture_start": start,
        "thigh_pairs_below_69mm": [
            {"hand": key[0], "thigh": key[1], "frame": key[2], "distance_mm": distance * 1000}
            for key, distance in distances.items()
            if distance <= 0.069
        ],
        "distance_scope": "模型手部包围体到大腿胶囊轴线的距离；不是实测皮肤间隙。69 mm 包含胶囊半径和采样余量。",
        "next_step": (
            "核对实测起点与模型间距；不要手改起点、碰撞阈值或通过字段。旧场景需用新版几何重新导出。"
            + ("桌面未完整入镜时，应检查路径是否落在导出的可观察工作区内。" if not scene_complete else "")
        ),
    }


def main():
    """Generate a reproducible offline plan from a recorded snapshot/target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--input", required=True, help="JSON: body, hands, grasp_center_m, obstacles, optional parameters"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from .alignment import compare_models
    from .calibration import binding
    from .collision import enable_acceleration
    from .model import RobotModel

    config = Config.load(args.config)
    source = json.loads(Path(args.input).read_text())
    model = RobotModel(config)
    enable_acceleration()
    planner = TrialPlanner(model, config)
    snapshot = SimpleNamespace(body=vector(source["body"], 29, "body"), hands=vector(source["hands"], 12, "hands"))
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; use a new plan filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    capture_start = {"body": snapshot.body.tolist(), "hands": snapshot.hands.tolist()}
    preparation = []
    stage = "trial"
    try:
        if np.max(np.abs(snapshot.body[15:] - REST)) <= 0.5:
            stage = "low_pose_preparation"
            entry = Planner(model, config)
            center = vector(source["grasp_center_m"], 3, "grasp") - [0, 0, 0.015]
            half = np.array([0.04, 0.04, 0.03])
            entry.recovery_obstacles = (
                *source.get("obstacles", ()),
                [(center - half).tolist(), (center + half).tolist()],
            )
            preparation = entry.ready(snapshot)
            snapshot.body[15:] = preparation[-1].target
            snapshot.hands = preparation[-1].hand_target.copy()
        stage = "trial"
        result = planner.plan_adaptive(
            snapshot, source["grasp_center_m"], obstacles=source.get("obstacles", ()), **source.get("parameters", {})
        )
    except (ValueError, RuntimeError) as exc:
        failure = failure_diagnostics(model, capture_start, exc, stage, source.get("scene_complete", False))
        failure["trial_phase"] = planner.last_phase
        failure["orientation_attempts"] = planner.orientation_attempts
        error_path = output.with_name(output.stem + "-error.json")
        error_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        parser.exit(1, f"Offline plan blocked ({stage}): {exc}\nReport: {error_path}\n")
    samples = [(snapshot.body, snapshot.hands)]
    for segment in preparation:
        body = np.asarray(capture_start["body"]).copy()
        body[15:] = segment.target
        samples.append((body, segment.hand_target))
    for phase in result["phases"]:
        q = snapshot.body.copy()
        q[15:] = phase["arms"]
        samples.append((q, np.asarray(phase["hands"])))
    result["alignment"] = compare_models(model, samples)
    if not result["alignment"]["passed"]:
        raise ValueError("trajectory_model_alignment_failed")
    result["binding"] = binding(config)
    result["input_sha256"] = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    result["scene_complete"] = bool(source.get("scene_complete", False))
    result["execution_scope"] = (
        "operator_cleared_table_workspace"
        if source.get("workspace_policy") == "operator_cleared"
        else (
            "observed_table_workspace"
            if config.table_observed is not None
            else "whole_table" if result["scene_complete"] else "offline_only"
        )
    )
    result["hardware_ready"] = False
    result["capture_start"] = capture_start
    result["preparation"] = [
        {
            "name": segment.name,
            "arms": segment.arms.tolist(),
            "hands": segment.hands.tolist(),
            "target": segment.target.tolist(),
            "hand_target": segment.hand_target.tolist(),
            "duration": segment.duration,
            "profile": segment.profile,
        }
        for segment in preparation
    ]
    result["trial_duration_seconds"] = result["duration_seconds"]
    result["duration_seconds"] += sum(segment.duration for segment in preparation)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": args.output,
                "geometry_passed": True,
                "hardware_ready": False,
                "phases": len(result["phases"]),
                "duration_seconds": result["duration_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
