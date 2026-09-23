"""Fixed-base URDF kinematics, articulated O6 bounds and local path checking."""

import hashlib
import json
import xml.etree.ElementTree as ET
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from ..kinematics import solve_arm_pose
from .collision import (
    accelerated_overlaps,
    capsule_hits_box,
    convex_hulls_violate_clearance,
    convex_hulls_within_distance,
    native_backend,
    oriented_box_hits_box,
    segment_box_distance,
    segment_distance,
)
from .config import ARM_NAMES, ASSETS, BODY_NAMES, HAND_PARTS, HAND_RANGES, vector
from .scene_coverage import check_observed_box
from .table import cube_penetrates
from .table import vertices as table_vertices


def overlaps(lo, hi, other_lo, other_hi, margin=0.0):
    """Test batches of bounding boxes."""
    native = accelerated_overlaps(lo, hi, other_lo, other_hi, margin)
    if native is not None:
        return native
    return bool(
        np.any(
            np.all(
                (np.atleast_2d(hi)[:, None] + margin >= np.atleast_2d(other_lo))
                & (np.atleast_2d(lo)[:, None] - margin <= np.atleast_2d(other_hi)),
                axis=-1,
            )
        )
    )


class RobotModel:
    """All positions are in pelvis coordinates; no assumed simulation root."""

    def __init__(self, config, *, environment_cleared=False):
        self.config = config
        self.environment_cleared = environment_cleared
        root = ET.parse(ASSETS / "g1_o6.urdf").getroot()
        self.joints = []
        self.limits = {}
        for node in root.findall("joint"):
            origin = node.find("origin")
            t = np.eye(4)
            if origin is not None:
                t[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                t[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            axis = node.find("axis")
            mimic = node.find("mimic")
            name = node.get("name")
            self.joints.append(
                {
                    "name": name,
                    "parent": node.find("parent").get("link"),
                    "child": node.find("child").get("link"),
                    "origin": t,
                    "axis": np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.array([1, 0, 0]),
                    "type": node.get("type"),
                    "mimic": dict(mimic.attrib) if mimic is not None else None,
                }
            )
            limit = node.find("limit")
            if limit is not None:
                self.limits[name] = (float(limit.get("lower", "-3.14")), float(limit.get("upper", "3.14")))
        children = {j["child"] for j in self.joints}
        self.root = next(j["parent"] for j in self.joints if j["parent"] not in children)
        # URDF topology is fixed for this model; retain the joint dictionaries
        # so geometry/axis updates are still reflected in every FK evaluation.
        available_frames = {self.root}
        available_q = set(BODY_NAMES) | {f"{side}_{part}" for side in ("left", "right") for part in HAND_PARTS}
        pending, self._pose_order = list(self.joints), []
        while pending:
            remaining = []
            for joint in pending:
                mimic = joint["mimic"]
                if joint["parent"] not in available_frames or (mimic and mimic["joint"] not in available_q):
                    remaining.append(joint)
                    continue
                self._pose_order.append(joint)
                available_frames.add(joint["child"])
                if mimic:
                    available_q.add(joint["name"])
            if len(remaining) == len(pending):
                raise ValueError("unresolved URDF tree/mimic")
            pending = remaining
        self._tcp_orders = {}
        by_child = {j["child"]: j for j in self._pose_order}
        by_name = {j["name"]: j for j in self._pose_order}
        for side in ("left", "right"):
            required, pending_frames = set(), [f"{side}_wrist_yaw_link"]
            while pending_frames:
                frame = pending_frames.pop()
                if frame == self.root or frame in required:
                    continue
                required.add(frame)
                joint = by_child[frame]
                pending_frames.append(joint["parent"])
                if joint["mimic"] and joint["mimic"]["joint"] in by_name:
                    pending_frames.append(by_name[joint["mimic"]["joint"]]["child"])
            self._tcp_orders[side] = [j for j in self._pose_order if j["child"] in required]
        # The trial speed bound needs only the two contact tips and wrist.
        required = set()
        pending_frames = ["left_index_distal", "left_thumb_distal", "left_wrist_yaw_link"]
        while pending_frames:
            frame = pending_frames.pop()
            if frame == self.root or frame in required:
                continue
            required.add(frame)
            joint = by_child[frame]
            pending_frames.append(joint["parent"])
            if joint["mimic"] and joint["mimic"]["joint"] in by_name:
                pending_frames.append(by_name[joint["mimic"]["joint"]]["child"])
        self._pinch_order = [j for j in self._pose_order if j["child"] in required]
        # Joint names survive serialization into the isolated planning worker.
        self._pose_layouts = {}
        for order in (self._pose_order, self._pinch_order, *self._tcp_orders.values()):
            frames = {self.root: 0} | {joint["child"]: index + 1 for index, joint in enumerate(order)}
            parents = np.array([frames[j["parent"]] for j in order], dtype=np.int64)
            kinds = np.array(
                [1 if j["type"] in ("revolute", "continuous") else 2 if j["type"] == "prismatic" else 0 for j in order],
                dtype=np.int64,
            )
            self._pose_layouts[tuple(j["name"] for j in order)] = parents, kinds
        self.bounds = json.loads((ASSETS / "hand_bounds.json").read_text())
        self.head_bounds = json.loads((ASSETS / "head_bounds.json").read_text())["entries"]
        body_metadata = json.loads((ASSETS / "body_bounds.json").read_text())
        self.body_bounds = body_metadata["entries"]
        hull_path = ASSETS / "body_hulls.npz"
        if hashlib.sha256(hull_path.read_bytes()).hexdigest() != body_metadata["hull_file_sha256"]:
            raise ValueError("body_hull_hash_mismatch")
        with np.load(hull_path, allow_pickle=False) as archive:
            self.body_hulls = {
                entry["hull_key"]: archive[entry["hull_key"]].astype(float) for entry in self.body_bounds
            }
        self.lower = np.array([self.limits[n][0] for n in ARM_NAMES])
        self.upper = np.array([self.limits[n][1] for n in ARM_NAMES])
        self.escape_geometry = None

    def _load_escape_geometry(self):
        if self.escape_geometry is not None:
            return
        metadata = json.loads((ASSETS / "escape_bounds.json").read_text())
        path = ASSETS / "escape_hulls.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["hull_file_sha256"]:
            raise ValueError("escape_hull_hash_mismatch")
        with np.load(path, allow_pickle=False) as archive:
            self.escape_geometry = [
                (entry["frame"], archive[entry["hull_key"]].astype(float)) for entry in metadata["entries"]
            ]
        parents = {joint["child"]: joint for joint in self.joints}
        for frame, vertices in self.escape_geometry:
            # Triangle inequality certifies every moving ancestor's radius for
            # all angles, once per model rather than once per path sample.
            radius, link = float(np.max(np.linalg.norm(vertices, axis=1))), frame
            while link in parents:
                joint = parents[link]
                if joint["type"] in ("revolute", "continuous") and joint["name"] not in BODY_NAMES[:15] and radius >= 1:
                    self.escape_geometry = None
                    raise ValueError("escape_mesh_outside_radius_bound")
                if joint["type"] == "prismatic":
                    self.escape_geometry = None
                    raise ValueError("escape_mesh_prismatic_ancestor")
                radius += float(np.linalg.norm(joint["origin"][:3, 3]))
                link = joint["parent"]

    def poses(self, body, hands):
        """Resolve all joints and mimic links using measured hardware order."""
        return self._poses(body, hands, self._pose_order)

    def pinch_poses(self, body, hands):
        """Resolve the left wrist and both contact tips with all their ancestors."""
        return self._poses(body, hands, self._pinch_order)

    def _poses(self, body, hands, order):
        q = dict(zip(BODY_NAMES, vector(body, 29, "body")))
        hands = vector(hands, 12, "hands")
        if np.any((hands < 0) | (hands > 255)):
            raise ValueError("hand raw range")
        for i, side in enumerate(("left", "right")):
            q.update(zip((f"{side}_{part}" for part in HAND_PARTS), (1 - hands[i * 6 : i * 6 + 6] / 255) * HAND_RANGES))
        for joint in order:
            mimic = joint["mimic"]
            if mimic is not None:
                q[joint["name"]] = q[mimic["joint"]] * float(mimic.get("multiplier", 1)) + float(mimic.get("offset", 0))
        native = native_backend()
        if native is not None:
            parents, kinds = self._pose_layouts[tuple(j["name"] for j in order)]
            motion = np.array([j["axis"] for j in order]) * np.array([q.get(j["name"], 0) for j in order])[:, None]
            rotations = Rotation.from_rotvec(motion[kinds == 1]).as_matrix()
            # Read current origins every time: native execution must not hide
            # a changed mounting transform behind cached kinematic results.
            origins = np.array([j["origin"] for j in order])
            matrices = native.compose_poses(origins, rotations, parents, kinds, motion)
            return {self.root: matrices[0]} | {j["child"]: matrices[i + 1] for i, j in enumerate(order)}
        rotating = [joint for joint in order if joint["type"] in ("revolute", "continuous")]
        # SciPy supports a batch of the same axis-angle conversions. This
        # removes thousands of repeated Python/C API calls during planning.
        rotations = iter(Rotation.from_rotvec([j["axis"] * q.get(j["name"], 0) for j in rotating]).as_matrix())
        poses = {self.root: np.eye(4)}
        for joint in order:
            t = joint["origin"].copy()
            if joint["type"] in ("revolute", "continuous"):
                t[:3, :3] = t[:3, :3] @ next(rotations)
            elif joint["type"] == "prismatic":
                t[:3, 3] += t[:3, :3] @ (joint["axis"] * q.get(joint["name"], 0))
            poses[joint["child"]] = poses[joint["parent"]] @ t
        return poses

    def tcp(self, body, hands, side="left"):
        """Return the calibrated wrist-local TCP and wrist rotation."""
        pose = self._poses(body, hands, self._tcp_orders[side])[f"{side}_wrist_yaw_link"]
        return pose[:3, 3] + pose[:3, :3] @ self.config.tcp_offset, pose[:3, :3]

    def boxes(self, poses, side):
        """Transform mesh-derived local corners; retain every finger link."""
        transforms = np.array([poses[entry["frame"]] for entry in self.bounds[side]])
        corners = np.array([entry["corners"] for entry in self.bounds[side]])
        points = corners @ transforms[:, :3, :3].transpose(0, 2, 1) + transforms[:, None, :3, 3]
        return np.stack((points.min(axis=1), points.max(axis=1)), axis=1)

    def check(self, body, hands, holding=False, extra_obstacles=(), thigh_mesh=False):
        """Reject table, torso, arm/hand and opposite-arm collisions."""
        poses = None
        if thigh_mesh:
            poses = self.poses(body, hands)
            self._check_escape_mesh_sweep(poses, poses, np.zeros(14), preparation=True)
        return self._check(body, hands, holding, extra_obstacles, thigh_mesh_poses=poses)

    def _check(
        self,
        body,
        hands,
        holding=False,
        extra_obstacles=(),
        thigh_minimums=None,
        recovery_start=None,
        thigh_mesh_poses=None,
        fixed_side_checks=None,
    ):
        """Internal checker; only a fully validated escape may supply pair minima."""
        if not self.config.scene_measured and not self.environment_cleared:
            raise ValueError("scene_not_measured")
        arms = np.asarray(body)[15:]
        lower = self.lower if recovery_start is None else np.minimum(self.lower, recovery_start)
        upper = self.upper if recovery_start is None else np.maximum(self.upper, recovery_start)
        if np.any(arms < lower) or np.any(arms > upper):
            raise ValueError("joint_limit")
        poses = self.poses(body, hands) if thigh_mesh_poses is None else thigh_mesh_poses
        boxes = {s: self.boxes(poses, s) for s in ("left", "right")}
        native = native_backend()
        capsules = {}
        for side in boxes:
            points = [poses[f"{side}_{name}_link"][:3, 3] for name in ("shoulder_yaw", "elbow", "wrist_yaw")]
            capsules[side] = [(points[0], points[1]), (points[1], points[2])]
        if fixed_side_checks is not None and fixed_side_checks[2]:
            body_boxes, head_boxes, thighs = fixed_side_checks[2][0]
        else:
            body_boxes = []
            for entry in self.body_bounds:
                pose = poses[entry["frame"]]
                points = np.asarray(entry["corners"]) @ pose[:3, :3].T + pose[:3, 3]
                body_boxes.append((entry, pose, points.min(0), points.max(0)))
            # Source head geometry follows the torso through the measured waist joints.
            head_boxes = []
            for entry in self.head_bounds:
                pose = poses[entry["frame"]]
                points = np.asarray(entry["corners"]) @ pose[:3, :3].T + pose[:3, 3]
                inverse = np.eye(4)
                inverse[:3, :3] = pose[:3, :3].T
                inverse[:3, 3] = -inverse[:3, :3] @ pose[:3, 3]
                head_boxes.append((entry, inverse, points.min(0), points.max(0)))
            thighs = [
                (poses[f"{side}_hip_yaw_link"][:3, 3], poses[f"{side}_knee_link"][:3, 3]) for side in ("left", "right")
            ]
            if fixed_side_checks is not None:
                fixed_side_checks[2].append((body_boxes, head_boxes, thighs))
        if self.environment_cleared and (holding or extra_obstacles):
            raise ValueError("cleared_environment_only_supports_empty_hand_preparation")
        obstacles = (
            []
            if self.environment_cleared
            else [(self.config.table_min, self.config.table_max), *self.config.obstacles, *extra_obstacles]
        )
        margin = self.config.clearance + 0.004  # covers the 0.001 rad path samples
        table = (
            table_vertices(self.config)
            if not self.environment_cleared and self.config.table_plane is not None
            else None
        )
        if table is not None:
            table_margin = margin + self.config.table_uncertainty
            expanded_table = (
                table[:, None, :] + np.array(list(product((-table_margin, table_margin), repeat=3)))
            ).reshape(-1, 3)
            table_lo, table_hi = table.min(0), table.max(0)
        for side, b in boxes.items():
            # Only reuse environment checks within this synchronous path.
            # Cross-arm/hand checks below always run, including the fixed side.
            if fixed_side_checks is None or side not in fixed_side_checks[1]:
                for lo, hi in b:
                    check_observed_box(self.config, lo, hi, margin)
                for a, z in capsules[side]:
                    check_observed_box(self.config, np.minimum(a, z), np.maximum(a, z), 0.035 + margin)
                for index, (lo, hi) in enumerate(obstacles):
                    if index == 0 and table is not None:
                        table_hand_overlap = overlaps(b[:, 0], b[:, 1], table_lo, table_hi, table_margin)
                        for entry, (hand_lo, hand_hi) in zip(self.bounds[side], b):
                            if not table_hand_overlap or not overlaps(
                                hand_lo, hand_hi, table_lo, table_hi, table_margin
                            ):
                                continue
                            pose = poses[entry["frame"]]
                            corners = np.asarray(entry["corners"]) @ pose[:3, :3].T + pose[:3, 3]
                            if convex_hulls_within_distance(corners, expanded_table, 0):
                                raise ValueError(f"hand_obstacle:{side}:table:{entry['frame']}")
                        for a, z in capsules[side]:
                            if capsule_hits_box(
                                a, z, table_lo, table_hi, 0.035 + table_margin
                            ) and convex_hulls_within_distance(np.array([a, z]), table, 0.035 + table_margin):
                                raise ValueError(f"arm_obstacle:{side}")
                        continue
                    if overlaps(b[:, 0], b[:, 1], lo, hi, margin):
                        for entry, (hand_lo, hand_hi) in zip(self.bounds[side], b):
                            if not overlaps(hand_lo, hand_hi, lo, hi, margin):
                                continue
                            local = np.asarray(entry["corners"])
                            if oriented_box_hits_box(local.min(0), local.max(0), poses[entry["frame"]], lo, hi, margin):
                                raise ValueError(f"hand_obstacle:{side}:{index}:{entry['frame']}")
                    if any(capsule_hits_box(a, z, lo, hi, 0.035 + margin) for a, z in capsules[side]):
                        raise ValueError(f"arm_obstacle:{side}")
                self._check_head(poses, side, b, capsules[side], head_boxes, margin)
                # The operator confirms that low resting hands can touch the soft
                # thigh surface. Once the shoulder is outward, do not insist on an
                # additional 5 mm air gap while curling/lifting. Keep the physical
                # envelope and 4 mm sweep reserve; elevated motions retain normal
                # clearance. The initial close escape is independently certified.
                outward = body[16] if side == "left" else -body[23]
                low_and_outward = poses[f"{side}_wrist_yaw_link"][2, 3] < 0 and outward >= 0.28 - 1e-10
                thigh_margin = 0.004 if low_and_outward else margin
                for thigh_side, (a, z) in zip(("left", "right"), thighs):
                    candidates = (
                        native.capsule_box_candidates(a, z, b, 0.065 + thigh_margin)
                        if native is not None
                        else [capsule_hits_box(a, z, lo, hi, 0.065 + thigh_margin) for lo, hi in b]
                    )
                    for entry, hit in zip(self.bounds[side], candidates):
                        if not hit:
                            continue
                        # The world AABB grows when a hand link rotates. Resolve
                        # broad-phase overlaps in the original mesh-bound frame,
                        # preserving the full thigh radius and clearance.
                        t = poses[entry["frame"]]
                        local = np.asarray(entry["corners"])
                        if capsule_hits_box(
                            t[:3, :3].T @ (a - t[:3, 3]),
                            t[:3, :3].T @ (z - t[:3, 3]),
                            local.min(0),
                            local.max(0),
                            0.065 + thigh_margin,
                        ):
                            key = (side, thigh_side, entry["frame"])
                            minimum = (thigh_minimums or {}).get(key)
                            distance = segment_box_distance(
                                t[:3, :3].T @ (a - t[:3, 3]),
                                t[:3, :3].T @ (z - t[:3, 3]),
                                local.min(0),
                                local.max(0),
                            )
                            if thigh_mesh_poses is None and (minimum is None or distance < minimum - 1e-10):
                                raise ValueError(f"hand_thigh:{side}")
                    if any(segment_distance(a, z, c, d) < 0.10 + margin for c, d in capsules[side]):
                        raise ValueError(f"arm_thigh:{side}")
                for body_entry, pose, lo, hi in body_boxes:
                    # URDF poses are rigid transforms. Avoid a generic matrix
                    # solve for every body hull at every path sample.
                    inverse = np.eye(4)
                    inverse[:3, :3] = pose[:3, :3].T
                    inverse[:3, 3] = -inverse[:3, :3] @ pose[:3, 3]
                    if overlaps(b[:, 0], b[:, 1], lo, hi, 0.005):
                        for entry in self.bounds[side]:
                            local = np.asarray(entry["corners"])
                            if oriented_box_hits_box(
                                local.min(0),
                                local.max(0),
                                inverse @ poses[entry["frame"]],
                                body_entry["lower"],
                                body_entry["upper"],
                                0.005,
                            ):
                                transform = inverse @ poses[entry["frame"]]
                                corners = local @ transform[:3, :3].T + transform[:3, 3]
                                if convex_hulls_within_distance(
                                    self.body_hulls[body_entry["hull_key"]], corners, 0.005 * np.sqrt(3)
                                ):
                                    raise ValueError(f"hand_torso:{side}")
                    if any(
                        capsule_hits_box(
                            inverse[:3, :3] @ a + inverse[:3, 3],
                            inverse[:3, :3] @ z + inverse[:3, 3],
                            body_entry["lower"],
                            body_entry["upper"],
                            0.035,
                        )
                        and convex_hulls_within_distance(
                            self.body_hulls[body_entry["hull_key"]],
                            np.array([a, z]) @ inverse[:3, :3].T + inverse[:3, 3],
                            0.035,
                        )
                        for a, z in capsules[side]
                    ):
                        raise ValueError(f"arm_torso:{side}")
                if fixed_side_checks is not None and side in fixed_side_checks[0]:
                    fixed_side_checks[1].add(side)
            other = "right" if side == "left" else "left"
            if native is not None:
                local = np.array([entry["corners"] for entry in self.bounds[side]])
                if native.capsules_hit_local_boxes(
                    np.array([*capsules[other], capsules[side][0]]),
                    np.array([poses[entry["frame"]] for entry in self.bounds[side]]),
                    local.min(axis=1),
                    local.max(axis=1),
                    0.040,
                ):
                    raise ValueError(f"hand_arm:{side}")
            else:
                for a, z in [*capsules[other], capsules[side][0]]:
                    for entry in self.bounds[side]:
                        t = poses[entry["frame"]]
                        local = np.asarray(entry["corners"])
                        if capsule_hits_box(
                            t[:3, :3].T @ (a - t[:3, 3]),
                            t[:3, :3].T @ (z - t[:3, 3]),
                            local.min(0),
                            local.max(0),
                            0.040,
                        ):
                            raise ValueError(f"hand_arm:{side}")
        if overlaps(boxes["left"][:, 0], boxes["left"][:, 1], boxes["right"][:, 0], boxes["right"][:, 1], margin):
            raise ValueError("hands_collide")
        if any(segment_distance(a, b, c, d) < 0.07 + margin for a, b in capsules["left"] for c, d in capsules["right"]):
            raise ValueError("arms_collide")
        if holding:
            center, _ = self.tcp(body, hands)
            # Upright cube centre is 15 mm below its measured top pinch point.
            center = center - [0, 0, 0.015]
            check_observed_box(self.config, center - 0.022, center + 0.022, margin)
            # Only the intended cube/table contact permits 2 mm measurement error.
            # Hand/arm table checks above retain the full clearance margin.
            if cube_penetrates(self.config, center):
                raise ValueError("held_cube_below_table")
            for lo, hi in obstacles[1:]:
                if overlaps(center - 0.022, center + 0.022, lo, hi, margin):
                    raise ValueError("held_cube_obstacle")

    def wrist_recovery_target(self, body):
        """Return inward targets for wrists within 10 mrad of a model limit.

        Only measured wrist overshoot up to 2 mrad is eligible. The model
        limits themselves remain unchanged; targets have 50 mrad clearance.
        """
        arms = vector(body, 29, "body")[15:].copy()
        wrist = np.zeros(14, dtype=bool)
        wrist[[4, 5, 6, 11, 12, 13]] = True
        outside = np.maximum(self.lower - arms, arms - self.upper)
        if np.any(outside > self.config.motion_checks.wrist_entry_overshoot_rad) or np.any(outside[~wrist] > 0):
            raise ValueError("wrist_recovery_entry_limit")
        target = arms.copy()
        low = wrist & (arms < self.lower + self.config.motion_checks.wrist_trigger_margin_rad)
        high = wrist & (arms > self.upper - self.config.motion_checks.wrist_trigger_margin_rad)
        target[low] = self.lower[low] + self.config.motion_checks.wrist_target_margin_rad
        target[high] = self.upper[high] - self.config.motion_checks.wrist_target_margin_rad
        if not np.any(low | high):
            raise ValueError("wrist_recovery_not_needed")
        return target

    def check_wrist_recovery(self, body, hands, target):
        """Check the actual inward curve, including its measured start geometry."""
        body = vector(body, 29, "body")
        target = vector(target, 14, "target")
        if not np.array_equal(target, self.wrist_recovery_target(body)):
            raise ValueError("wrist_recovery_target_mismatch")
        start = body[15:].copy()
        count = max(1, int(np.ceil(np.max(np.abs(target - start)) / self.config.motion_checks.path_step_rad)))
        for fraction in np.linspace(0, 1, count + 1):
            sample = body.copy()
            sample[15:] = start + fraction * (target - start)
            self._check(sample, hands, recovery_start=start)
        self.check(sample, hands)

    def _thigh_distances(self, poses, keys=None):
        distances = {}
        for side in ("left", "right"):
            for thigh_side in ("left", "right"):
                a, z = (poses[f"{thigh_side}_{part}_link"][:3, 3] for part in ("hip_yaw", "knee"))
                for entry in self.bounds[side]:
                    key = (side, thigh_side, entry["frame"])
                    if keys is not None and key not in keys:
                        continue
                    t = poses[entry["frame"]]
                    local = np.asarray(entry["corners"])
                    distances[key] = segment_box_distance(
                        t[:3, :3].T @ (a - t[:3, 3]),
                        t[:3, :3].T @ (z - t[:3, 3]),
                        local.min(0),
                        local.max(0),
                    )
        return distances

    def _check_escape_mesh_sweep(self, previous, current, step, *, preparation=False):
        """Certify close-to-thigh withdrawal with source convex meshes.

        The capsule is a broad approximation, not a physical skin surface.
        Enclose each moving mesh between its endpoint hulls, adding the signed
        preparation clearance (escape: 0.25 mm) and a rotational chord bound.
        Escape uses shoulder-only steps. Preparation includes every arm/finger
        and mimic angle; each moving ancestor must satisfy the 1 m radius bound.
        """
        self._load_escape_geometry()
        legs, hands = [], []
        for frame, vertices in self.escape_geometry:
            pose = current[frame]
            points = vertices @ pose[:3, :3].T + pose[:3, 3]
            if any(part in frame for part in ("_hip_", "_knee_")):
                if preparation and not np.allclose(previous[frame], pose, atol=1e-12, rtol=0):
                    raise ValueError("prepare_thigh_sweep_requires_fixed_body")
                legs.append(points)
            else:
                old = previous[frame]
                start = vertices @ old[:3, :3].T + old[:3, 3]
                hands.append((frame, np.vstack((start, points))))
        margin = (self.config.motion_checks.prepare_thigh_clearance_m if preparation else 0.00025) + 0.5 * float(
            np.sum(np.abs(step))
        ) ** 2
        if not hands or not legs:
            return
        # Bounds belong to a mesh, not a mesh pair. Compute each once for this
        # sample, then screen all pairs together without caching across poses.
        hand_lo = np.array([mesh.min(0) for _, mesh in hands])
        hand_hi = np.array([mesh.max(0) for _, mesh in hands])
        leg_lo = np.array([mesh.min(0) for mesh in legs])
        leg_hi = np.array([mesh.max(0) for mesh in legs])
        candidates = np.argwhere(
            np.all(hand_hi[:, None] + margin >= leg_lo, axis=-1) & np.all(hand_lo[:, None] - margin <= leg_hi, axis=-1)
        )
        for hand_index, leg_index in candidates:
            frame, hand = hands[hand_index]
            if convex_hulls_violate_clearance(hand, legs[leg_index], margin):
                prefix = "prepare_hand_thigh_mesh_collision" if preparation else "thigh_escape_mesh_collision"
                raise ValueError(f"{prefix}:{frame}")

    def _path_joint_steps(self, start, end, hand_start, hand_end):
        """Include every driven and mimic rotation in the swept chord bound."""
        delta = dict(zip(BODY_NAMES, np.r_[np.zeros(15), np.asarray(end) - start]))
        for i, side in enumerate(("left", "right")):
            angles = (np.asarray(hand_start)[i * 6 : i * 6 + 6] - np.asarray(hand_end)[i * 6 : i * 6 + 6]) / 255
            delta.update(zip((f"{side}_{part}" for part in HAND_PARTS), angles * HAND_RANGES))
        pending = [joint for joint in self.joints if joint["mimic"] is not None]
        while pending:
            remaining = []
            for joint in pending:
                mimic = joint["mimic"]
                if mimic["joint"] not in delta:
                    remaining.append(joint)
                else:
                    delta[joint["name"]] = delta[mimic["joint"]] * float(mimic.get("multiplier", 1))
            if len(remaining) == len(pending):
                raise ValueError("unresolved_sweep_mimic")
            pending = remaining
        return np.array(list(delta.values()))

    def check_thigh_escape(self, body, hands, target, extra_obstacles=()):
        """Validate outward withdrawal from a measured low pose, including close contact.

        Source hull sweeps resolve capsule false positives. Every initially
        deficient capsule distance must be nondecreasing and the endpoint must
        pass ordinary full-clearance checks. No general collision bypass exists.
        """
        body = vector(body, 29, "body")
        hands = vector(hands, 12, "hands")
        target = vector(target, 14, "arms")
        delta = target - body[15:]
        fixed = np.ones(14, dtype=bool)
        fixed[[1, 2, 8, 9]] = False
        if (
            np.any(np.abs(delta[fixed]) > 1e-12)
            or delta[1] < 0
            or delta[8] > 0
            or np.max(np.abs(delta[[1, 8]])) > self.config.motion_checks.escape_shoulder_roll_rad
            or np.max(np.abs(delta[[2, 9]])) > self.config.motion_checks.escape_shoulder_yaw_rad
        ):
            raise ValueError("thigh_escape_requires_small_outward_shoulder_motion")
        poses = self.poses(body, hands)
        if np.max(np.abs(body[[18, 25]] - 0.98)) > self.config.motion_checks.escape_elbow_entry_rad or any(
            poses[f"{side}_wrist_yaw_link"][2, 3] > 0 for side in ("left", "right")
        ):
            raise ValueError("thigh_escape_requires_low_rest_pose")
        full_radius = 0.065 + self.config.clearance + 0.004
        initial = {k: d for k, d in self._thigh_distances(poses).items() if d <= full_radius + 1e-12}
        refine_capsule = any(distance <= 0.069 for distance in initial.values())
        previous = initial
        previous_poses = poses
        previous_outward = np.array([poses["left_wrist_yaw_link"][1, 3], -poses["right_wrist_yaw_link"][1, 3]])
        count = max(1, int(np.ceil(np.max(np.abs(delta)) / self.config.motion_checks.path_step_rad)))
        for fraction in np.linspace(0, 1, count + 1):
            q = body.copy()
            q[15:] += fraction * delta
            current_poses = self.poses(q, hands)
            outward = np.array(
                [current_poses["left_wrist_yaw_link"][1, 3], -current_poses["right_wrist_yaw_link"][1, 3]]
            )
            if np.any(outward < previous_outward - 1e-10):
                raise ValueError("thigh_escape_wrist_moved_inward")
            distances = self._thigh_distances(current_poses, initial) if initial else {}
            if any(d < previous[k] - 1e-10 for k, d in distances.items()):
                raise ValueError("thigh_escape_clearance_decreased")
            if refine_capsule:
                self._check_escape_mesh_sweep(previous_poses, current_poses, delta / count)
            self._check(q, hands, extra_obstacles=extra_obstacles, thigh_minimums=initial)
            previous = distances
            previous_poses = current_poses
            previous_outward = outward
        self.check(q, hands, extra_obstacles=extra_obstacles)

    def _check_head(self, poses, side, boxes, capsules, head_boxes, margin):
        """Check source head boxes in their moving link frames, including clearance."""
        for head, inverse, head_lo, head_hi in head_boxes:
            hand_overlap = overlaps(boxes[:, 0], boxes[:, 1], head_lo, head_hi, np.sqrt(3) * margin)
            for entry, (lower, upper) in zip(self.bounds[side], boxes):
                # Local per-axis padding may project onto any world axis.
                if not hand_overlap or not overlaps(lower, upper, head_lo, head_hi, np.sqrt(3) * margin):
                    continue
                local = np.asarray(entry["corners"])
                if oriented_box_hits_box(
                    local.min(0),
                    local.max(0),
                    inverse @ poses[entry["frame"]],
                    head["lower"],
                    head["upper"],
                    margin,
                ):
                    raise ValueError(f"head_collision:{side}:{entry['frame']}")
            for index, (start, end) in enumerate(capsules):
                if capsule_hits_box(
                    inverse[:3, :3] @ start + inverse[:3, 3],
                    inverse[:3, :3] @ end + inverse[:3, 3],
                    head["lower"],
                    head["upper"],
                    0.035 + margin,
                ):
                    raise ValueError(f"head_collision:{side}:arm_{index}")

    def check_path(self, body, start, end, hand_start, hand_end, **kwargs):
        """Sweep arms and articulated hands together at bounded angular increments."""
        delta = max(float(np.max(np.abs(end - start))), float(np.max(np.abs(hand_end - hand_start))) / 255 * 1.6)
        n = max(1, int(np.ceil(delta / self.config.motion_checks.path_step_rad)))
        previous_poses = None
        step = self._path_joint_steps(start, end, hand_start, hand_end) / n if kwargs.get("thigh_mesh") else None
        checks = {key: value for key, value in kwargs.items() if key != "thigh_mesh"}
        fixed_sides = {
            side
            for i, side in enumerate(("left", "right"))
            if np.array_equal(start[i * 7 : (i + 1) * 7], end[i * 7 : (i + 1) * 7])
            and np.array_equal(hand_start[i * 6 : (i + 1) * 6], hand_end[i * 6 : (i + 1) * 6])
        }
        # Legs, waist and torso are fixed by this joint-space path definition.
        fixed_side_checks = (fixed_sides, set(), [])
        for t in np.linspace(0, 1, n + 1):
            q = np.array(body, float)
            q[15:] = start + t * (end - start)
            hand = hand_start + t * (hand_end - hand_start)
            if step is not None:
                poses = self.poses(q, hand)
                self._check_escape_mesh_sweep(
                    poses if previous_poses is None else previous_poses,
                    poses,
                    np.zeros_like(step) if previous_poses is None else step,
                    preparation=True,
                )
                self._check(q, hand, thigh_mesh_poses=poses, **checks)
                previous_poses = poses
            else:
                self._check(q, hand, fixed_side_checks=fixed_side_checks, **kwargs)

    def solve(self, body, hands, target, rotation, *, local_point=None):
        """Solve left-arm IK locally, preserving waist and the opposite arm."""
        seed = np.array(body, float)
        target = vector(target, 3, "target")
        initial = seed[15:22].copy()

        def fk(angles):
            q = seed.copy()
            q[15:22] = angles
            p, r = self.tcp(q, hands)
            if local_point is not None:
                p = p + r @ (np.asarray(local_point) - self.config.tcp_offset)
            pose = np.eye(4)
            pose[:3, 3], pose[:3, :3] = p, r
            return pose

        solution = solve_arm_pose(fk, initial, self.lower[:7], self.upper[:7], target, rotation)
        seed[15:22] = solution
        p, r = self.tcp(seed, hands)
        if local_point is not None:
            p = p + r @ (np.asarray(local_point) - self.config.tcp_offset)
        position_error = float(np.linalg.norm(p - target))
        rotation_error = float(Rotation.from_matrix(rotation.T @ r).magnitude())
        checks = self.config.motion_checks
        if position_error > checks.ik_position_error_m or rotation_error > checks.ik_rotation_error_rad:
            raise ValueError(
                f"IK_residual:position_mm={position_error * 1000:.2f}/{checks.ik_position_error_m * 1000:.2f},"
                f"rotation_rad={rotation_error:.4f}/{checks.ik_rotation_error_rad:.4f}"
            )
        jump = float(np.max(np.abs(solution - initial)))
        if jump > checks.ik_branch_jump_rad:
            raise ValueError(f"IK_branch_jump:step_rad={jump:.4f}/{checks.ik_branch_jump_rad:.4f}")
        return seed
