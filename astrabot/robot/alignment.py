"""Offline model correspondence checks; never imports a hardware adapter."""

import numpy as np
from scipy.spatial.transform import Rotation

from ..kinematics import Kinematics
from .config import BODY_NAMES, HAND_PARTS, HAND_RANGES


def simulation_joints(body, hands, model=None):
    """Map measured body/raw fingers to the simulator's joint and mimic names."""
    joints = dict(zip(BODY_NAMES, body))
    for index, prefix in enumerate(("lh", "rh")):
        angles = (1 - np.asarray(hands[index * 6 : index * 6 + 6]) / 255) * HAND_RANGES
        joints.update({f"{prefix}_{part}": float(value) for part, value in zip(HAND_PARTS, angles)})
        for finger in ("pinky", "ring", "middle", "index"):
            joints[f"{prefix}_{finger}_dip"] = joints[f"{prefix}_{finger}_mcp_pitch"] * 0.89
        joints[f"{prefix}_thumb_ip"] = joints[f"{prefix}_thumb_cmc_pitch"] * 2.29
    if model is not None:
        for joint in model.joints:
            mimic = joint["mimic"]
            if mimic is not None:
                name = joint["name"].replace("left_", "lh_", 1).replace("right_", "rh_", 1)
                parent = mimic["joint"].replace("left_", "lh_", 1).replace("right_", "rh_", 1)
                joints[name] = joints[parent] * float(mimic.get("multiplier", 1)) + float(mimic.get("offset", 0))
    return joints


def physical_simulator_profile(model):
    """Apply explicit URDF hand mounts to a private simulator FK instance.

    The packaged native simulator model remains unchanged. This overlay is for
    replaying real hardware trajectories and does not claim a live scene edit.
    """
    sim = Kinematics([0, 0, 0], [1, 0, 0, 0])
    poses = model.poses(np.zeros(29), np.full(12, 255.0))
    mounts = {}
    for side, prefix in (("left", "lh"), ("right", "rh")):
        joint = sim.by_child[f"{prefix}_hand_base_link"]
        physical = np.linalg.inv(poses[f"{side}_wrist_yaw_link"]) @ poses[f"{side}_hand_base_link"]
        mounts[side] = {
            "native_transform": (joint["f0"] @ joint["invf1"]).tolist(),
            "physical_transform": physical.tolist(),
        }
        joint["f0"], joint["invf1"] = physical, np.eye(4)
    return sim, mounts


def simulation_poses(sim, q):
    """Resolve each simulator link once per sample, including mount overlays."""
    poses = {}

    def resolve(link):
        if link in poses:
            return poses[link]
        if link not in sim.by_child:
            poses[link] = sim.root.copy()
        else:
            joint = sim.by_child[link]
            motion = np.eye(4)
            if joint["type"] == "PhysicsRevoluteJoint":
                axis = np.eye(3)["XYZ".index(joint["axis"])]
                motion[:3, :3] = Rotation.from_rotvec(axis * q.get(joint["name"], 0.0)).as_matrix()
            poses[link] = resolve(joint["parent"]) @ joint["f0"] @ motion @ joint["invf1"]
        return poses[link]

    for link in sim.by_child:
        resolve(link)
    resolve("pelvis")
    return poses


def compare_models(model, samples, physical_profile=True):
    """Compare both FK trees in the same pelvis frame, including every hand link."""
    sim, mounts = physical_simulator_profile(model)
    if not physical_profile:
        sim = Kinematics([0, 0, 0], [1, 0, 0, 0])
    positions, angles = [], []
    for body, hands in samples:
        q = simulation_joints(body, hands, model if physical_profile else None)
        poses = model.poses(body, hands)
        simulator_poses = simulation_poses(sim, q)
        frames = [name.replace("_joint", "_link") for name in BODY_NAMES]
        frames += ["pelvis", "torso_link", "head_link"]
        frames += [entry["frame"] for entries in model.bounds.values() for entry in entries]
        pairs = []
        for frame in set(frames):
            if frame not in poses:
                continue
            sim_frame = frame
            if frame not in sim.by_child and frame != "pelvis":
                sim_frame = frame.replace("left_", "lh_", 1).replace("right_", "rh_", 1)
            if sim_frame not in sim.by_child and sim_frame != "pelvis":
                raise ValueError(f"unmapped_frame:{frame}")
            pairs.append((poses[frame], simulator_poses[sim_frame]))
        if pairs:
            pairs = np.asarray(pairs)
            actual, expected = pairs[:, 0], pairs[:, 1]
            positions.extend(np.linalg.norm(actual[:, :3, 3] - expected[:, :3, 3], axis=1))
            relative = actual[:, :3, :3].transpose(0, 2, 1) @ expected[:, :3, :3]
            angles.extend(Rotation.from_matrix(relative).magnitude())
    if not positions:
        raise ValueError("alignment_samples_required")
    maximum_position = float(max(positions))
    maximum_angle = float(max(angles))
    limit_error = max(
        abs(model.limits[name][i] - sim.by_name[name][key])
        for name in BODY_NAMES[15:]
        for i, key in enumerate(("lower", "upper"))
    )
    return {
        "passed": maximum_position <= 2e-6 and maximum_angle <= 2e-6 and limit_error <= 2e-6,
        "comparisons": len(positions),
        "position_max_m": maximum_position,
        "rotation_max_rad": maximum_angle,
        "arm_limit_max_rad": limit_error,
        "physical_profile": physical_profile,
        "hand_mount_overlay": mounts,
        "right_thumb_mimic": next(j["mimic"] for j in model.joints if j["name"] == "right_thumb_ip"),
        "scope": "kinematics and arm limits; not contact physics, thermal behavior or hardware calibration",
    }
