"""Shared simulation/real O6 pinch geometry and source motion sequence."""

import numpy as np
from scipy.spatial.transform import Rotation

NAMES = ["lh_index_mcp_pitch", "lh_index_dip", "lh_thumb_cmc_yaw", "lh_thumb_cmc_pitch", "lh_thumb_ip"]
OPEN = np.array([0.5, 0.5 * 0.89, 1.0, 0.2, 0.2 * 2.29])
CLOSED = np.array([0.72, 0.72 * 0.89, 1.0, 0.32, 0.32 * 2.29])
REST_FINGER_ANGLE = 1.3
MOTOR_RANGES = np.array([1.6, 1.6, 1.6, 1.6, 0.58, 1.36])
INDEX_TIP = np.array([0.006, 0, 0.03, 1])
THUMB_TIP = np.array([-0.005, 0, 0.04, 1])


def raw_hand(phase):
    """Convert the shared angles to pinky/ring/middle/index/thumb-pitch/yaw raw."""
    if phase not in ("open", "closed"):
        raise ValueError("unknown_pinch_phase")
    preset = OPEN if phase == "open" else CLOSED
    angles = np.array([REST_FINGER_ANGLE] * 3 + [preset[0], preset[3], preset[2]])
    return 255 * (1 - angles / MOTOR_RANGES)


def pinch_rotation(index, thumb, yaw_degrees=0):
    """Orient the wrist-local pinch axis horizontally in the planning frame."""
    d = np.asarray(index, float) - thumb
    if np.linalg.norm(d) < 1e-6:
        raise ValueError("degenerate_pinch_axis")
    d /= np.linalg.norm(d)
    z = np.array([0.0, 0, 1])
    z -= d * (z @ d)
    if np.linalg.norm(z) < 1e-6:
        raise ValueError("vertical_local_pinch_axis")
    z /= np.linalg.norm(z)
    base = np.diag([1.0, -1, -1]) @ np.column_stack([d, z, np.cross(d, z)]).T
    return Rotation.from_euler("z", yaw_degrees, degrees=True).as_matrix() @ base


def source_waypoints(grasp, transit_yaw, clearance_height, hover_height=0.148, contact_yaw=90):
    """Shared approach, clearance-height turn, descent and pinch phases.

    Yaw is expressed in the caller's explicit planning frame, not implicitly
    copied between simulator world and real pelvis coordinates.
    """
    grasp = np.asarray(grasp, float)
    if grasp.shape != (3,) or not np.isfinite(grasp).all() or not 0 < clearance_height <= hover_height:
        raise ValueError("invalid_source_waypoints")
    hover = grasp + [0, 0, hover_height]
    clearance = grasp + [0, 0, clearance_height]
    count = max(1, int(np.ceil(abs(contact_yaw - transit_yaw) / 10)))
    turns = np.linspace(transit_yaw, contact_yaw, count + 1)[1:]
    return [
        ("approach", hover, transit_yaw, OPEN),
        ("clearance", clearance, transit_yaw, OPEN),
        *[("turn", clearance, float(angle), OPEN) for angle in turns],
        ("descend", grasp, contact_yaw, OPEN),
        ("close", grasp, contact_yaw, CLOSED),
    ]
