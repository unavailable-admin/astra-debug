"""Validated physical configuration and explicit hardware layouts."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..pinch import MOTOR_RANGES, raw_hand
from .motion_checks import MotionChecks

ASSETS = Path(__file__).parent / "assets"
ARM_NAMES = tuple(
    f"{side}_{part}_joint"
    for side in ("left", "right")
    for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
)
BODY_NAMES = (
    tuple(
        f"{side}_{part}_joint"
        for side in ("left", "right")
        for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    )
    + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    + ARM_NAMES
)
HAND_PARTS = (
    "pinky_mcp_pitch",
    "ring_mcp_pitch",
    "middle_mcp_pitch",
    "index_mcp_pitch",
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
)
HAND_RANGES = MOTOR_RANGES.copy()
GRIP = np.tile([0.0, 0.0, 0.0, 0.0, 102.0, 18.0], 2)
RAISE = np.array([0.8, 0.3, 0, 0.35, 0, -1.55, 0, 0.8, -0.3, 0, 0.35, 0, -1.55, 0])
# Holding open fingers needs a rearward arc to keep fingertips behind the table
# edge until elevated. This waypoint still requires a full scene/path check.
OPEN_RAISE = np.array([1.0, 0.3, 0, 0.35, 0, -1.55, 0, 1.0, -0.3, 0, 0.35, 0, -1.55, 0])
REST = np.array([0, 0.3, 0, 1.33, 0, 0, 0, 0, -0.3, 0, 1.33, 0, 0, 0])
READY = np.array(
    [
        -0.06736489,
        0.20515722,
        0.08011083,
        -0.57397872,
        0.01750336,
        -0.02328863,
        0.017542,
        -0.06736489,
        -0.20515722,
        -0.08011083,
        -0.57397872,
        -0.01750336,
        -0.02328863,
        -0.017542,
    ]
)


def vector(value, size, name):
    """Require a finite vector without silent broadcasting."""
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return result


@dataclass
class Config:
    """Installation values; unmeasured geometry never enables motion."""

    motion_checks: MotionChecks = field(default_factory=MotionChecks)
    standing_mode: int = 4  # Unitree G1 StandUp FSM; Start(500) is a transition command.
    interface: str = "enP2p1s0"
    left_can: str = "can4"
    right_can: str = "can5"
    camera_device: str = "/dev/video0"
    camera_width: int = 3840
    camera_height: int = 1200
    camera_fps: float = 30.0
    joint_speed: float = 0.3
    cartesian_speed: float = 0.02
    hand_speed: float = 100.0  # canonical raw units / second
    feedback_timeout: float = 0.25
    lease_timeout: float = 2.0
    tracking_error: float = 0.08
    arrival_error: float = 0.03
    hand_arrival_error: float = 5.0
    settle_timeout: float = 5.0
    motor_temperature_limit: float = 120.0
    shell_temperature_limit: float = 90.0
    hand_temperature_limit: float = 90.0
    hand_torque_raw: int = 10
    scene_measured: bool = False
    table_min: list = field(default_factory=lambda: [0.25, -0.6, -0.85])
    table_max: list = field(default_factory=lambda: [1.0, 0.6, -0.12])
    table_plane: list | None = None  # Optional measured z = a*x + b*y + c in pelvis frame.
    table_footprint: list | None = None  # Ordered convex XY corners; bounds remain broad-phase only.
    table_observed: list | None = None  # Convex XY workspace: stereo-visible or explicitly operator-cleared.
    table_uncertainty: float = 0.0
    clearance: float = 0.005
    obstacles: list = field(default_factory=list)
    ready_arms: list = field(default_factory=lambda: READY.tolist())
    ready_hands: list = field(default_factory=lambda: np.tile(raw_hand("open"), 2).tolist())
    tcp_offset: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    tcp_measured: bool = False
    grasp_rotation_xyzw: list = field(default_factory=lambda: [0, 0, 0, 1])
    open_left: list = field(default_factory=lambda: raw_hand("open").tolist())
    closed_left: list = field(default_factory=lambda: raw_hand("closed").tolist())
    target_origin: list = field(default_factory=lambda: [0.35, 0.18, -0.10])
    target_axis: list = field(default_factory=lambda: [0, -1, 0])
    target_spacing: float = 0.09
    calibration: str = ""
    calibration_report: str = ""
    trial_lift_height_m: float = 0.02
    trial_speed_scale: float = 1.0
    trial_grip_mode: str = "tactile"
    trial_require_grip_confirmation: bool = True
    trial_grip_gap_m: float = 0.038
    trial_thumb_yaw_offset_raw: float = 0.0
    trial_grasp_offset: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    tactile_threshold: float = 8.0
    output: str = "outputs/robot"

    def trial_hand(self, phase):
        """Trial-only left thumb abduction; other finger presets stay unchanged."""
        hand = raw_hand(phase)
        hand[5] += self.trial_thumb_yaw_offset_raw
        return hand

    def pinch_reference_hands(self, hands):
        """Keep the original grasp reference when independently yawing the thumb."""
        result = np.asarray(hands, float).copy()
        if self.trial_thumb_yaw_offset_raw:
            result[5] = raw_hand("open")[5]
        return result

    def __post_init__(self):
        if isinstance(self.motion_checks, dict):
            self.motion_checks = MotionChecks(**self.motion_checks)
        if not isinstance(self.motion_checks, MotionChecks):
            raise TypeError("invalid motion_checks")
        for name in (
            "joint_speed",
            "cartesian_speed",
            "hand_speed",
            "feedback_timeout",
            "lease_timeout",
            "tracking_error",
            "arrival_error",
            "hand_arrival_error",
            "settle_timeout",
            "motor_temperature_limit",
            "shell_temperature_limit",
            "hand_temperature_limit",
            "camera_fps",
            "clearance",
            "target_spacing",
            "tactile_threshold",
        ):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (float, int)) or not np.isfinite(val) or val <= 0:
                raise ValueError(f"invalid {name}")
        if (
            isinstance(self.trial_speed_scale, bool)
            or not np.isfinite(self.trial_speed_scale)
            or not 1 <= self.trial_speed_scale <= 4
        ):
            raise ValueError("trial_speed_scale_outside_1_to_4")
        if (
            isinstance(self.trial_lift_height_m, bool)
            or not np.isfinite(self.trial_lift_height_m)
            or not 0 < self.trial_lift_height_m <= 0.12
        ):
            raise ValueError("trial_lift_height_outside_0_to_120mm")
        if self.joint_speed > 0.45 or self.cartesian_speed > 0.03:
            raise ValueError("commissioning speed exceeds 0.45 rad/s or 0.03 m/s")
        for name in ("scene_measured", "tcp_measured", "trial_require_grip_confirmation"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("table_min", "table_max", "tcp_offset", "target_origin", "target_axis"):
            vector(getattr(self, name), 3, name)
        if np.any(np.asarray(self.table_min) >= self.table_max):
            raise ValueError("unordered table bounds")
        if self.table_footprint is not None:
            points = np.asarray(self.table_footprint, float)
            if points.shape != (4, 2) or not np.isfinite(points).all() or self.table_plane is None:
                raise ValueError("invalid_table_footprint")
            edges = np.roll(points, -1, axis=0) - points
            following = np.roll(edges, -1, axis=0)
            turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
            if not (np.all(turns > 1e-6) or np.all(turns < -1e-6)):
                raise ValueError("table_footprint_must_be_ordered_convex")
            if np.any(points < np.asarray(self.table_min)[:2] - 1e-8) or np.any(
                points > np.asarray(self.table_max)[:2] + 1e-8
            ):
                raise ValueError("table_footprint_outside_bounds")
        if not np.isfinite(self.table_uncertainty) or not 0 <= self.table_uncertainty <= 0.01:
            raise ValueError("invalid_table_uncertainty")
        if self.table_observed is not None:
            from .scene_coverage import halfplanes

            points = np.asarray(self.table_observed, float)
            if points.ndim != 2 or points.shape[1:] != (2,) or not 3 <= len(points) <= 64:
                raise ValueError("invalid_table_observed")
            if self.table_plane is None or not np.isfinite(points).all():
                raise ValueError("invalid_table_observed")
            if np.any(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1) < 1e-6):
                raise ValueError("invalid_table_observed_edges")
            normals, offsets = halfplanes(points)
            if not np.isfinite(normals).all() or np.any(points @ normals.T + offsets < -1e-8):
                raise ValueError("table_observed_must_be_convex")
            if abs(np.linalg.det(np.array([points[1] - points[0], points[2] - points[0]]))) < 1e-8:
                raise ValueError("table_observed_degenerate")
        if self.table_plane is not None:
            plane = vector(self.table_plane, 3, "table_plane")
            if np.linalg.norm(plane[:2]) > 0.2:
                raise ValueError("table_tilt_exceeds_commissioning_range")
            heights = [
                plane @ [x, y, 1]
                for x in (self.table_min[0], self.table_max[0])
                for y in (self.table_min[1], self.table_max[1])
            ]
            if min(heights) <= self.table_min[2]:
                raise ValueError("table_plane_below_base")
        vector(self.ready_arms, 14, "ready_arms")
        ready_hands = vector(self.ready_hands, 12, "ready_hands")
        if np.any((ready_hands < 0) | (ready_hands > 255)):
            raise ValueError("invalid ready_hands")
        for name in ("open_left", "closed_left"):
            value = vector(getattr(self, name), 6, name)
            if np.any((value < 0) | (value > 255)):
                raise ValueError(f"invalid {name}")
        quat = vector(self.grasp_rotation_xyzw, 4, "grasp_rotation_xyzw")
        if abs(np.linalg.norm(quat) - 1) > 1e-6:
            raise ValueError("grasp rotation must be a unit xyzw quaternion")
        if abs(np.linalg.norm(self.target_axis) - 1) > 1e-6 or abs(self.target_axis[2]) > 1e-6:
            raise ValueError("target_axis must be a horizontal unit vector")
        if self.camera_width < 128 or self.camera_width % 2 or self.camera_height < 64:
            raise ValueError("invalid stereo image dimensions")
        if self.trial_grip_mode not in ("tactile", "supervised_position"):
            raise ValueError("invalid_trial_grip_mode")
        if not np.isfinite(self.trial_grip_gap_m) or not 0.036 <= self.trial_grip_gap_m <= 0.04:
            raise ValueError("trial_grip_gap_requires_36_to_40mm")
        if not np.isfinite(self.trial_thumb_yaw_offset_raw) or abs(self.trial_thumb_yaw_offset_raw) > 30:
            raise ValueError("trial_thumb_yaw_offset_exceeds_30raw")
        offset = vector(self.trial_grasp_offset, 3, "trial_grasp_offset")
        if np.linalg.norm(offset[:2]) > 0.01 + 1e-12 or abs(offset[2]) > 1e-12:
            raise ValueError("trial_grasp_offset_requires_horizontal_at_most_10mm")
        if self.left_can == self.right_can or not 1 <= self.hand_torque_raw <= 20:
            raise ValueError("invalid CAN mapping or commissioning hand torque")
        for bounds in self.obstacles:
            a = np.asarray(bounds, float)
            if a.shape != (2, 3) or not np.isfinite(a).all() or np.any(a[0] >= a[1]):
                raise ValueError("invalid obstacle box")

    @classmethod
    def load(cls, path):
        """Load JSON; resolve data paths against the configuration file."""
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        for key in ("calibration", "calibration_report", "output"):
            if data.get(key):
                p = Path(data[key])
                data[key] = str(p if p.is_absolute() else path.parent / p)
        return cls(**data)


def file_digest(path):
    """Fingerprint model/calibration inputs for audit and validation binding."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
