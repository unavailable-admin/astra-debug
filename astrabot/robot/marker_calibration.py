"""Offline camera/rigid-hand-marker fitting; no device access or robot commands."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .calibration import rotation


def _observations(records, minimum, reference_frame="pelvis"):
    if len(records) < minimum:
        raise ValueError(f"at_least_{minimum}_marker_poses_required")
    identifiers, transforms, points = [], [], []
    for record in records:
        identifier = str(record["sample_id"])
        if identifier in identifiers:
            raise ValueError("duplicate_marker_pose_id")
        identifiers.append(identifier)
        if record.get("frame") != "left_hand_base_link" or record.get("marker_ids") != ["A", "B"]:
            raise ValueError("rigid_hand_frame_and_consistent_marker_ids_required")
        transform = np.asarray(record[f"hand_to_{reference_frame}"], float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("invalid_hand_to_reference")
        rotation(transform[:3, :3])
        observed = np.asarray(record["camera_points_m"], float)
        if observed.shape != (2, 3) or not np.isfinite(observed).all() or np.any(observed[:, 2] <= 0):
            raise ValueError("two_finite_positive_depth_marker_points_required")
        transforms.append(transform)
        points.append(observed)
    return identifiers, np.array(transforms), np.array(points)


def fit_rigid_hand_markers(training, validation, measured_separation_m, *, reference_frame="pelvis"):
    """Fit camera-to-reference and two unknown marker locations on the rigid hand.

    Inputs contain paired A/B stereo points and synchronized, stationary FK poses.
    Marker coordinates are learned from training only. The ruler measurement is
    reserved for checking scale, not used to force the fitted marker separation.
    Validation errors never modify the fit, and a report never enables motion.
    Use torso_link for a torso-mounted camera when manually positioning the waist;
    each input must then contain hand_to_torso_link from its measured body pose.
    """
    if reference_frame not in ("pelvis", "torso_link"):
        raise ValueError("unsupported_camera_reference_frame")
    if not np.isfinite(measured_separation_m) or not 0.02 <= measured_separation_m <= 0.08:
        raise ValueError("measured_marker_separation_must_be_20_to_80_mm")
    train_ids, hand, camera = _observations(training, 8, reference_frame)
    valid_ids, valid_hand, valid_camera = _observations(validation, 3, reference_frame)
    if set(train_ids) & set(valid_ids):
        raise ValueError("training_and_validation_pose_ids_overlap")
    # Distinct labels alone do not make repeated/adjacent poses independent.
    for pose in valid_hand:
        relative = np.swapaxes(hand[:, :3, :3], 1, 2) @ pose[:3, :3]
        angle = Rotation.from_matrix(relative).magnitude()
        displacement = np.linalg.norm(hand[:, :3, 3] - pose[:3, 3], axis=1)
        if np.any((angle < 0.1) & (displacement < 0.02)):
            raise ValueError("validation_pose_too_close_to_training")
    for poses in (hand, valid_hand):
        for i, pose in enumerate(poses):
            for previous in poses[:i]:
                angle = Rotation.from_matrix(previous[:3, :3].T @ pose[:3, :3]).magnitude()
                if angle < 0.03 and np.linalg.norm(previous[:3, 3] - pose[:3, 3]) < 0.005:
                    raise ValueError("duplicate_or_nearly_identical_marker_pose")

    # Initialize rotation by aligning marker trajectories to hand-origin motion.
    source = camera.mean(axis=1)
    target = hand[:, :3, 3]
    u, _, vt = np.linalg.svd((source - source.mean(axis=0)).T @ (target - target.mean(axis=0)))
    camera_rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    # For a fixed camera rotation, camera translation and both marker offsets
    # are linear unknowns. Constant hand orientation makes this system singular.
    matrix, rhs = [], []
    for pose, observed in zip(hand, camera):
        for marker in range(2):
            row = np.zeros((3, 9))
            row[:, :3] = np.eye(3)
            row[:, 3 + 3 * marker : 6 + 3 * marker] = -pose[:3, :3]
            matrix.append(row)
            rhs.append(pose[:3, 3] - camera_rotation @ observed[marker])
    matrix = np.vstack(matrix)
    if np.linalg.matrix_rank(matrix, tol=1e-6) != 9:
        raise ValueError("marker_pose_rotation_diversity_is_degenerate")
    linear = np.linalg.lstsq(matrix, np.concatenate(rhs), rcond=None)[0]
    initial = np.r_[Rotation.from_matrix(camera_rotation).as_rotvec(), linear]

    def residual(parameters):
        camera_rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        marker = parameters[6:].reshape(2, 3)
        expected = np.einsum("nij,mj->nmi", hand[:, :3, :3], marker) + hand[:, None, :3, 3]
        observed = camera @ camera_rotation.T + parameters[3:6]
        return (observed - expected).ravel()

    result = least_squares(residual, initial, loss="soft_l1", f_scale=0.002, x_scale="jac", max_nfev=500)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(result.x[:3]).as_matrix()
    transform[:3, 3] = result.x[3:6]
    marker = result.x[6:].reshape(2, 3)
    singular = np.linalg.svd(result.jac, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))

    def evaluate(identifiers, transforms, observations):
        reports = []
        for identifier, pose, observed_camera in zip(identifiers, transforms, observations):
            expected = marker @ pose[:3, :3].T + pose[:3, 3]
            observed = observed_camera @ transform[:3, :3].T + transform[:3, 3]
            error = np.linalg.norm(observed - expected, axis=1)
            reports.append(
                {
                    "sample_id": identifier,
                    "marker_errors_m": error.tolist(),
                    "max_error_m": float(error.max()),
                    "separation_error_m": float(
                        abs(np.linalg.norm(observed_camera[0] - observed_camera[1]) - measured_separation_m)
                    ),
                }
            )
        return reports

    training_report = evaluate(train_ids, hand, camera)
    validation_report = evaluate(valid_ids, valid_hand, valid_camera)
    blockers = []
    if not result.success or not np.isfinite(result.x).all():
        blockers.append("rigid_marker_fit_not_converged")
    if not np.isfinite(condition) or condition > 1e5 or singular[-1] < 1e-6:
        blockers.append("rigid_marker_fit_not_observable")
    if np.any(marker < [-0.04, -0.06, -0.02]) or np.any(marker > [0.04, 0.06, 0.14]):
        blockers.append("fitted_markers_outside_rigid_hand")
    if max(item["max_error_m"] for item in validation_report) > 0.008:
        blockers.append("held_out_marker_error")
    if max(item["separation_error_m"] for item in validation_report) > 0.002:
        blockers.append("held_out_marker_scale_error")
    if abs(np.linalg.norm(marker[0] - marker[1]) - measured_separation_m) > 0.002:
        blockers.append("fitted_marker_scale_error")
    return {
        f"camera_to_{reference_frame}": transform.tolist(),
        "reference_frame": reference_frame,
        "frame": "left_hand_base_link",
        "marker_ids": ["A", "B"],
        "local_points": marker.tolist(),
        "training_sample_ids": train_ids,
        "validation_sample_ids": valid_ids,
        "training": training_report,
        "validation": validation_report,
        "jacobian_condition": condition,
        "blockers": blockers,
        "geometric_validation_passed": not blockers,
        "calibrated_for_manipulation": False,
        "remaining": "Verify physical marker identity, image/FK provenance, actual cube, TCP, and final table measurements.",
    }
