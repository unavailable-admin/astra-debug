"""Fit an enclosing rectangle from the straight portions of a rounded table.

Inputs are calibrated ray intersections with the measured tabletop plane, in
metres. The reference corner is a line intersection, not a point on the arc.
"""

import numpy as np


def _line(points: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    if points.ndim != 2 or points.shape[1] != 3 or not 3 <= len(points) <= 8 or not np.isfinite(points).all():
        raise ValueError(f"table_edge_invalid_samples:{name}")
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    direction = axes[0]
    if (points[-1] - points[0]) @ direction < 0:
        direction = -direction
    distances = (points - center) @ direction
    span = float(np.ptp(distances))
    if span < 0.15:
        raise ValueError(f"table_edge_span_below_150mm:{name}:{span * 1000:.1f}")
    if np.any(np.diff(distances) < 0.02):
        raise ValueError(f"table_edge_samples_unordered_or_too_close:{name}")
    residual = float(np.linalg.norm(points - center - distances[:, None] * direction, axis=1).max())
    if residual > 0.01:
        raise ValueError(f"table_edge_line_residual_exceeds_10mm:{name}:{residual * 1000:.1f}")
    return center, direction, span, residual


def fit_table_edges(near: np.ndarray, left: np.ndarray, plane: np.ndarray, width: float, depth: float) -> tuple:
    """Return rectangle corners and diagnostics after adjacent-edge validation.

    Require ordered samples, long straight segments, near-perpendicular lines,
    bounded corner extrapolation, and agreement with the measured dimensions.
    Each boundary gets a baseline 10 mm plus its residual and extrapolation growth;
    it is a geometric allowance, not a calibrated confidence interval.
    """
    near, left, plane = np.asarray(near, float), np.asarray(left, float), np.asarray(plane, float)
    if plane.shape != (3,) or not np.isfinite(plane).all():
        raise ValueError("table_edge_invalid_plane")
    if not np.isfinite([width, depth]).all() or width <= 0 or depth <= 0:
        raise ValueError("table_edge_invalid_dimensions")
    near_center, near_direction, near_span, near_residual = _line(near, "near")
    left_center, left_direction, left_span, left_residual = _line(left, "left")
    normal = np.r_[-plane[:2], 1.0]
    normal_size = np.linalg.norm(normal)
    normal /= normal_size
    if np.max(np.abs(np.r_[near @ normal, left @ normal] - plane[2] / normal_size)) > 1e-5:
        raise ValueError("table_edge_samples_not_on_measured_plane")
    angle_error = float(np.degrees(np.arcsin(np.clip(abs(near_direction @ left_direction), 0, 1))))
    if angle_error > 5:
        raise ValueError(f"table_edges_not_perpendicular:deviation_deg={angle_error:.2f}")

    # Average the two orientation estimates, then fit the offsets of an exactly
    # orthogonal rectangle. Perspective angles are never tested in image space.
    across_from_left = np.cross(left_direction, normal)
    if across_from_left @ near_direction < 0:
        across_from_left = -across_from_left
    across = near_direction + across_from_left
    across /= np.linalg.norm(across)
    inward = np.cross(normal, across)
    if inward @ left_direction < 0:
        inward = -inward
    origin = across * (left_center @ across) + inward * (near_center @ inward) + normal * (plane[2] / normal_size)
    near_error = float(np.max(np.abs((near - origin) @ inward)))
    left_error = float(np.max(np.abs((left - origin) @ across)))
    rectangle_error = max(near_error, left_error)
    # Rectangle residual is a boundary uncertainty, not a height error.
    # Carry it into the expanded table volume checked along every path rather
    # than rejecting central, elevated motions at scene construction time.

    extrapolation = {}
    padding_by_edge = {}
    for name, points, axis, length in (("near", near, across, width), ("left", left, inward, depth)):
        offsets = (points - origin) @ axis
        if offsets.min() < -0.01 or offsets.max() > length + 0.01:
            raise ValueError(f"table_edge_samples_outside_measured_extent:{name}")
        gap = max(0.0, float(offsets.min()))
        if gap > min(0.3, length / 2):
            raise ValueError(f"table_corner_extrapolation_too_far:{name}:{gap * 1000:.1f}mm")
        extrapolation[name] = gap
        # Bound this line over its own dimension, including the unsampled ends.
        # A distant side-edge sample does not add its angular growth to the
        # measured near edge. One global pad needlessly erases low-entry space.
        error, span = (near_error, near_span) if name == "near" else (left_error, left_span)
        end_gap = max(gap, length - float(offsets.max()), 0)
        padding_by_edge[name] = 0.01 + max(0.002, error) * (1 + 2 * end_gap / span)

    # Extrapolating short/noisy segments can magnify angular error at far corners.
    # Refuse excessive allowance rather than silently trusting that rectangle.
    padding = max(padding_by_edge.values())
    if padding > 0.05:
        raise ValueError(f"table_edge_padding_exceeds_50mm:{padding * 1000:.1f}")
    corners = np.array(
        [origin, origin + width * across, origin + width * across + depth * inward, origin + depth * inward]
    )
    side_pad, near_pad = padding_by_edge["left"], padding_by_edge["near"]
    padded = corners + np.array(
        [
            -side_pad * across - near_pad * inward,
            side_pad * across - near_pad * inward,
            side_pad * across + near_pad * inward,
            -side_pad * across + near_pad * inward,
        ]
    )
    report = {
        "mode": "adjacent_straight_edges",
        "reference_corner_source": "intersection_of_fitted_straight_edges_not_physical_arc",
        "reference_corner_m": origin.tolist(),
        "near_samples_m": near.tolist(),
        "left_samples_m": left.tolist(),
        "span_mm": {"near": near_span * 1000, "left": left_span * 1000},
        "line_residual_max_mm": {"near": near_residual * 1000, "left": left_residual * 1000},
        "perpendicular_deviation_deg": angle_error,
        "rectangle_residual_max_mm": rectangle_error * 1000,
        "rectangle_residual_above_10mm": bool(rectangle_error > 0.01),
        "boundary_acceptance": "bounded_padding_then_full_path_collision_check",
        "corner_extrapolation_mm": {name: value * 1000 for name, value in extrapolation.items()},
        "footprint_padding_m": padding,
        "boundary_padding_m": {"near_far": near_pad, "left_right": side_pad},
        "padded_corners_m": padded.tolist(),
        "padding_source": "per_boundary_residual_and_unsampled_end_growth_not_statistical_confidence",
        "rounded_corners_enclosed": True,
    }
    return corners, report


def object_extent_report(letters, corners, width, depth, *, target_only=False):
    """Check required objects while retaining non-target measurement diagnostics."""
    corners = np.asarray(corners, float)
    across, inward = (corners[1] - corners[0]) / width, (corners[3] - corners[0]) / depth
    rows = []
    for name, record in letters.items():
        center = np.asarray(record["estimated_xyz"], float) - corners[0]
        u, v = float(center @ across), float(center @ inward)
        inside = bool(np.isfinite([u, v]).all() and 0.02 <= u <= width - 0.02 and 0.02 <= v <= depth - 0.02)
        rows.append(
            {
                "object": name,
                "u_mm": u * 1000,
                "v_mm": v * 1000,
                "inside": inside,
                "required": not target_only or name == "A",
            }
        )
    required = [row for row in rows if row["required"]]
    return {
        "scope": "target_A_only" if target_only else "all_modeled_objects",
        "samples": rows,
        "passed": bool(required) and "A" in letters and all(row["inside"] for row in required),
        "failed_objects": [row["object"] for row in required if not row["inside"]],
        "diagnostic_only_outside": [row["object"] for row in rows if not row["required"] and not row["inside"]],
    }
