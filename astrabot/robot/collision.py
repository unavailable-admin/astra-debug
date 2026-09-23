"""Oriented-box and capsule collision tests with explicit clearance margins."""

from itertools import combinations, pairwise

import numpy as np

_FAST = None


def native_backend():
    """Return an already warmed task-side backend, or None for reference math."""
    return _FAST


def enable_acceleration():
    """Enable optional native kernels in this task process after compilation."""
    global _FAST
    if _FAST is None:
        try:
            from . import collision_fast
        except ImportError:
            return False
        collision_fast.warmup()
        _FAST = collision_fast
    return True


def accelerated_overlaps(lo, hi, other_lo, other_hi, margin):
    """Return None when native task-side acceleration has not been enabled."""
    if _FAST is None:
        return None
    arrays = [np.ascontiguousarray(v, dtype=float).reshape(-1, 3) for v in (lo, hi, other_lo, other_hi)]
    return bool(_FAST.overlaps(*arrays, float(margin)))


def _nearest_simplex(points):
    """Closest point in a small convex simplex, including degenerate faces."""
    best_distance, best_point, best_face = float("inf"), None, None
    for count in range(1, min(4, len(points)) + 1):
        for indices in combinations(range(len(points)), count):
            face = points[list(indices)]
            if count == 1:
                weights = np.ones(1)
            else:
                gram = face @ face.T
                # Preserve the KKT system and solver while avoiding recursive
                # array concatenation for every face of every path sample.
                matrix = np.ones((count + 1, count + 1))
                matrix[:count, :count] = gram
                matrix[count, count] = 0
                rhs = np.zeros(count + 1)
                rhs[count] = 1
                result = np.linalg.lstsq(matrix, rhs, rcond=1e-12)[0]
                weights = result[:count]
                if np.any(weights < -1e-9) or abs(weights.sum() - 1) > 1e-7:
                    continue
                weights = np.maximum(weights, 0)
                weights /= weights.sum()
            point = weights @ face
            distance = point @ point
            if distance < best_distance:
                best_distance, best_point = distance, point
                best_face = face[weights > 1e-10]
    return best_point, best_face


def convex_hulls_within_distance(first, second, margin):
    """Conservative GJK distance test with a separating-plane certificate.

    False requires a support-plane gap strictly larger than the requested
    margin. A simplex point inside the margin proves intersection; numerical
    nonconvergence conservatively returns True. Vertices may define a point,
    segment, box, or mesh convex hull.
    """
    first, second = np.asarray(first, float), np.asarray(second, float)
    if (
        first.ndim != 2
        or second.ndim != 2
        or first.shape[1:] != (3,)
        or second.shape[1:] != (3,)
        or not len(first)
        or not len(second)
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or not np.isfinite(margin)
        or margin < 0
    ):
        raise ValueError("invalid_convex_geometry")

    if _FAST is not None:
        return bool(_FAST.convex_distance(np.ascontiguousarray(first), np.ascontiguousarray(second), float(margin)))

    def support(direction):
        return first[np.argmax(first @ direction)] - second[np.argmin(second @ direction)]

    direction = first.mean(0) - second.mean(0)
    if np.linalg.norm(direction) < 1e-12:
        direction = np.array([1.0, 0, 0])
    simplex = np.array([support(-direction)])
    point = simplex[0]
    for _ in range(64):
        distance = np.linalg.norm(point)
        if distance <= margin + 1e-7:
            return True
        vertex = support(-point)
        if (point @ vertex) / distance > margin + 1e-7:
            return False
        if np.any(np.linalg.norm(simplex - vertex, axis=1) < 1e-10):
            return True
        point, simplex = _nearest_simplex(np.vstack([simplex, vertex]))
    return True


def convex_hulls_violate_clearance(first, second, clearance):
    """Check signed clearance; negative values bound translational penetration.

    For negative clearance, a separating-axis certificate must demonstrate
    that translation by at most the allowed depth separates the swept hulls.
    Hull construction failure conservatively reports a collision.
    """
    if not np.isfinite(clearance):
        raise ValueError("invalid_convex_clearance")
    if clearance >= 0:
        return convex_hulls_within_distance(first, second, clearance)
    if not convex_hulls_within_distance(first, second, 0):
        return False
    from scipy.spatial import ConvexHull, QhullError

    first, second = np.asarray(first, float), np.asarray(second, float)
    try:
        hulls = [ConvexHull(points) for points in (first, second)]
    except QhullError:
        return True

    def separates(axes):
        lengths = np.linalg.norm(axes, axis=1)
        axes = axes[lengths > 1e-12] / lengths[lengths > 1e-12, None]
        if not len(axes):
            return False
        a, b = first @ axes.T, second @ axes.T
        # Both directions matter, including one hull contained in the other.
        depth = np.minimum(a.max(0) - b.min(0), b.max(0) - a.min(0))
        return bool(np.any(depth <= -clearance - 1e-9))

    if separates(np.vstack([h.equations[:, :3] for h in hulls])):
        return False
    edges = []
    for points, hull in zip((first, second), hulls):
        pairs = {tuple(sorted(pair)) for face in hull.simplices for pair in combinations(face, 2)}
        edges.append(np.array([points[b] - points[a] for a, b in pairs]))
    for edge in edges[0]:
        if separates(np.cross(edge, edges[1])):
            return False
    return True


def oriented_box_hits_box(local_lower, local_upper, pose, lower, upper, margin=0.0):
    """Test an oriented box against a world box, retaining the full world margin.

    Use the separating-axis theorem after expanding the world box by ``margin``
    on each axis. Contact counts as intersection. Unlike a world AABB of the
    rotated object, this test does not fill the empty corners around it.
    """
    local_lower, local_upper, pose, lower, upper = (
        np.asarray(value, dtype=float) for value in (local_lower, local_upper, pose, lower, upper)
    )
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("invalid_box_margin")
    if any(value.shape != (3,) for value in (local_lower, local_upper, lower, upper)) or pose.shape != (4, 4):
        raise ValueError("invalid_box_shape")
    if not all(np.isfinite(value).all() for value in (local_lower, local_upper, pose, lower, upper)):
        raise ValueError("invalid_box_geometry")
    if np.any(local_lower > local_upper) or np.any(lower > upper):
        raise ValueError("invalid_box_bounds")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0):
        raise ValueError("invalid_box_rotation")
    center = rotation @ ((local_lower + local_upper) / 2) + pose[:3, 3]
    extent = (local_upper - local_lower) / 2
    other_center = (lower + upper) / 2
    other_extent = (upper - lower) / 2 + margin
    world_axes = np.eye(3)
    axes = [*world_axes, *rotation.T]
    axes.extend(np.cross(first, second) for first in world_axes for second in rotation.T)
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-10:
            continue
        axis = axis / length
        radius = other_extent @ np.abs(axis) + extent @ np.abs(rotation.T @ axis)
        if abs((center - other_center) @ axis) > radius + 1e-12:
            return False
    return True


def segment_hits_box(start, end, lower, upper, radius: float) -> bool:
    """Conservative capsule/AABB intersection using a radius-expanded box."""
    lower = np.asarray(lower) - radius
    upper = np.asarray(upper) + radius
    delta = end - start
    near, far = 0.0, 1.0
    for axis in range(3):
        if abs(delta[axis]) < 1e-12:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
        else:
            bounds = sorted(((lower[axis] - start[axis]) / delta[axis], (upper[axis] - start[axis]) / delta[axis]))
            near, far = max(near, bounds[0]), min(far, bounds[1])
            if near > far:
                return False
    return True


def segment_distance(a, b, c, d) -> float:
    """Distance between finite segments, including degenerate hand spheres."""
    if _FAST is not None:
        return float(_FAST.segment_distance(*(np.ascontiguousarray(v, dtype=float) for v in (a, b, c, d))))
    # Alternating projection converges poorly for nearly parallel segments; solve
    # the interior optimum and all four endpoint projections explicitly.
    u, v = b - a, d - c
    candidates = []
    for point, first, direction in ((a, c, v), (b, c, v), (c, a, u), (d, a, u)):
        factor = np.clip(np.dot(point - first, direction) / max(np.dot(direction, direction), 1e-15), 0, 1)
        candidates.append(np.linalg.norm(point - first - factor * direction))
    matrix = np.array([[u @ u, -(u @ v)], [-(u @ v), v @ v]])
    if np.linalg.det(matrix) > 1e-15:
        factors = np.linalg.solve(matrix, [-u @ (a - c), v @ (a - c)])
        if np.all((factors >= 0) & (factors <= 1)):
            candidates.append(np.linalg.norm(a + factors[0] * u - c - factors[1] * v))
    return float(min(candidates))


def segment_box_distance(start, end, lower, upper) -> float:
    """Return exact segment/box distance, including degenerate segments.

    Squared distance is quadratic between crossings of the six box planes.
    Minimize each interval analytically rather than expanding box corners.
    """
    start, end, lower, upper = (np.asarray(value, dtype=float) for value in (start, end, lower, upper))
    if _FAST is not None:
        return float(_FAST.segment_box_distance(*(np.ascontiguousarray(v) for v in (start, end, lower, upper))))
    delta = end - start
    crossings = [0.0, 1.0]
    for axis in range(3):
        if abs(delta[axis]) > 1e-12:
            crossings.extend(
                float(t)
                for t in ((lower[axis] - start[axis]) / delta[axis], (upper[axis] - start[axis]) / delta[axis])
                if 0 < t < 1
            )
    crossings = sorted(set(crossings))
    distance = float("inf")
    for left, right in pairwise(crossings):
        midpoint = start + (left + right) * 0.5 * delta
        active = (midpoint < lower) | (midpoint > upper)
        if not np.any(active):
            return 0.0
        boundary = np.where(midpoint < lower, lower, upper)
        a, b = (start - boundary)[active], delta[active]
        t = float(np.clip(-(a @ b) / max(b @ b, 1e-30), left, right))
        distance = min(distance, float(np.linalg.norm(a + t * b)))
    return distance


def capsule_hits_box(start, end, lower, upper, radius: float) -> bool:
    """Test exact capsule/box intersection after conservative broad phase."""
    if _FAST is not None:
        return bool(
            _FAST.capsule_box(
                np.ascontiguousarray(start, dtype=float),
                np.ascontiguousarray(end, dtype=float),
                np.ascontiguousarray(lower, dtype=float),
                np.ascontiguousarray(upper, dtype=float),
                float(radius),
            )
        )
    start, end, lower, upper = (np.asarray(value, dtype=float) for value in (start, end, lower, upper))
    if not segment_hits_box(start, end, lower, upper, radius):
        return False
    return segment_box_distance(start, end, lower, upper) <= radius + 1e-12
