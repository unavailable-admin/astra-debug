"""Optional compiled geometry kernels for task-side planning only.

Keep IEEE arithmetic (no fastmath), the same geometric tolerances and the
conservative GJK nonconvergence result. Import/compile before enabling dispatch.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def overlaps(lo, hi, other_lo, other_hi, margin):
    for i in range(len(lo)):
        for j in range(len(other_lo)):
            if np.all(hi[i] + margin >= other_lo[j]) and np.all(lo[i] - margin <= other_hi[j]):
                return True
    return False


@njit(cache=True)
def nearest_simplex(points):
    best_distance = np.inf
    best_point, best_face = points[0].copy(), points[:1].copy()
    for count in range(1, min(4, len(points)) + 1):
        indices = np.arange(count)
        while True:
            face = points[indices]
            valid = True
            weights = np.ones(1)
            if count > 1:
                matrix = np.ones((count + 1, count + 1))
                matrix[:count, :count] = face @ face.T
                matrix[count, count] = 0
                rhs = np.zeros(count + 1)
                rhs[count] = 1
                weights = np.linalg.lstsq(matrix, rhs, rcond=1e-12)[0][:count]
                valid = not np.any(weights < -1e-9) and abs(weights.sum() - 1) <= 1e-7
                if valid:
                    weights = np.maximum(weights, 0)
                    weights /= weights.sum()
            if valid:
                point = weights @ face
                distance = point @ point
                if distance < best_distance:
                    best_distance, best_point = distance, point
                    best_face = face[weights > 1e-10]
            # Lexicographic combinations, matching itertools.combinations.
            index = count - 1
            while index >= 0 and indices[index] == len(points) - count + index:
                index -= 1
            if index < 0:
                break
            indices[index] += 1
            for j in range(index + 1, count):
                indices[j] = indices[j - 1] + 1
    return best_point, best_face


@njit(cache=True)
def support(first, second, direction):
    return first[np.argmax(first @ direction)] - second[np.argmin(second @ direction)]


@njit(cache=True)
def convex_distance(first, second, margin):
    direction = np.sum(first, axis=0) / len(first) - np.sum(second, axis=0) / len(second)
    if np.linalg.norm(direction) < 1e-12:
        direction = np.array([1.0, 0.0, 0.0])
    point = support(first, second, -direction)
    simplex = point.reshape(1, 3).copy()
    for _ in range(64):
        distance = np.linalg.norm(point)
        if distance <= margin + 1e-7:
            return True
        vertex = support(first, second, -point)
        if (point @ vertex) / distance > margin + 1e-7:
            return False
        for row in simplex:
            if np.linalg.norm(row - vertex) < 1e-10:
                return True
        expanded = np.empty((len(simplex) + 1, 3))
        expanded[:-1] = simplex
        expanded[-1] = vertex
        point, simplex = nearest_simplex(expanded)
    return True


@njit(cache=True)
def segment_box_distance(start, end, lower, upper):
    delta = end - start
    crossings = np.empty(8)
    crossings[0], crossings[1] = 0.0, 1.0
    count = 2
    for axis in range(3):
        if abs(delta[axis]) > 1e-12:
            for bound in (lower[axis], upper[axis]):
                t = (bound - start[axis]) / delta[axis]
                if 0 < t < 1:
                    crossings[count] = t
                    count += 1
    crossings = np.sort(crossings[:count])
    distance = np.inf
    for index in range(count - 1):
        left, right = crossings[index], crossings[index + 1]
        if left == right:
            continue
        midpoint = start + (left + right) * 0.5 * delta
        active = (midpoint < lower) | (midpoint > upper)
        if not np.any(active):
            return 0.0
        boundary = np.where(midpoint < lower, lower, upper)
        a, b = (start - boundary)[active], delta[active]
        t = min(right, max(left, -(a @ b) / max(b @ b, 1e-30)))
        distance = min(distance, np.linalg.norm(a + t * b))
    return distance


@njit(cache=True)
def capsule_box(start, end, lower, upper, radius):
    delta = end - start
    near, far = 0.0, 1.0
    for axis in range(3):
        lo, hi = lower[axis] - radius, upper[axis] + radius
        if abs(delta[axis]) < 1e-12:
            if start[axis] < lo or start[axis] > hi:
                return False
        else:
            a, b = (lo - start[axis]) / delta[axis], (hi - start[axis]) / delta[axis]
            near, far = max(near, min(a, b)), min(far, max(a, b))
            if near > far:
                return False
    return segment_box_distance(start, end, lower, upper) <= radius + 1e-12


@njit(cache=True)
def segment_distance(a, b, c, d):
    u, v = b - a, d - c
    best = np.inf
    for point, first, direction in ((a, c, v), (b, c, v), (c, a, u), (d, a, u)):
        factor = min(1.0, max(0.0, np.dot(point - first, direction) / max(np.dot(direction, direction), 1e-15)))
        best = min(best, np.linalg.norm(point - first - factor * direction))
    matrix = np.array([[u @ u, -(u @ v)], [-(u @ v), v @ v]])
    if np.linalg.det(matrix) > 1e-15:
        rhs = np.array([-u @ (a - c), v @ (a - c)])
        factors = np.linalg.solve(matrix, rhs)
        if np.all((factors >= 0) & (factors <= 1)):
            best = min(best, np.linalg.norm(a + factors[0] * u - c - factors[1] * v))
    return best


@njit(cache=True)
def compose_poses(origins, rotations, parents, kinds, motion):
    result = np.empty((len(origins) + 1, 4, 4))
    result[0] = np.eye(4)
    rotation_index = 0
    for index in range(len(origins)):
        local = origins[index].copy()
        if kinds[index] == 1:
            local[:3, :3] = local[:3, :3] @ rotations[rotation_index]
            rotation_index += 1
        elif kinds[index] == 2:
            local[:3, 3] += local[:3, :3] @ motion[index]
        result[index + 1] = result[parents[index]] @ local
    return result


@njit(cache=True)
def capsules_hit_local_boxes(capsules, poses, lower, upper, radius):
    """Check capsule/OBB pairs without Python dispatch per finger link."""
    for a, z in capsules:
        for index in range(len(poses)):
            rotation = poses[index, :3, :3].T
            translation = poses[index, :3, 3]
            if capsule_box(
                rotation @ (a - translation), rotation @ (z - translation), lower[index], upper[index], radius
            ):
                return True
    return False


@njit(cache=True)
def capsule_box_candidates(a, z, boxes, radius):
    """Keep all capsule/AABB hits for unchanged reference narrow-phase checks."""
    hits = np.zeros(len(boxes), dtype=np.bool_)
    for index in range(len(boxes)):
        hits[index] = capsule_box(a, z, boxes[index, 0], boxes[index, 1], radius)
    return hits


def warmup():
    """Compile all dispatched C-contiguous float64 signatures before use."""
    p = np.zeros(3)
    q = np.ones(3)
    box = np.array([[0.0, 0, 0], [1.0, 1, 1]])
    overlaps(box, box, box, box, 0.0)
    convex_distance(box, box, 0.0)
    segment_box_distance(p, q, p, q)
    capsule_box(p, q, p, q, 0.0)
    segment_distance(p, q, p, q)
    capsules_hit_local_boxes(np.array([[p, q]]), np.eye(4)[None].copy(), box[:1], box[1:], 0.04)
    capsule_box_candidates(p, q, box[None].copy(), 0.065)
    compose_poses(
        np.eye(4)[None].copy(),
        np.eye(3)[None].copy(),
        np.zeros(1, dtype=np.int64),
        np.ones(1, dtype=np.int64),
        np.zeros((1, 3)),
    )
