"""Bound table work to the scene-authorized region; derive stereo coverage when required."""

from functools import lru_cache
from itertools import product

import cv2
import numpy as np

from ..geometry import undistort_checked


def clip_polygon(points, normal, offset):
    """Clip a convex XY polygon to normal @ XY + offset >= 0."""
    points = np.asarray(points, float).reshape(-1, 2)
    result = []
    for first, second in zip(points, np.roll(points, -1, axis=0)):
        a, b = first @ normal + offset, second @ normal + offset
        if a >= -1e-10:
            result.append(first)
        if (a < 0) != (b < 0):
            result.append(first + a / (a - b) * (second - first))
    unique = []
    for point in result:
        if not unique or np.linalg.norm(point - unique[-1]) > 1e-9:
            unique.append(point)
    if len(unique) > 1 and np.linalg.norm(unique[0] - unique[-1]) <= 1e-9:
        unique.pop()
    return np.asarray(unique).reshape(-1, 2)


def halfplanes(polygon):
    """Return inward unit halfplanes of an ordered convex polygon."""
    return _halfplanes(tuple(tuple(point) for point in polygon))


@lru_cache(maxsize=128)
def _halfplanes(polygon):
    points = np.asarray(polygon, float)
    edges = np.roll(points, -1, axis=0) - points
    area = np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))
    normals = np.column_stack((-edges[:, 1], edges[:, 0])) * np.sign(area)
    if abs(area) < 1e-10 or np.any(np.linalg.norm(normals, axis=1) < 1e-10):
        raise ValueError("degenerate_coverage_polygon")
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals, -np.sum(normals * points, axis=1)


def observed_table(footprint, plane, geometry, transform):
    """Intersect the table with both camera views at the cube base and raised top.

    Retain a 32 pixel image border, check inverse distortion along each edge,
    and reserve 40 mm at visibility boundaries for partially unseen cubes.
    The 0–80 mm height band covers 40 mm cubes, the 20 mm trial lift and error.
    Anything above unobserved table columns remains forbidden at every height.
    """
    region = np.asarray(footprint, float)
    width, height = geometry.image_size_wh
    for right in (False, True):
        intrinsic, distortion = (geometry.K_right, geometry.D_right) if right else (geometry.K_left, geometry.D_left)
        limits = np.array(
            [
                (32 - intrinsic[0, 2]) / intrinsic[0, 0],
                (width - 33 - intrinsic[0, 2]) / intrinsic[0, 0],
                (32 - intrinsic[1, 2]) / intrinsic[1, 1],
                (height - 33 - intrinsic[1, 2]) / intrinsic[1, 1],
            ]
        )
        # Build an inscribed undistorted rectangle from forward rays. The image
        # corners can lie outside the reliable inverse-distortion domain.
        for scale in np.linspace(1.0, 0.2, 81):
            xmin, xmax, ymin, ymax = limits * scale
            xs, ys = np.linspace(xmin, xmax, 65), np.linspace(ymin, ymax, 65)
            rays = np.array(list(product(xs, ys)))
            pixels = cv2.projectPoints(
                np.c_[rays, np.ones(len(rays))], np.zeros(3), np.zeros(3), intrinsic, distortion
            )[0].reshape(-1, 2)
            if np.any(pixels < 32) or np.any(pixels > [width - 33, height - 33]):
                continue
            try:
                inverse = undistort_checked(pixels, intrinsic, distortion)
            except ValueError:
                continue
            if np.max(np.abs(inverse - rays)) < 0.0002:
                break
        else:
            raise ValueError("no_reliable_table_view")
        rotation = transform[:3, :3].T
        translation = -rotation @ transform[:3, 3]
        if right:
            rotation, translation = geometry.R @ rotation, geometry.R @ translation + geometry.T
        for elevation in (0.0, 0.08):
            xy_map = rotation @ np.array([[1, 0], [0, 1], plane[:2]])
            origin = rotation @ [0, 0, plane[2] + elevation] + translation
            for direction in ([1, 0, -xmin], [-1, 0, xmax], [0, 1, -ymin], [0, -1, ymax], [0, 0, 1]):
                normal = np.asarray(direction) @ xy_map
                offset = np.asarray(direction) @ origin - 0.04 * np.linalg.norm(normal)
                region = clip_polygon(region, normal, offset)
                if len(region) < 3:
                    raise ValueError("no_common_table_view")
    return region.tolist()


def check_observed_box(config, lower, upper, margin=0.0):
    """Check swept bounds against stereo coverage or the operator-cleared footprint."""
    if config.table_observed is None:
        return
    footprint = config.table_footprint or [
        [config.table_min[0], config.table_min[1]],
        [config.table_max[0], config.table_min[1]],
        [config.table_max[0], config.table_max[1]],
        [config.table_min[0], config.table_max[1]],
    ]
    if _workspace_covers_table(tuple(map(tuple, config.table_observed)), tuple(map(tuple, footprint))):
        # The original check clips the swept box to this table footprint.
        # If the entire footprint is authorized, every possible intersection
        # is authorized too. Physical table collision is checked separately.
        return
    lower, upper = np.asarray(lower) - margin, np.asarray(upper) + margin
    if np.any(upper[:2] < np.asarray(config.table_min)[:2]) or np.any(lower[:2] > np.asarray(config.table_max)[:2]):
        return
    xy = np.array([[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]])
    view_normals, view_offsets = halfplanes(config.table_observed)
    if np.all(xy @ view_normals.T + view_offsets >= -1e-8):
        return
    plane = np.asarray(config.table_plane)
    if upper[2] < min(plane @ [x, y, 1] for x, y in product((lower[0], upper[0]), (lower[1], upper[1]))):
        return
    normals, offsets = halfplanes(footprint)
    for normal, offset in zip(normals, offsets):
        xy = clip_polygon(xy, normal, offset)
        if len(xy) == 0:
            return
    if np.any(xy @ view_normals.T + view_offsets < -1e-8):
        raise ValueError("motion_enters_unobserved_table_region")


@lru_cache(maxsize=64)
def _workspace_covers_table(workspace, footprint):
    normals, offsets = halfplanes(workspace)
    return bool(np.all(np.asarray(footprint) @ normals.T + offsets >= -1e-8))
