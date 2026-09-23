"""Measured table volume, including small pelvis-relative tabletop inclination."""

from itertools import product

import numpy as np


def height(config, x, y):
    """Return measured tabletop height, preserving legacy horizontal configs."""
    if config.table_plane is None:
        return config.table_max[2]
    a, b, c = config.table_plane
    return a * x + b * y + c


def vertices(config):
    """Convex volume below the measured plane inside the configured XY footprint."""
    lo, hi = config.table_min, config.table_max
    footprint = config.table_footprint or list(product((lo[0], hi[0]), (lo[1], hi[1])))
    return np.array([[x, y, z] for x, y in footprint for z in (lo[2], height(config, x, y))])


def cube_penetrates(config, center, half=0.02):
    """Check the complete upright cube bottom; only table contact gets 3 mm tolerance."""
    center = np.asarray(center)
    lo, hi = center - half, center + half
    lower, upper = np.asarray(config.table_min), np.asarray(config.table_max)
    if not (np.all(hi[:2] >= lower[:2]) and np.all(lo[:2] <= upper[:2])):
        return False
    if config.table_plane is None:
        return lo[2] < config.table_max[2] - 0.003
    # Cubes rest on the observed table, so their bottom face follows its normal.
    # Treating the cube as pelvis-upright invents penetration on a tilted table.
    a, b, _ = config.table_plane
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    x_axis = np.array([1.0, 0, a])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    bottom = center - half * normal
    corners = [bottom + x * x_axis + y * y_axis for x, y in product((-half, half), repeat=2)]
    return any(point[2] < height(config, point[0], point[1]) - 0.003 for point in corners)
