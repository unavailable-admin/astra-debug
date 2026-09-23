"""Conservative hand envelopes for the existing sampled fixed-grip sweep."""

from copy import copy
from itertools import product

import numpy as np


def envelope_model(model, body, measured, commanded):
    """Enclose every existing grip sample in each measured link's local box.

    The finger samples and arm samples stay unchanged. A successful envelope
    check covers all finger samples at every arm pose. If the larger boxes
    overlap, callers fall back to checking the original individual shapes.
    """
    count = max(
        1, int(np.ceil(np.max(np.abs(commanded - measured)) / 255 * 1.6 / model.config.motion_checks.path_step_rad))
    )
    reference = model.poses(body, measured)
    samples = [model.poses(body, measured + alpha * (commanded - measured)) for alpha in np.linspace(0, 1, count + 1)]
    result = copy(model)
    result.bounds = {}
    for side, entries in model.bounds.items():
        result.bounds[side] = []
        for entry in entries:
            frame = entry["frame"]
            base = reference[frame]
            vertices = []
            for poses in samples:
                pose = poses[frame]
                world = np.asarray(entry["corners"]) @ pose[:3, :3].T + pose[:3, 3]
                vertices.extend((world - base[:3, 3]) @ base[:3, :3])
            vertices = np.array(vertices)
            lo, hi = vertices.min(0) - 1e-12, vertices.max(0) + 1e-12
            result.bounds[side].append({**entry, "corners": np.array(list(product(*zip(lo, hi)))).tolist()})
    return result


def covers_grip(planner, body, segment, measured, commanded, obstacles, cache):
    """Use a conservative certificate when possible, otherwise retain exact checks."""
    key = (measured.tobytes(), commanded.tobytes())
    if key not in cache:
        cache[key] = envelope_model(planner.model, body, measured, commanded)
    try:
        cache[key].check_path(body, segment.arms, segment.target, measured, measured, extra_obstacles=obstacles)
    except ValueError:
        return False
    return True
