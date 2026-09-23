"""Bind looser tracking of distant cubes to larger planning obstacles."""

import numpy as np


def cube_obstacles(letters):
    """Keep A's 150 mm neighborhood precise; enclose distant cubes' 20 mm drift.

    Distance only selects a tracking tolerance. Every resulting box remains an
    obstacle for preparation, both arms, grasping and return planning.
    """
    target = np.asarray(letters["A"]["estimated_xyz"])
    obstacles = []
    for name, record in letters.items():
        center = np.asarray(record["estimated_xyz"])
        coarse = name != "A" and np.linalg.norm(center[:2] - target[:2]) > 0.15
        record["tracking_tolerance_m"] = 0.02 if coarse else 0.01
        record["depth_spread_limit_m"] = 0.02 if coarse else 0.008
        if name != "A":
            half = np.array([0.04 / np.sqrt(2) + 0.005] * 2 + [0.025])
            half += record.get("position_uncertainty_m", 0)
            if coarse:
                half += 0.02
            obstacles.append([(center - half).tolist(), (center + half).tolist()])
    return obstacles
