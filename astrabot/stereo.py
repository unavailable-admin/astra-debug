"""Cyan upright cube localization from raw stereo RGB, calibration and robot FK.

No scene layout or object pose files. Identity comes from the caller's current
VLM detections. Assumptions: level table, upright 40 mm cubes, cyan top glyphs.
"""

from pathlib import Path

import cv2
import numpy as np

from .geometry import StereoGeometry
from .vision import mask


class StereoTracker:
    def __init__(self, sim=None, cube_size=0.04, geometry=None):
        self.sim = sim
        self.geometry = geometry or StereoGeometry.load("sim")
        self.cube_size = cube_size
        g = self.geometry
        self.R1, R2, self.P1, P2, self.Q, _, _ = cv2.stereoRectify(
            g.K_left, g.D_left, g.K_right, g.D_right, g.image_size_wh, g.R, g.T.reshape(3, 1), alpha=0
        )
        self.maps = [
            cv2.initUndistortRectifyMap(K, D, R, P, g.image_size_wh, cv2.CV_32FC1)
            for K, D, R, P in (
                (g.K_left, g.D_left, self.R1, self.P1),
                (g.K_right, g.D_right, R2, P2),
            )
        ]

    def locate(self, image, identified=None):
        if not identified:
            return {"ok": False, "letters": {}, "reason": "Current VLM identities required"}
        image = Path(image)
        left = cv2.imread(str(image))
        right = cv2.imread(str(image.with_name(image.stem + "_right.jpg")))
        if (
            left is None
            or right is None
            or left.shape != (*self.geometry.image_size_wh[::-1], 3)
            or right.shape != left.shape
        ):
            raise ValueError("Synchronized stereo pair at calibration resolution required")
        T = self.sim.kin.fk(self.sim.full_q(), "head_link") @ self.geometry.T_head_left_cv
        return self.locate_images(left, right, identified, T)

    def locate_images(
        self, left, right, identified, T, *, require_common_height=True, glyph_mask=None, coarse_objects=()
    ):
        """Measure glyph surfaces, optionally allowing 20 mm spread for coarse obstacles."""
        coarse_objects = set(coarse_objects)
        if "A" in coarse_objects or not coarse_objects.issubset(identified):
            raise ValueError("invalid_coarse_obstacle_selection")
        g = self.geometry
        if left.shape != (*g.image_size_wh[::-1], 3) or right.shape != left.shape:
            raise ValueError("Image resolution differs from calibration")
        rect = [cv2.remap(im, *maps, cv2.INTER_LINEAR) for im, maps in zip((left, right), self.maps)]
        gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in rect]

        def matcher(min_disp):
            return cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=128,
                blockSize=5,
                P1=8 * 25,
                P2=32 * 25,
                disp12MaxDiff=1,
                uniquenessRatio=12,
                speckleWindowSize=40,
                speckleRange=1,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )

        disparity = matcher(0).compute(*gray).astype(np.float32) / 16
        reverse = matcher(-128).compute(gray[1], gray[0]).astype(np.float32) / 16
        yy, xx = np.indices(disparity.shape)
        rx = xx - disparity
        sampled = cv2.remap(reverse, rx.astype(np.float32), yy.astype(np.float32), cv2.INTER_LINEAR)
        consistent = (disparity > 1) & (rx >= 0) & (rx < g.image_size_wh[0]) & (np.abs(disparity + sampled) < 0.75)
        with np.errstate(invalid="ignore", divide="ignore"):
            camera = cv2.reprojectImageTo3D(disparity, self.Q) @ self.R1
            world = camera @ T[:3, :3].T + T[:3, 3]
        binary = mask(left, mode=g.mode) if glyph_mask is None else np.asarray(glyph_mask)
        if binary.shape != left.shape[:2] or binary.dtype != np.uint8:
            raise ValueError("invalid_scoped_glyph_mask")
        count, labels, stats, _centers = cv2.connectedComponentsWithStats(binary)
        letters = {}
        errors = {}
        used = set()
        for letter, pixel in identified.items():
            try:
                choices = [i for i in range(1, count) if stats[i, 4] >= 15]
                box_centers = stats[:, :2] + stats[:, 2:4] / 2
                idx = min(choices, key=lambda i: np.linalg.norm(box_centers[i] - pixel))
                if idx in used or np.linalg.norm(box_centers[idx] - pixel) > 10:
                    raise ValueError("No unique cyan component at VLM detection")
                used.add(idx)
                component = (labels == idx).astype(np.uint8) * 255
                rect_mask = cv2.remap(component, *self.maps[0], cv2.INTER_NEAREST) > 0
                good = rect_mask & consistent & np.isfinite(world).all(axis=2)
                points = world[good]
                if len(points) < 10:
                    raise ValueError("Fewer than 10 left/right-consistent glyph pixels")
                z = float(np.median(points[:, 2]))
                coarse = letter in coarse_objects
                keep = np.abs(points[:, 2] - z) < (0.024 if coarse else 0.008)
                matched_yx = np.column_stack(np.where(good))[keep]
                points = points[keep]
                if len(points) < 10 or keep.mean() < 0.7:
                    raise ValueError("Glyph depth is not a consistent horizontal surface")
                z = float(np.median(points[:, 2]))
                spread = float(np.quantile(np.abs(points[:, 2] - z), 0.9))
                if spread > (0.02 if coarse else 0.008):
                    raise ValueError(f"Glyph depth spread exceeds {20 if coarse else 8}mm")
                # Use bounding-box center rather than the ink's asymmetric centroid.
                x, y, w, h, _ = stats[idx]
                u, v = x + w / 2, y + h / 2
                normalized = cv2.undistortPoints(np.array([[[u, v]]], dtype=float), g.K_left, g.D_left)[0, 0]
                ray = T[:3, :3] @ np.r_[normalized, 1.0]
                distance = (z - T[2, 3]) / ray[2]
                if distance <= 0:
                    raise ValueError("Surface intersection behind camera")
                top = T[:3, 3] + distance * ray
                center = top - [0, 0, self.cube_size / 2]
                grasp = top - [0, 0, 0.005]
                sample = matched_yx[np.linspace(0, len(matched_yx) - 1, min(12, len(matched_yx))).astype(int)]
                correspondences = [
                    {
                        "left_rectified_px": [int(px), int(py)],
                        "right_rectified_px": [float(px - disparity[py, px]), int(py)],
                        "lr_error_px": float(abs(disparity[py, px] + sampled[py, px])),
                    }
                    for py, px in sample
                ]
                letters[letter] = {
                    "estimated_xyz": center.tolist(),
                    "top_center_xyz": top.tolist(),
                    "grasp_center_xyz": grasp.tolist(),
                    "pixel": [float(u), float(v)],
                    "valid_depth_pixels": len(points),
                    "depth_p90_spread_m": spread,
                    "source": "stereo_sgbm_lr_checked_glyph_surface",
                    "cube_size_m": self.cube_size,
                    "correspondence_samples": correspondences,
                }
            except ValueError as exc:
                errors[letter] = str(exc)
        if letters:
            tops = np.array([v["top_center_xyz"][2] for v in letters.values()])
            table_z = float(np.median(tops) - self.cube_size)
            if require_common_height and np.ptp(tops) > 0.015:
                return {
                    "ok": False,
                    "letters": {},
                    "errors": errors,
                    "reason": "Upright cube top heights disagree by >15mm",
                }
        else:
            table_z = None
        return {
            "ok": bool(letters),
            "letters": letters,
            "errors": errors,
            "table_z": table_z,
            "table_source": "median measured upright cube tops minus known cube size",
        }
