"""Fast fresh-image verification for one previously identified letter cube."""

import asyncio
import json
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from ..pinch import INDEX_TIP, THUMB_TIP
from ..stereo import StereoTracker
from ..vision import known_glyph_mask
from .calibration import load
from .camera import capture
from .config import file_digest


class TrialObserver:
    """Track identified glyphs, with stereo displacement and model occlusion checks.

    The initial scene must already include the non-cube obstacles. No image is
    accepted from a cached capture; unexplained loss of a visible obstacle stops
    the trial. Occlusion by the observed robot is recorded, not called a fresh
    remeasurement of that object.
    """

    def __init__(self, client, model, config, initial_letters, initial_image, output):
        self.client, self.model, self.config = client, model, config
        self.initial = initial_letters
        self.output = Path(output)
        self.image_hash = file_digest(initial_image)
        self.template = self._glyph(cv2.imread(str(initial_image)), initial_letters["A"]["pixel"])
        self.reference = np.asarray(initial_letters["A"]["estimated_xyz"])
        self.index = 0

    @staticmethod
    def _glyph(image, pixel):
        count, labels, stats, centers = cv2.connectedComponentsWithStats(known_glyph_mask(image, [pixel]))
        choices = [i for i in range(1, count) if stats[i, 4] >= 15]
        if not choices:
            raise ValueError("no_glyph_pixels")
        closest = min(choices, key=lambda i: np.linalg.norm(centers[i] - pixel))
        if np.linalg.norm(centers[closest] - pixel) > 10:
            raise ValueError("identified_glyph_not_visible")
        x, y, w, h, _ = stats[closest]
        return (labels[y : y + h, x : x + w] == closest).astype(np.uint8) * 255

    @staticmethod
    def project(point, g, T, right=False):
        camera = T[:3, :3].T @ (np.asarray(point) - T[:3, 3])
        if right:
            camera = g.R @ camera + g.T
        if camera[2] <= 0:
            raise ValueError("point_behind_camera")
        K, D = (g.K_right, g.D_right) if right else (g.K_left, g.D_left)
        return cv2.projectPoints(camera.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)[0][0, 0]

    def occluded(self, pixel, poses, g, T, right=False):
        target = np.atleast_2d(pixel)
        target_lo, target_hi = target.min(0), target.max(0)
        for entries in self.model.bounds.values():
            for entry in entries:
                pose = poses[entry["frame"]]
                corners = np.array(entry["corners"]) @ pose[:3, :3].T + pose[:3, 3]
                projected = np.array([self.project(x, g, T, right=right) for x in corners])
                if np.all(target_hi >= projected.min(0) - 5) and np.all(target_lo <= projected.max(0) + 5):
                    return True
        return False

    @staticmethod
    def reacquire_pixels(image, projected, search_radii=None):
        """Resolve approximate projections to actual unique glyph box centers.

        StereoTracker expects detected box centers, not model projections. A
        Wider pixel search never relaxes the subsequent per-object scene check.
        """
        search_radii = search_radii or {}
        binary = known_glyph_mask(image, projected.values())
        for name, radius in search_radii.items():
            binary |= known_glyph_mask(image, [projected[name]], radius=int(np.ceil(radius)) + 32)
        count, _labels, stats, _centers = cv2.connectedComponentsWithStats(binary)
        boxes = stats[:, :2] + stats[:, 2:4] / 2
        choices = [i for i in range(1, count) if stats[i, 4] >= 15]
        found, used = {}, set()
        for name, pixel in projected.items():
            if not choices:
                raise ValueError(f"projected_glyph_missing:{name}")
            closest = min(choices, key=lambda i: np.linalg.norm(boxes[i] - pixel))
            if closest in used or np.linalg.norm(boxes[closest] - pixel) > search_radii.get(name, 25):
                raise ValueError(f"projected_glyph_ambiguous_or_missing:{name}")
            used.add(closest)
            found[name] = boxes[closest]
        return found, binary

    async def __call__(self, stage):
        self.index += 1
        directory = self.output / f"{self.index:02d}-{stage}"
        left_path, right_path, meta = await asyncio.to_thread(capture, self.config, directory, self.client)
        state = meta["body_after"]
        g, T, _ = load(self.config.calibration, state["body"])
        tracker = StereoTracker(geometry=g)
        image = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        poses = self.model.poses(state["body"], state["hands"])
        pixels = {}
        search_radii = {}
        hidden = []
        hidden_cameras = {}
        for name, record in self.initial.items():
            top = np.asarray(record["top_center_xyz"]).copy()
            if name == "A" and stage == "lifted":
                top[2] += 0.02
            pixel = self.project(top, g, T)
            # A partly covered top cannot be required to yield complete stereo
            # texture. Bound the 40 mm square for arbitrary tabletop yaw.
            half = 0.02 * np.sqrt(2)
            top_corners = [top + [x, y, 0] for x, y in product((-half, half), repeat=2)]
            left_hidden = self.occluded([self.project(x, g, T) for x in top_corners], poses, g, T)
            right_hidden = self.occluded(
                [self.project(x, g, T, right=True) for x in top_corners], poses, g, T, right=True
            )
            if name != "A" and (left_hidden or right_hidden):
                hidden.append(name)
                hidden_cameras[name] = [
                    side for side, blocked in (("left", left_hidden), ("right", right_hidden)) if blocked
                ]
                continue
            pixels[name] = pixel
            if record.get("tracking_tolerance_m") == 0.02:
                shifted = [self.project(top + np.asarray(delta), g, T) for delta in product((-0.02, 0.02), repeat=3)]
                search_radii[name] = max(25, float(np.max(np.linalg.norm(np.asarray(shifted) - pixel, axis=1))) + 2)
        pixels, glyph_mask = self.reacquire_pixels(image, pixels, search_radii)
        target_glyph = self._glyph(image, pixels["A"])
        # In a small projected window A is unique; also reject major shape loss.
        score = cv2.matchShapes(self.template, target_glyph, cv2.CONTOURS_MATCH_I1, 0)
        if not np.isfinite(score) or score > 0.25:
            raise ValueError("A_glyph_changed_or_occluded")
        measured = await asyncio.to_thread(
            tracker.locate_images,
            image,
            right,
            pixels,
            T,
            require_common_height=False,
            glyph_mask=glyph_mask,
            coarse_objects={name for name in pixels if self.initial[name].get("depth_spread_limit_m") == 0.02},
        )
        (directory / "measurements.json").write_text(json.dumps(measured, indent=2))
        if set(measured["letters"]) != set(pixels):
            raise ValueError(f'current_scene_incomplete:{measured.get("errors",{})}')
        displacements = {}
        for name, record in measured["letters"].items():
            if name == "A":
                continue
            tolerance = self.initial[name].get("tracking_tolerance_m", 0.01)
            delta = np.array(record["estimated_xyz"]) - self.initial[name]["estimated_xyz"]
            displacements[name] = {
                "delta_m": delta.tolist(),
                "distance_m": float(np.linalg.norm(delta)),
                "limit_m": tolerance,
            }
        (directory / "scene-displacements.json").write_text(json.dumps(displacements, indent=2))
        for name, displacement in displacements.items():
            if displacement["distance_m"] > displacement["limit_m"]:
                raise ValueError(
                    f"scene_object_moved:{name}:distance_mm={displacement['distance_m'] * 1000:.2f}"
                    f":limit_mm={displacement['limit_m'] * 1000:.2f}"
                )
        overlay = image.copy()
        projected_tips = {}
        for name, frame, tip in (
            ("index", "left_index_distal", INDEX_TIP),
            ("thumb", "left_thumb_distal", THUMB_TIP),
        ):
            pixel = self.project((poses[frame] @ tip)[:3], g, T)
            projected_tips[name] = pixel.tolist()
            point = tuple(np.rint(pixel).astype(int))
            cv2.drawMarker(overlay, point, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(overlay, name, (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        overlay_path = directory / "model-fingertips.jpg"
        if not cv2.imwrite(str(overlay_path), overlay):
            raise OSError("cannot_write_fingertip_overlay")
        return {
            "center": measured["letters"]["A"]["estimated_xyz"],
            "acquired_monotonic": meta["acquired_monotonic"],
            "scene_clear": True,
            "scene_clear_source": self.initial["A"].get("workspace_policy", "vision_checked"),
            "occluded_objects": hidden,
            "occluded_cameras": hidden_cameras,
            "capture": str(directory / "capture.json"),
            "model_fingertip_overlay": str(overlay_path),
            "model_fingertip_pixels": projected_tips,
            "source_identity_image_sha256": self.image_hash,
            "glyph_shape_distance": float(score),
            "coarse_objects": [name for name in pixels if self.initial[name].get("depth_spread_limit_m") == 0.02],
            "scope": (
                "fresh A only; non-target objects are operator-cleared, not visually checked"
                if self.initial["A"].get("workspace_policy") == "operator_cleared"
                else "fresh target and visible known cubes; modeled robot occlusions recorded; non-cube scene assumed stationary"
            ),
        }
