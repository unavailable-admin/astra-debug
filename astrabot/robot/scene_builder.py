"""Build a single-A scene from one recorded stereo pair and identified pixels.

This module has no robot client or motion interface. Pixels come from local
glyph components; API evidence or a manual annotation supplies identity/table edges.
"""

import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from ..stereo import StereoTracker
from ..vision import known_glyph_mask
from .calibration import binding, load
from .capture_state import CAPTURE_BODY_DRIFT_RAD, CAPTURE_HAND_DRIFT_RAW, stationary_capture_state
from .config import Config, file_digest, vector
from .scene_coverage import observed_table
from .table_edges import fit_table_edges, object_extent_report

TARGET_TABLE_HEIGHT_TOLERANCE_M = 0.008


def target_height_check(record, plane):
    """Compare measured A top with table + 40 mm, independently of plane inliers."""
    top = vector(record["top_center_xyz"], 3, "A_top")
    table_z = float(np.asarray(plane) @ [top[0], top[1], 1])
    error = float(top[2] - table_z - 0.04)
    return {
        "top_above_table_mm": float((top[2] - table_z) * 1000),
        "expected_mm": 40.0,
        "error_mm": error * 1000,
        "limit_mm": TARGET_TABLE_HEIGHT_TOLERANCE_M * 1000,
        "passed": abs(error) <= TARGET_TABLE_HEIGHT_TOLERANCE_M + 1e-12,
    }


def discover(image):
    """Enumerate blue glyphs without assuming their identity or table location."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ink = cv2.inRange(hsv, np.array([85, 160, 40]), np.array([115, 255, 255]))
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    boxes = [
        list(map(int, stats[i]))
        for i in range(1, count)
        if stats[i, 4] >= 20 and all(5 <= size <= 90 for size in stats[i, 2:4])
    ]
    boxes.sort(key=lambda b: (b[1], b[0]))
    return [
        {"id": index, "pixel": [x + w / 2, y + h / 2], "bbox": [x, y, w, h], "area": area}
        for index, (x, y, w, h, area) in enumerate(boxes, 1)
    ]


def read_capture(directory, config):
    """Require intact images and stable, healthy robot feedback at acquisition."""
    directory = Path(directory).resolve()
    meta = json.loads((directory / "capture.json").read_text())
    if meta.get("capture_valid") is False:
        raise ValueError(f"capture_rejected:{meta.get('validation_error', 'incomplete_validation')}")
    for name in ("left.jpg", "right.jpg", "stereo.jpg"):
        if meta.get("image_sha256", {}).get(name) != file_digest(directory / name):
            raise ValueError(f"capture_image_changed:{name}")
    before, after = meta.get("body_before"), meta.get("body_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise TypeError("capture_requires_robot_state:recapture_with_--socket")
    if not before.get("session") or before["session"] != after.get("session"):
        raise ValueError("capture_session_changed")
    if type(before.get("generation")) is not int or before["generation"] != after.get("generation"):
        raise ValueError("capture_generation_changed")
    for state in (before, after):
        vector(state.get("body"), 29, "capture_body")
        vector(state.get("hands"), 12, "capture_hands")
        if state.get("health") != "" or state.get("error") != "" or state.get("mode") != config.standing_mode:
            raise ValueError("capture_feedback_unhealthy")
        if not stationary_capture_state(state):
            raise ValueError("stationary_capture_required")
    if np.max(np.abs(np.array(before["body"]) - after["body"])) > CAPTURE_BODY_DRIFT_RAD:
        raise ValueError("capture_body_moved")
    if np.max(np.abs(np.array(before["hands"]) - after["hands"])) > CAPTURE_HAND_DRIFT_RAW:
        raise ValueError("capture_fingers_moved")
    left, right = (cv2.imread(str(directory / name)) for name in ("left.jpg", "right.jpg"))
    expected = (config.camera_height, config.camera_width // 2, 3)
    if left is None or right is None or left.shape != expected or right.shape != expected:
        raise ValueError("capture_image_size_mismatch")
    if (meta.get("width"), meta.get("height"), meta.get("device")) != (
        config.camera_width,
        config.camera_height,
        config.camera_device,
    ):
        raise ValueError("capture_config_mismatch")
    return meta, left, right


def fit_table(letters, *, operator_cleared_workspace=False, diagnostics=None):
    """Fit a dominant plane; return every rejected sample for explicit modeling."""
    if len(letters) < 4:
        raise ValueError("至少需要四块分散摆放的积木来拟合桌面")
    tops = np.array([v["top_center_xyz"] for v in letters.values()], float)
    if tops.shape != (len(letters), 3) or not np.isfinite(tops).all():
        raise ValueError("invalid_cube_top_measurements")
    centered = tops[:, :2] - tops[:, :2].mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[-1] / np.sqrt(len(tops)) < 0.025:
        raise ValueError("积木分布过窄或共线，无法稳定估计桌面；请分散后重拍")
    design = np.c_[tops[:, :2], np.ones(len(tops))]
    heights = tops[:, 2] - 0.04
    triples = list(combinations(range(len(tops)), 3))
    choices = np.linspace(0, len(triples) - 1, min(512, len(triples))).astype(int)
    minimum_consensus = len(tops) // 2 + 1 if operator_cleared_workspace else np.ceil(0.8 * len(tops))
    minimum_consensus = max(4, minimum_consensus)
    best = best_valid = None
    seen = set()
    rejected_unstable = 0
    for index in choices:
        rows = list(triples[index])
        if np.linalg.matrix_rank(design[rows]) < 3:
            continue
        candidate = np.linalg.solve(design[rows], heights[rows])
        errors = np.abs(design @ candidate - heights)
        inliers = errors <= 0.004
        score = (int(inliers.sum()), -float(np.median(errors)))
        if best is None or score > best[0]:
            best = (score, inliers)
        signature = tuple(np.flatnonzero(inliers))
        if inliers.sum() < minimum_consensus or signature in seen:
            continue
        seen.add(signature)
        fitted = np.linalg.lstsq(design[inliers], heights[inliers], rcond=None)[0]
        fitted_errors = heights - design @ fitted
        spread = tops[inliers, :2] - tops[inliers, :2].mean(axis=0)
        spread_m = np.linalg.svd(spread, compute_uv=False)[-1] / np.sqrt(inliers.sum())
        if np.max(np.abs(fitted_errors[inliers])) > 0.005 or spread_m < 0.025 or np.linalg.norm(fitted[:2]) > 0.2:
            rejected_unstable += 1
            continue
        # A near-collinear set can fit its own points almost perfectly while
        # predicting an arbitrary tabletop tilt. Rank only valid refitted
        # consensus sets; never relax the required majority or metric limits.
        valid_score = (int(inliers.sum()), -float(np.median(np.abs(fitted_errors))))
        if best_valid is None or valid_score > best_valid[0]:
            best_valid = (valid_score, inliers)
    # In operator-cleared mode these are redundant plane samples, not an
    # inventory of obstacles whose geometry all needs to be reconstructed.
    if best_valid is not None:
        best = best_valid
    if best is None or best[1].sum() < minimum_consensus:
        raise ValueError("没有足够一致的桌面深度；检查误选标记或重拍")
    inliers = best[1]
    plane = np.linalg.lstsq(design[inliers], heights[inliers], rcond=None)[0]
    residual = tops[:, 2] - 0.04 - design @ plane
    spread = tops[inliers, :2] - tops[inliers, :2].mean(axis=0)
    spread_m = float(np.linalg.svd(spread, compute_uv=False)[-1] / np.sqrt(inliers.sum()))
    if diagnostics is not None:
        diagnostics.update(
            sample_count=len(tops),
            minimum_consensus=int(minimum_consensus),
            inliers=[name for name, keep in zip(letters, inliers) if keep],
            rejected=[name for name, keep in zip(letters, inliers) if not keep],
            narrow_spread_mm=spread_m * 1000,
            minimum_spread_mm=25.0,
            plane=plane.tolist(),
            residual_mm={name: float(error * 1000) for name, error in zip(letters, residual)},
            rejected_unstable_candidates=rejected_unstable,
        )
    if np.max(np.abs(residual[inliers])) > 0.005:
        raise ValueError("桌面拟合内点残差超过 5 mm；检查标记或重拍")
    if spread_m < 0.025:
        raise ValueError(f"可靠深度的空间分布不足，不能外推桌面：{spread_m * 1000:.2f} mm < 25 mm")
    if np.linalg.norm(plane[:2]) > 0.2:
        raise ValueError("table_tilt_exceeds_commissioning_range")
    return plane, residual, {name for name, keep in zip(letters, inliers) if keep}


def on_plane(pixel, plane, geometry, transform):
    ray = cv2.undistortPoints(np.array([[pixel]], float), geometry.K_left, geometry.D_left)[0, 0]
    direction = transform[:3, :3] @ np.r_[ray, 1.0]
    normal = np.r_[-plane[:2], 1.0]
    denominator = normal @ direction
    if abs(denominator) < 1e-6:
        raise ValueError("table_ray_parallel")
    distance = (plane[2] - normal @ transform[:3, 3]) / denominator
    if not 0 < distance < 4:
        raise ValueError("table_edge_ray_out_of_range")
    return transform[:3, 3] + distance * direction


def table_corners(edge_pixels, plane, geometry, transform, letters, width, depth):
    """First point is the visible near-left corner; second is along the near edge."""
    origin, other = [on_plane(pixel, plane, geometry, transform) for pixel in edge_pixels]
    across = other - origin
    if np.linalg.norm(across) < 0.15:
        raise ValueError("桌边两点距离太近；第二点请沿近边向右选远一些")
    across /= np.linalg.norm(across)
    normal = np.r_[-plane[:2], 1.0]
    normal /= np.linalg.norm(normal)
    inward = np.cross(normal, across)
    centers = np.array([v["estimated_xyz"] for v in letters.values()])
    if (centers.mean(axis=0) - origin) @ inward < 0:
        inward = -inward
    offsets = centers - origin
    u, v = offsets @ across, offsets @ inward
    if np.any(u < 0.02) or np.any(u > width - 0.02) or np.any(v < 0.02) or np.any(v > depth - 0.02):
        raise ValueError("部分积木不在所标桌面内；检查近左角、近边方向、桌宽深或误选的候选")
    return np.array(
        [origin, origin + width * across, origin + width * across + depth * inward, origin + depth * inward]
    )


def project(points, geometry, transform):
    camera = (np.asarray(points) - transform[:3, 3]) @ transform[:3, :3]
    if np.any(camera[:, 2] <= 0):
        raise ValueError("scene_projection_behind_camera")
    return cv2.projectPoints(camera, np.zeros(3), np.zeros(3), geometry.K_left, geometry.D_left)[0][:, 0]


class SceneBuilder:
    """Prepare a draft, then export with explicit automatic or manual provenance."""

    def __init__(self, config_path, capture_dir, output, *, operator_cleared_workspace=False, target_letter="A"):
        self.target_letter = target_letter
        self.operator_cleared_workspace = operator_cleared_workspace
        self.config_path = Path(config_path).resolve()
        self.config = Config.load(self.config_path)
        self.capture_dir, self.output = Path(capture_dir).resolve(), Path(output).resolve()
        self.meta, self.left, self.right = read_capture(self.capture_dir, self.config)
        self.geometry, self.transform, _ = load(self.config.calibration, self.meta["body_after"]["body"])
        self.candidates = discover(self.left)
        self.source_hash = file_digest(self.capture_dir / "capture.json")
        self.config_hash = file_digest(self.config_path)
        self.calibration_hash = file_digest(self.config.calibration)
        self.model_binding = binding(self.config)
        self.draft = None
        self.draft_hash = None
        self.exported = False
        self.output.mkdir(parents=True, exist_ok=False)

    def verify_sources(self):
        if (
            file_digest(self.capture_dir / "capture.json") != self.source_hash
            or file_digest(self.config_path) != self.config_hash
            or file_digest(self.config.calibration) != self.calibration_hash
            or binding(self.config) != self.model_binding
        ):
            raise ValueError("source_config_calibration_or_model_changed:重新建立场景")
        read_capture(self.capture_dir, self.config)

    def build(self, annotation, *, identification=None):
        if self.exported:
            raise ValueError("scene_already_exported:新试验使用新目录")
        self.draft = self.draft_hash = None
        self.verify_sources()
        operator_cleared = getattr(self, "operator_cleared_workspace", False)
        if (annotation.get("workspace_policy") == "operator_cleared") != operator_cleared:
            raise ValueError("scene_workspace_policy_mismatch")
        ids, target = annotation.get("cube_ids"), annotation.get("target_id")
        if not isinstance(ids, list) or not 4 <= len(ids) <= 64 or any(type(i) is not int for i in ids):
            raise ValueError("请选择至少四块积木，包含 A；最多 64 块")
        if type(target) is not int or target not in ids or len(ids) != len(set(ids)):
            raise ValueError("请选择唯一的 A，且积木候选不得重复")
        candidates = {item["id"]: item for item in self.candidates}
        if any(i not in candidates for i in ids):
            raise ValueError("unknown_candidate_id")
        edge_mode = annotation.get("table_edge_mode", "visible_corner")
        if edge_mode not in ("visible_corner", "adjacent_straight_edges"):
            raise ValueError("scene_unknown_table_edge_mode")
        edge_names = ("near", "left") if edge_mode == "adjacent_straight_edges" else ("near",)
        count_min, count_max = (3, 8) if len(edge_names) == 2 else (2, 2)
        h, w = self.left.shape[:2]
        edges = {}
        for name in edge_names:
            edge = np.asarray(annotation.get(f"{name}_edge_px"), float)
            if edge.ndim != 2 or edge.shape[1] != 2 or not count_min <= len(edge) <= count_max:
                raise ValueError(f"table_edge_invalid_point_count:{name}")
            if not np.isfinite(edge).all() or np.any(edge < 0) or np.any(edge >= [w, h]):
                raise ValueError("table_edge_outside_image")
            edges[name] = edge
        dimensions = []
        for name, lower, upper in (
            ("table_width_mm", 300, 2500),
            ("table_depth_mm", 200, 1500),
            ("table_thickness_mm", 1, 100),
        ):
            value = annotation.get(name)
            if type(value) not in (int, float) or not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"invalid_{name}")
            dimensions.append(value / 1000)
        width, depth, thickness = dimensions
        pixels = {("A" if i == target else f"cube_{i:03d}"): candidates[i]["pixel"] for i in ids}
        measured = StereoTracker(geometry=self.geometry).locate_images(
            self.left,
            self.right,
            pixels,
            self.transform,
            require_common_height=False,
            glyph_mask=known_glyph_mask(self.left, pixels.values()),
        )
        (self.output / "measurements.json").write_text(json.dumps(measured, indent=2) + "\n")
        if "A" not in measured["letters"]:
            raise ValueError(f"A 必须有可靠双目深度：{measured.get('errors', {}).get('A')}")
        minimum_depths = 4 if operator_cleared else max(4, np.ceil(len(pixels) * 0.75))
        if len(measured["letters"]) < minimum_depths:
            requirement = "至少四个分散桌面测量点" if operator_cleared else "75% 的积木"
            raise ValueError(f"可靠双目深度不足，需要{requirement}，请重拍：{measured.get('errors', {})}")
        plane_diagnostics = {}
        try:
            plane, residual, inliers = fit_table(
                measured["letters"], operator_cleared_workspace=operator_cleared, diagnostics=plane_diagnostics
            )
        except ValueError as exc:
            plane_diagnostics["error"] = str(exc)
            raise
        finally:
            (self.output / "table-plane-check.json").write_text(
                json.dumps(plane_diagnostics, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            )
        target_height = target_height_check(measured["letters"]["A"], plane)
        (self.output / "target-height-check.json").write_text(
            json.dumps(target_height, indent=2, allow_nan=False) + "\n"
        )
        if not target_height["passed"]:
            raise ValueError(
                f"A 顶面应在桌面上方 40 mm；实测 {target_height['top_above_table_mm']:.2f} mm，"
                f"偏差 {target_height['error_mm']:+.2f} mm，允许 ±{target_height['limit_mm']:.1f} mm"
            )
        letters = dict(measured["letters"])
        # A retains its measured stereo pose even when it is not a plane-fit
        # inlier. Only non-target obstacles may use a plane-derived fallback.
        modeled = [] if operator_cleared else sorted(set(pixels) - inliers - {"A"})
        for name in modeled:
            top_plane = plane.copy()
            top_plane[2] += 0.04
            top = on_plane(pixels[name], top_plane, self.geometry, self.transform)
            letters[name] = {
                "pixel": pixels[name],
                "top_center_xyz": top.tolist(),
                "estimated_xyz": (top - [0, 0, 0.02]).tolist(),
                "grasp_center_xyz": (top - [0, 0, 0.005]).tolist(),
                "cube_size_m": 0.04,
                "source": "table_plane_and_known_cube_height",
                "position_uncertainty_m": 0.02,
                "stereo_error": measured.get("errors", {}).get(name, "table_fit_outlier"),
            }
        if edge_mode == "adjacent_straight_edges":
            samples = {
                name: np.array([on_plane(pixel, plane, self.geometry, self.transform) for pixel in points])
                for name, points in edges.items()
            }
            corners, edge_report = fit_table_edges(samples["near"], samples["left"], plane, width, depth)
            extent = object_extent_report(letters, corners, width, depth, target_only=operator_cleared)
            (self.output / "table-object-extent.json").write_text(
                json.dumps(extent, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            )
            if not extent["passed"]:
                raise ValueError(
                    "积木不在相邻直线重建的桌面范围内："
                    + ",".join(extent["failed_objects"])
                    + "；检查两条边的选择、方向及桌面尺寸"
                )
        else:
            corners = table_corners(edges["near"], plane, self.geometry, self.transform, letters, width, depth)
            edge_report = {"mode": "visible_corner", "footprint_padding_m": 0.01}
        (self.output / "table-edges.json").write_text(json.dumps(edge_report, indent=2, allow_nan=False) + "\n")
        # Keep the oriented, padded footprint for collision narrow phase. Its
        # AABB is only a broad-phase bound; still exclude the space below it.
        footprint = None
        if "padded_corners_m" in edge_report:
            padded = np.array(edge_report["padded_corners_m"])
            footprint = padded[:, :2].tolist()
            bounds_min, bounds_max = padded.min(axis=0), padded.max(axis=0)
            bounds_min[2] -= max(0.74, thickness)
        else:
            bounds_min, bounds_max = corners.min(axis=0), corners.max(axis=0)
            bounds_min -= [0.01, 0.01, max(0.74, thickness)]
            bounds_max += [0.01, 0.01, 0]
        data = asdict(self.config)
        data.update(
            scene_measured=True,
            table_min=bounds_min.tolist(),
            table_max=bounds_max.tolist(),
            table_plane=plane.tolist(),
            table_footprint=footprint,
            table_uncertainty=0.005,
            obstacles=[],
            output=str(self.output / "executor"),
        )
        workspace = footprint or corners[:, :2].tolist()
        # The existing executor interprets this polygon as the permitted table
        # workspace. Operator clearance authorizes the entire measured footprint,
        # not just its camera-visible subset. Physical collision geometry above
        # is unchanged, and scene_complete stays false (no visual certification).
        data["table_observed"] = (
            workspace
            if operator_cleared
            else observed_table(workspace, plane, self.geometry, self.transform) if identification is not None else None
        )
        # Preserve candidate/TCP validity flags. Scene review is not calibration.
        data["target_origin"][2] = float(plane @ [*data["target_origin"][:2], 1] + 0.02)
        scene_config = Config(**data)
        from .observation_policy import cube_obstacles

        table_support_letters = letters
        if operator_cleared:
            # Non-target samples establish the plane only. Neither planning
            # nor observation receives them as validated scene obstacles.
            letters = {"A": dict(letters["A"], workspace_policy="operator_cleared")}
        obstacles = cube_obstacles(letters)
        state = self.meta["body_after"]
        source = {
            "body": state["body"],
            "hands": state["hands"],
            "grasp_center_m": letters["A"]["grasp_center_xyz"],
            "target_letter": getattr(self, "target_letter", "A"),
            "obstacles": obstacles,
            "observation_policy_version": 1,
            "workspace_policy": "operator_cleared" if operator_cleared else "vision_checked",
            "scene_complete": False,
            "parameters": {},
            "binding": binding(scene_config),
            "source_capture_sha256": self.source_hash,
            "source_capture": str(self.capture_dir / "capture.json"),
            "source_session": state["session"],
            "source_generation": state["generation"],
            "capture_wall_time": self.meta.get("wall_time"),
            "start_source": "measured_capture_pose_not_assumed_ready",
        }
        draft = {
            "config": data,
            "input": source,
            "letters": letters,
            "table_support_letters": table_support_letters if operator_cleared else {},
            "annotation": annotation,
            "table_corners_m": corners.tolist(),
            "table_edge_model": edge_report,
            "table_observed": data["table_observed"],
            "table_residual_max_mm": float(
                max(abs(r) for name, r in zip(measured["letters"], residual) if name in inliers) * 1000
            ),
            "table_all_samples_residual_max_mm": float(np.max(np.abs(residual)) * 1000),
            "table_inliers": sorted(inliers),
            "target_height_check": target_height,
            "modeled_obstacles": modeled,
            "source_config_sha256": self.config_hash,
            "source_calibration_sha256": self.calibration_hash,
            "hardware_ready": False,
            "api_calls": identification.get("api_calls", 1) if identification is not None else 0,
            "identification": identification,
        }
        overlay = self.left.copy()
        outline = np.round(project(corners, self.geometry, self.transform)).astype(np.int32)
        cv2.polylines(overlay, [outline], True, (0, 210, 255), 3)
        if data["table_observed"] is not None:
            domain = np.asarray(data["table_observed"])
            domain = np.c_[domain, domain @ plane[:2] + plane[2]]
            visible_outline = np.round(project(domain, self.geometry, self.transform)).astype(np.int32)
            cv2.polylines(overlay, [visible_outline], True, (255, 255, 0), 3)
        for points in edges.values():
            for pixel in points:
                cv2.circle(overlay, tuple(np.round(pixel).astype(int)), 5, (255, 0, 255), 2)
        for name, record in letters.items():
            point = tuple(np.round(record["pixel"]).astype(int))
            color = (0, 230, 0) if name == "A" else ((0, 128, 255) if name in modeled else (255, 160, 0))
            cv2.circle(overlay, point, 22, color, 2)
            cv2.putText(overlay, name, (point[0] + 24, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if not cv2.imwrite(str(self.output / "scene-review.jpg"), overlay):
            raise OSError("scene_overlay_write_failed")
        (self.output / "draft.json").write_text(json.dumps(draft, indent=2, allow_nan=False) + "\n")
        self.draft, self.draft_hash = draft, file_digest(self.output / "draft.json")
        return {
            "draft_hash": self.draft_hash,
            "cube_count": len(letters),
            "A_center_mm": (np.array(letters["A"]["estimated_xyz"]) * 1000).round(1).tolist(),
            "grasp_center_mm": (np.array(source["grasp_center_m"]) * 1000).round(1).tolist(),
            "table_residual_max_mm": draft["table_residual_max_mm"],
            "table_edge_model": edge_report,
            "modeled_obstacles": modeled,
            "hardware_ready": False,
        }

    def approve(self, request):
        """Export the exact draft explicitly confirmed through the manual UI."""
        if self.exported or self.draft is None or request.get("draft_hash") != self.draft_hash:
            raise ValueError("review_the_current_draft_before_export")
        if any(request.get(key) is not True for key in ("target_confirmed", "all_cubes_marked", "table_confirmed")):
            raise ValueError("请确认 A、所有积木与清空杂物、以及桌面范围")
        if self.draft.get("identification") is not None:
            raise ValueError("automatic_draft_requires_automatic_export")
        return self._export(dict(request, mode="manual", api_calls=0))

    def export_automatic(self):
        """Export validated API evidence without fabricating operator confirmations."""
        from .scene_identification import annotation_from_decision

        if self.exported or self.draft is None or not self.draft.get("identification"):
            raise ValueError("automatic_identification_required")
        evidence = self.draft["identification"]
        path = Path(evidence["path"])
        if file_digest(path) != evidence["sha256"]:
            raise ValueError("scene_identification_file_changed")
        decision = json.loads(path.read_text())
        operator_cleared = getattr(self, "operator_cleared_workspace", False)
        annotation = annotation_from_decision(
            decision,
            self.candidates,
            self.left.shape,
            self.draft["annotation"],
            operator_cleared_workspace=operator_cleared,
        )
        if annotation != self.draft["annotation"]:
            raise ValueError("scene_identification_draft_mismatch")
        return self._export(
            {
                "mode": "automatic_vision_and_local_stereo",
                "api_calls": self.draft.get("api_calls", 1),
                "identification": evidence,
                "draft_hash": self.draft_hash,
                "operator_confirmed": False,
                "completeness_source": (
                    "operator_managed_workspace_not_visually_verified"
                    if operator_cleared
                    else "vision_assessment_of_recorded_image"
                ),
                "tabletop_fully_visible": decision["tabletop_fully_visible"],
                "table_edge_model": self.draft.get("table_edge_model"),
                "workspace_policy": "operator_cleared" if operator_cleared else "vision_checked",
            }
        )

    def _export(self, request):
        self.verify_sources()
        if file_digest(self.output / "draft.json") != self.draft_hash:
            raise ValueError("draft_file_changed")
        draft = self.draft
        operator_cleared = draft["input"].get("workspace_policy") == "operator_cleared"
        scene_complete = False if operator_cleared else request.get("tabletop_fully_visible", True)
        execution_scope = (
            "operator_cleared_table_workspace"
            if operator_cleared
            else (
                "observed_table_workspace"
                if draft["config"].get("table_observed") is not None
                else "whole_table" if scene_complete else "offline_only"
            )
        )
        source = dict(
            draft["input"],
            scene_complete=scene_complete,
            review_sha256=self.draft_hash,
            scene_source=request["mode"],
            execution_scope=execution_scope,
        )
        files = {
            "scene-config.json": draft["config"],
            "scene-input.json": source,
            "scene-letters.json": {"letters": draft["letters"], "identity_image": str(self.capture_dir / "left.jpg")},
            "scene-review.json": {"review": request, "draft_sha256": self.draft_hash, "hardware_ready": False},
        }
        for name in files:
            if (self.output / name).exists():
                raise ValueError("scene_output_already_exists")
        for name, value in files.items():
            with (self.output / name).open("x") as stream:
                json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
        manifest = {
            "files": {name: file_digest(self.output / name) for name in files},
            "hardware_ready": False,
            "mode": request["mode"],
            "api_calls": request["api_calls"],
            "scene_complete": scene_complete,
            "execution_scope": execution_scope,
            "workspace_policy": draft["input"].get("workspace_policy", "vision_checked"),
        }
        (self.output / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self.exported = True
        return {
            "output": str(self.output),
            "files": list(files),
            "hardware_ready": False,
            "mode": request["mode"],
            "api_calls": request["api_calls"],
            "scene_complete": scene_complete,
            "execution_scope": execution_scope,
            "workspace_policy": draft["input"].get("workspace_policy", "vision_checked"),
        }
