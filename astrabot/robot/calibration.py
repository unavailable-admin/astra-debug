"""Calibration is enabled by independent measurements, not a configured pass flag."""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..geometry import StereoGeometry
from .config import ASSETS, Config, file_digest
from .manual_capture import load_body_evidence
from .model import RobotModel


def rotation(value):
    """Require a proper rotation, avoiding silent projection of invalid matrices."""
    r = np.asarray(value, float)
    if (
        r.shape != (3, 3)
        or not np.isfinite(r).all()
        or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
        or abs(np.linalg.det(r) - 1) > 1e-6
    ):
        raise ValueError("invalid_rotation")
    return r


def load(path, body=None):
    """Read stereo fields; return the camera-to-pelvis pose for measured body joints.

    A torso-mounted candidate requires an explicit body pose. Its reference must
    never silently be treated as pelvis coordinates.
    """
    d = json.loads(Path(path).read_text())
    size = tuple(d["image_size"])
    if len(size) != 2 or any(type(v) is not int or v < 64 for v in size):
        raise ValueError("invalid_calibration_size")
    matrices = []
    for name in ("K1", "K2"):
        k = np.asarray(d[name], float)
        if (
            k.shape != (3, 3)
            or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1])
        ):
            raise ValueError("invalid_intrinsics")
        matrices.append(k)
    distortions = []
    for name in ("D1", "D2"):
        v = np.asarray(d[name], float).reshape(-1)
        if len(v) not in (4, 5, 8, 12, 14) or not np.isfinite(v).all():
            raise ValueError("invalid_pinhole_distortion")
        distortions.append(v)
    if d.get("distortion_model", "pinhole") != "pinhole":
        raise ValueError("pinhole_calibration_required")
    t = np.asarray(d["T_right_from_left_m"], float).reshape(-1)
    if t.shape != (3,) or not np.isfinite(t).all() or not 0.03 <= np.linalg.norm(t) <= 0.15:
        raise ValueError("invalid_stereo_baseline")
    torso_reference = d.get("T_torso_left_cv") is not None
    transform = np.asarray(d.get("T_torso_left_cv") if torso_reference else d.get("T_pelvis_left_cv"), float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("camera_to_pelvis_extrinsic_missing_or_invalid")
    rotation(transform[:3, :3])
    if torso_reference:
        if body is None:
            raise ValueError("torso_camera_requires_measured_body")
        transform = RobotModel(Config()).poses(body, np.zeros(12))["torso_link"] @ transform
    g = StereoGeometry("real", size, *matrices, *distortions, rotation(d["R_right_from_left"]), t)
    return g, transform, d


def binding(config):
    """Bind every physical setting and geometry asset to the measured report."""
    d = asdict(config)
    # Include default motion checks too: v6 changes pacing semantics, so legacy
    # reports must not silently acquire the same binding as a newly checked path.
    if d.get("table_footprint") is None:
        d.pop("table_footprint", None)  # Preserve bindings of legacy AABB scene exports.
    if d.get("table_observed") is None:
        d.pop("table_observed", None)
    for key in ("output", "calibration", "calibration_report"):
        d.pop(key)
    assets = {
        p.name: file_digest(p)
        for p in (
            ASSETS / "g1_o6.urdf",
            ASSETS / "hand_bounds.json",
            ASSETS / "head_bounds.json",
            ASSETS / "body_bounds.json",
            ASSETS / "body_hulls.npz",
            ASSETS / "escape_bounds.json",
            ASSETS / "escape_hulls.npz",
            ASSETS / "gravity.xml",
        )
    }
    return hashlib.sha256(json.dumps({"config": d, "assets": assets}, sort_keys=True).encode()).hexdigest()


def fit_rigid(camera_points, pelvis_points):
    """Fit a candidate SE(3); callers must reserve separate held-out validation data."""
    a, b = np.asarray(camera_points, float), np.asarray(pelvis_points, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 6 or not np.isfinite([a, b]).all():
        raise ValueError("six_matching_3D_points_required")
    if np.linalg.matrix_rank(a - a.mean(0), tol=0.005) < 2:
        raise ValueError("degenerate_extrinsic_samples")
    u, _, vt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
    r = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = b.mean(0) - r @ a.mean(0)
    return t


def validate(config, samples_path):
    """Check 40 mm edges, table height and FK landmarks at independent postures."""
    # This initial transform is not used for measurements. Torso candidates are
    # transformed using each capture's observed body pose below.
    geometry, transform, source = load(config.calibration, np.zeros(29))
    if geometry.image_size_wh != (config.camera_width // 2, config.camera_height):
        raise ValueError("calibration_capture_resolution_mismatch")
    if not config.scene_measured or not config.tcp_measured:
        raise ValueError("scene_and_tcp_measurements_required")
    samples_path = Path(samples_path).resolve()
    data = json.loads(samples_path.read_text())
    if data.get("purpose") != "held_out_validation":
        raise ValueError("independent_validation_samples_required")
    model = RobotModel(config)
    errors, edge_errors, table_errors, postures, ids, hashes = [], [], [], [], [], {}
    baseline = None
    torso_reference = source.get("T_torso_left_cv") is not None
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    for sample in data["samples"]:
        identifier = str(sample["id"])
        if identifier in ids or identifier in list(map(str, source.get("training_sample_ids", []))):
            raise ValueError("reused_training_or_duplicate_sample")
        ids.append(identifier)
        capture_path = (samples_path.parent / sample["capture"]).resolve()
        meta = json.loads(capture_path.read_text())
        if (
            meta["device"] != config.camera_device
            or meta["boot_id"] != boot
            or not 0 <= time.time() - meta["wall_time"] <= 86400
        ):
            raise ValueError("capture_device_boot_or_age_mismatch")
        if (
            meta["width"] != config.camera_width
            or meta["height"] != config.camera_height
            or meta["timestamp_source"] != "v4l2_monotonic"
        ):
            raise ValueError("capture_geometry_or_timestamp_mismatch")
        for name, digest in meta["image_sha256"].items():
            if file_digest(capture_path.parent / name) != digest:
                raise ValueError("image_hash_mismatch")
        hashes[str(capture_path)] = file_digest(capture_path)
        passive = meta.get("acquisition_source") == "passive_dds_body_v1"
        if passive:
            body, trace_hashes = load_body_evidence(meta, capture_path)
            hashes.update(trace_hashes)
            if sample["kind"] == "hand" and sample.get("frame") != "left_hand_base_link":
                raise ValueError("passive_capture_requires_rigid_hand_base")
            # FK placeholder only; fingers were not measured.
            hands = np.zeros(12)
        else:
            state = meta.get("body_after")
            if not state or state["health"]:
                raise ValueError("capture_missing_healthy_robot_snapshot")
            body, hands = np.asarray(state["body"]), np.asarray(state["hands"])
        # Hand observations in a torso reference can span waist poses. Table/cube
        # observations still anchor the fixed scene and the runtime body baseline.
        if not torso_reference or sample["kind"] == "cube":
            baseline = body[:15] if baseline is None else baseline
            if np.max(np.abs(body[:15] - baseline)) > 0.02:
                raise ValueError("base_changed_during_calibration")
        if torso_reference:
            transform = model.poses(body, hands)["torso_link"] @ np.asarray(source["T_torso_left_cv"])
        camera = geometry.triangulate(
            sample["left_px"], sample["right_px"], image_size_wh=geometry.image_size_wh, max_reprojection_px=1.5
        )
        xyz = camera @ transform[:3, :3].T + transform[:3, 3]
        if sample["kind"] == "cube":
            edges = sample["edges"]
            if len(edges) < 4:
                raise ValueError("four_measured_top_edges_required")
            for a, b in edges:
                if a == b:
                    raise ValueError("degenerate_edge")
                edge_errors.append(abs(np.linalg.norm(xyz[a] - xyz[b]) - 0.04))
            table_errors.extend(abs(xyz[:, 2] - 0.04 - config.table_max[2]))
        elif sample["kind"] == "hand":
            local = np.asarray(sample["local_points"], float)
            if local.shape != xyz.shape or len(local) < 2 or not np.isfinite(local).all():
                raise ValueError("matching_hand_landmarks_required")
            frame = sample["frame"]
            if not frame.startswith("left_") or frame not in {e["frame"] for e in model.bounds["left"]}:
                raise ValueError("left_hand_landmark_frame_required")
            pose = model.poses(body, hands)[frame]
            expected = local @ pose[:3, :3].T + pose[:3, 3]
            errors.extend(np.linalg.norm(xyz - expected, axis=1))
            postures.append(body[15:22])
        else:
            raise ValueError("unknown_validation_sample_kind")
    if len(edge_errors) < 12 or len(postures) < 3 or len(errors) < 6:
        raise ValueError("three_cube_views_and_three_hand_postures_required")
    if any(np.max(np.abs(a - b)) < 0.1 for i, a in enumerate(postures) for b in postures[i + 1 :]):
        raise ValueError("hand_validation_postures_too_similar")
    metrics = {
        "edge_max_m": float(max(edge_errors)),
        "workspace_max_m": float(max(errors)),
        "table_max_m": float(max(table_errors)),
    }
    passed = metrics["edge_max_m"] <= 0.002 and metrics["workspace_max_m"] <= 0.008 and metrics["table_max_m"] <= 0.005
    return {
        "passed": bool(passed),
        "metrics": metrics,
        "thresholds": {"edge_max_m": 0.002, "workspace_max_m": 0.008, "table_max_m": 0.005},
        "binding": binding(config),
        "calibration_sha256": file_digest(config.calibration),
        "samples_path": str(samples_path),
        "samples_sha256": file_digest(samples_path),
        "capture_hashes": hashes,
        "wall_time": time.time(),
        "boot_id": boot,
        "baseline_body": baseline.tolist(),
        "sample_ids": ids,
    }


def require_valid(config, body):
    """Recompute independent checks; changing only a report's pass flag cannot enable motion."""
    if not config.calibration_report:
        raise ValueError("calibration_validation_report_required")
    report = json.loads(Path(config.calibration_report).read_text())
    if report["binding"] != binding(config) or report["calibration_sha256"] != file_digest(config.calibration):
        raise ValueError("calibration_or_configuration_changed")
    if file_digest(report["samples_path"]) != report["samples_sha256"]:
        raise ValueError("validation_samples_changed")
    fresh = validate(config, report["samples_path"])
    if not fresh["passed"] or fresh["capture_hashes"] != report["capture_hashes"]:
        raise ValueError("calibration_validation_failed")
    if np.max(np.abs(np.asarray(body)[:15] - fresh["baseline_body"])) > 0.02:
        raise ValueError("robot_base_moved_since_calibration")
    return load(config.calibration, body)[:2]
