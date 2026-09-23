"""Reuse checked preparation artifacts only with the installed, valid scene."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .calibration import binding
from .config import file_digest
from .motion_checks import MotionChecks
from .scene_bundle import SceneBundle
from .trajectory import Segment


def implementation_digest():
    """Invalidate saved plans when any robot algorithm or shared geometry changes."""
    root = Path(__file__).parent
    paths = sorted(root.glob("*.py")) + [
        root.parent / name for name in ("stereo.py", "kinematics.py", "pinch.py", "geometry.py", "vision.py")
    ]
    return hashlib.sha256("".join(file_digest(path) for path in paths).encode()).hexdigest()


def pose_matches(actual, expected, *, body_tolerance=None, checks=None):
    """Bound saved-path start drift using fresh complete body and hand feedback."""
    checks = checks or MotionChecks()
    body_tolerance = checks.scene_pose_rad if body_tolerance is None else body_tolerance
    for key, size, tolerance in (("body", 29, body_tolerance), ("hands", 12, checks.scene_hand_raw)):
        a, b = np.asarray(actual.get(key, [])), np.asarray(expected.get(key, []))
        if a.shape != (size,) or b.shape != (size,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            return False
        if np.max(np.abs(a - b)) > tolerance:
            return False
    return True


def cache_path(config_path):
    """Keep the active preparation pointer alongside its operator configuration."""
    path = Path(config_path).resolve()
    return path.with_name(path.stem + "-prepared.json")


def save_prepared(config_path, output, bundle):
    """Bind files, settings, code and session after successful no-motion install."""
    output = Path(output).resolve()
    data = {
        "version": 2,
        "scene": str(bundle.directory.resolve()),
        "scene_id": bundle.digest,
        "session": bundle.source["source_session"],
        "implementation": implementation_digest(),
        "config_sha256": file_digest(config_path),
        "binding": binding(bundle.config),
        "files": {
            name: {"path": str(output / name), "sha256": file_digest(output / name)}
            for name in ("preparation-plan.json",)
        },
    }
    path = cache_path(config_path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_prepared(config_path, config, state):
    """Reject changed files, interrupted scenes and prior boots before any motion."""
    try:
        data = json.loads(cache_path(config_path).read_text())
        if (
            data["version"] != 2
            or data["scene_id"] != state.get("scene_id")
            or data["session"] != state.get("session")
            or state.get("recovery_scene_valid") is not True
            or data["implementation"] != implementation_digest()
            or data["config_sha256"] != file_digest(config_path)
        ):
            raise ValueError("prepared_scene_or_settings_changed")
        bundle = SceneBundle.load(data["scene"])
        bundle.check_installation(config)
        if (
            bundle.digest != data["scene_id"]
            or bundle.source["source_session"] != state["session"]
            or bundle.source.get("observation_policy_version") != 1
            or binding(bundle.config) != data["binding"]
        ):
            raise ValueError("prepared_geometry_changed")
        # P may have moved the arms to READY. A shifted base still needs a new scene.
        if (
            np.max(np.abs(np.asarray(state["body"][:15]) - bundle.source["body"][:15]))
            > config.motion_checks.scene_pose_rad
        ):
            raise ValueError("prepared_base_pose_changed")
        files = {}
        for name, entry in data["files"].items():
            if file_digest(entry["path"]) != entry["sha256"]:
                raise ValueError("prepared_plan_file_changed")
            files[name] = json.loads(Path(entry["path"]).read_text())
        preparation = files["preparation-plan.json"]
        if preparation["scene_id"] != bundle.digest:
            raise ValueError("prepared_plan_not_checked")
        segments = []
        for item in preparation["segments"]:
            item = dict(item)
            for key in ("arms", "hands", "target", "hand_target"):
                item[key] = np.asarray(item[key])
            segments.append(Segment(**item))
        return bundle, segments, preparation["predicted_ready"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"prepare_required:请先运行 prepare；{exc}") from exc
