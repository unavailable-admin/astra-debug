"""Read and bind completed scene exports for supervised preparation and trials."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .calibration import binding
from .config import Config, file_digest, vector
from .scene_builder import read_capture

SCENE_FIELDS = {
    "scene_measured",
    "table_min",
    "table_max",
    "table_plane",
    "table_footprint",
    "table_observed",
    "table_uncertainty",
    "obstacles",
    "target_origin",
    "output",
}


@dataclass
class SceneBundle:
    """Verified files, original capture state, and the corresponding runtime config."""

    directory: Path
    config: Config
    source: dict
    letters: dict
    identity_image: Path
    digest: str

    @classmethod
    def load(cls, directory: Path, *, require_complete=True):
        """Reject incomplete, edited, or differently calibrated scene exports."""
        directory = Path(directory).resolve()
        manifest = json.loads((directory / "export.json").read_text())
        for name in ("scene-config.json", "scene-input.json", "scene-letters.json", "scene-review.json"):
            if manifest.get("files", {}).get(name) != file_digest(directory / name):
                raise ValueError(f"scene_export_file_changed:{name}")
        config = Config.load(directory / "scene-config.json")
        source = json.loads((directory / "scene-input.json").read_text())
        review = json.loads((directory / "scene-review.json").read_text())
        draft = json.loads((directory / "draft.json").read_text())
        if (
            require_complete and source.get("scene_complete") is not True and config.table_observed is None
        ) or not config.scene_measured:
            raise ValueError("complete_measured_scene_required")
        draft_hash = file_digest(directory / "draft.json")
        if source.get("review_sha256") != draft_hash or review.get("draft_sha256") != draft_hash:
            raise ValueError("scene_draft_changed")
        if source.get("binding") != binding(config):
            raise ValueError("scene_model_or_config_changed")
        if draft.get("source_calibration_sha256") != file_digest(config.calibration):
            raise ValueError("scene_calibration_changed")
        capture_path = Path(source["source_capture"])
        if file_digest(capture_path) != source.get("source_capture_sha256"):
            raise ValueError("scene_capture_changed")
        meta, _, _ = read_capture(capture_path.parent, config)
        state = meta["body_after"]
        if state["session"] != source.get("source_session"):
            raise ValueError("scene_capture_session_mismatch")
        for key, size in (("body", 29), ("hands", 12)):
            if not np.array_equal(vector(source[key], size, key), state[key]):
                raise ValueError(f"scene_capture_pose_mismatch:{key}")
        data = json.loads((directory / "scene-letters.json").read_text())
        identity_image = Path(data["identity_image"]).resolve()
        if identity_image != (capture_path.parent / "left.jpg").resolve():
            raise ValueError("scene_identity_image_mismatch")
        letters = data["letters"]
        a = vector(letters["A"]["estimated_xyz"], 3, "A_center")
        if np.linalg.norm(vector(source["grasp_center_m"], 3, "grasp") - a - [0, 0, 0.015]) > 1e-6:
            raise ValueError("scene_target_geometry_mismatch")
        # All normal executor motions see non-target cubes. Preparation/return
        # additionally sees A; intended finger contact must not collide with A's box.
        runtime = asdict(config)
        runtime["obstacles"] = [*config.obstacles, *source["obstacles"]]
        return cls(
            directory, Config(**runtime), source, letters, identity_image, file_digest(directory / "export.json")
        )

    def target_obstacle(self) -> list:
        """Enclose A including arbitrary tabletop yaw and a position allowance."""
        center = vector(self.letters["A"]["estimated_xyz"], 3, "A_center")
        half = np.array([0.04, 0.04, 0.03])
        return [(center - half).tolist(), (center + half).tolist()]

    def check_installation(self, current: Config):
        """Allow scene updates without changing hardware, speed or TCP settings."""
        old, new = asdict(current), asdict(self.config)
        changed = [key for key in old if key not in SCENE_FIELDS and old[key] != new[key]]
        if changed:
            raise ValueError(f"scene_installation_settings_changed:{','.join(changed)}")

    def check_pose(self, state: dict):
        """Require the current session and the captured pose before scene install."""
        if state.get("session") != self.source["source_session"]:
            raise ValueError("scene_from_different_executor_session:重新拍照定位")
        for key, size, tolerance in (
            ("body", 29, self.config.motion_checks.scene_pose_rad),
            ("hands", 12, self.config.motion_checks.scene_hand_raw),
        ):
            if np.max(np.abs(vector(state[key], size, key) - self.source[key])) > tolerance:
                raise ValueError(f"scene_capture_pose_changed:{key}:重新拍照定位")


def install_scene(executor, message: dict) -> dict:
    """Prepare geometry outside the control lock, then install at the same idle generation."""
    from .model import RobotModel
    from .trajectory import Planner

    bundle = SceneBundle.load(message["directory"])
    planner = Planner(RobotModel(bundle.config), bundle.config)
    planner.recovery_obstacles = (bundle.target_obstacle(),)
    with executor.lock:
        state = executor.status()
        if any(message.get(key) != state[key] for key in ("session", "generation")):
            raise ValueError("expired_scene_install_generation")
        if (
            state["state"] not in ("DISARMED", "HOLD", "READY")
            or state["task_active"]
            or state["planning_active"]
            or state["remaining_segments"]
            or state["takeover_active"]
            or state["handoff_pending"]
            or state["thermal_handoff_required"]
            or state["health"]
            or state["error"]
        ):
            raise ValueError("scene_install_requires_healthy_idle_executor")
        bundle.check_installation(executor.config)
        bundle.check_pose(state)
        executor.config, executor.planner = bundle.config, planner
        executor.scene_id = bundle.digest
        executor.recovery_scene_valid = True
        executor.recovery_notice = ""
        executor.generation += 1
        executor.event("scene_installed", scene_id=bundle.digest, directory=str(bundle.directory))
        return executor.status()
