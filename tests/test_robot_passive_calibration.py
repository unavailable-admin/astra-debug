"""Validate torso compensation and raw passive evidence without robot hardware."""

import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from astrabot.robot.calibration import load, require_valid, validate
from astrabot.robot.config import Config, file_digest
from astrabot.robot.model import RobotModel


class PassiveCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = Config(scene_measured=True, tcp_measured=True, table_max=[1, 0.6, -0.12])
        self.model = RobotModel(self.config)
        # Simple synthetic optics keep projected landmarks inside the image.
        self.mount = np.eye(4)
        self.mount[:3, 3] = [0, 0, -1]
        self.K = np.array([[700, 0, 960], [0, 700, 600], [0, 0, 1.0]])
        source = {
            "image_size": [1920, 1200],
            "K1": self.K.tolist(),
            "K2": self.K.tolist(),
            "D1": [0.0] * 5,
            "D2": [0.0] * 5,
            "R_right_from_left": np.eye(3).tolist(),
            "T_right_from_left_m": [-0.06, 0, 0],
            "T_torso_left_cv": self.mount.tolist(),
            "T_pelvis_left_cv": None,
            "training_sample_ids": [],
        }
        self.config.calibration = str(self.root / "candidate.json")
        Path(self.config.calibration).write_text(json.dumps(source))
        self.samples = []
        self.local = np.array([[-0.018, 0, 0.04], [-0.018, 0, 0.09]])
        for i in range(3):
            body = np.zeros(29)
            body[12] = i * 0.15
            body[15] = i * 0.2
            poses = self.model.poses(body, np.zeros(12))
            hand = poses["left_hand_base_link"]
            points = self.local @ hand[:3, :3].T + hand[:3, 3]
            sample = self.add_sample(f"hand-{i}", body, points, "hand")
            sample.update(frame="left_hand_base_link", local_points=self.local.tolist())
        for i in range(3):
            points = np.array([[0.25, 0, -0.08], [0.29, 0, -0.08], [0.29, 0.04, -0.08], [0.25, 0.04, -0.08]])
            points[:, 1] += i * 0.08
            sample = self.add_sample(f"cube-{i}", np.zeros(29), points, "cube")
            sample["edges"] = [[0, 1], [1, 2], [2, 3], [3, 0]]
        self.manifest = self.root / "samples.json"
        self.save_samples()

    def save_samples(self):
        self.manifest.write_text(json.dumps({"purpose": "held_out_validation", "samples": self.samples}))

    def add_sample(self, name, body, pelvis_points, kind):
        directory = self.root / name
        directory.mkdir()
        exposure = time.monotonic() - 1
        trace = [
            {"joint_time": exposure + (i - 20) * 0.01, "tick": 1000 + 10 * i, "body": body.tolist(), "joint_fault": ""}
            for i in range(41)
        ]
        (directory / "joint-trace.json").write_text(json.dumps(trace))
        # Hash-bound image artifacts: projection consistency is exercised below.
        for image in ("stereo.jpg", "left.jpg", "right.jpg"):
            (directory / image).write_bytes((name + image).encode())
        meta = {
            "device": self.config.camera_device,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "wall_time": time.time(),
            "width": 3840,
            "height": 1200,
            "timestamp_source": "v4l2_monotonic",
            "acquired_monotonic": exposure,
            "image_sha256": {name: file_digest(directory / name) for name in ("stereo.jpg", "left.jpg", "right.jpg")},
            "purpose": "manual_rigid_hand_observation",
            "acquisition_source": "passive_dds_body_v1",
            "robot_commands_sent": 0,
            "fingers_observed": False,
            "joint_reader": {"alive": True, "fault": ""},
            "joint_trace": {"path": "joint-trace.json", "sha256": file_digest(directory / "joint-trace.json")},
            "rigid_body_stationarity": {"passed": True},
        }
        (directory / "capture.json").write_text(json.dumps(meta))
        camera_pose = self.model.poses(body, np.zeros(12))["torso_link"] @ self.mount
        camera_points = (pelvis_points - camera_pose[:3, 3]) @ camera_pose[:3, :3]

        def project(points):
            return cv2.projectPoints(points, np.zeros(3), np.zeros(3), self.K, np.zeros(5))[0].reshape(-1, 2).tolist()

        sample = {
            "id": name,
            "kind": kind,
            "capture": f"{name}/capture.json",
            "left_px": project(camera_points),
            "right_px": project(camera_points + [-0.06, 0, 0]),
        }
        self.samples.append(sample)
        return sample

    def test_torso_motion_compensated_and_scene_baseline_preserved(self):
        report = validate(self.config, self.manifest)
        self.assertTrue(report["passed"], report["metrics"])
        self.assertLess(report["metrics"]["workspace_max_m"], 1e-10)
        np.testing.assert_array_equal(report["baseline_body"], np.zeros(15))
        self.assertEqual(len(report["capture_hashes"]), 12)
        self.config.calibration_report = str(self.root / "report.json")
        Path(self.config.calibration_report).write_text(json.dumps(report))
        require_valid(self.config, np.zeros(29))
        moved = np.zeros(29)
        moved[12] = 0.1
        with self.assertRaisesRegex(ValueError, "base_moved"):
            require_valid(self.config, moved)

    def test_torso_camera_needs_body_and_returns_correct_pelvis_pose(self):
        with self.assertRaisesRegex(ValueError, "requires_measured_body"):
            load(self.config.calibration)
        body = np.zeros(29)
        body[12] = 0.3
        _, pose, _ = load(self.config.calibration, body)
        np.testing.assert_allclose(pose, self.model.poses(body, np.zeros(12))["torso_link"] @ self.mount)

    def test_unobserved_fingers_are_not_valid_landmarks(self):
        self.samples[0]["frame"] = "left_index_link"
        self.save_samples()
        with self.assertRaisesRegex(ValueError, "requires_rigid_hand_base"):
            validate(self.config, self.manifest)

    def test_trace_hash_and_saved_pass_flag_cannot_hide_motion(self):
        trace_path = self.root / "hand-0/joint-trace.json"
        trace = json.loads(trace_path.read_text())
        trace[20]["body"][18] = 0.1
        trace_path.write_text(json.dumps(trace))
        with self.assertRaisesRegex(ValueError, "trace_hash_mismatch"):
            validate(self.config, self.manifest)
        meta_path = self.root / "hand-0/capture.json"
        meta = json.loads(meta_path.read_text())
        meta["joint_trace"]["sha256"] = file_digest(trace_path)
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "body_moved"):
            validate(self.config, self.manifest)

    def test_duplicate_ticks_are_not_new_joint_evidence(self):
        trace_path = self.root / "hand-0/joint-trace.json"
        trace = json.loads(trace_path.read_text())
        trace[20]["tick"] = trace[19]["tick"]
        trace_path.write_text(json.dumps(trace))
        meta_path = self.root / "hand-0/capture.json"
        meta = json.loads(meta_path.read_text())
        meta["joint_trace"]["sha256"] = file_digest(trace_path)
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "duplicate_or_regressed"):
            validate(self.config, self.manifest)


if __name__ == "__main__":
    unittest.main()
