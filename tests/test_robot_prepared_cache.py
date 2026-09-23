"""Saved plans cannot outlive their scene, settings, files or executor boot."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from astrabot.robot.config import Config
from astrabot.robot.pick_a import write_json
from astrabot.robot.prepared_cache import load_prepared, pose_matches, save_prepared
from astrabot.robot.scene_bundle import SceneBundle
from astrabot.robot.trajectory import Segment


class PreparedCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_path = self.root / "config.json"
        self.config = Config(scene_measured=True)
        self.config_path.write_text(json.dumps(asdict(self.config)))
        self.state = {
            "session": "boot",
            "scene_id": "digest",
            "recovery_scene_valid": True,
            "generation": 2,
            "body": [0.0] * 29,
            "hands": [255.0] * 12,
        }
        self.bundle = SceneBundle(
            self.root / "scene",
            self.config,
            {"source_session": "boot", "body": [0.0] * 29, "hands": [255.0] * 12, "observation_policy_version": 1},
            {},
            self.root / "left.jpg",
            "digest",
        )
        segment = Segment(np.zeros(14), np.full(12, 255), np.zeros(14), np.full(12, 255), 1)
        write_json(
            self.root / "preparation-plan.json",
            {"scene_id": "digest", "segments": [asdict(segment)], "predicted_ready": self.state},
        )
        save_prepared(self.config_path, self.root, self.bundle)

    def load(self, state=None):
        with patch.object(SceneBundle, "load", return_value=self.bundle):
            return load_prepared(self.config_path, self.config, self.state if state is None else state)

    def test_console_pause_and_arm_movement_do_not_discard_same_scene(self):
        self.state["generation"] += 1
        self.state["body"][15] += 0.2
        bundle, segments, _ = self.load()
        self.assertEqual(bundle.digest, "digest")
        self.assertEqual(segments[0].duration, 1)
        self.assertFalse(pose_matches(self.state, self.bundle.source))
        self.assertFalse((self.root / "trial-plan-predicted.json").exists())

    def test_restart_interruption_replaced_scene_or_shifted_base_rejects(self):
        for key, value in (
            ("session", "new-boot"),
            ("scene_id", "other"),
            ("recovery_scene_valid", False),
            ("body", [0.01] + [0.0] * 28),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "prepare_required"):
                self.load(dict(self.state, **{key: value}))

    def test_changed_plan_config_or_code_rejects(self):
        with (
            patch("astrabot.robot.prepared_cache.implementation_digest", return_value="changed"),
            self.assertRaisesRegex(ValueError, "prepare_required"),
        ):
            self.load()
        self.config_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "prepare_required"):
            self.load()
        self.config_path.write_text(json.dumps(asdict(self.config)))
        (self.root / "preparation-plan.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "plan_file_changed"):
            self.load()

    def test_pose_match_rejects_missing_nonfinite_and_excessive_drift(self):
        self.assertTrue(pose_matches(self.state, self.state))
        for body in (None, [0.0] * 28, [float("nan")] * 29, [0.02] * 29):
            self.assertFalse(pose_matches(dict(self.state, body=body), self.state))
