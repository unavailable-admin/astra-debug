"""Configuration bounds, transport and actual control/planning threshold behavior."""

import json
import tempfile
import unittest
from concurrent.futures import Future
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from astrabot.robot.calibration import binding
from astrabot.robot.config import Config
from astrabot.robot.executor import Executor
from astrabot.robot.hardware import MockHardware
from astrabot.robot.model import RobotModel
from astrabot.robot.motion_checks import MotionChecks
from astrabot.robot.prepared_cache import pose_matches
from astrabot.robot.scene_bundle import SceneBundle
from astrabot.robot.trajectory import Segment


class MotionCheckTests(unittest.TestCase):
    def test_json_round_trip_partial_defaults_and_unknown_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"motion_checks": {"tracking_wait_ratio": 0.8}}))
            config = Config.load(path)
            self.assertEqual(config.motion_checks.tracking_wait_ratio, 0.8)
            self.assertEqual(config.motion_checks.escape_recheck_rad, 0.001)
            path.write_text(json.dumps(asdict(config)))
            self.assertEqual(Config.load(path).motion_checks, config.motion_checks)
            path.write_text('{"motion_checks": {"unknown": 1}}')
            with self.assertRaises(TypeError):
                Config.load(path)

    def test_invalid_values_and_unsafe_sampling_rejected(self):
        for value in (True, None, "0.5", float("nan"), float("inf"), 0, -1, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MotionChecks(tracking_wait_ratio=value)
        for changes in (
            {"planning_refreshes": 1.5},
            {"planning_refreshes": True},
            {"path_step_rad": 0.1},
            {"escape_tracking_hard_rad": 0.5},
            {"escape_recheck_rad": 0.05},
            {"escape_tracking_hard_rad": 0.004},
            {"planning_body_drift_rad": 0.02},
            {"pinch_step_m": 0.001},
            {"pinch_arrival_m": 0.0016},
            {"transfer_arrival_m": 0.0041},
            {"release_position_tolerance_m": 0.0081},
            {"pinch_arrival_m": 0},
            {"pinch_substep_m": 0.0051},
            {"tracking_resume_ratio": 1.0},
            {"tracking_resume_speed_rad_s": 0},
            {"tracking_resume_stable_s": 0.2, "tracking_wait_timeout_s": 0.1},
            {"tracking_speed_window_s": 0.1, "tracking_resume_stable_s": 0.05},
            {"tracking_resume_peak_speed_rad_s": 0.01},
            {"adaptive_brake_horizon_s": 0},
            {"adaptive_brake_horizon_s": 0.21},
            {"adaptive_brake_speed_rad_s": 0},
            {"adaptive_brake_speed_rad_s": 0.11},
            {"prepare_command_lead_rad": 0.081},
            {"prepare_arrival_timeout_s": 5.1},
            {"prepare_thigh_clearance_m": -0.001001},
            {"prepare_thigh_clearance_m": 0.005},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                MotionChecks(**changes)

    def test_signed_prepare_clearance_is_scoped_and_round_trips(self):
        for value in (-0.001, 0, 0.001):
            config = Config(motion_checks={"prepare_thigh_clearance_m": value})
            restored = Config(**json.loads(json.dumps(asdict(config))))
            self.assertEqual(restored.motion_checks.prepare_thigh_clearance_m, value)
        with self.assertRaises(ValueError):
            MotionChecks(path_step_rad=-0.001)

    def test_changed_threshold_is_bound_and_scene_cannot_hot_swap_it(self):
        config = Config()
        changed = replace(config, motion_checks=replace(config.motion_checks, tracking_wait_ratio=0.8))
        self.assertNotEqual(binding(config), binding(changed))
        self.assertEqual(binding(config), binding(Config(**asdict(config))))
        bundle = SceneBundle(Path("unused"), changed, {}, {}, Path("unused"), "")
        with self.assertRaisesRegex(ValueError, "installation_settings_changed:motion_checks"):
            bundle.check_installation(config)

    def test_model_path_sampling_uses_configuration(self):
        model = object.__new__(RobotModel)
        model.config = Config(motion_checks={"path_step_rad": 0.0005})
        model._check = Mock()
        body, start, end, hands = np.zeros(29), np.zeros(14), np.zeros(14), np.zeros(12)
        end[0] = 0.002
        model.check_path(body, start, end, hands, hands)
        self.assertEqual(model._check.call_count, 5)

    def test_cache_pose_match_honors_custom_tolerance(self):
        before = {"body": [0.0] * 29, "hands": [255.0] * 12}
        after = {"body": [0.002] * 29, "hands": [255.0] * 12}
        self.assertTrue(pose_matches(before, after))
        self.assertFalse(pose_matches(before, after, checks=MotionChecks(scene_pose_rad=0.001)))

    def executor(self, **checks):
        self.now = 10.0
        hardware = MockHardware(lambda: self.now)
        executor = Executor(hardware, Mock(), Config(motion_checks=checks), lambda: self.now)
        self.addCleanup(executor.close)
        executor.request({"op": "begin"})
        executor.weight = 1
        return executor, hardware

    def test_wait_ratio_changes_waiting_not_hard_stop(self):
        for ratio, waits in ((0.4, True), (0.8, False)):
            executor, hardware = self.executor(tracking_wait_ratio=ratio)
            q = hardware.body[15:].copy()
            executor.queue.append(Segment(q, hardware.hands.copy(), q.copy(), hardware.hands.copy(), 1, "clear_thighs"))
            hardware.body[16] -= 0.003
            self.now += 0.004
            executor.tick()
            self.assertEqual(executor.tracking_wait_started is not None, waits)
            self.assertEqual(executor.status()["motion_checks"]["tracking_wait_ratio"], ratio)
            hardware.body[16] = q[1] - 0.011
            self.now += 0.004
            executor.tick()
            self.assertEqual(executor.error, "arm_tracking_error")

    def test_general_planning_recheck_uses_config_and_reports_actual_limit(self):
        executor, hardware = self.executor(planning_recheck_rad=0.02)
        snapshot = hardware.snapshot()
        q = snapshot.body[15:].copy()
        planned = Future()
        planned.set_result([Segment(q, snapshot.hands.copy(), q.copy(), snapshot.hands.copy(), 1)])
        executor.future = (executor.generation, planned, snapshot, self.now)
        executor.planning_request = ("prepare", {})
        executor.planning_origin = snapshot
        executor.state = "PLANNING"
        hardware.body[15] += 0.025
        with patch.object(executor.pool, "submit", return_value=Future()):
            self.now += 0.004
            executor.tick()
        self.assertEqual(executor.planning_refreshes, 1)
        self.assertFalse(executor.queue)
        event = next(event for event in executor.events if event["kind"] == "planning_start_refreshed")
        self.assertEqual(event["threshold_rad"], 0.02)
        self.assertEqual(event["joint_index"], 15)

    def test_bound_scene_retains_task_body_and_hand_drift_limits(self):
        for body_drift, hand_drift in ((0.006, 0.0), (0.0, 1.1)):
            with self.subTest(body_drift=body_drift, hand_drift=hand_drift):
                executor, hardware = self.executor(scene_task_rad=0.005, scene_task_hand_raw=1.0)
                snapshot = hardware.snapshot()
                q = snapshot.body[15:].copy()
                planned = Future()
                planned.set_result([Segment(q, snapshot.hands.copy(), q.copy(), snapshot.hands.copy(), 1)])
                executor.future = (executor.generation, planned, snapshot, self.now)
                executor.scene_id = "current"
                executor.planning_request = ("move", {"expected_scene_id": "current"})
                executor.planning_origin = snapshot
                executor.state = "PLANNING"
                hardware.body[15] += body_drift
                hardware.hands[0] -= hand_drift
                with patch.object(executor.pool, "submit", return_value=Future()):
                    self.now += 0.004
                    executor.tick()
                self.assertEqual(executor.planning_refreshes, 1)
                self.assertFalse(executor.queue)
                self.assertEqual(executor.planning_last_refresh["threshold_rad"], 0.005)

    def test_recorded_one_raw_roundoff_does_not_exhaust_refresh_budget(self):
        executor, hardware = self.executor(scene_task_rad=0.005, scene_task_hand_raw=1.0)
        snapshot = hardware.snapshot()
        q = snapshot.body[15:].copy()
        planned = Future()
        planned.set_result([Segment(q, snapshot.hands.copy(), q.copy(), snapshot.hands.copy(), 1)])
        executor.future = (executor.generation, planned, snapshot, self.now)
        executor.scene_id = "current"
        executor.planning_request = ("move", {"expected_scene_id": "current"})
        executor.planning_origin = snapshot
        executor.planning_refreshes = executor.config.motion_checks.planning_refreshes
        executor.state = "PLANNING"
        hardware.body[15] += 0.0022170841693878174
        hardware.hands[0] -= 1.0000000000000284
        self.now += 0.004
        executor.tick()
        self.assertFalse(executor.error)
        self.assertTrue(executor.queue)
        self.assertEqual(executor.planning_refreshes, executor.config.motion_checks.planning_refreshes)

    def test_real_hand_drift_still_fails_when_refresh_budget_is_exhausted(self):
        executor, hardware = self.executor(scene_task_rad=0.005, scene_task_hand_raw=1.0)
        snapshot = hardware.snapshot()
        q = snapshot.body[15:].copy()
        planned = Future()
        planned.set_result([Segment(q, snapshot.hands.copy(), q.copy(), snapshot.hands.copy(), 1)])
        executor.future = (executor.generation, planned, snapshot, self.now)
        executor.scene_id = "current"
        executor.planning_request = ("move", {"expected_scene_id": "current"})
        executor.planning_origin = snapshot
        executor.planning_refreshes = executor.config.motion_checks.planning_refreshes
        executor.state = "PLANNING"
        hardware.hands[0] -= 1.1
        self.now += 0.004
        executor.tick()
        self.assertFalse(executor.queue)
        self.assertIn("planning_start_did_not_settle_after_refresh", executor.error)
        self.assertIn("hand_raw=1.100000000", executor.error)
