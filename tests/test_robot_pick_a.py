"""Preparation/trial safety boundaries and report semantics without a robot."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from astrabot.robot.config import Config
from astrabot.robot.executor import Executor
from astrabot.robot.hardware import MockHardware
from astrabot.robot.pick_a import outcome_checks, run, wait_ready
from astrabot.robot.scene_bundle import SceneBundle, install_scene
from astrabot.robot.tracking import TRACKING_POLICY_VERSION
from astrabot.robot.trajectory import Planner, Segment


class SceneInstallTests(unittest.TestCase):
    def setUp(self):
        self.clock = lambda: 10.0
        self.hardware = MockHardware(self.clock)
        self.executor = Executor(self.hardware, None, Config(), self.clock)
        self.addCleanup(self.executor.close)
        state = self.executor.status()
        self.bundle = SceneBundle(
            Path("scene"),
            Config(scene_measured=True),
            {"source_session": state["session"], "body": state["body"], "hands": state["hands"]},
            {"A": {"estimated_xyz": [0.5, 0.1, 0.1]}},
            Path("left.jpg"),
            "scene-digest",
        )
        self.message = {"directory": "scene", "session": state["session"], "generation": state["generation"]}

    def install(self):
        with patch.object(SceneBundle, "load", return_value=self.bundle), patch("astrabot.robot.model.RobotModel"):
            return install_scene(self.executor, self.message)

    def test_idle_scene_install_changes_geometry_without_motor_writes(self):
        state = self.install()
        self.assertEqual(self.hardware.writes, [])
        self.assertFalse(state["attached"])
        self.assertEqual(state["state"], "DISARMED")
        self.assertEqual(state["scene_id"], "scene-digest")
        self.assertEqual(state["generation"], self.message["generation"] + 1)
        self.assertEqual(len(self.executor.planner.recovery_obstacles), 1)

    def test_paused_generation_cannot_be_overwritten_by_slow_scene_load(self):
        self.executor.request({"op": "pause"})
        with self.assertRaisesRegex(ValueError, "expired_scene_install_generation"):
            self.install()
        self.assertIsNone(self.executor.scene_id)

    def test_active_task_rejects_scene_swap_without_cancelling_task(self):
        state = self.executor.request({"op": "begin"})
        self.message["generation"] = state["generation"]
        with self.assertRaisesRegex(ValueError, "healthy_idle_executor"):
            self.install()
        self.assertIsNotNone(self.executor.lease)
        self.assertEqual(self.executor.generation, state["generation"])

    def test_scene_cannot_change_speeds_or_installation(self):
        self.bundle.config = replace(self.bundle.config, joint_speed=0.1)
        with self.assertRaisesRegex(ValueError, "installation_settings_changed:joint_speed"):
            self.install()
        self.assertFalse(self.executor.config.scene_measured)

    def test_changed_capture_pose_or_session_blocks_install(self):
        self.hardware.body[15] += 0.01
        with self.assertRaisesRegex(ValueError, "capture_pose_changed"):
            self.install()
        self.hardware.body[15] -= 0.01
        self.bundle.source["source_session"] = "previous-boot"
        with self.assertRaisesRegex(ValueError, "different_executor_session"):
            self.install()

    def test_unmeasured_prepare_and_stale_prepare_never_attach(self):
        with self.assertRaisesRegex(ValueError, "prepare_requires_measured_scene"):
            self.executor.request({"op": "prepare"})
        self.assertFalse(self.executor.attached)
        self.install()
        with self.assertRaisesRegex(ValueError, "expired_task_generation"):
            self.executor.request({"op": "prepare", **{k: self.message[k] for k in ("session", "generation")}})
        self.assertFalse(self.executor.attached)

    def test_recovery_includes_target_cube_in_every_planned_segment(self):
        config = Config(scene_measured=True)
        model = Mock()
        planner = Planner(model, config)
        planner.recovery_obstacles = (self.bundle.target_obstacle(),)
        body = np.zeros(29)
        body[15:] = config.ready_arms
        planner.ready(SimpleNamespace(body=body, hands=np.asarray(config.ready_hands)))
        self.assertGreater(model.check_path.call_count, 0)
        for call in model.check_path.call_args_list:
            self.assertEqual(call.kwargs["extra_obstacles"], planner.recovery_obstacles)

    def test_interrupted_trial_requires_fresh_scene_before_return(self):
        self.install()
        begun = self.executor.request({"op": "begin"})
        owned = {key: begun[key] for key in ("session", "generation")}
        self.executor.request({"op": "end", **owned})
        self.assertFalse(self.executor.recovery_scene_valid)
        with self.assertRaisesRegex(ValueError, "prepare_requires_refreshed_scene"):
            self.executor.request({"op": "prepare"})
        state = self.executor.request({"op": "reset"})
        self.assertEqual(state["state"], "RELEASING")
        self.assertFalse(self.executor.reset_requested)
        self.assertIn("refresh scene", state["operator_action"])
        self.assertIsNone(self.executor.future)
        self.executor.request({"op": "pause"})
        self.message["generation"] = self.executor.generation
        self.assertTrue(self.install()["recovery_scene_valid"])

    def test_scene_restoration_requires_current_lease_and_measured_ready(self):
        self.install()
        begun = self.executor.request({"op": "begin"})
        restore = {
            "op": "scene_restored",
            "scene_id": self.bundle.digest,
            **{key: begun[key] for key in ("session", "generation")},
        }
        with self.assertRaisesRegex(ValueError, "verified_trial_return"):
            self.executor.request(restore)
        self.hardware.body[15:] = self.bundle.config.ready_arms
        self.hardware.hands[:] = self.bundle.config.ready_hands
        with self.assertRaisesRegex(ValueError, "verified_trial_return"):
            self.executor.request({**restore, "scene_id": "old-scene"})
        self.assertTrue(self.executor.request(restore)["recovery_scene_valid"])
        self.executor.request({**restore, "op": "end"})
        with self.assertRaisesRegex(ValueError, "expired_task_generation"):
            self.executor.request(restore)
        self.executor.request({"op": "reset"})
        self.assertTrue(self.executor.reset_requested)

    def test_shutdown_after_interruption_does_not_plan_using_stale_scene(self):
        self.install()
        self.executor.request({"op": "begin"})
        self.executor.request({"op": "shutdown"})
        self.assertTrue(self.executor.shutdown_requested)
        self.assertFalse(self.executor.reset_requested)
        self.assertIsNone(self.executor.future)
        self.executor.release_started = 9.0
        self.hardware.hands[:] = 255
        self.executor.tick()
        self.assertEqual(self.executor.state, "FAULT")
        self.assertIsNone(self.executor.future)
        self.assertTrue(self.executor.attached)


class TrialScriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_executor_rejected_before_capture_or_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                config=root / "config.json",
                output=root,
                execute=True,
                socket=None,
                load_scene_only=False,
                capture_plan=False,
            )
            args.config.write_text("{}")
            client = Mock()
            for version in (9, 10):
                for scene_only in (False, True):
                    args.load_scene_only = scene_only
                    args.execute = not scene_only
                    client.request.return_value = {"scene_protocol_version": 3, "tracking_policy_version": version}
                    with (
                        self.subTest(version=version, scene_only=scene_only),
                        patch("astrabot.robot.service.Client", return_value=client),
                        self.assertRaisesRegex(ValueError, "executor_restart_required_for_motion_checks"),
                    ):
                        await run(args, {})
            self.assertTrue(all(call.args == ("status",) for call in client.request.call_args_list))

    async def test_recorded_low_pose_can_prepare_without_client(self):
        source = json.loads((Path(__file__).parent / "fixtures/robot_round_table_low_entry.json").read_text())
        config = Config(scene_measured=True, **source["corrected_table"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(asdict(config)))
            bundle = SceneBundle(
                root / "scene",
                config,
                {"body": source["body"], "hands": source["hands"], "scene_complete": False},
                {"A": {"estimated_xyz": [0.344, 0.045, 0.039]}},
                root / "left.jpg",
                "digest",
            )
            args = SimpleNamespace(
                config=config_path,
                scene=bundle.directory,
                output=root,
                execute=False,
                load_scene_only=False,
                capture_plan=False,
                prepare_only=True,
            )
            report = {"success_verified": False}
            with (
                patch.object(SceneBundle, "load", return_value=bundle),
                patch("astrabot.robot.service.Client") as client,
                patch("astrabot.robot.pick_a.plan_trial") as trial_plan,
            ):
                await run(args, report)
                client.assert_not_called()
                trial_plan.assert_not_called()
            saved = json.loads((root / "report.json").read_text())
            self.assertEqual(saved["stage"], "offline_passed")
            self.assertTrue(saved["offline_checks_passed"])
            self.assertFalse(saved["scene_complete"])
            self.assertFalse(saved["success_verified"])
            preparation = json.loads((root / "preparation-plan.json").read_text())
            np.testing.assert_array_equal(preparation["segments"][0]["arms"], source["body"][15:])
            np.testing.assert_allclose(preparation["predicted_ready"]["body"][15:], config.ready_arms)

    async def test_from_ready_captures_fresh_scene_without_cached_plan_or_raise(self):
        config = Config()
        state = {
            "session": "s",
            "generation": 2,
            "state": "HOLD",
            "body": [0.0] * 15 + config.ready_arms,
            "hands": config.ready_hands,
            "attached": True,
            "weight": 1,
            "task_active": False,
            "planning_active": False,
            "remaining_segments": 0,
            "scene_protocol_version": 3,
            "tracking_policy_version": TRACKING_POLICY_VERSION,
            "cleared_environment_version": 1,
        }
        client = Mock()
        client.request.return_value = state
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text("{}")
            args = SimpleNamespace(
                config=config_path,
                execute=True,
                from_ready=True,
                scene=None,
                load_scene_only=False,
                capture_plan=False,
                prepare_only=False,
                socket=None,
                output=root,
                api_timeout=120,
                table_width_mm=1200,
                table_depth_mm=600,
                table_thickness_mm=10,
            )
            report = {}
            with (
                patch("astrabot.robot.service.Client", return_value=client),
                patch(
                    "astrabot.robot.preflight.inspect", return_value={"startup_checks_passed": True, "snapshot": state}
                ),
                patch("astrabot.robot.pick_a.load_prepared", side_effect=AssertionError("old cache")),
                patch("astrabot.robot.pick_a.plan_preparation", side_effect=AssertionError("raise planner")),
                patch(
                    "astrabot.robot.recover.refresh_scene",
                    new_callable=AsyncMock,
                    side_effect=ValueError("fresh capture reached"),
                ) as refresh,
            ):
                with self.assertRaisesRegex(ValueError, "fresh capture reached"):
                    await run(args, report)
                refresh.assert_awaited_once()
                self.assertTrue(report["preparation_skipped"])
                self.assertNotIn("prepare", [call.args[0] for call in client.request.call_args_list])
                state["body"] = [0.0] * 29
                refresh.reset_mock()
                with self.assertRaisesRegex(ValueError, "prepare_required"):
                    await run(args, {})
                refresh.assert_not_called()

    async def test_complete_script_sequence_and_failure_cleanup(self):
        for cleared, (fail, capture_plan, reused, scene_only) in product(
            (False, True),
            (
                (False, False, False, False),
                (True, False, False, False),
                (False, True, False, False),
                (False, False, True, False),
                (False, False, False, True),
                ("refresh_failure", False, False, False),
                ("cancel_during_refresh", False, False, False),
                ("planning_failure", False, False, False),
            ),
        ):
            with self.subTest(fail=fail, capture_plan=capture_plan), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_path = root / "config.json"
                config_path.write_text("{}")
                config = Config(scene_measured=True)
                body = [0.0] * 15 + config.ready_arms
                hands = config.ready_hands
                state = {
                    "session": "s",
                    "generation": 0,
                    "body": body,
                    "hands": hands,
                    "state": "DISARMED",
                    "scene_id": "digest" if reused else None,
                    "recovery_scene_valid": True,
                    "scene_protocol_version": 3,
                    "tracking_policy_version": TRACKING_POLICY_VERSION,
                    "attached": False,
                    "weight": 0,
                    "task_active": False,
                    "planning_active": False,
                    "remaining_segments": 0,
                    "health": "",
                    "error": "",
                    "grasp_ready": False,
                }
                operations = []

                def request(op, *, state=state, operations=operations, **values):
                    operations.append(op)
                    if "generation" in values and any(values[k] != state[k] for k in ("session", "generation")):
                        raise RuntimeError("expired_task_generation")
                    if op == "load_scene":
                        state.update(scene_id="digest", generation=state["generation"] + 1)
                    elif op == "prepare":
                        state.update(
                            generation=state["generation"] + 1, state="READY", attached=True, weight=1, grasp_ready=True
                        )
                    elif op == "begin":
                        state.update(generation=state["generation"] + 1, state="RUNNING", task_active=True)
                    elif op in ("end", "pause"):
                        state.update(generation=state["generation"] + 1, state="HOLD", task_active=False)
                    return dict(state)

                client = SimpleNamespace(request=request)
                bundle = SceneBundle(
                    root / "scene",
                    config,
                    {"source_session": "s", "body": body, "hands": hands, "scene_complete": True},
                    {"A": {"estimated_xyz": [0.4, 0.1, 0]}},
                    root / "left.jpg",
                    "digest",
                )
                bundle.source["workspace_policy"] = "operator_cleared" if cleared else "vision_checked"
                segment = Segment(np.array(body[15:]), np.array(hands), np.array(body[15:]), np.array(hands), 1)
                saved_plan = {"geometry_passed": True, "contact_yaw_deg": 0, "start_body": body, "start_hands": hands}
                fresh_body = list(body)
                fresh_body[15] += 0.001
                fresh_bundle = SceneBundle(
                    root / "after-ready/scene",
                    config,
                    {"source_session": "s", "body": fresh_body, "hands": hands, "scene_complete": True},
                    {"A": {"estimated_xyz": [0.41, 0.1, 0]}},
                    root / "fresh-left.jpg",
                    "fresh-digest",
                )
                fresh_bundle.source["workspace_policy"] = bundle.source["workspace_policy"]

                class FakeTask:
                    def __init__(self, client, expected):
                        result = client.request("begin", **expected)
                        self.expected = {k: result[k] for k in ("session", "generation")}

                    async def wait(self):
                        return request("status")

                    def status(self):
                        return request("status")

                    def close(self):
                        return request("end", **self.expected)

                class FakeTrial:
                    def __init__(self, checked, model, config, observe):
                        self.observe = observe
                        self.events = []

                    async def run(self, plan, fail=fail):
                        self.events.append({"contact_verified": True})
                        for stage in (
                            "before_approach",
                            "lifted",
                            "replaced",
                            "released",
                            "returned",
                        ):
                            if stage == "before_approach":
                                await self.observe(stage)
                                self.events.append({"visual": stage})
                            else:
                                self.events.append({"visual_skipped": stage})
                            if fail and stage == "lifted":
                                raise ValueError("injected_after_lift_failure")
                        await self.observe("result_confirmation")
                        self.events.append(
                            {"operator_result_verified": True, "stages": ["lifted", "replaced", "released", "returned"]}
                        )
                        return {"success_verified": True}

                args = SimpleNamespace(
                    config=config_path,
                    execute=not (capture_plan or scene_only),
                    load_scene_only=scene_only,
                    capture_plan=capture_plan,
                    scene=None if capture_plan or reused or scene_only else bundle.directory,
                    output=root,
                    prepare_only=False,
                    socket=None,
                    hover_timeout=120,
                    table_width_mm=1200,
                    table_depth_mm=600,
                    table_thickness_mm=10,
                    api_timeout=120,
                    operator_cleared_workspace=cleared,
                )
                report = {"success_verified": False}
                observer = AsyncMock(return_value={"center": [0.4, 0.1, 0], "scene_clear": True})

                async def refresh_at_ready(
                    refresh_args,
                    refresh_client,
                    expected,
                    *,
                    state=state,
                    operations=operations,
                    fresh_bundle=fresh_bundle,
                    fresh_body=fresh_body,
                    fail=fail,
                    cleared=cleared,
                ):
                    self.assertTrue(state["grasp_ready"])
                    self.assertEqual(refresh_args.operator_cleared_workspace, cleared)
                    self.assertEqual(state["state"], "READY")
                    self.assertNotIn("plan_trial", operations)
                    refresh_args.output.mkdir(parents=True)
                    operations.append("refresh_scene")
                    if fail == "refresh_failure":
                        raise ValueError("injected_refresh_failure")
                    installed = refresh_client.request("load_scene", **expected)
                    state.update(scene_id=fresh_bundle.digest, body=fresh_body)
                    installed = dict(state)
                    expected["generation"] = installed["generation"]
                    if fail == "cancel_during_refresh":
                        state.update(generation=state["generation"] + 1, state="HOLD")
                    return installed

                def plan_at_ready(
                    *values,
                    state=state,
                    operations=operations,
                    fresh_bundle=fresh_bundle,
                    fail=fail,
                    saved_plan=saved_plan,
                ):
                    self.assertTrue(state["grasp_ready"])
                    self.assertIn("refresh_scene", operations)
                    self.assertEqual(values[1]["body"], state["body"])
                    self.assertIs(values[0], fresh_bundle)
                    operations.append("plan_trial")
                    if fail == "planning_failure":
                        raise ValueError("injected_planning_failure")
                    return saved_plan

                with (
                    patch("astrabot.robot.service.Client", return_value=client),
                    patch(
                        "astrabot.robot.preflight.inspect",
                        return_value={"startup_checks_passed": True, "snapshot": dict(state)},
                    ),
                    patch.object(
                        SceneBundle,
                        "load",
                        side_effect=lambda directory, fresh_bundle=fresh_bundle, bundle=bundle, **kwargs: (
                            fresh_bundle if "after-ready" in str(directory) else bundle
                        ),
                    ),
                    patch("astrabot.robot.camera.capture") as capture,
                    patch(
                        "astrabot.robot.pick_a.load_prepared",
                        return_value=(bundle, [segment], {"body": body, "hands": hands}),
                    ),
                    patch(
                        "astrabot.robot.scene_builder.SceneBuilder",
                        return_value=SimpleNamespace(output=bundle.directory),
                    ) as builder_factory,
                    patch("astrabot.robot.scene_review.automatic_scene") as vision,
                    patch("astrabot.robot.pick_a.RobotModel"),
                    patch(
                        "astrabot.robot.pick_a.plan_preparation",
                        return_value=([segment], {"body": body, "hands": hands}),
                    ) as preparation_planner,
                    patch("astrabot.robot.pick_a.plan_trial", side_effect=plan_at_ready) as trial_planner,
                    patch("astrabot.robot.recover.refresh_scene", side_effect=refresh_at_ready) as refresh,
                    patch("astrabot.robot.trial_observer.TrialObserver", return_value=observer) as observer_factory,
                    patch("astrabot.robot.task.Task", FakeTask),
                    patch(
                        "astrabot.robot.scene_task.SceneCheckedTask",
                        return_value=SimpleNamespace(checks=[], close=Mock()),
                    ),
                    patch("astrabot.robot.trial_runtime.SingleTrial", FakeTrial),
                    patch(
                        "astrabot.robot.pick_a.confirm_result",
                        new_callable=AsyncMock,
                        return_value={"result_verified": True},
                    ),
                ):
                    if fail:
                        failure = "cancelled" if fail == "cancel_during_refresh" else "injected_"
                        with self.assertRaisesRegex(ValueError, failure):
                            await run(args, report)
                    else:
                        await run(args, report)
                    if reused:
                        capture.assert_not_called()
                        vision.assert_not_called()
                        preparation_planner.assert_called_once()
                        trial_planner.assert_called_once()
                        observer.assert_any_await("before_preparation")
                        self.assertFalse(report["ready_plan_reused"])
                        self.assertTrue(report["prepared_result_reused"])
                no_motion = capture_plan or scene_only
                self.assertEqual(report["workspace_policy"], bundle.source["workspace_policy"])
                if no_motion:
                    self.assertEqual(builder_factory.call_args.kwargs["operator_cleared_workspace"], cleared)
                early_failure = isinstance(fail, str)
                self.assertEqual(report["success_verified"], not fail and not no_motion)
                self.assertEqual(state["state"], "DISARMED" if no_motion else "HOLD")
                self.assertEqual(
                    operations.count("load_scene"),
                    (0 if reused else 1) + (0 if no_motion or fail == "refresh_failure" else 1),
                )
                self.assertEqual(operations.count("prepare"), 0 if no_motion else 1)
                self.assertEqual(operations.count("begin"), 0 if no_motion or early_failure else 1)
                self.assertEqual(operations.count("end"), 0 if no_motion or early_failure else 1)
                self.assertEqual(operations.count("scene_restored"), 0 if fail or no_motion else 1)
                if scene_only:
                    self.assertEqual(report["stage"], "scene_loaded_no_motion")
                    preparation_planner.assert_not_called()
                    trial_planner.assert_not_called()
                    observer.assert_not_awaited()
                    capture.assert_called_once()
                    vision.assert_called_once()
                    self.assertNotIn("move", operations)
                    self.assertNotIn("pause", operations)
                if capture_plan:
                    self.assertEqual(report["stage"], "prepared_no_motion")
                    self.assertTrue(report["offline_checks_passed"])
                    trial_planner.assert_not_called()
                    refresh.assert_not_awaited()
                    observer.assert_not_awaited()
                    self.assertNotIn("move", operations)
                    self.assertNotIn("pause", operations)
                self.assertNotIn("reset", operations)
                self.assertNotIn("release", operations)
                if not no_motion and not early_failure:
                    self.assertLess(operations.index("prepare"), operations.index("refresh_scene"))
                    self.assertLess(operations.index("refresh_scene"), operations.index("plan_trial"))
                    self.assertLess(operations.index("plan_trial"), operations.index("begin"))
                    trial_planner.assert_called_once()
                    self.assertEqual(observer_factory.call_args.args[3], fresh_bundle.letters)
                    self.assertEqual(report["scene_id"], fresh_bundle.digest)
                    self.assertFalse((root / "trial-plan-predicted.json").exists())
                if early_failure:
                    self.assertNotIn("begin", operations)
                    self.assertNotIn("move", operations)
                    if fail != "planning_failure":
                        trial_planner.assert_not_called()
                self.assertTrue((root / "report.json").exists())

    async def test_prepare_ack_is_not_ready_and_pause_cancels_wait(self):
        client = Mock()
        client.request.side_effect = [
            {"session": "s", "generation": 3, "state": "PLANNING", "grasp_ready": False},
            {"session": "s", "generation": 4, "state": "HOLD", "grasp_ready": False},
        ]
        with self.assertRaisesRegex(ValueError, "cancelled"):
            await wait_ready(client, {"session": "s", "generation": 3}, 2)

    async def test_startup_messages_distinguish_takeover_planning_and_motion(self):
        token = {"session": "s", "generation": 3}
        client = Mock()
        client.request.side_effect = [
            dict(token, planning_active=True, planning_waiting_for_stability=True),
            dict(token, planning_active=True, planning_waiting_for_stability=False),
            dict(token, remaining_segments=1),
            dict(token, grasp_ready=True),
        ]
        with patch("builtins.print") as output, patch("astrabot.robot.pick_a.asyncio.sleep", new_callable=AsyncMock):
            await wait_ready(client, token, 2)
        messages = [call.args[0] for call in output.call_args_list]
        self.assertEqual(len(messages), 3)
        self.assertIn("接管并保持", messages[0])
        self.assertIn("起点已稳定", messages[1])
        self.assertIn("开始执行", messages[2])

    async def test_offline_failure_never_creates_executor_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}")
            args = SimpleNamespace(
                config=config,
                execute=False,
                load_scene_only=False,
                capture_plan=False,
                scene=root / "scene",
                output=root,
                prepare_only=False,
            )
            report = {"success_verified": False}
            with (
                patch.object(SceneBundle, "load", side_effect=ValueError("scene_export_file_changed")),
                patch("astrabot.robot.service.Client") as client,
            ):
                with self.assertRaisesRegex(ValueError, "export_file_changed"):
                    await run(args, report)
                client.assert_not_called()
            self.assertTrue((root / "report.json").exists())
            self.assertFalse(report["success_verified"])

    def test_success_requires_release_withdrawal_and_our_final_hold(self):
        ready = {"body": [0.0] * 29, "hands": [255.0] * 12}
        final = dict(
            ready,
            session="s",
            generation=3,
            state="HOLD",
            attached=True,
            weight=1,
            task_active=False,
            planning_active=False,
            remaining_segments=0,
            health="",
            error="",
        )
        events = [{"contact_verified": True}] + [
            {"visual": stage} for stage in ("lifted", "replaced", "released", "returned")
        ]
        self.assertTrue(all(outcome_checks(events, ready, final, Config(), final).values()))
        for stage in ("lifted", "replaced", "released", "returned"):
            incomplete = [event for event in events if event.get("visual") != stage]
            self.assertFalse(all(outcome_checks(incomplete, ready, final, Config(), final).values()))
        interrupted = dict(final, generation=4)
        self.assertFalse(outcome_checks(events, ready, interrupted, Config(), final)["final_control_hold"])
        self.assertFalse(outcome_checks(events, ready, final, Config(), None)["final_control_hold"])


if __name__ == "__main__":
    unittest.main()
