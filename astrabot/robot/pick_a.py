"""Repeatable supervised Pick A with fresh scene, preparation and final reporting."""

import argparse
import asyncio
import json
import select
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .alignment import compare_models
from .calibration import binding
from .collision import enable_acceleration
from .config import BODY_NAMES, Config, vector
from .model import RobotModel
from .motion_checks import MotionChecks
from .prepared_cache import load_prepared, pose_matches, save_prepared
from .scene_bundle import SceneBundle
from .tracking import TRACKING_POLICY_VERSION
from .trajectory import Planner
from .trial_plan import TrialPlanner, failure_diagnostics


def write_json(path: Path, value: dict):
    """Persist numeric records atomically, rejecting nonfinite values."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=lambda item: item.tolist()) + "\n"
    )
    temporary.replace(path)


def token(state: dict) -> dict:
    """Bind actions to the exact session and cancellation generation."""
    return {key: state[key] for key in ("session", "generation")}


def same_generation(state: dict, expected: dict):
    """Never adopt a generation created by an operator interruption."""
    if state.get("session") == expected.get("session") and (state.get("health") or state.get("error")):
        raise ValueError(state.get("health") or state["error"])
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("pick_a_cancelled_or_executor_restarted")
    if state.get("health") or state.get("error"):
        raise ValueError(state.get("health") or state["error"])


def snapshot(state: dict) -> SimpleNamespace:
    """Validate the observed inputs to the offline planners."""
    return SimpleNamespace(body=vector(state["body"], 29, "body"), hands=vector(state["hands"], 12, "hands"))


def plan_trial(bundle: SceneBundle, state: dict, model: RobotModel) -> dict:
    """Build the full trial from this observed or explicitly predicted ready pose."""
    started = time.monotonic()
    accelerated = enable_acceleration()
    warmed = time.monotonic()
    start = snapshot(state)
    result = TrialPlanner(model, bundle.config).plan_adaptive(
        start, bundle.source["grasp_center_m"], **bundle.source.get("parameters", {})
    )
    planned = time.monotonic()
    samples = [(start.body, start.hands)]
    for phase in result["phases"]:
        body = start.body.copy()
        body[15:] = phase["arms"]
        samples.append((body, np.asarray(phase["hands"])))
    result["alignment"] = compare_models(model, samples)
    if not result["alignment"]["passed"]:
        raise ValueError("trajectory_model_alignment_failed")
    result.update(
        planning_backend="numba" if accelerated else "numpy",
        planning_timing_s={
            "backend_warmup": warmed - started,
            "path_planning": planned - warmed,
            "alignment": time.monotonic() - planned,
            "total": time.monotonic() - started,
        },
        binding=binding(bundle.config),
        scene_complete=bundle.source["scene_complete"],
        workspace_policy=bundle.source.get("workspace_policy", "vision_checked"),
        hardware_ready=False,
        execution_scope=(
            "operator_cleared_table_workspace"
            if bundle.source.get("workspace_policy") == "operator_cleared"
            else "observed_table_workspace" if bundle.config.table_observed is not None else "whole_table"
        ),
    )
    return result


def plan_preparation(bundle: SceneBundle, model: RobotModel, state=None) -> tuple[list, dict]:
    """Preflight low-entry preparation including A, then predict its endpoint."""
    enable_acceleration()
    planner = Planner(model, bundle.config)
    planner.recovery_obstacles = (bundle.target_obstacle(),)
    state = bundle.source if state is None else state
    segments = planner.ready(snapshot(state))
    ready = {key: list(state[key]) for key in ("body", "hands")}
    ready["body"][15:] = segments[-1].target.tolist()
    ready["hands"] = segments[-1].hand_target.tolist()
    return segments, ready


async def wait_ready(client, expected: dict, timeout: float, checks=None, *, require_console=False) -> dict:
    """Require measured ready feedback instead of the prepare acknowledgement."""
    checks = checks or MotionChecks()
    deadline = time.monotonic() + checks.prepare_planning_timeout_s + timeout
    planning_deadline = time.monotonic() + checks.prepare_planning_timeout_s
    motion_started = False
    reported_refreshes = 0
    reported_stability_wait = False
    reported_planning_start = False
    while time.monotonic() < deadline:
        state = client.request("status")
        same_generation(state, expected)
        if require_console and state.get("console_channel", {}).get("connected") is not True:
            raise ValueError("prepare_requires_connected_console")
        if state.get("planning_waiting_for_stability") and not reported_stability_wait:
            print("正在接管并保持双臂，可能有小幅调整；等待起点稳定后计算准备路径。", flush=True)
            reported_stability_wait = True
        if (
            state.get("planning_active")
            and not state.get("planning_waiting_for_stability")
            and not reported_planning_start
        ):
            print("接管完成、起点已稳定，正在计算准备路径。", flush=True)
            reported_planning_start = True
        refreshes = state.get("planning_refreshes", 0)
        if refreshes > reported_refreshes:
            print(f"起点姿态有变化，正在按最新反馈重新检查准备路径（第 {refreshes} 次）。", flush=True)
            detail = state.get("planning_last_refresh")
            if detail:
                print(
                    f"  {BODY_NAMES[detail['joint_index']]} 变化 {detail['drift_rad']:.6f} rad，"
                    f"重算门限 {detail['threshold_rad']:.6f} rad。",
                    flush=True,
                )
            reported_refreshes = refreshes
        if state.get("planning_active") and not motion_started and time.monotonic() >= planning_deadline:
            raise TimeoutError("preparation_planning_timeout")
        if not motion_started and state.get("remaining_segments", 0) > 0:
            motion_started = True
            deadline = time.monotonic() + timeout
            print("准备路径已通过，开始执行到预备位的动作。", flush=True)
        if state.get("grasp_ready") is True:
            return state
        await asyncio.sleep(0.1)
    raise TimeoutError("preparation_did_not_reach_ready")


async def confirm_hover(task, observation: dict, timeout: float):
    """Wait for human fingertip verification while respecting console cancellation."""
    print(f"下降前核对指尖：{observation['model_fingertip_overlay']}", flush=True)
    print(f"确认真实食指/拇指与 A 对齐后输入 VERIFIED；{timeout:g} 秒内未确认则停止并保持。", flush=True)
    if not sys.stdin.isatty():
        raise ValueError("hover_review_requires_interactive_terminal")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task.status()
        if select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip() != "VERIFIED":
                raise ValueError("hover_review_not_verified")
            task.status()
            observation["pinch_alignment_verified"] = True
            return
        await asyncio.sleep(0.1)
    raise TimeoutError("hover_review_timeout")


async def confirm_grip(task, timeout: float):
    """Wait for human fingertip verification while respecting console cancellation."""
    print(f"确认积木已稳定夹住且现场手已撤离后输入 GRIP_OK；{timeout:g} 秒内未确认则停止并保持。", flush=True)
    if not sys.stdin.isatty():
        raise ValueError("grip_review_requires_interactive_terminal")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task.status()
        if select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip() != "GRIP_OK":
                raise ValueError("grip_review_not_verified")
            task.status()
            return {"grip_verified": True}
        await asyncio.sleep(0.1)
    raise TimeoutError("grip_review_timeout")


async def confirm_result(task, timeout: float):
    """Wait for human outcome verification while respecting console cancellation."""
    print(
        f"人工确认积木已提起、放回、松手留在原位，手臂已撤回且桌面恢复后输入 RESULT_OK；{timeout:g} 秒内未确认则停止并保持。",
        flush=True,
    )
    if not sys.stdin.isatty():
        raise ValueError("result_review_requires_interactive_terminal")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task.status()
        if select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip() != "RESULT_OK":
                raise ValueError("result_review_not_verified")
            task.status()
            return {"result_verified": True}
        await asyncio.sleep(0.1)
    raise TimeoutError("result_review_timeout")


def outcome_checks(events: list, ready: dict, final: dict, config: Config, end: dict | None) -> dict:
    """Whole-trial success requires every stage and our confirmed final HOLD."""
    visual = {event.get("visual") for event in events}
    manual = {
        stage for event in events if event.get("operator_result_verified") is True for stage in event.get("stages", [])
    }
    verified = visual | manual
    auto_grip = config.trial_grip_mode == "supervised_position" and not config.trial_require_grip_confirmation
    grip_check = (
        (
            any(event.get("position_grip_closure_finished") is True for event in events)
            and any(event.get("position_grip_auto_advance") is True for event in events)
        )
        if auto_grip
        else any(
            event.get("contact_verified") is True
            or (config.trial_grip_mode == "supervised_position" and event.get("operator_grip_verified") is True)
            for event in events
        )
    )
    return {
        (
            "position_closure_completed"
            if auto_grip
            else "operator_grip" if config.trial_grip_mode == "supervised_position" else "opposing_contact"
        ): grip_check,
        "lift_verified": "lifted" in verified,
        "replacement_verified": "replaced" in verified,
        "release_verified": "released" in verified,
        "withdrawal_verified": "returned" in verified,
        "ready_pose_observed": bool(
            final.get("body") is not None
            and final.get("hands") is not None
            and np.max(np.abs(np.asarray(final["body"]) - ready["body"])) <= config.arrival_error
            and np.max(np.abs(np.asarray(final["hands"]) - ready["hands"])) <= config.hand_arrival_error
        ),
        "final_control_hold": bool(
            end
            and token(end) == token(final)
            and final.get("state") == "HOLD"
            and final.get("attached") is True
            and final.get("weight") == 1
            and final.get("task_active") is False
            and final.get("planning_active") is False
            and final.get("remaining_segments") == 0
            and not final.get("health")
            and not final.get("error")
        ),
    }


async def run(args, report: dict):
    """Run one selected mode; offline mode never creates an executor client."""
    client = task = trial = checked = None
    owned = ready = bundle = None
    reused = False
    report_path = args.output / "report.json"
    try:
        config = Config.load(args.config)
        report["motion_checks"] = asdict(config.motion_checks)
        expected_policy = "operator_cleared" if getattr(args, "operator_cleared_workspace", False) else "vision_checked"
        report["workspace_policy"] = expected_policy
        if args.execute or args.load_scene_only or args.capture_plan:
            from .preflight import failure_summary, inspect
            from .service import Client

            client = Client(args.socket)
            executor_status = client.request("status")
            if executor_status.get("scene_protocol_version") != 3:
                raise ValueError("executor_restart_required_for_scene_loading:先按文档安全交接再启动新版执行器")
            # Scene bindings include the v11 head geometry. An old process
            # cannot validate those exports, even for a no-motion scene load.
            if executor_status.get("tracking_policy_version", 0) < TRACKING_POLICY_VERSION:
                raise ValueError(
                    "executor_restart_required_for_motion_checks:新版头部几何及执行策略需正常交接后重启执行器"
                )
            report["preflight"] = await asyncio.to_thread(inspect, client)
            if not report["preflight"]["startup_checks_passed"]:
                raise ValueError(f"startup_preflight_blocked:{failure_summary(report['preflight'])}")
            initial_token = token(report["preflight"]["snapshot"])
        if getattr(args, "from_ready", False):
            ready = report["preflight"]["snapshot"]
            if (
                ready.get("cleared_environment_version", 0) < 1
                or not ready.get("attached")
                or ready.get("weight") != 1
                or ready.get("task_active")
                or ready.get("planning_active")
                or ready.get("remaining_segments")
                or np.max(np.abs(np.asarray(ready["body"])[15:] - config.ready_arms))
                > min(config.motion_checks.prepare_arrival_rad, config.tracking_error)
                or np.max(np.abs(np.asarray(ready["hands"]) - config.ready_hands)) > config.hand_arrival_error
            ):
                raise ValueError("prepare_required:先移走桌子并运行 prepare 抬手到位")
            owned = token(ready)
            checks = config.motion_checks
            report.update(ready=ready, hardware_ready=True, scene=None, preparation_skipped=True)
        else:
            if args.execute and not args.scene:
                bundle, segments, predicted_ready = load_prepared(args.config, config, report["preflight"]["snapshot"])
                directory = bundle.directory
                reused = True
                report["prepared_result_reused"] = True
                print("复用准备场景与抬手路径；抓取场景和路径将在实测抬手到位后重新生成。", flush=True)
            elif args.scene:
                directory = args.scene
            else:
                from .camera import capture
                from .scene_builder import SceneBuilder
                from .scene_review import automatic_scene

                report["stage"] = "capture_and_scene"
                write_json(report_path, report)
                await asyncio.to_thread(capture, config, args.output / "capture", client)
                builder = SceneBuilder(
                    args.config,
                    args.output / "capture",
                    args.output / "scene",
                    operator_cleared_workspace=getattr(args, "operator_cleared_workspace", False),
                )
                await asyncio.to_thread(
                    automatic_scene,
                    builder,
                    {
                        "table_width_mm": args.table_width_mm,
                        "table_depth_mm": args.table_depth_mm,
                        "table_thickness_mm": args.table_thickness_mm,
                    },
                    args.api_timeout,
                )
                directory = builder.output
            if bundle is None:
                bundle = SceneBundle.load(
                    directory, require_complete=args.execute or args.load_scene_only or args.capture_plan
                )
            bundle.check_installation(config)
            if bundle.source.get("workspace_policy", "vision_checked") != expected_policy:
                raise ValueError("workspace_policy_changed:按当前实验模式重新 prepare")
            report.update(
                scene=str(bundle.directory),
                scene_id=bundle.digest,
                scene_complete=bundle.source["scene_complete"],
                execution_scope=bundle.source.get("execution_scope"),
                workspace_policy=expected_policy,
            )
            model = RobotModel(bundle.config)
            if client is not None:
                state = client.request("status")
                same_generation(state, initial_token)
                if not reused:
                    bundle.check_pose(state)
            if client is not None and not args.capture_plan and not (args.load_scene_only and not args.scene):
                from .trial_observer import TrialObserver

                observer = TrialObserver(
                    client, model, bundle.config, bundle.letters, bundle.identity_image, args.output / "observations"
                )
                report["stage"] = "verify_prepared_scene"
                write_json(report_path, report)
                observation = await observer("before_preparation")
                if (
                    observation.get("scene_clear") is not True
                    or np.linalg.norm(np.asarray(observation["center"]) - bundle.letters["A"]["estimated_xyz"])
                    > bundle.config.motion_checks.target_observation_m
                ):
                    raise ValueError("scene_changed_before_preparation")
                report["before_preparation"] = observation
            if not args.load_scene_only:
                report["stage"] = "offline_preparation"
                write_json(report_path, report)
                try:
                    if not reused:
                        segments, predicted_ready = await asyncio.to_thread(plan_preparation, bundle, model)
                    elif not pose_matches(state, bundle.source, checks=bundle.config.motion_checks):
                        print("当前双臂姿态已变化，只重算到预备位的路径。", flush=True)
                        segments, predicted_ready = await asyncio.to_thread(plan_preparation, bundle, model, state)
                except (ValueError, RuntimeError) as exc:
                    report["planning_failure"] = failure_diagnostics(
                        model,
                        {key: bundle.source[key] for key in ("body", "hands")},
                        exc,
                        report["stage"],
                        report["scene_complete"],
                    )
                    raise
                write_json(
                    args.output / "preparation-plan.json",
                    {
                        "segments": [asdict(segment) for segment in segments],
                        "predicted_ready": predicted_ready,
                        "scene_id": bundle.digest,
                        "hardware_ready": False,
                    },
                )
                if not args.prepare_only and not args.execute and not args.capture_plan:
                    report["stage"] = "offline_trial"
                    write_json(report_path, report)
                    print("离线预览：正在检查预测预备位的抓取路径；实机不会复用此规划。", flush=True)
                    trial_plan = await asyncio.to_thread(plan_trial, bundle, predicted_ready, model)
                    write_json(args.output / "trial-plan-predicted.json", trial_plan)
                    report["selected_contact_yaw_deg"] = trial_plan["contact_yaw_deg"]
                report.update(stage="offline_passed", offline_checks_passed=True)
                if not args.execute and not args.capture_plan:
                    return
            same_generation(client.request("status"), initial_token)
            report["stage"] = "install_scene"
            write_json(report_path, report)
            from .service import Client

            if reused:
                installed = client.request("status")
                same_generation(installed, initial_token)
                if installed.get("recovery_scene_valid") is not True:
                    raise ValueError("prepare_required:场景已失效，请重新 prepare")
            else:
                installed = await asyncio.to_thread(
                    Client(args.socket, timeout=30).request,
                    "load_scene",
                    directory=str(bundle.directory),
                    **initial_token,
                )
            owned = token(installed)
            if installed.get("scene_id") != bundle.digest:
                raise ValueError("executor_scene_install_mismatch")
            report["scene_installed"] = installed
            if args.load_scene_only or args.capture_plan:
                report["stage"] = "prepared_no_motion" if args.capture_plan else "scene_loaded_no_motion"
                if args.capture_plan and not args.prepare_only:
                    save_prepared(args.config, args.output, bundle)
                owned = None
                return
            report["stage"] = "prepare"
            write_json(report_path, report)
            previous = owned
            owned = {**owned, "generation": owned["generation"] + 1}
            acknowledgement = client.request("prepare", **previous)
            print("准备命令已受理；先接管保持，再检查路径。路径通过后开始向预备位移动。", flush=True)
            if token(acknowledgement) != owned:
                raise ValueError("prepare_generation_mismatch")
            checks = bundle.config.motion_checks
            ready = await wait_ready(
                client,
                owned,
                max(
                    checks.prepare_motion_timeout_s,
                    sum(segment.duration for segment in segments) + checks.prepare_motion_extra_s,
                ),
                checks,
            )
            report.update(ready=ready, hardware_ready=True)
            if args.prepare_only:
                report.update(stage="ready", preparation_verified=True)
                owned = None
                return
        # Raising can change the camera/body frame. Refresh and install the
        # entire scene before planning, rather than reusing low-pose geometry.
        from .recover import refresh_scene
        from .trial_observer import TrialObserver

        report["stage"] = "refresh_scene_after_ready"
        write_json(report_path, report)
        print("手臂和手型已到位；正在重新拍照建模，随后仅从当前实测姿态规划抓取。", flush=True)
        refresh_args = SimpleNamespace(
            config=args.config,
            socket=args.socket,
            output=args.output / "after-ready",
            api_timeout=args.api_timeout,
            table_width_mm=args.table_width_mm,
            table_depth_mm=args.table_depth_mm,
            table_thickness_mm=args.table_thickness_mm,
            operator_cleared_workspace=getattr(args, "operator_cleared_workspace", False),
        )
        refresh_started = time.monotonic()
        refreshed = await refresh_scene(refresh_args, client, owned)
        report["scene_refresh_elapsed_s"] = time.monotonic() - refresh_started
        same_generation(refreshed, owned)
        bundle = SceneBundle.load(refresh_args.output / "scene")
        bundle.check_installation(config)
        if bundle.source.get("workspace_policy", "vision_checked") != expected_policy:
            raise ValueError("refreshed_workspace_policy_mismatch")
        if refreshed.get("scene_id") != bundle.digest:
            raise ValueError("executor_scene_install_mismatch")
        ready = client.request("status")
        same_generation(ready, owned)
        bundle.check_pose(ready)
        model = RobotModel(bundle.config)
        observer = TrialObserver(
            client, model, bundle.config, bundle.letters, bundle.identity_image, args.output / "ready-observations"
        )
        report.update(
            stage="plan_from_measured_ready",
            preparation_scene=report["scene"],
            scene=str(bundle.directory),
            scene_id=bundle.digest,
            scene_complete=bundle.source["scene_complete"],
            execution_scope=bundle.source.get("execution_scope"),
            grasp_planning_start=ready,
            ready_plan_reused=False,
        )
        write_json(report_path, report)
        print("新场景已就绪；正在规划完整抓取路径（含抬起、放回和退路）。", flush=True)
        planning_started = time.monotonic()
        trial_plan = await asyncio.to_thread(plan_trial, bundle, ready, model)
        report["grasp_planning_elapsed_s"] = time.monotonic() - planning_started
        print(f"抓取规划完成，用时 {report['grasp_planning_elapsed_s']:.1f} 秒；接下来复查并执行动作。", flush=True)
        report["selected_contact_yaw_deg"] = trial_plan["contact_yaw_deg"]
        write_json(args.output / "trial-plan.json", trial_plan)
        current = client.request("status")
        same_generation(current, owned)
        if (
            np.max(np.abs(np.asarray(current["body"]) - ready["body"])) > checks.scene_pose_rad
            or np.max(np.abs(np.asarray(current["hands"]) - ready["hands"])) > checks.scene_hand_raw
        ):
            raise ValueError("ready_pose_changed_during_planning")
        # Legacy cache is only relevant to the original prepare-in-trial route.
        # Every subsequent trial still builds a fresh grasp plan after READY.
        if not getattr(args, "from_ready", False):
            ready_segments, predicted_ready = await asyncio.to_thread(plan_preparation, bundle, model, ready)
            same_generation(client.request("status"), owned)
            write_json(
                refresh_args.output / "preparation-plan.json",
                {
                    "segments": [asdict(segment) for segment in ready_segments],
                    "predicted_ready": predicted_ready,
                    "scene_id": bundle.digest,
                    "hardware_ready": False,
                },
            )
            save_prepared(args.config, refresh_args.output, bundle)
        from .scene_task import SceneCheckedTask
        from .task import Task
        from .trial_runtime import SingleTrial

        previous = owned
        owned = {**owned, "generation": owned["generation"] + 1}
        task = Task(client, expected=previous)
        await task.wait()
        checked = SceneCheckedTask(task, model, bundle.config)
        checked.executor_scene_id = bundle.digest

        async def observe(stage):
            report["stage"] = stage
            write_json(report_path, report)
            if stage == "grip_confirmation":
                return await confirm_grip(task, args.hover_timeout)
            if stage == "result_confirmation":
                report["motion_completed"] = True
                report["verification_source"] = "operator"
                write_json(report_path, report)
                return await confirm_result(task, args.hover_timeout)
            result = await observer(stage)
            return result

        report["stage"] = "trial"
        trial = SingleTrial(checked, model, bundle.config, observe)
        report["result"] = await trial.run(trial_plan)
        report["scene_restored"] = client.request("scene_restored", scene_id=bundle.digest, **owned)
        report["stage"] = "finish_hold"
    finally:
        if trial is not None:
            report["events"] = trial.events
        if checked is not None:
            report["scene_checks"] = checked.checks
        try:
            if task is not None:
                report["end"] = task.close()
            elif owned is not None:
                report["end"] = client.request("pause", **owned)
        except (OSError, ValueError, RuntimeError) as exc:
            report["cleanup_error"] = str(exc)
        if checked is not None:
            checked.close()
        if client is not None:
            try:
                report["final"] = client.request("status")
            except (OSError, ValueError, RuntimeError) as exc:
                report["final_status_error"] = str(exc)
        if report.get("stage") == "finish_hold" and ready is not None:
            report["checks"] = outcome_checks(
                report.get("events", []), ready, report.get("final", {}), bundle.config, report.get("end")
            )
            report["success_verified"] = all(report["checks"].values()) and not report.get("cleanup_error")
            if not report["success_verified"]:
                report["error"] = "trial_final_verification_failed"
        report["finished_wall_time"] = time.time()
        write_json(report_path, report)


def main(argv=None):
    """Default to offline checks; physical preparation/trial requires --execute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", type=Path, help="Existing export; execution checks fresh observations")
    parser.add_argument("--output", type=Path, help="New directory; defaults to outputs/pick-a-TIME")
    parser.add_argument("--socket", type=Path)
    parser.add_argument(
        "--from-ready",
        action="store_true",
        help="Use already raised arms; capture a fresh scene without preparation motion",
    )
    parser.add_argument("--execute", action="store_true", help="Prepare and run one supervised physical trial")
    parser.add_argument(
        "--operator-cleared-workspace",
        action="store_true",
        help="Operator keeps non-target objects out of the path; vision verifies A and table geometry only",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true", help="Only plan/execute preparation")
    modes.add_argument("--load-scene-only", action="store_true", help="Update idle executor scene without motion")
    modes.add_argument(
        "--capture-plan", action="store_true", help="Check startup, capture, build/plan and load scene without motion"
    )
    parser.add_argument("--hover-timeout", type=float, default=120)
    parser.add_argument("--api-timeout", type=float, default=120)
    parser.add_argument("--table-width-mm", type=float, default=1200)
    parser.add_argument("--table-depth-mm", type=float, default=600)
    parser.add_argument("--table-thickness-mm", type=float, default=10)
    args = parser.parse_args(argv)
    if args.from_ready and (not args.execute or args.scene or args.prepare_only):
        parser.error("--from-ready requires --execute without --scene or --prepare-only")
    if not args.scene and not args.execute and not args.capture_plan and not args.load_scene_only:
        parser.error("离线检查或仅加载场景需要 --scene；新拍照并执行试验需要 --execute")
    if (args.load_scene_only or args.capture_plan) and args.execute:
        parser.error("仅加载场景/拍照规划模式不与 --execute 同用")
    if args.capture_plan and args.scene:
        parser.error("--capture-plan 会新拍照定位，不接受旧 --scene")
    if args.execute and not args.prepare_only and not sys.stdin.isatty():
        parser.error("实机试验需要交互终端完成下降前指尖确认")
    if not 1 <= args.api_timeout <= 300 or not 1 <= args.hover_timeout <= 300:
        parser.error("timeout must be between 1 and 300 seconds")
    from .scene_identification import validate_dimensions

    validate_dimensions(args.table_width_mm, args.table_depth_mm, args.table_thickness_mm)
    args.output = (args.output or Path("outputs") / f"pick-a-{time.time_ns()}").resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "success_verified": False,
        "hardware_ready": False,
        "started_wall_time": time.time(),
        "mode": (
            "load_scene"
            if args.load_scene_only
            else "capture_plan" if args.capture_plan else "execute" if args.execute else "offline"
        ),
    }
    try:
        asyncio.run(run(args, report))
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- persist any trial failure after guarded cleanup
        report["error"] = f"{type(exc).__name__}:{exc}"
        if hasattr(exc, "orientation_attempts"):
            report["planning_failure"] = {
                "orientation_attempts": exc.orientation_attempts,
                "next_step": (
                    "试抬积木底面检查失败：核对 A 顶面高度与桌面拟合；场景高度通过不等于积木底面通过，勿强行试抬。"
                    if any("trial_cube_below_table" in a["error"] for a in exc.orientation_attempts)
                    else (
                        "接近路径 IK 未通过：核对 A 的位置与可达范围；可离线验证更靠近机器人的摆放位置，勿放宽 IK 门限。"
                        if all(
                            a["phase"] == "approach_compact" and a["error"].startswith("IK_")
                            for a in exc.orientation_attempts
                        )
                        else "检查各朝向失败阶段；闭指桌面间隙失败需核对手型与桌面，不能仅按最后一个 IK 错误处理。"
                    )
                ),
            }
        report["success_verified"] = False
        write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "report": str(args.output / "report.json"),
                "scene": report.get("scene"),
                "scene_complete": report.get("scene_complete"),
                "workspace_policy": report.get("workspace_policy"),
                "execution_scope": report.get("execution_scope"),
                "selected_contact_yaw_deg": report.get("selected_contact_yaw_deg"),
                "offline_checks_passed": report.get("offline_checks_passed", False),
                "stage": report.get("stage"),
                "hardware_ready": report.get("hardware_ready", False),
                "success_verified": report["success_verified"],
                "motion_completed": report.get("motion_completed", False),
                "verification_source": report.get("verification_source"),
                "error": report.get("error"),
                "next_step": report.get("planning_failure", {}).get("next_step"),
            },
            ensure_ascii=False,
        )
    )
    if report.get("stage") == "prepared_no_motion" and not report.get("error"):
        print("PREPARED：准备场景和抬手路径已检查并加载，尚未运动。trial 将抬手到位后重新建模并规划抓取。")
    if report.get("stage") == "scene_loaded_no_motion" and not report.get("error"):
        print("SCENE_READY：场景已加载，尚未计算或执行路径。只测抬手用 raise；整轮抓取仍需 prepare。")
    return 1 if report.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
