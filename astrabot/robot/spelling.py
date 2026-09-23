"""Operator-interruptible front-row spelling; object success is assumed, not verified."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..stereo import StereoTracker
from ..vision import AstraVision, known_glyph_mask
from .calibration import load
from .camera import capture
from .config import Config
from .model import RobotModel
from .pick_a import plan_trial, same_generation, token, write_json
from .preflight import failure_summary, inspect
from .recover import refresh_scene
from .scene_builder import discover, project, read_capture
from .scene_bundle import SceneBundle
from .scene_identification import candidate_images
from .scene_task import SceneCheckedTask
from .service import Client
from .task import Task
from .transfer import front_slots, preflight_transfer
from .trial_observer import TrialObserver
from .trial_runtime import SingleTrial


def current_target(config, directory, letter, placed, timeout):
    """Identify and triangulate only this letter; retained placements are not measured."""
    meta, left, right = read_capture(directory / "capture", config)
    geometry, transform, _ = load(config.calibration, meta["body_after"]["body"])
    candidates = discover(left)
    # Exclude known occupied slots geometrically, including repeated-letter words.
    # The slots come from commanded placements, not a new success observation.
    for item in placed:
        center = np.asarray(item["cube_center"]) + [0, 0, 0.02]
        corners = [center + [x, y, 0] for x in (-0.035, 0.035) for y in (-0.035, 0.035)]
        pixels = project(corners, geometry, transform)
        lo, hi = pixels.min(axis=0), pixels.max(axis=0)
        candidates = [
            c for c in candidates if not np.all((np.asarray(c["pixel"]) >= lo) & (np.asarray(c["pixel"]) <= hi))
        ]
    vision = AstraVision(directory / "vision", timeout=timeout, glyph_mode="real")
    builder = SimpleNamespace(left=left, candidates=candidates, capture_dir=directory / "capture")
    images = candidate_images(builder, vision.directory)
    decision = vision._call(
        "current_letter",
        (
            f"Select one upright 40 mm cube bearing letter {letter}. Inspect only this target. "
            "Candidate IDs mark eligible source glyphs; already placed cubes were excluded. "
            "Do not verify other letters, placed cubes, table edges, or overall word success. "
            "Return JSON with target_letter, target_id, target_confidence (0..1), "
            "target_verified (boolean), target_upright (boolean). If unavailable or ambiguous, say so."
        ),
        images,
    )
    known = {c["id"]: c for c in candidates}
    confidence = decision.get("target_confidence")
    if (
        decision.get("target_letter") != letter
        or decision.get("target_verified") is not True
        or decision.get("target_upright") is not True
        or type(confidence) not in (int, float)
        or not 0.9 <= confidence <= 1
        or type(decision.get("target_id")) is not int
        or decision["target_id"] not in known
    ):
        raise ValueError(f"current_letter_not_identified:{letter}")
    pixel = known[decision["target_id"]]["pixel"]
    measured = StereoTracker(geometry=geometry).locate_images(
        left,
        right,
        {"A": pixel},
        transform,
        require_common_height=False,
        glyph_mask=known_glyph_mask(left, [pixel]),
    )
    write_json(directory / "target-measurement.json", measured)
    if "A" not in measured["letters"]:
        raise ValueError(f"current_letter_depth_failed:{letter}:{measured.get('errors')}")
    record = dict(measured["letters"]["A"], workspace_policy="operator_cleared")
    return record, directory / "capture/left.jpg"


def make_plan(config, source, state, placement, placed):
    """Reuse the existing grasp planner and checks, then check the new transfer."""
    started = time.monotonic()
    model = RobotModel(config)
    parameters = dict(source.get("parameters", {}))
    parameters["obstacles"] = [*parameters.get("obstacles", []), *[p["obstacle"] for p in placed]]
    from .trial_plan import TrialPlanningError

    candidates = (
        [None]
        if any(k in parameters for k in ("transit_yaw", "contact_yaw"))
        else [0, -15, 15, -30, 30, -45, 45, -60, 60, -90, 90]
    )
    attempts = []
    for yaw in candidates:
        values = dict(parameters)
        if yaw is not None:
            values.update(transit_yaw=yaw, contact_yaw=yaw)
        bundle = SimpleNamespace(config=config, source={**source, "parameters": values})
        attempt_started = time.monotonic()
        try:
            plan = plan_trial(bundle, state, model)
            plan["placement"] = placement
            plan["transfer_preflight"] = preflight_transfer(model, config, plan)
        except ValueError as exc:
            if not isinstance(exc, TrialPlanningError) and not str(exc).startswith(
                (
                    "hand_",
                    "arm_",
                    "head_",
                    "joint_limit",
                    "ik_",
                    "IK_",
                    "trial_cube_",
                    "motion_enters_",
                    "pinch_step_too_large",
                )
            ):
                raise
            attempts.append(
                {
                    "yaw_deg": yaw,
                    "phase": "complete_transfer",
                    "error": str(exc),
                    "goal": getattr(exc, "transfer_goal", None),
                    "elapsed_s": time.monotonic() - attempt_started,
                }
            )
            continue
        plan["transfer_orientation_attempts"] = [
            *attempts,
            {"yaw_deg": yaw, "geometry_passed": True, "elapsed_s": time.monotonic() - attempt_started},
        ]
        plan["complete_planning_elapsed_s"] = time.monotonic() - started
        return plan
    raise TrialPlanningError(attempts)


async def run(args, report):
    """Run the queue under one cancellable task lease, with no success prompts."""
    task = checked = trial = None
    client = None
    owned = None
    output = args.output
    config = Config.load(args.config)
    report.update(
        word=args.word,
        success_verified=False,
        verification_source=None,
        motion_completed=False,
        placements=[],
        stage="startup",
    )
    try:
        if not args.execute:
            bundle = SceneBundle.load(args.scene)
            if bundle.source.get("target_letter", "A") != args.word[0]:
                raise ValueError("offline_scene_target_mismatch")
            slots = front_slots(bundle.config, len(args.word))
            state = {"body": bundle.source["body"], "hands": bundle.source["hands"]}
            plan = await asyncio.to_thread(
                make_plan, bundle.config, bundle.source, state, {"xy": slots[0], "height_above_table": None}, []
            )
            write_json(output / "first-transfer-plan.json", plan)
            report.update(
                stage="offline_passed",
                offline_checks_passed=True,
                slots=slots,
                offline_scope="first_letter_transfer_only",
            )
            return
        client = Client(args.socket)
        preflight = await asyncio.to_thread(inspect, client)
        report["preflight"] = preflight
        if not preflight["startup_checks_passed"]:
            raise ValueError(f"startup_preflight_blocked:{failure_summary(preflight)}")
        state = preflight["snapshot"]
        owned = token(state)
        if state.get("transfer_path_version", 0) < 1:
            raise ValueError("executor_restart_required_for_transfer")
        if (
            state.get("task_active")
            or state.get("planning_active")
            or state.get("remaining_segments")
            or not state.get("attached")
            or state.get("weight") != 1
            or np.max(np.abs(np.asarray(state["body"])[15:] - config.ready_arms))
            > min(config.motion_checks.prepare_arrival_rad, config.tracking_error)
            or np.max(np.abs(np.asarray(state["hands"]) - config.ready_hands)) > config.hand_arrival_error
        ):
            raise ValueError("prepare_required")
        report["stage"] = "first_letter_and_table"
        first_args = SimpleNamespace(
            **{
                **vars(args),
                "output": output / "first-scene",
                "target_letter": args.word[0],
                "operator_cleared_workspace": True,
            }
        )
        await refresh_scene(first_args, client, owned)
        bundle = SceneBundle.load(first_args.output / "scene")
        config = bundle.config
        model = RobotModel(config)
        slots = front_slots(config, len(args.word))
        report.update(slots=slots, scene=str(bundle.directory), hardware_ready=True)
        previous = dict(owned)
        owned["generation"] += 1
        task = Task(client, expected=previous)
        same_generation(task.status(), owned)
        checked = SceneCheckedTask(task, model, config)
        checked.executor_scene_id = bundle.digest
        height = None
        for index, letter in enumerate(args.word):
            task.status()
            directory = output / f"{index + 1:02d}-{letter}"
            directory.mkdir()
            report.update(stage="identify_current_letter", current_letter=letter)
            print(f"字母 {letter}（{index + 1}/{len(args.word)}）：定位当前目标。", flush=True)
            item = {"letter": letter, "index": index, "stage": "identify"}
            report.setdefault("letters", []).append(item)
            write_json(output / "report.json", report)
            if index == 0:
                record, identity_image = bundle.letters["A"], bundle.identity_image
            else:
                await asyncio.to_thread(capture, config, directory / "capture", client)
                task.status()
                record, identity_image = await asyncio.to_thread(
                    current_target, config, directory, letter, report["placements"], args.api_timeout
                )
                task.status()
            source = {**bundle.source, "grasp_center_m": record["grasp_center_xyz"]}
            state = task.status()
            item["stage"] = "plan_transfer"
            report["stage"] = "plan_transfer"
            print(f"字母 {letter}：检查抓取、搬运及撤回路径。", flush=True)
            planning_started = time.monotonic()
            plan = await asyncio.get_running_loop().run_in_executor(
                checked.pool,
                make_plan,
                config,
                source,
                state,
                {"xy": slots[index], "height_above_table": height},
                report["placements"],
            )
            item["planning_elapsed_s"] = time.monotonic() - planning_started
            print(f"字母 {letter}：完整路径检查完成，用时 {item['planning_elapsed_s']:.1f} 秒。", flush=True)
            write_json(directory / "transfer-plan.json", plan)
            task.status()
            observer = TrialObserver(client, model, config, {"A": record}, identity_image, directory / "before-grasp")

            async def observe(stage, observer=observer):
                task.status()
                if stage != "before_approach":
                    raise ValueError(f"unexpected_spelling_observation:{stage}")
                return await observer(stage)

            item["stage"] = "transfer"
            report["stage"] = "transfer"
            print(f"字母 {letter}：开始搬运，console 空格可随时停止。", flush=True)
            trial = SingleTrial(checked, model, config, observe)
            result = await trial.run(plan)
            height = result["height_above_table"]
            placement = {
                "letter": letter,
                "index": index,
                "cube_center": result["cube_center"],
                "obstacle": result["obstacle"],
                "assumed": True,
                "success_verified": False,
            }
            report["placements"].append(placement)
            print(f"字母 {letter}：动作完成，按目标位置记录并继续。", flush=True)
            item.update(stage="completed", result=result)
            write_json(output / "report.json", report)
            trial = None
        report.update(
            stage="completed",
            motion_completed=True,
            height_above_table=height,
            success_basis="operator_requested_assumption",
        )
    finally:
        if trial is not None:
            report["letters"][-1]["events"] = trial.events
        if checked is not None:
            report["scene_checks"] = checked.checks
        try:
            if task is not None:
                report["end"] = task.close()
            elif client is not None and owned is not None:
                report["end"] = client.request("pause", **owned)
        except (OSError, RuntimeError, ValueError) as exc:
            report["cleanup_error"] = str(exc)
        if checked is not None:
            checked.close()
        if client is not None:
            try:
                report["final"] = client.request("status")
            except (OSError, RuntimeError, ValueError) as exc:
                report["final_status_error"] = str(exc)
        write_json(output / "report.json", report)


async def spell(
    word: str,
    *,
    config: str | Path,
    execute: bool = False,
    scene: str | Path | None = None,
    socket: str | Path | None = None,
    output: str | Path | None = None,
    api_timeout: float = 120,
    table_width_mm: float = 1200,
    table_depth_mm: float = 600,
    table_thickness_mm: float = 10,
) -> dict:
    """Plan or execute a letter queue and return its persisted report.

    Args:
        word: Letters A-Z in placement order; lowercase is normalized.
        config: Robot configuration file.
        execute: Enable hardware execution using the existing console/task lease.
        scene: Saved scene required for offline first-letter planning.
        socket: Optional executor socket override.
        output: New output directory; an existing directory is never overwritten.
        api_timeout: Target identification API timeout in seconds.
        table_width_mm: Table width used by the existing scene builder.
        table_depth_mm: Table depth used by the existing scene builder.
        table_thickness_mm: Table thickness used by the existing scene builder.

    Returns:
        Report including its path, motion_completed, success_verified and error
        when execution fails. Object success is always an operator assumption.

    Raises:
        ValueError: Invalid word or missing offline scene, before any motion.
        asyncio.CancelledError: Caller cancellation, after task cleanup.
    """
    word = word.strip().upper()
    if not word or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in word):
        raise ValueError("word must contain letters A-Z")
    if not execute and scene is None:
        raise ValueError("offline mode requires scene")
    directory = (Path(output) if output is not None else Path("outputs") / f"spelling-{time.time_ns()}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(
        word=word,
        config=Path(config),
        execute=execute,
        scene=Path(scene) if scene is not None else None,
        socket=Path(socket) if socket is not None else None,
        output=directory,
        api_timeout=api_timeout,
        table_width_mm=table_width_mm,
        table_depth_mm=table_depth_mm,
        table_thickness_mm=table_thickness_mm,
    )
    report = {
        "report": str(directory / "report.json"),
        "word": word,
        "started_wall_time": time.time(),
        "mode": "execute" if execute else "offline",
        "motion_completed": False,
        "success_verified": False,
    }
    try:
        await run(args, report)
    except asyncio.CancelledError:
        report.update(error="CancelledError:caller_cancelled", motion_completed=False, success_verified=False)
        raise
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- persist failures after guarded cleanup
        report.update(error=f"{type(exc).__name__}:{exc}", motion_completed=False, success_verified=False)
    finally:
        report["finished_wall_time"] = time.time()
        write_json(directory / "report.json", report)
    return report


def main(argv=None):
    """Default to offline planning; hardware motion requires --execute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--word", default="ACE")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--api-timeout", type=float, default=120)
    parser.add_argument("--table-width-mm", type=float, default=1200)
    parser.add_argument("--table-depth-mm", type=float, default=600)
    parser.add_argument("--table-thickness-mm", type=float, default=10)
    args = parser.parse_args(argv)
    try:
        report = asyncio.run(spell(**vars(args)))
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "report": report["report"],
                "stage": report.get("stage"),
                "motion_completed": report.get("motion_completed"),
                "error": report.get("error"),
            },
            ensure_ascii=False,
        )
    )
    return 1 if report.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
