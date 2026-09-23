"""Standalone left-hand spelling: measured stereo, bounded motion, fresh verification."""

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ..stereo import StereoTracker
from ..vision import AstraVision, identified_pixels
from .calibration import require_valid
from .camera import capture
from .config import Config
from .model import RobotModel
from .service import Client
from .task import Task


def target_positions(word, config):
    """Fixed row in measured pelvis coordinates, never Scene11 coordinates."""
    if not 1 <= len(word) <= 4 or len(set(word)) != len(word) or not word.isascii() or not word.isalpha():
        raise ValueError("one_to_four_unique_ASCII_letters_required")
    origin = np.asarray(config.target_origin)
    if abs(origin[2] - config.table_max[2] - 0.02) > 0.005:
        raise ValueError("target_centres_must_be_20mm_above_measured_table")
    return {
        letter: origin + i * config.target_spacing * np.asarray(config.target_axis) for i, letter in enumerate(word)
    }


def detached(model, state, points):
    """Require fresh observed hand geometry at least 100 mm from cube centres."""
    poses = model.poses(state["body"], state["hands"])
    boxes = np.concatenate([model.boxes(poses, side) for side in ("left", "right")])
    for point in points:
        distance = np.linalg.norm(np.maximum(np.maximum(boxes[:, 0] - point, point - boxes[:, 1]), 0), axis=1)
        if distance.min() < 0.1:
            return False
    return True


async def close_on_contact(task, config):
    """Small feedback-based increments, bounded torque, fresh opposing contacts."""
    for _ in range(120):
        state = task.status()
        contact = (
            state["tactile_age"] <= config.feedback_timeout
            and len(state["tactile"]) >= 2
            and min(state["tactile"][:2]) >= config.tactile_threshold
        )
        if contact:
            # Confirm contact over separate sensor updates, not repeated cached reads.
            stamp = time.monotonic() - state["tactile_age"]
            await asyncio.sleep(0.08)
            next_state = task.status()
            if (
                time.monotonic() - next_state["tactile_age"] > stamp + 0.02
                and next_state["tactile_age"] <= config.feedback_timeout
                and min(next_state["tactile"][:2]) >= config.tactile_threshold
            ):
                return next_state
        hands = np.asarray(state["hands"])
        difference = np.asarray(config.closed_left) - hands[:6]
        if np.max(np.abs(difference)) < 2:
            raise ValueError("closed_without_verified_opposing_contact")
        hands[:6] += np.clip(difference, -2.0, 2.0)
        await task.move(hands=hands.tolist())
    raise TimeoutError("grasp_contact_timeout")


async def run(args):
    """Run one standalone attempt; any failed gate holds and reports the cause."""
    if not args.robot_config:
        raise ValueError("--backend real requires --robot-config")
    config = Config.load(args.robot_config)
    output = args.output or Path(config.output) / f"spelling-{time.time_ns()}"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "backend": "real",
        "word": args.word.upper(),
        "success_verified": False,
        "config": asdict(config),
        "started_wall_time": time.time(),
        "observations": [],
        "motions": [],
    }
    task = None
    try:
        client = Client(args.robot_socket)
        state = client.request("status")
        if state["health"]:
            raise ValueError(state["health"])
        geometry, transform = require_valid(config, state["body"])
        targets = target_positions(args.word.upper(), config)
        model = RobotModel(config)
        tracker = StereoTracker(geometry=geometry)
        api = AstraVision(output / "api", timeout=args.api_timeout, glyph_mode="real")
        if not args.inspect_only:
            if np.max(np.abs(np.asarray(state["body"])[15:] - config.ready_arms)) > 0.03:
                raise ValueError("run_checked_prepare_before_spelling")
            task = Task(client)
            await task.wait()

        async def observe(only=None):
            if task:
                task.status()
            frame, right, meta = await asyncio.to_thread(
                capture, config, output / f'frame-{len(report["observations"]):03d}', client
            )
            decision = await api.scene_candidates(frame, report["word"])
            if task:
                task.status()  # Reject API results after an operator interruption.
            image = cv2.imread(str(frame))
            pixels = identified_pixels(decision, image.shape, report["word"])
            if only:
                pixels = {k: v for k, v in pixels.items() if k == only}
            measured = tracker.locate_images(image, cv2.imread(str(right)), pixels, transform)
            if not measured["ok"] or (only and only not in measured["letters"]):
                raise ValueError(f"stereo_observation_failed:{measured}")
            report["observations"].append({"capture": meta, "decision": decision, "measured": measured})
            (output / "report.json").write_text(json.dumps(report, indent=2))
            return measured["letters"], decision

        async def cartesian(point, holding=False, obstacles=()):
            current, _ = model.tcp(task.status()["body"], task.status()["hands"])
            point = np.asarray(point)
            # Every waypoint is preflighted from the current measured pose.
            count = max(1, int(np.ceil(np.linalg.norm(point - current) / 0.18)))
            for t in np.linspace(1 / count, 1, count):
                xyz = current + t * (point - current)
                state = await task.move(xyz=xyz.tolist(), holding=holding, obstacles=obstacles)
                report["motions"].append({"xyz": xyz.tolist(), "holding": holding, "state": state})

        for skill in range(args.max_skills + 1):
            letters, decision = await observe()
            if set(letters) != set(targets):
                raise ValueError("all_target_letters_must_be_visible_upright")
            points = {k: np.asarray(v["estimated_xyz"]) for k, v in letters.items()}
            if max(abs(v[2] - 0.02 - config.table_max[2]) for v in points.values()) > 0.008:
                raise ValueError("observed_cube_heights_disagree_with_measured_table")
            errors = {k: float(np.linalg.norm(points[k] - targets[k])) for k in targets}
            report["placement_errors_m"] = errors
            if args.inspect_only:
                report["reason"] = "inspection_complete"
                break
            if max(errors.values()) <= 0.018:
                if detached(model, task.status(), list(points.values())):
                    report["success_verified"] = True
                    report["reason"] = "fresh_stereo_upright_row_and_hands_clear"
                    break
                raise ValueError("hands_not_clear_of_completed_row")
            if skill == args.max_skills:
                raise ValueError("skill_budget_exhausted")
            if decision.get("next_action", {}).get("action") == "stop":
                raise ValueError("vision_requested_stop")
            letter = next(k for k in targets if errors[k] > 0.018)
            if any(np.linalg.norm(points[k][:2] - targets[letter][:2]) < 0.05 for k in points if k != letter):
                raise ValueError("target_occupied_by_another_cube")
            obstacles = [[(v - 0.022).tolist(), (v + 0.022).tolist()] for k, v in points.items() if k != letter]
            pickup = np.asarray(letters[letter]["grasp_center_xyz"])
            destination = targets[letter] + [0, 0, 0.015]
            clearance = max(pickup[2], destination[2]) + 0.12
            hands = np.asarray(task.status()["hands"])
            hands[:6] = config.open_left
            await task.move(hands=hands.tolist())
            await cartesian([*pickup[:2], clearance], obstacles=obstacles)
            await cartesian(pickup, obstacles=obstacles)
            await close_on_contact(task, config)
            await cartesian(pickup + [0, 0, 0.02], holding=True, obstacles=obstacles)
            lifted, _ = await observe(only=letter)
            displacement = np.asarray(lifted[letter]["estimated_xyz"]) - points[letter]
            if abs(displacement[2] - 0.02) > 0.01 or np.linalg.norm(displacement[:2]) > 0.01:
                raise ValueError("trial_lift_not_visually_verified")
            await cartesian([*pickup[:2], clearance], holding=True, obstacles=obstacles)
            await cartesian([*destination[:2], clearance], holding=True, obstacles=obstacles)
            await cartesian(destination, holding=True, obstacles=obstacles)
            hands = np.asarray(task.status()["hands"])
            hands[:6] = config.open_left
            await task.move(hands=hands.tolist())
            await cartesian([*destination[:2], clearance], obstacles=obstacles)
            await task.move(arms=config.ready_arms, hands=np.full(12, 255.0).tolist(), obstacles=obstacles)
            # Next loop always takes a new image after opening and retreating.
    except Exception as exc:  # noqa: BLE001 -- record any task failure before closing the control lease
        report["reason"] = f"{type(exc).__name__}:{exc}"
    finally:
        if task:
            task.close()
        report["finished_wall_time"] = time.time()
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {"output": str(output), "success_verified": report["success_verified"], "reason": report.get("reason")},
                indent=2,
            )
        )
    return report
