"""Bounded API retries and checks on current measured letter positions."""

import asyncio
import urllib.error

import numpy as np


async def observe_scene(api, sim, frame, word, max_attempts=10):
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    for attempt in range(max_attempts):
        await sim.check_live_status()
        try:
            return await api.scene_candidates(frame, word)
        except (urllib.error.HTTPError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (502, 503, 504):
                raise
            if attempt + 1 == max_attempts:
                raise
            delay = min(30, 2 ** (min(attempt, 4) + 1))
            print(
                "API_RETRY",
                f"next={attempt + 2}/{max_attempts}",
                f"wait={delay}s",
                type(exc).__name__,
                str(exc),
                flush=True,
            )
            await asyncio.sleep(delay)


def placement_errors(estimates, targets):
    return {
        letter: float(
            np.linalg.norm(
                np.array(value["estimated_xyz"][:2])
                - [targets[letter]["target_x"], targets[letter]["target_y"]]
            )
        )
        for letter, value in estimates.items()
        if letter in targets
    }


def select_action(decision, estimates, targets, tolerance=0.018):
    """A model decision needs current visual coordinates and code-side checks."""
    errors = placement_errors(estimates, targets)
    action = decision.get("next_action", {})
    if not isinstance(action, dict):
        return "stop", None, "invalid_action"
    if action.get("action") == "finish":
        if decision.get("row_complete") is True and all(
            errors.get(letter, float("inf")) <= tolerance for letter in targets
        ):
            return "finish", None, "visual_and_position_checks_passed"
        return "stop", None, "finish_not_supported_by_measurements"
    letter = action.get("letter")
    if action.get("action") == "pick_place" and letter in targets and letter in estimates:
        if errors[letter] <= tolerance:
            return "stop", None, "selected_letter_already_at_target"
        x, y, _ = estimates[letter]["estimated_xyz"]
        in_workspace = -0.35 <= x <= 0.25 and 1.63 <= y <= 1.96
        if not in_workspace:
            return "stop", None, "outside_skill_workspace"
        return "pick_place", letter, "validated"
    return "stop", None, "uncertain_or_invalid_scene_decision"
