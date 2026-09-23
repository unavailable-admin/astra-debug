"""Stationary synchronized stereo capture with kernel timestamps and exact size."""

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .capture_state import CAPTURE_BODY_DRIFT_RAD, CAPTURE_HAND_DRIFT_RAW, stationary_capture_state
from .config import ASSETS, file_digest, vector
from .service import runtime_dir


def _check_stationary_state(before, after):
    if before["generation"] != after["generation"] or before["session"] != after["session"]:
        raise ValueError("capture_invalidated_by_operator")
    for state in (before, after):
        if state["health"] or state.get("error") or state["state"] == "FAULT":
            raise ValueError("capture_robot_feedback_unhealthy")
        if not stationary_capture_state(state):
            raise ValueError("stationary_capture_required")
        vector(state["body"], 29, "capture_body")
        vector(state["hands"], 12, "capture_hands")


def _wait_stable(config, client, initial, directory):
    """Require 300 ms of quiet feedback within a bounded, cancellable wait."""
    started = time.monotonic()
    window_start = started
    states, window = [initial], [initial]
    try:
        _check_stationary_state(initial, initial)
        while time.monotonic() - started < config.settle_timeout:
            time.sleep(min(0.05, max(0, config.settle_timeout - (time.monotonic() - started))))
            current = client.request("status")
            states.append(current)
            _check_stationary_state(initial, current)
            now = time.monotonic()
            if now - started >= config.settle_timeout:
                break
            window.append(current)
            body_span = np.max(np.ptp([s["body"] for s in window], axis=0))
            hand_span = np.max(np.ptp([s["hands"] for s in window], axis=0))
            if body_span > 0.0015 or hand_span > 1:
                window, window_start = [current], now
            elif now - window_start >= 0.3:
                return current
        raise TimeoutError("robot_did_not_settle_before_capture")
    finally:
        (directory / "capture-settle.json").write_text(
            json.dumps({"elapsed_s": time.monotonic() - started, "states": states}, indent=2) + "\n"
        )


def capture(config, directory, client=None):
    """Record one fresh side-by-side frame, optionally bound to stable robot state."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source = ASSETS / "capture.cpp"
    executable = runtime_dir() / f"capture-{file_digest(source)[:16]}"
    if not executable.exists():
        temporary = executable.with_suffix(f".{time.time_ns()}.tmp")
        subprocess.run(
            ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(temporary)],
            check=True,
            timeout=30,
        )
        temporary.replace(executable)
    before = client.request("status") if client else None
    if before is not None:
        before = _wait_stable(config, client, before, directory)
    started = time.monotonic()
    full, metadata_path = directory / "stereo.jpg", directory / "capture.json"
    command = [
        str(executable),
        config.camera_device,
        str(config.camera_width),
        str(config.camera_height),
        str(int(config.camera_fps)),
        str(full),
        str(metadata_path),
    ]
    try:
        subprocess.run(command, check=True, timeout=10, capture_output=True, text=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = exc.stderr or (
            "capture helper timed out without an error message"
            if isinstance(exc, subprocess.TimeoutExpired)
            else f"capture helper exited with code {exc.returncode} without an error message"
        )
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        failure = {
            "device": config.camera_device,
            "error": detail.strip(),
            "returncode": getattr(exc, "returncode", None),
            "command": command,
        }
        (directory / "capture-error.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise RuntimeError(
            f"camera_capture_failed:{config.camera_device}: {detail.strip()}; "
            f"details: {directory / 'capture-error.json'}; retry with a new output directory"
        ) from exc
    after = client.request("status") if client else None
    meta = json.loads(metadata_path.read_text())
    meta.update(
        device=config.camera_device,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        body_before=before,
        body_after=after,
        wall_time=time.time(),
        capture_valid=False,
    )
    # Persist the actual exposure snapshots even when validation rejects the
    # image. Rejected evidence must never become an accepted scene capture.
    metadata_path.write_text(json.dumps(meta, indent=2))
    try:
        if not started <= meta["acquired_monotonic"] <= time.monotonic():
            raise ValueError("frame_timestamp_outside_capture_window")
        if time.monotonic() - meta["acquired_monotonic"] > config.feedback_timeout:
            raise ValueError("captured_frame_stale")
        image = cv2.imread(str(full))
        if image is None or image.shape != (config.camera_height, config.camera_width, 3):
            raise ValueError("unexpected_capture_resolution")
        if before is not None:
            _check_stationary_state(before, after)
            body_delta = np.abs(np.asarray(before["body"]) - after["body"])
            hand_delta = np.abs(np.asarray(before["hands"]) - after["hands"])
            meta["body_delta_rad"] = body_delta.tolist()
            meta["hand_delta_raw"] = hand_delta.tolist()
            if np.max(body_delta) > CAPTURE_BODY_DRIFT_RAD or np.max(hand_delta) > CAPTURE_HAND_DRIFT_RAW:
                raise ValueError("robot_moved_during_capture")
    except Exception as exc:
        meta["validation_error"] = f"{type(exc).__name__}:{exc}"
        metadata_path.write_text(json.dumps(meta, indent=2))
        raise
    left, right = directory / "left.jpg", directory / "right.jpg"
    mid = config.camera_width // 2
    for path, data in ((left, image[:, :mid]), (right, image[:, mid:])):
        if not cv2.imwrite(str(path), data, [cv2.IMWRITE_JPEG_QUALITY, 98]):
            raise OSError("image_write_failed")
    meta.update(
        {
            "capture_valid": True,
            "image_sha256": {p.name: file_digest(p) for p in (full, left, right)},
            "wall_time": time.time(),
        }
    )
    metadata_path.write_text(json.dumps(meta, indent=2))
    return left, right, meta
