"""Read-only stereo/body/finger/tactile evidence for manually placed grasps."""

import argparse
import fcntl
import json
import threading
import time
from pathlib import Path

import numpy as np

from .config import Config, file_digest, vector
from .hand_process import HandProcess
from .manual_capture import run as capture_body
from .service import runtime_dir


def check_hand_trace(records, exposure, feedback_timeout):
    """Require fresh, stationary finger feedback bracketing the exposure."""
    if not records or not np.isfinite(exposure):
        raise ValueError("missing_hand_trace")
    times = np.array([r["recorded_monotonic"] for r in records])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("unordered_hand_trace")
    if times[0] > exposure - 0.1 or times[-1] < exposure + 0.1:
        raise ValueError("hand_exposure_not_bracketed")
    start = max(0, int(np.searchsorted(times, exposure - 0.1)) - 1)
    end = min(len(times), int(np.searchsorted(times, exposure + 0.1)) + 1)
    selected = records[start:end]
    if len(selected) < 3 or np.max(np.diff(times[start:end])) > 0.08:
        raise ValueError("hand_trace_gap")
    for sample in selected:
        if sample.get("fault") or sample.get("command_writes") != 0:
            raise ValueError("hand_trace_fault_or_command_write")
        vector(sample.get("hands"), 12, "capture_hands")
        vector(sample.get("tactile"), 10, "capture_tactile")
        for key in ("hand_time", "tactile_time"):
            age = sample["recorded_monotonic"] - sample.get(key, 0)
            if not np.isfinite(age) or not 0 <= age <= feedback_timeout:
                raise ValueError("hand_or_tactile_feedback_stale")
    hands = np.array([r["hands"] for r in selected])
    span = float(np.max(np.ptp(hands, axis=0)))
    if np.any((hands < 0) | (hands > 255)) or span > 2:
        raise ValueError("fingers_moved_during_exposure")
    nearest = min(selected, key=lambda r: abs(r["recorded_monotonic"] - exposure))
    tactile = np.array([r["tactile"] for r in selected])
    return {
        "passed": True,
        "hands_at_exposure": nearest["hands"],
        "max_finger_range_raw": span,
        "tactile_at_exposure": nearest["tactile"],
        "tactile_min": tactile.min(0).tolist(),
        "tactile_max": tactile.max(0).tolist(),
        "tactile_order": "left thumb/index/middle/ring/pinky, then right",
        "tactile_units": "sensor raw values; no Newton calibration or contact-label assumption",
    }


def run(config, directory):
    """Collect evidence without creating any actuator command interface."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    # The CAN worker must not compete with a live executor. The service uses
    # this same advisory lock and cannot start while this capture is active.
    with (runtime_dir() / "hardware.lock").open("a") as ownership:
        try:
            fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("executor_active_use_existing_feedback_or_finish_handoff") from exc
        worker = HandProcess(config, read_only=True)
        stop, records = threading.Event(), []

        def observe():
            while not stop.is_set():
                packet = worker.snapshot()
                records.append(dict(packet, recorded_monotonic=time.monotonic()))
                stop.wait(0.02)

        thread = threading.Thread(target=observe, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                sample = worker.snapshot()
                now = time.monotonic()
                if (
                    not sample.get("fault")
                    and sample.get("command_writes") == 0
                    and all(
                        0 <= now - sample.get(key, 0) <= config.feedback_timeout
                        for key in ("hand_time", "tactile_time")
                    )
                ):
                    break
                time.sleep(0.02)
            else:
                raise ValueError(f"manual_grasp_hand_feedback_unavailable:{sample.get('fault', '')}")
            time.sleep(0.15)
            meta = capture_body(config, directory)
            time.sleep(0.15)
        finally:
            stop.set()
            thread.join(timeout=1)
            worker.close()
            if directory.exists():
                (directory / "hand-trace.json").write_text(json.dumps(records, indent=2) + "\n")
        trace_path = directory / "hand-trace.json"
        meta["purpose"] = "manual_grasp_observation"
        meta["hand_trace"] = {"path": trace_path.name, "sha256": file_digest(trace_path)}
        try:
            meta["hand_stationarity"] = check_hand_trace(records, meta["acquired_monotonic"], config.feedback_timeout)
        except ValueError as exc:
            meta["hand_stationarity"] = {"passed": False, "error": str(exc)}
        meta["fingers_observed"] = meta["hand_stationarity"]["passed"]
        meta["calibration_sample_accepted"] = False
        (directory / "capture.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta


def main():
    """Save one manually supported observation; never move or enable the robot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(Config.load(args.config), args.output)
    print(json.dumps({key: result[key] for key in ("rigid_body_stationarity", "hand_stationarity")}, indent=2))
    if not result["rigid_body_stationarity"]["passed"] or not result["hand_stationarity"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
