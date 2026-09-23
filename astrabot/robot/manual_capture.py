"""Passive stereo/body acquisition for manually supported rigid-hand calibration.

Creates a DDS subscriber only: no command publisher, CAN driver or mode client.
This process does not stop or change any separately running robot controller.
"""

import argparse
import json
import threading
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from .camera import capture
from .config import Config, file_digest
from .state_reader import LatestStateReader


def check_trace(records, exposure, window=0.1, max_gap=0.05, max_delta=0.003):
    """Validate measured stationarity around exposure, including intermediate motion."""
    if not records or not np.isfinite(exposure):
        raise ValueError("missing_joint_trace")
    times = np.asarray([r["joint_time"] for r in records], float)
    body = np.asarray([r["body"] for r in records], float)
    if body.shape != (len(records), 29) or not np.isfinite(body).all() or not np.isfinite(times).all():
        raise ValueError("invalid_joint_trace")
    if np.any(np.diff(times) <= 0):
        raise ValueError("unordered_joint_trace")
    if times[0] > exposure - window or times[-1] < exposure + window:
        raise ValueError("exposure_not_bracketed")
    start = max(0, int(np.searchsorted(times, exposure - window)) - 1)
    end = min(len(times), int(np.searchsorted(times, exposure + window)) + 1)
    selected = records[start:end]
    gaps = np.diff(times[start:end])
    if len(gaps) < 2 or gaps.max() > max_gap:
        raise ValueError("joint_trace_gap")
    if any(r.get("joint_fault") for r in selected):
        raise ValueError("joint_feedback_fault")
    delta = float(np.ptp(body[start:end], axis=0).max())
    if delta > max_delta:
        raise ValueError("body_moved_during_exposure")
    nearest = int(np.argmin(abs(times - exposure)))
    return {
        "passed": True,
        "scope": "body_joints_only_no_finger_measurement",
        "window_seconds": window,
        "max_gap_seconds": float(gaps.max()),
        "max_joint_range_rad": delta,
        "samples_in_window": len(selected),
        "nearest_joint_time": float(times[nearest]),
        "body_at_exposure": body[nearest].tolist(),
    }


def load_body_evidence(meta, capture_path):
    """Recompute body-only evidence from its hashed raw trace, ignoring saved pass flags.

    The caller also checks camera geometry, images, device and boot/age. No finger
    positions are returned: these samples support only rigid hand-base landmarks.
    """
    if (
        meta.get("acquisition_source") != "passive_dds_body_v1"
        or meta.get("purpose") != "manual_rigid_hand_observation"
        or meta.get("robot_commands_sent") != 0
        or meta.get("fingers_observed") is not False
    ):
        raise ValueError("invalid_passive_capture_source")
    diagnostics = meta.get("joint_reader", {})
    if diagnostics.get("alive") is not True or diagnostics.get("fault") != "":
        raise ValueError("passive_capture_reader_failed")
    if set(meta.get("image_sha256", {})) != {"stereo.jpg", "left.jpg", "right.jpg"}:
        raise ValueError("passive_capture_images_missing")
    trace_info = meta["joint_trace"]
    if trace_info["path"] != "joint-trace.json":
        raise ValueError("invalid_joint_trace_path")
    path = Path(capture_path).resolve().parent / trace_info["path"]
    digest = file_digest(path)
    if digest != trace_info["sha256"]:
        raise ValueError("joint_trace_hash_mismatch")
    records = json.loads(path.read_text())
    ticks = [r.get("tick") for r in records]
    if any(type(t) is not int or not 0 <= t < 2**32 for t in ticks):
        raise ValueError("invalid_joint_trace_ticks")
    if any(not 0 < (b - a) % 2**32 < 2**31 for a, b in pairwise(ticks)):
        raise ValueError("duplicate_or_regressed_joint_trace_tick")
    if any("joint_fault" not in r for r in records):
        raise ValueError("joint_trace_fault_evidence_missing")
    result = check_trace(records, meta["acquired_monotonic"])
    return np.asarray(result["body_at_exposure"]), {str(path): digest}


class BodyObserver:
    """Keep a short raw body trace without opening a robot control interface."""

    def __init__(self, interface):
        from unitree_sdk2py.core import channel
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
            "/tmp/cdds.LOG", "stderr"
        ).replace("<Verbosity>config</Verbosity>", "<Verbosity>warning</Verbosity>")
        ChannelFactoryInitialize(0, interface)
        self.lock = threading.Lock()
        self.records = []
        self.reader = LatestStateReader(self._receive)

    def _receive(self, sample, acquired):
        record = {
            "joint_time": acquired,
            "tick": int(sample.tick),
            "body": [float(m.q) for m in sample.motor_state[:29]],
            "joint_fault": "unitree_motor_fault" if any(m.motorstate for m in sample.motor_state[12:29]) else "",
        }
        with self.lock:
            self.records.append(record)
            self.records = self.records[-10000:]

    def trace(self):
        with self.lock:
            return list(self.records)

    def close(self):
        self.reader.close()


def run(config, directory):
    """Save raw evidence even when the pose is unsuitable for calibration."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    observer = BodyObserver(config.interface)
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            records = observer.trace()
            if len(records) >= 30 and records[-1]["joint_time"] - records[0]["joint_time"] >= 0.15:
                break
            if observer.reader.diagnostics()["fault"]:
                raise ValueError(observer.reader.diagnostics()["fault"])
            time.sleep(0.01)
        else:
            raise ValueError("body_feedback_unavailable")
        _, _, meta = capture(config, directory)
        time.sleep(0.15)
        records = observer.trace()
        trace_path = directory / "joint-trace.json"
        trace_path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        diagnostics = observer.reader.diagnostics()
        meta.update(
            purpose="manual_rigid_hand_observation",
            acquisition_source="passive_dds_body_v1",
            robot_commands_sent=0,
            fingers_observed=False,
            joint_trace={"path": trace_path.name, "sha256": file_digest(trace_path)},
            joint_reader=diagnostics,
            calibration_sample_accepted=False,
        )
        try:
            if diagnostics["fault"] or not diagnostics["alive"]:
                raise ValueError("joint_reader_failed")
            meta["rigid_body_stationarity"] = check_trace(records, meta["acquired_monotonic"])
        except ValueError as exc:
            meta["rigid_body_stationarity"] = {"passed": False, "error": str(exc)}
        (directory / "capture.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
        return meta
    finally:
        observer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(Config.load(args.config), args.output)
    print(json.dumps(result["rigid_body_stationarity"], indent=2))
    if not result["rigid_body_stationarity"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
