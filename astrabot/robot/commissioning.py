"""Read-only device diagnostics and measured calibration commissioning."""

import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .hardware import RealHardware, probe_hand


def run(args):
    """Diagnostic construction never sends arm/hand position commands."""
    config = Config.load(args.config) if args.config else Config()
    if args.command == "capture":
        from .camera import capture
        from .service import Client

        if not args.output:
            raise ValueError("capture requires --output new-directory")
        _, _, metadata = capture(config, args.output, Client(args.socket) if args.socket else None)
        print(json.dumps(metadata, indent=2))
        return
    if args.command == "validate-calibration":
        from .calibration import validate

        if not args.samples or not args.output:
            raise ValueError("validation requires --samples and --output")
        result = validate(config, args.samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise ValueError("calibration validation failed; motion remains gated")
        return
    result = {"motor_commands_sent": 0, "config": asdict(config), "uid": os.getuid(), "devices": {}}
    for device in ("/dev/video0", "/dev/video1"):
        result["devices"][device] = {
            "exists": Path(device).exists(),
            "read_write": os.access(device, os.R_OK | os.W_OK),
        }
    for interface in (config.left_can, config.right_can):
        proc = subprocess.run(
            ["ip", "-json", "-details", "link", "show", interface],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        result["devices"][interface] = json.loads(proc.stdout) if proc.returncode == 0 else proc.stderr
        try:
            result["devices"][interface + "_hand_identity"] = probe_hand(interface)
        except (OSError, ValueError) as exc:
            result["devices"][interface + "_error"] = str(exc)
    hardware = RealHardware(config, hands=False, read_only=True)
    try:
        samples = []
        for _ in range(40):
            time.sleep(0.1)
            state = hardware.snapshot()
            if state:
                samples.append(
                    {
                        "body": state.body.tolist(),
                        "mode": state.mode,
                        "temperature": state.temperature,
                        "fault": state.fault,
                        "joint_age": time.monotonic() - state.joint_time,
                    }
                )
        result["robot_samples"] = samples
        result["lowstate_received"] = bool(samples)
    finally:
        hardware.close()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
