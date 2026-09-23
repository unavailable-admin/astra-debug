"""Standalone commissioning, executor control and independent stop console."""

import argparse
import json
import select
import sys
import termios
import tty
import uuid
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .service import Client, serve


def console(client):
    """Space holds; O opens; R checks a return. Q leaves the executor alive."""
    if not sys.stdin.isatty():
        raise ValueError("console_requires_terminal")
    print(
        "SPACE: pause/hold | P: prepare from current pose | O: release/hold | R: release/return | Q: leave console",
        flush=True,
    )
    print("P/R require current scene geometry and a feasible path; R opens both hands first.", flush=True)
    # Verify connection before entering raw input mode.
    print(json.dumps(client.request("status")), flush=True)
    original = termios.tcgetattr(sys.stdin)
    console_id = uuid.uuid4().hex
    try:
        tty.setcbreak(sys.stdin.fileno())
        last_alarm = ""
        was_ready = False
        while True:
            try:
                status = client.request("console_heartbeat", console_id=console_id)
                ready = status.get("grasp_ready", False)
                if ready and not was_ready:
                    print("\nREADY: arm pose and pinch hand shape confirmed by fresh feedback", flush=True)
                was_ready = ready
            except (OSError, RuntimeError) as exc:
                print(f"Console connection lost: {exc}", flush=True)
            if not select.select([sys.stdin], [], [], 1.0)[0]:
                try:
                    alarm = client.request("status").get("operator_action", "")
                    if alarm and alarm != last_alarm:
                        print("\n" + alarm + "\a", flush=True)
                    last_alarm = alarm
                except (OSError, RuntimeError) as exc:
                    print(f"Console connection lost: {exc}", flush=True)
                continue
            key = sys.stdin.read(1).lower()
            if key == "q":
                return
            op = {" ": "pause", "p": "prepare", "o": "release", "r": "reset"}.get(key)
            if op:
                try:
                    result = client.request(op, console_id=console_id)
                    print(
                        f"{op}: {result['state']} generation={result['generation']} "
                        f"ack={result['ack_seconds']*1000:.1f}ms error={result['error']}",
                        flush=True,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    print(f"NO ACK: {exc}; use the robot's hardware stop if needed", flush=True)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)
        try:
            client.request("console_detach", console_id=console_id)
        except (OSError, RuntimeError):
            pass


def main(argv=None):
    """Keep operator recovery commands separate from task-generation commands."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=(
            "serve",
            "console",
            "status",
            "stop",
            "pause",
            "release",
            "reset",
            "prepare",
            "shutdown",
            "disarm",
            "init-config",
            "diagnose",
            "capture",
            "validate-calibration",
        ),
    )
    p.add_argument("--config", type=Path)
    p.add_argument("--socket", type=Path)
    p.add_argument("--mock", action="store_true", help="Fake hardware only; never claims a physical pass")
    p.add_argument("--output", type=Path)
    p.add_argument("--samples", type=Path)
    args = p.parse_args(argv)
    try:
        if args.command == "init-config":
            print(json.dumps(asdict(Config()), indent=2))
        elif args.command == "serve":
            if not args.config and not args.mock:
                p.error("real executor requires --config")
            serve(Config.load(args.config) if args.config else Config(), args.mock, args.socket)
        elif args.command == "console":
            console(Client(args.socket))
        elif args.command in ("capture", "diagnose", "validate-calibration"):
            from .commissioning import run

            run(args)
        else:
            result = Client(args.socket).request("pause" if args.command == "stop" else args.command)
            print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        p.exit(1, f"Robot command failed: {exc}\n")


if __name__ == "__main__":
    main()
