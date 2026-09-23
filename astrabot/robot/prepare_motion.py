"""Supervised preparation-only motion using the executor's current checked scene."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from .motion_checks import MotionChecks
from .pick_a import same_generation, token, wait_ready, write_json
from .preflight import failure_summary, inspect
from .service import Client
from .tracking import TRACKING_POLICY_VERSION


async def run(client, report):
    """Check startup, request a fresh preparation path and wait for measured READY.

    This command has no camera, scene-builder or grasp-planner dependency. A
    missing or invalid scene must be explicitly refreshed before another try.
    """
    owned = None
    try:
        initial = client.request("status")
        if initial.get("tracking_policy_version", 0) < TRACKING_POLICY_VERSION:
            raise ValueError("executor_restart_required_for_motion_checks")
        report["preflight"] = await asyncio.to_thread(inspect, client)
        if not report["preflight"]["startup_checks_passed"]:
            raise ValueError(f"startup_preflight_blocked:{failure_summary(report['preflight'])}")
        state = report["preflight"]["snapshot"]
        same_generation(state, token(initial))
        if not state.get("scene_id") or state.get("recovery_scene_valid") is not True:
            raise ValueError("scene_required:先运行 scene 加载当前场景，再按空格运行 raise")
        checks = MotionChecks(**state["motion_checks"])
        report.update(scene_id=state["scene_id"], stage="prepare", motion_checks=state["motion_checks"])
        owned = {**token(state), "generation": state["generation"] + 1}
        acknowledgement = client.request("prepare", **token(state))
        same_generation(acknowledgement, owned)
        print("只测试到预备位：接管稳定后检查当前路径，通过后抬手。空格可取消。", flush=True)
        ready = await wait_ready(client, owned, checks.prepare_motion_timeout_s, checks, require_console=True)
        report.update(stage="ready", ready=ready, hardware_ready=True)
        owned = None
    finally:
        if owned is not None:
            try:
                report["end"] = client.request("pause", **owned)
            except (OSError, RuntimeError, ValueError) as exc:
                report["cleanup_error"] = str(exc)
        report["finished_wall_time"] = time.time()


def main(argv=None):
    """Keep this explicit physical action separate from no-motion scene loading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = (args.output or Path("outputs") / f"prepare-motion-{time.time_ns()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"hardware_ready": False, "success_verified": False, "started_wall_time": time.time()}
    try:
        asyncio.run(run(Client(args.socket), report))
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- record failures after guarded cancellation
        report["error"] = f"{type(exc).__name__}:{exc}"
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "report": str(output / "report.json"),
                **{key: report.get(key) for key in ("stage", "hardware_ready", "error")},
            },
            ensure_ascii=False,
        )
    )
    if report["hardware_ready"]:
        print("READY：实测已到预备位；没有执行抓取，整轮实验前仍需 prepare。")
    return 0 if report["hardware_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
