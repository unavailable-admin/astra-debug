"""Explicit, cancellable release and checked return without full trial planning."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from .config import Config
from .motion_checks import MotionChecks
from .pick_a import same_generation, token, write_json
from .preflight import inspect
from .scene_bundle import SceneBundle
from .service import Client
from .tracking import TRACKING_POLICY_VERSION


def connected(state):
    """The physical operator must retain the independent pause console."""
    if state.get("console_channel", {}).get("connected") is not True:
        raise ValueError("recovery_requires_connected_console")


async def wait_state(client, expected, predicate, timeout):
    """Cancellation, live faults and console loss take precedence over completion."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.request("status")
        same_generation(state, expected)
        connected(state)
        if predicate(state):
            return state
        await asyncio.sleep(0.1)
    raise TimeoutError("recovery_wait_timeout")


async def refresh_scene(args, client, expected):
    """Acquire current geometry only; never search grasp orientations."""
    from .camera import capture
    from .scene_builder import SceneBuilder
    from .scene_review import automatic_scene

    config = Config.load(args.config)
    same_generation(client.request("status"), expected)
    await asyncio.to_thread(capture, config, args.output / "capture", client)
    builder = SceneBuilder(
        args.config,
        args.output / "capture",
        args.output / "scene",
        operator_cleared_workspace=getattr(args, "operator_cleared_workspace", False),
        target_letter=getattr(args, "target_letter", "A"),
    )
    await asyncio.to_thread(
        automatic_scene,
        builder,
        {
            name: getattr(args, name, default)
            for name, default in (("table_width_mm", 1200), ("table_depth_mm", 600), ("table_thickness_mm", 10))
        },
        args.api_timeout,
    )
    bundle = SceneBundle.load(builder.output)
    bundle.check_installation(config)
    state = client.request("status")
    same_generation(state, expected)
    connected(state)
    bundle.check_pose(state)
    previous = dict(expected)
    expected["generation"] += 1
    installed = await asyncio.to_thread(
        Client(args.socket, timeout=30).request, "load_scene", directory=str(bundle.directory), **previous
    )
    same_generation(installed, expected)
    if installed.get("scene_id") != bundle.digest or installed.get("recovery_scene_valid") is not True:
        raise ValueError("recovery_scene_install_mismatch")
    return installed


async def run(args, report, client=None):
    """Only the explicit recover command invokes this physical action sequence."""
    client = client or Client(args.socket)
    owned = None
    try:
        initial = client.request("status")
        if initial.get("tracking_policy_version", 0) < TRACKING_POLICY_VERSION:
            raise ValueError("executor_restart_required_for_recovery")
        connected(initial)
        checks = MotionChecks(**initial["motion_checks"])
        if initial.get("health") or initial.get("thermal_handoff_required") or initial.get("handoff_pending"):
            raise ValueError(initial.get("health") or "recovery_handoff_pending")
        report["stage"] = "release"
        # Reserve the expected generation before sending so ambiguous replies
        # can only pause our action, never a newer operator command.
        owned = {**token(initial), "generation": initial["generation"] + 1}
        released = client.request("release", **token(initial))
        same_generation(released, owned)
        print("已取消原动作；正在松手并保持双臂。", flush=True)
        await wait_state(client, owned, lambda s: s["state"] in ("HOLD", "DISARMED"), checks.release_timeout_s)
        preflight = await asyncio.to_thread(inspect, client)
        same_generation(preflight["snapshot"], owned)
        connected(preflight["snapshot"])
        # Recovery itself cancels the old generation. It does not forge a
        # console Space receipt: that receipt is required for starting trials,
        # while explicit recovery requires a live console and owned generation.
        startup_checks = [c for c in preflight["checks"] if c["name"] != "暂停控制台与空格回执"]
        report["recovery_preflight"] = startup_checks
        if not startup_checks or not all(c["passed"] for c in startup_checks):
            raise ValueError("recovery_preflight_blocked")
        state = preflight["snapshot"]
        if not state.get("scene_id") or state.get("recovery_scene_valid") is not True:
            report["stage"] = "refresh_scene"
            print("场景已失效，正在重新拍照建模；只检查回位路径。", flush=True)
            state = await refresh_scene(args, client, owned)
            report["scene_refreshed"] = True
        else:
            report["scene_refreshed"] = False
        same_generation(state, owned)
        connected(state)
        report["scene_id"] = state["scene_id"]
        previous = owned
        owned = {**owned, "generation": owned["generation"] + 1}
        result = client.request("prepare", **previous)
        same_generation(result, owned)
        report["stage"] = "return_ready"
        print("正在检查并执行回预备位路径；控制台空格可随时取消。", flush=True)
        report["ready"] = await wait_state(
            client, owned, lambda s: s.get("grasp_ready") is True, checks.recovery_timeout_s
        )
        report.update(stage="ready", recovery_verified=True)
        owned = None
    finally:
        if owned is not None:
            try:
                report["end"] = client.request("pause", **owned)
            except (OSError, RuntimeError, ValueError) as exc:
                report["cleanup_error"] = str(exc)
        report["finished_wall_time"] = time.time()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--api-timeout", type=float, default=120)
    parser.add_argument("--operator-cleared-workspace", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.api_timeout <= 300:
        parser.error("api timeout must be between 1 and 300 seconds")
    args.output = (args.output or Path("outputs") / f"recover-{time.time_ns()}").resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"recovery_verified": False, "started_wall_time": time.time()}
    try:
        asyncio.run(run(args, report))
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- persist failure after cancellation cleanup
        report["error"] = f"{type(exc).__name__}:{exc}"
    write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "report": str(args.output / "report.json"),
                **{key: report.get(key) for key in ("stage", "recovery_verified", "scene_refreshed", "error")},
            },
            ensure_ascii=False,
        )
    )
    if report["recovery_verified"]:
        print("READY：已确认回到预备位。场景若已更新，下轮实验前请重新 prepare。")
    return 0 if report["recovery_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
