"""Operator-confirmed empty workspace preparation and handoff, without vision."""

import argparse
import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path

from .model import RobotModel
from .motion_checks import MotionChecks
from .pick_a import same_generation, token, wait_ready, write_json
from .preflight import failure_summary, inspect
from .recover import connected
from .service import Client
from .trajectory import Planner


def check_ownership(state, expected):
    """Allow an old latched planning error to be cleared, never another generation."""
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("cleared_action_cancelled_or_executor_restarted")


def install_cleared_environment(executor, message):
    """Discard visual geometry only for empty-hand preparation and handoff."""
    if message.get("operator_confirmed_empty") is not True:
        raise ValueError("operator_empty_workspace_confirmation_required")
    with executor.lock:
        state = executor.status()
        check_ownership(state, token(message))
        connected(state)
        if (
            state["state"] not in ("HOLD", "DISARMED", "READY")
            or state["task_active"]
            or state["planning_active"]
            or state["remaining_segments"]
            or state["takeover_active"]
            or state["handoff_pending"]
            or state["thermal_handoff_required"]
            or state["health"]
        ):
            raise ValueError("cleared_environment_requires_healthy_idle_executor")
        config = replace(
            executor.config,
            scene_measured=False,
            table_plane=None,
            table_footprint=None,
            table_observed=None,
            table_uncertainty=0.0,
            obstacles=[],
        )
    # Model construction is outside the control lock. Recheck ownership before installation.
    planner = Planner(RobotModel(config, environment_cleared=True), config)
    with executor.lock:
        current = executor.status()
        check_ownership(current, token(state))
        connected(current)
        if current["health"] or current["thermal_handoff_required"] or current["state"] != state["state"]:
            raise ValueError("cleared_environment_state_changed")
        executor.config, executor.planner = config, planner
        executor.scene_id = None
        executor.recovery_scene_valid = False
        executor.recovery_notice = ""
        executor.error = ""
        executor.generation += 1
        executor.event("operator_cleared_environment", visual_verified=False)
        return executor.status()


def stopped_receipt(path, expected):
    """Accept only a final audit record proving this generation handed off."""
    if not path:
        return None
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 1024 * 1024))
            lines = stream.read().splitlines()
        record = json.loads(lines[-1])
        state = record["status"]
        if (
            record.get("final") is True
            and all(state.get(k) == v for k, v in expected.items())
            and state.get("state") == "STOPPED"
            and state.get("attached") is False
            and state.get("weight") == 0
            and not state.get("error")
        ):
            return state
    except (OSError, ValueError, KeyError, IndexError):
        pass
    return None


async def run(client, action, report):
    """Run an explicitly confirmed action with generation-owned cancellation."""
    owned = None
    try:
        initial = client.request("status")
        if initial.get("cleared_environment_version", 0) < 1:
            raise ValueError("executor_restart_required_for_cleared_environment")
        connected(initial)
        if initial.get("health") or initial.get("thermal_handoff_required"):
            raise ValueError(initial.get("health") or "thermal_handoff_required")
        if action == "prepare":
            preflight = await asyncio.to_thread(inspect, client)
            report["preflight"] = preflight
            if not preflight["startup_checks_passed"]:
                raise ValueError(f"startup_preflight_blocked:{failure_summary(preflight)}")
            same_generation(preflight["snapshot"], token(initial))
        owned = {**token(initial), "generation": initial["generation"] + 1}
        paused = client.request("pause", **token(initial))
        check_ownership(paused, owned)
        previous = owned
        owned = {**owned, "generation": owned["generation"] + 1}
        installed = await asyncio.to_thread(
            Client(client.path, timeout=30).request,
            "clear_environment",
            operator_confirmed_empty=True,
            **previous,
        )
        same_generation(installed, owned)
        report.update(environment_cleared=True, visual_verified=False, scene_id=None)
        previous = owned
        increment = 0 if action == "shutdown" and not installed.get("attached") else 1
        owned = {**owned, "generation": owned["generation"] + increment}
        op = {"prepare": "prepare", "recover": "reset", "shutdown": "shutdown"}[action]
        ack = client.request(op, operator_confirmed_empty=True, **previous)
        same_generation(ack, owned)
        checks = MotionChecks(**installed["motion_checks"])
        report["stage"] = action
        print("已确认空手且桌子/障碍物移走；正在检查并执行动作，不拍照、不调用视觉 API。", flush=True)
        if action != "shutdown":
            final = await wait_ready(client, owned, checks.recovery_timeout_s, checks, require_console=True)
        else:
            deadline = time.monotonic() + checks.recovery_timeout_s
            while time.monotonic() < deadline:
                # The server closes after STOPPED. A disconnected socket alone is not proof of handoff.
                try:
                    final = client.request("status")
                except OSError:
                    final = stopped_receipt(initial.get("audit_path"), owned)
                    if final is None:
                        # Final audit flush follows socket teardown; allow a bounded grace period.
                        for _ in range(20):
                            await asyncio.sleep(0.1)
                            final = stopped_receipt(initial.get("audit_path"), owned)
                            if final is not None:
                                break
                    if final is None:
                        raise
                same_generation(final, owned)
                if final.get("state") == "STOPPED" and not final.get("attached") and final.get("weight") == 0:
                    break
                connected(final)
                await asyncio.sleep(0.02)
            else:
                raise TimeoutError("shutdown_wait_timeout")
        report.update(completed=True, final=final, stage="stopped" if action == "shutdown" else "ready")
        owned = None
    finally:
        if owned is not None:
            try:
                report["end"] = client.request("pause", **owned)
            except (OSError, RuntimeError, ValueError) as exc:
                report["cleanup_error"] = str(exc)
        report["finished_wall_time"] = time.time()


def main(argv=None):
    """Require an operator's physical clearance before each no-vision operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "recover", "shutdown"))
    parser.add_argument("--socket", type=Path)
    args = parser.parse_args(argv)
    print("确认手中无积木，桌子和障碍物已移出双臂完整运动范围，保持 console 开启。")
    if input("完成后在 console 按空格，再输入 CLEARED（其他输入取消）：").strip() != "CLEARED":
        return 1
    output = Path("outputs") / f"cleared-{args.action}-{time.time_ns()}"
    output.mkdir(parents=True)
    report = {"completed": False, "operator_confirmed_empty": True, "started_wall_time": time.time()}
    try:
        asyncio.run(run(Client(args.socket), args.action, report))
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- report after cancellation cleanup
        report["error"] = f"{type(exc).__name__}:{exc}"
    write_json(output / "report.json", report)
    print(json.dumps({"report": str(output / "report.json"), **{k: report.get(k) for k in ("completed", "error")}}))
    if report["completed"] and args.action != "shutdown":
        print("READY：双臂已到预备位。桌子放回并摆稳后，按空格，再运行 trial 重新拍照建模。")
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
