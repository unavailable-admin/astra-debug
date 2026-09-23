"""Read-only startup report from the existing executor; never takes control."""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from .service import Client


def finite(value):
    """Reject missing/boolean/nonfinite telemetry rather than inferring a pass."""
    return type(value) in (int, float) and math.isfinite(value)


def evaluate(before, state):
    """Report startup gates separately from scene, calibration and path readiness."""
    checks = []

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "实机执行器", state.get("hardware_backend") == "RealHardware", state.get("hardware_backend", "旧执行器缺少遥测")
    )
    check(
        "执行器会话",
        bool(state.get("session")) and before.get("session") == state.get("session"),
        state.get("session"),
    )
    mode, expected = state.get("mode"), state.get("standing_mode")
    check(
        "当前模式",
        type(mode) is int and type(expected) is int and mode == expected,
        {"current": mode, "required_standing": expected, "known_modes": {"0": "零力矩", "1": "阻尼", "4": "站立"}},
    )
    timeout = state.get("feedback_timeout")
    for stream in ("joint", "hand"):
        age = state.get(f"{stream}_age")
        previous, current = before.get(f"{stream}_time"), state.get(f"{stream}_time")
        check(
            f"{stream} 反馈",
            finite(timeout)
            and timeout > 0
            and finite(age)
            and 0 <= age <= timeout
            and finite(previous)
            and finite(current)
            and current > previous,
            {"age_seconds": age, "limit_seconds": timeout, "previous": previous, "current": current},
        )
    tick_age = state.get("tick_age")
    check("控制循环", finite(tick_age) and 0 <= tick_age < 0.1, {"tick_age": tick_age})
    idle = (
        state.get("task_active") is False
        and state.get("planning_active") is False
        and state.get("remaining_segments") == 0
        and state.get("takeover_active") is False
        and state.get("handoff_pending") is False
        and state.get("thermal_handoff_required") is False
    )
    detached = (
        state.get("state") == "DISARMED"
        and state.get("attached") is False
        and state.get("weight") == 0
        and state.get("arm_output") == "disabled"
    )
    held = (
        state.get("state") in ("HOLD", "READY")
        and state.get("attached") is True
        and state.get("weight") == 1
        and state.get("arm_output") in ("hold", "active")
    )
    check(
        "当前执行器控制权",
        idle and (detached or held),
        {key: state.get(key) for key in ("state", "attached", "weight", "arm_output", "disarm_zero_frames")},
    )
    channel = state.get("console_channel") or {}
    pause_age = channel.get("pause_age")
    check(
        "暂停控制台与空格回执",
        channel.get("connected") is True
        and finite(pause_age)
        and 0 <= pause_age <= 300
        and channel.get("pause_generation") == state.get("generation"),
        dict(channel, instruction="保持控制台开启；开始测试前在该控制台按空格，再运行本检查"),
    )
    limits = state.get("temperature_limits") or {}
    for channel, key in (("motor", "temperature"), ("shell", "shell_temperature"), ("hand", "hand_temperature")):
        value, limit = state.get(key), limits.get(channel)
        check(
            f"温度 {channel}",
            finite(value) and finite(limit) and limit > 0 and value < limit,
            {"reading_c": value, "limit_c": limit},
        )
    check(
        "故障",
        all(state.get(key) == "" for key in ("health", "error", "feedback_fault")),
        {key: state.get(key) for key in ("health", "error", "feedback_fault", "operator_action")},
    )
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "startup_checks_passed": all(item["passed"] for item in checks),
        "pick_ready": False,
        "scope": "只检查启动状态；场景、指尖标定和轨迹由抓取程序另行验证。不会接管、暂停或清零。",
        "ownership_scope": "控制权字段仅描述当前执行器；不能证明旧进程 SDK 目标已清零，也不检测其他 DDS 发布者。",
        "checks": checks,
        "snapshot": state,
    }


def inspect(client):
    """Use two read-only samples to detect feedback that has stopped advancing."""
    before = client.request("status")
    time.sleep(0.15)
    return evaluate(before, client.request("status"))


def failure_summary(report):
    """Explain failed startup gates without implying that Space fixes every fault."""
    state = report.get("snapshot", {})
    failed = [item for item in report.get("checks", []) if not item["passed"]]
    details = []
    for item in failed:
        if item["name"] == "当前模式":
            mode = state.get("mode")
            label = {0: "零力矩", 1: "阻尼", 4: "站立"}.get(mode, "未知")
            details.append(f"当前为{label}模式(mode={mode})，需要站立模式(mode={state.get('standing_mode')})")
        elif item["name"] == "暂停控制台与空格回执":
            details.append("请保持 console 开启，并在其中按空格")
        elif item["name"] == "故障":
            details.append("故障=" + str(item["detail"]))
        else:
            details.append(f"{item['name']}未通过")
    return "；".join(details) or "查看报告中的 preflight.checks"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--output", type=Path, help="New JSON report path; existing reports are never overwritten")
    args = parser.parse_args(argv)
    output = args.output or Path("outputs/preflight") / f"startup-{time.time_ns()}.json"
    try:
        # Reserve the report before requesting telemetry, so failures are recorded too.
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            try:
                report = inspect(Client(args.socket))
            except (OSError, RuntimeError, ValueError) as exc:
                report = {
                    "startup_checks_passed": False,
                    "pick_ready": False,
                    "error": str(exc),
                    "next_step": "确认执行器已启动，且使用相同容器、操作 UID 和 socket；不要强杀仍接管的旧执行器。",
                }
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    except OSError as exc:
        parser.exit(2, f"Cannot write report: {exc}\n")
    for item in report.get("checks", []):
        print(
            f"{'PASS' if item['passed'] else 'BLOCK'} {item['name']}: {json.dumps(item['detail'], ensure_ascii=False)}"
        )
    if report.get("error"):
        print(f"BLOCK: {report['error']}\n{report['next_step']}")
    print(f"Report: {output.resolve()}")
    print("STARTUP PASS（不代表 Pick A 已就绪）" if report["startup_checks_passed"] else "STARTUP BLOCK")
    return 0 if report["startup_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
