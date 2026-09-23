"""Read-only load test of separated DDS/CAN processes, without a command server."""

import argparse
import fcntl
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .config import Config
from .executor import Executor
from .hardware import ArmCommandEncoder, RealHardware
from .service import runtime_dir


def udp_sockets():
    """Read receive queue/drop counters only for sockets owned by this process."""
    inodes = set()
    for fd in Path("/proc/self/fd").iterdir():
        try:
            link = fd.readlink().as_posix()
        except OSError:
            continue
        if link.startswith("socket:["):
            inodes.add(link[8:-1])
    result = {}
    for name in ("udp", "udp6"):
        for line in Path(f"/proc/self/net/{name}").read_text().splitlines()[1:]:
            fields = line.split()
            if fields[9] in inodes:
                result[fields[9]] = {
                    "port": int(fields[1].split(":")[1], 16),
                    "rx_bytes": int(fields[4].split(":")[1], 16),
                    "drops": int(fields[-1]),
                }
    return result


def run(config, seconds, output, command_load=False):
    """Measure production readers under load while actuation is disabled twice."""
    output.mkdir(parents=True, exist_ok=False)
    with (runtime_dir() / "hardware.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        encoder = ArmCommandEncoder() if command_load else None
        hardware = RealHardware(config, read_only=True)
        executor = Executor(hardware, None, config)
        try:
            # Count startup failures separately from steady-state acceptance.
            startup = time.monotonic()
            while time.monotonic() - startup < 5:
                if not executor.health(hardware.snapshot()):
                    break
                time.sleep(0.02)
            initial = hardware.diagnostics()
            initial_udp = udp_sockets()
            started = previous = time.monotonic()
            gaps, joint_ages, hand_ages, mode_ages, ipc_ages = [], [], [], [], []
            faults = Counter()
            encoding_times = []
            encoded_frames = encoded_bytes = 0
            held_arms = None
            next_record = started
            with (output / "samples.jsonl").open("w") as log:
                while time.monotonic() - started < seconds:
                    now = time.monotonic()
                    gaps.append(now - previous)
                    previous = now
                    executor.tick()
                    s = hardware.snapshot()
                    fault = executor.health(s)
                    if encoder is not None and s is not None:
                        if held_arms is None:
                            held_arms = s.body[15:].copy()
                        with hardware.lock:
                            low = hardware.low
                        encoding_started = time.monotonic()
                        # The production encoder and IDL serializer run here, but
                        # this diagnostic has no arm publisher or command server.
                        payload = encoder.encode(low, held_arms, 0.5).serialize()
                        encoding_times.append(time.monotonic() - encoding_started)
                        encoded_frames += 1
                        encoded_bytes += len(payload)
                    if fault:
                        faults[fault] += 1
                    if now >= next_record:
                        d = hardware.diagnostics()
                        record = {"time": now, "fault": fault, "diagnostics": d}
                        if s is not None:
                            joint_ages.append(now - s.joint_time)
                            hand_ages.append(now - s.hand_time)
                            record.update(mode=s.mode, joint_age=joint_ages[-1], hand_age=hand_ages[-1])
                        mode_ages.append(d["mode_query"]["age"])
                        ipc_ages.append(d["hand_worker"].get("ipc_age", 0))
                        log.write(json.dumps(record, allow_nan=False) + "\n")
                        next_record = now + 0.05
                    time.sleep(max(0, 0.004 - (time.monotonic() - now)))
            final = hardware.diagnostics()
            final_udp = udp_sockets()
            mode = final["mode_query"]
            counts = {k: mode[k] - initial["mode_query"][k] for k in ("attempts", "successes", "errors")}
            drops = {
                inode: row["drops"] - initial_udp.get(inode, {}).get("drops", 0) for inode, row in final_udp.items()
            }
            result = {
                "duration": time.monotonic() - started,
                "pid": os.getpid(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "state": executor.state,
                "command_load": command_load,
                "encoded_frames": encoded_frames,
                "encoded_bytes": encoded_bytes,
                "encoding_seconds_p95": float(np.percentile(encoding_times, 95)) if encoding_times else None,
                "fault_samples": dict(faults),
                "mode_counts": counts,
                "mode_latency_p95": mode["latency_p95"],
                "max_tick_gap": max(gaps),
                "max_joint_age": max(joint_ages, default=None),
                "max_hand_age": max(hand_ages, default=None),
                "max_mode_age": max(mode_ages),
                "max_ipc_age": max(ipc_ages),
                "udp_drop_deltas": drops,
                "initial_udp": initial_udp,
                "final_udp": final_udp,
                "initial": initial,
                "final": final,
            }
            result["passed"] = (
                not faults
                and counts["attempts"] > 0
                and counts["errors"] == 0
                and not any(drops.values())
                and max(gaps) < 0.1
                and final["arm_command_writes"] == 0
                and final["hand_worker"].get("command_writes") == 0
            )
            (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            print(json.dumps(result, indent=2), flush=True)
            return result
        finally:
            executor.close()
            hardware.close()


def main():
    """Run a read-only acceptance interval and save raw samples plus a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--command-load", action="store_true", help="Also compute and serialize production arm frames in memory"
    )
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    result = run(Config.load(args.config) if args.config else Config(), args.seconds, args.output, args.command_load)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
