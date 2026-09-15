"""Resumable, cold-reset ACE speed benchmark. No object ground truth is used."""

import argparse
import csv
import hashlib
import json
import os
import random
import signal
import statistics
import subprocess
import sys
from pathlib import Path

from .paths import DEFAULT_URI, ROOT

PACKAGE = Path(__file__).parent
SPEEDS = [1.2, 1.5, 2.0, 2.5, 3.0]


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def classify(report):
    if report.get("success_verified"):
        return "success"
    error = report.get("error", "").lower()
    if any(
        s in error
        for s in ["timeout", "connection", "websocket", "http", "api", "worker", "scene", "status"]
    ):
        return "infrastructure"
    if error:
        return "motion_or_runtime_error"
    if report.get("skills_completed", 0) == 0:
        return "perception_or_decision_stop"
    return "placement_or_perception_failure"


def aggregate(out):
    records = [json.loads(p.read_text()) for p in sorted(out.glob("trial_*/result.json"))]
    rows = []
    for speed in SPEEDS:
        group = [r for r in records if r["speed"] == speed]
        passed = [r["simulation_seconds"] for r in group if r["success"]]
        rows.append(
            dict(
                speed=speed,
                completed=len(group),
                successes=len(passed),
                success_rate=len(passed) / len(group) if group else None,
                successful_sim_seconds_median=statistics.median(passed) if passed else None,
                successful_sim_seconds_min=min(passed) if passed else None,
                successful_sim_seconds_max=max(passed) if passed else None,
                failures={
                    c: sum(r["category"] == c for r in group)
                    for c in sorted({r["category"] for r in group if not r["success"]})
                },
            )
        )
    write_json(
        out / "summary.json",
        {"completed": len(records), "planned": 50, "groups": rows, "trials": records},
    )
    if records:
        with (out / "trials.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return rows


def command(cmd, log, timeout):
    with log.open("w") as stream:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return 124


def main(argv=None):
    parser = argparse.ArgumentParser(prog="astrabot benchmark", description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uri", default=DEFAULT_URI)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent controllers when resuming the same batch.
    import fcntl

    lock = (out / "batch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = sorted(PACKAGE.rglob("*.py")) + sorted((PACKAGE / "config").glob("*"))
    hashes = {
        str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_hashes"] != hashes or manifest["uri"] != args.uri:
            raise RuntimeError("Source changed; resume requires identical benchmark code")
    else:
        rng = random.Random(20260914)
        schedule = []
        for repetition in range(1, 11):
            speeds = SPEEDS.copy()
            rng.shuffle(speeds)
            schedule.extend(
                [
                    {"number": len(schedule) + i + 1, "repetition": repetition, "speed": s}
                    for i, s in enumerate(speeds)
                ]
            )
        manifest = {
            "word": "ACE",
            "uri": args.uri,
            "max_skills": 4,
            "api_max_attempts": 10,
            "schedule": schedule,
            "source_hashes": hashes,
            "reset": "arm26 then full36 worker restart before every episode",
            "metric": "sim_time last frame minus initial frame; reset and API waiting excluded",
            "success": "Current VLM identities and upright cubes; stereo XY errors <=18mm; hands >=100mm away",
        }
        write_json(manifest_path, manifest)
    for trial in manifest["schedule"]:
        trialdir = out / f"trial_{trial['number']:02d}_{trial['speed']:g}x"
        trialdir.mkdir(exist_ok=True)
        if (trialdir / "result.json").exists():
            continue
        if (trialdir / "episode").exists():
            raise RuntimeError(f"Interrupted episode requires explicit audit: {trialdir}")
        print("START", json.dumps(trial), flush=True)
        reset_code = command(
            [
                sys.executable,
                "-m",
                "astrabot",
                "reset",
                "--uri",
                args.uri,
                "--output",
                str(trialdir / "reset_status.json"),
            ],
            trialdir / "reset.log",
            900,
        )
        if reset_code:
            write_json(
                out / "batch_status.json",
                {"state": "reset_failed", "trial": trial, "returncode": reset_code},
            )
            raise RuntimeError(f"Cold reset failed; trial not started: {trialdir}")
        write_json(out / "batch_status.json", {"state": "running", "trial": trial})
        code = command(
            [
                sys.executable,
                "-m",
                "astrabot",
                "run",
                "--uri",
                args.uri,
                "--word",
                "ACE",
                "--speed",
                str(trial["speed"]),
                "--max-skills",
                "4",
                "--output",
                str(trialdir / "episode"),
            ],
            trialdir / "episode.log",
            3600,
        )
        reportfile = trialdir / "episode/report.json"
        report = (
            json.loads(reportfile.read_text())
            if reportfile.exists()
            else {"error": "No report produced"}
        )
        if code == 124:
            report.update(success_verified=False, error="Episode timeout")
        result = {
            **trial,
            "success": bool(report.get("success_verified")),
            "category": classify(report),
            "simulation_seconds": report.get("simulation_seconds"),
            "skills_completed": report.get("skills_completed", 0),
            "observations": sum(e["state"] == "observe" for e in report.get("events", [])),
            "reason": report.get("reason", ""),
            "error": report.get("error", ""),
            "returncode": code,
            "wall_seconds": report.get("wall_seconds"),
            "report": str(reportfile),
        }
        write_json(trialdir / "result.json", result)
        aggregate(out)
        print("DONE", json.dumps(result), flush=True)
    write_json(out / "batch_status.json", {"state": "complete", "trials": 50})
    print("SUMMARY", json.dumps(aggregate(out)), flush=True)
