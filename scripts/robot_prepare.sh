#!/usr/bin/env bash
# Steps 4-7 against an already running executor and pause console; no motion.
set -euo pipefail
robot_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$robot_repo_root"
exec "${ASTRABOT_PYTHON:-python3}" -m astrabot.robot.pick_a --capture-plan "$@"
