#!/usr/bin/env bash
# Without --execute, only check a supplied --scene offline.
set -euo pipefail
robot_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$robot_repo_root"
exec "${ASTRABOT_PYTHON:-python3}" -m astrabot.robot.pick_a "$@"
