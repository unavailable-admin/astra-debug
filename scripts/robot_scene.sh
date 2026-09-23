#!/usr/bin/env bash
# One vision API request plus local stereo; --manual enables browser annotation.
# Does not contact the robot executor or send motion commands.
set -euo pipefail
robot_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$robot_repo_root"
exec "${ASTRABOT_PYTHON:-python3}" -m astrabot.robot.scene_review "$@"
