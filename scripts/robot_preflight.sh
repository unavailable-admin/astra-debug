#!/usr/bin/env bash
# Run as the executor's operator UID inside the container. Read-only IPC.
set -euo pipefail
robot_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$robot_repo_root"
exec "${ASTRABOT_PYTHON:-python3}" -m astrabot.robot.preflight "$@"
