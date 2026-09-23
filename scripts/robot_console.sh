#!/usr/bin/env bash
set -euo pipefail
robot_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$robot_repo_root"
exec "${ASTRABOT_PYTHON:-python3}" -m astrabot.robot console "$@"
