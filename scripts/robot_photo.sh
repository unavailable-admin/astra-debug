#!/usr/bin/env bash
# Capture one diagnostic stereo frame from the Thor host; no robot commands.
set -euo pipefail
robot_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
robot_container=${ASTRA_CONTAINER:-ros2humble-wmp}
robot_uid=${ASTRA_OPERATOR_UID:-3100}
robot_config=${ASTRA_CONFIG:-$robot_repo_root/outputs/operator-startup-20260917T080408Z/config.json}
robot_python=${ASTRABOT_PYTHON:-python3}
if (( $# > 0 )); then
  echo '用法：bash scripts/robot_photo.sh（自动创建独立照片目录）' >&2
  exit 2
fi
exec docker exec -i -u "$robot_uid" -w "$robot_repo_root" "$robot_container" \
  "$robot_python" - "$robot_config" <<'PY'
import sys
import time
from pathlib import Path

from astrabot.robot.camera import capture
from astrabot.robot.config import Config

output = (Path("outputs") / f"photo-{time.time_ns()}").resolve()
try:
    left, right, _ = capture(Config.load(sys.argv[1]), output)
except Exception as exc:
    print(f"拍照失败：{exc}\n诊断目录：{output}", file=sys.stderr)
    sys.exit(1)
print(f"左图：{left}\n右图：{right}\n双目拼图：{output / 'stereo.jpg'}", flush=True)
PY
