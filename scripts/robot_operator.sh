#!/usr/bin/env bash
# Thor host entry point. Empty-workspace preparation executes the checked raise.
set -euo pipefail
robot_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
robot_container=${ASTRA_CONTAINER:-ros2humble-wmp}
robot_uid=${ASTRA_OPERATOR_UID:-3100}
robot_config=${ASTRA_CONFIG:-$robot_repo_root/outputs/operator-startup-20260917T080408Z/config.json}
robot_python=${ASTRABOT_PYTHON:-python3}
robot_action=${1:-help}
if { [[ "$robot_action" != spell ]] && (( $# > 1 )); } || (( $# > 2 )); then
  echo '每次只指定一个操作：start、shutdown、disarm、status、console、scene、raise、prepare、recover、trial 或 spell [WORD]。' >&2
  exit 2
fi

# This experimental launcher uses the operator's placement guarantee. The
# Python entry points remain strict unless this explicit flag is supplied.
robot_workspace_args=()
case "$robot_action" in
  scene|trial)
    case "${ASTRA_OPERATOR_CLEARED_WORKSPACE:-1}" in
      1)
        robot_workspace_args=(--operator-cleared-workspace)
        echo '人工清空区域实验模式：只复核 A 和桌面几何，不检查其他物体；保持 console 可用。'
        ;;
      0) ;;
      *) echo 'ASTRA_OPERATOR_CLEARED_WORKSPACE 必须为 0 或 1。' >&2; exit 2 ;;
    esac
    ;;
esac

case "$robot_action" in
  start)
    # A live executor must retain its ownership; never restart it implicitly.
    if docker exec -u "$robot_uid" -w "$robot_repo_root" "$robot_container" \
      "$robot_python" -m astrabot.robot status >/dev/null 2>&1; then
      echo '执行器已在运行；无需再次启动。继续打开 console，并运行 prepare。'
      exit 0
    fi
    docker exec -u root "$robot_container" bash "$robot_repo_root/scripts/robot_device_access.sh" "$robot_uid"
    exec docker exec -it -u "$robot_uid" -w "$robot_repo_root" "$robot_container" \
      "$robot_python" -m astrabot.robot serve --config "$robot_config"
    ;;
  console)
    exec docker exec -it -u "$robot_uid" -w "$robot_repo_root" "$robot_container" \
      "$robot_python" -m astrabot.robot console
    ;;
  disarm|status)
    # Existing server gates require confirmed damping for disarm. Never kill.
    exec docker exec -u "$robot_uid" -w "$robot_repo_root" "$robot_container" \
      "$robot_python" -m astrabot.robot "$robot_action"
    ;;
  prepare|recover|shutdown)
    exec docker exec -it -u "$robot_uid" -w "$robot_repo_root" "$robot_container" \
      "$robot_python" -m astrabot.robot.cleared_environment "$robot_action"
    ;;
  scene)
    exec docker exec -it -u "$robot_uid" -w "$robot_repo_root" "$robot_container" \
      "$robot_python" -m astrabot.robot.pick_a "${robot_workspace_args[@]}" --config "$robot_config" --load-scene-only
    ;;
  raise)
    exec docker exec -it -u "$robot_uid" -w "$robot_repo_root" "$robot_container" \
      "$robot_python" -m astrabot.robot.prepare_motion
    ;;
  trial)
    exec docker exec -it -u "$robot_uid" -w "$robot_repo_root" "$robot_container" \
      "$robot_python" -m astrabot.robot.pick_a "${robot_workspace_args[@]}" --config "$robot_config" --execute --from-ready
    ;;
  spell)
    exec docker exec -it -u "$robot_uid" -w "$robot_repo_root" "$robot_container" \
      "$robot_python" -m astrabot.robot.spelling --config "$robot_config" --execute --word "${2:-ACE}"
    ;;
  help|--help|-h)
    echo '在 Thor 宿主机使用：robot_operator.sh start | shutdown | disarm | status | console | scene | raise | prepare | recover | trial | spell [WORD]'
    echo 'start：设备授权并启动执行器；shutdown：请求松手及检查回位交接，等执行器正常退出；console：暂停控制台；prepare：清空桌子后实际抬手；recover：清空桌子后回预备位；shutdown：清空桌子后收手交接；三者不拍照、不调用 API。trial：放回桌子后重新建模抓取。'
    echo 'spell ACE：从预备位开始，按机器人视角左到右摆到前排；只识别当前字母，自动继续，无 RESULT_OK。console 空格可随时停止。'
    echo 'disarm：仅限现场可靠支撑并进入阻尼后交还 SDK 控制；status：查看交接状态。'
    echo 'prepare/recover/shutdown 先确认空手并移走桌子及障碍物，输入 CLEARED；trial 前放回桌子并按空格。'
    echo '本实验默认由操作者清空抓取区域，仅复核 A 和桌面；设 ASTRA_OPERATOR_CLEARED_WORKSPACE=0 恢复完整场景检查。'
    ;;
  *)
    echo "未知操作：$robot_action；使用 start、shutdown、disarm、status、console、scene、raise、prepare、recover、trial 或 spell [WORD]。" >&2
    exit 2
    ;;
esac
