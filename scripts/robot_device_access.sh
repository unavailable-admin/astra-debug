#!/usr/bin/env bash
# Run as root in the robot's existing container. Does not command any motors.
set -euo pipefail
if [[ $(id -u) != 0 ]]; then
  echo 'Run as root in the container that owns /dev/video0 and can4/can5.' >&2
  exit 1
fi
if pgrep -f '[/]sense_dexterous_hand/lib/sense_dexterous_hand/dexterous_hand_node' >/dev/null; then
  echo 'Existing hand node detected; refusing to change device access.' >&2
  exit 1
fi
robot_operator_uid=${1:-3100}
[[ "$robot_operator_uid" =~ ^[0-9]+$ ]]
# Check the complete device set before changing anything. UP only describes
# the interface; hand identity/feedback is checked by the running executor.
for robot_camera in /dev/video0 /dev/video1; do
  if [[ ! -c "$robot_camera" ]]; then
    echo "Missing camera node: $robot_camera. Check USB connection/container device mapping." >&2
    exit 1
  fi
done
for robot_can_interface in can4 can5; do
  ip link show "$robot_can_interface" >/dev/null
done
setfacl -m "u:${robot_operator_uid}:rw" /dev/video0 /dev/video1
for robot_can_interface in can4 can5; do
  if ip -brief link show "$robot_can_interface" | grep -q 'UP'; then
    echo "$robot_can_interface already UP; leaving its settings intact."
  else
    ip link set "$robot_can_interface" type can bitrate 1000000 restart-ms 100
    ip link set "$robot_can_interface" up
  fi
done
getfacl -cp /dev/video0 /dev/video1
ip -details link show can4
ip -details link show can5
echo 'Device access configured. Next: start the executor/console and run robot_preflight.sh.'
