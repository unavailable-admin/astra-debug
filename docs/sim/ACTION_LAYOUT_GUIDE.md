# G1+O6 仿真 Server — Action Layout 指南

适用 server: `scene_ws_action_server_g1o6.py`（pool worker）。本文档说明该 server 支持的两种 action 维度布局（layout）、它们的语义差异，以及 client 端要怎么对齐 cfg。

---

## 1. 两种 layout 一览

| layout | dim | hand DoFs | 行为 | 用途 |
| --- | --- | --- | --- | --- |
| `full36` (默认) | 36 | 22（每指 mcp+dip + 拇指 cmc_yaw/cmc_pitch/ip） | action term 驱动全部 36 个 joint | 历史模型 / 旧 H5 数据 |
| `arm26` | 26 | 12（每指 mcp_pitch + 拇指 cmc_yaw/cmc_pitch） | action term 仅驱动 26 个 joint；其余 10 个 dip/ip 由 articulation PD 控制器在默认/初始姿态附近撑住 | 新一代 26-D 模型 / 真机 underactuated 灵巧手对齐 |

**关键事实**（已通过 plan §4.1 实测验证）：

- arm26 模式下，提交一段 30 步小幅 action（仅 left_shoulder_pitch 上有 1° 量级正弦），10 个未列 joint 在 30 步后 max|Δ| = **0.00004 rad**。即 IsaacLab `JointPositionActionTerm` 真的让未列 joint 维持 articulation 的默认 setpoint（PD-held），不会漂移、不会瘫倒。
- arm26 模式下，**手指无法主动合拢做精细抓取**（DIP/IP 是被动的），适合策略本身就只输出 mcp 层级（典型 Inspire/Unitree 真机 underactuated 接口）的场景。

---

## 2. 36-D/26-D action 与 51-D state 的精确 joint 顺序

**36-D**（`full36`，joint 顺序由 `source/easim/scripts/g1o6_action_layout.py` 定义）：

```
0  left_shoulder_pitch_joint
1  left_shoulder_roll_joint
2  left_shoulder_yaw_joint
3  left_elbow_joint
4  left_wrist_roll_joint
5  left_wrist_pitch_joint
6  left_wrist_yaw_joint
7  right_shoulder_pitch_joint
8  right_shoulder_roll_joint
9  right_shoulder_yaw_joint
10 right_elbow_joint
11 right_wrist_roll_joint
12 right_wrist_pitch_joint
13 right_wrist_yaw_joint
14 lh_index_mcp_pitch
15 lh_index_dip          *
16 lh_middle_mcp_pitch
17 lh_middle_dip         *
18 lh_pinky_mcp_pitch
19 lh_pinky_dip          *
20 lh_ring_mcp_pitch
21 lh_ring_dip           *
22 lh_thumb_cmc_yaw
23 lh_thumb_cmc_pitch
24 lh_thumb_ip           *
25 rh_index_mcp_pitch
26 rh_index_dip          *
27 rh_middle_mcp_pitch
28 rh_middle_dip         *
29 rh_pinky_mcp_pitch
30 rh_pinky_dip          *
31 rh_ring_mcp_pitch
32 rh_ring_dip           *
33 rh_thumb_cmc_yaw
34 rh_thumb_cmc_pitch
35 rh_thumb_ip           *
```

带 `*` 的 10 个就是 arm26 模式下被丢掉的关节。

**26-D**（`arm26`）：14 arm（同上 0–13）+ 12 hand：

```
0..13  arm 14 个（与 full36 完全一致）
14 lh_index_mcp_pitch
15 lh_middle_mcp_pitch
16 lh_pinky_mcp_pitch
17 lh_ring_mcp_pitch
18 lh_thumb_cmc_yaw
19 lh_thumb_cmc_pitch
20 rh_index_mcp_pitch
21 rh_middle_mcp_pitch
22 rh_pinky_mcp_pitch
23 rh_ring_mcp_pitch
24 rh_thumb_cmc_yaw
25 rh_thumb_cmc_pitch
```

**arm26 → full36 的 gather 索引表**（即"26 个 arm26 列对应 full36 哪 26 个槽位"）：

```
[0,1,2,3,4,5,6,7,8,9,10,11,12,13, 14,16,18,20,22,23, 25,27,29,31,33,34]
```

被丢的 10 列在 full36 里的索引：`[15,17,19,21,24, 26,28,30,32,35]`。

完整名字 + 索引表的唯一可信源是 [`source/easim/scripts/g1o6_action_layout.py`](source/easim/scripts/g1o6_action_layout.py) 中的 `FULL_36_JOINT_NAMES / ARM_26_JOINT_NAMES / ARM26_TO_FULL36_INDICES / FULL36_DROPPED_FOR_ARM26 / UNACTUATED_JOINT_NAMES_FOR_ARM26`。

### 2.1 返回 state 的 51-D 顺序

`step_result.frames[*].state.joint_position_51` 和 `joint_velocity_51` 使用 `full_body_51` 布局。这个布局来自 IsaacLab articulation 的 `robot.data.joint_names`，不是 `full36` action 的顺序。client 应优先读取 `status_response.full_body_joint_names`；当前环境顺序如下：

```text
0  left_hip_pitch_joint
1  right_hip_pitch_joint
2  waist_yaw_joint
3  left_hip_roll_joint
4  right_hip_roll_joint
5  waist_roll_joint
6  left_hip_yaw_joint
7  right_hip_yaw_joint
8  waist_pitch_joint
9  left_knee_joint
10 right_knee_joint
11 left_shoulder_pitch_joint
12 right_shoulder_pitch_joint
13 left_ankle_pitch_joint
14 right_ankle_pitch_joint
15 left_shoulder_roll_joint
16 right_shoulder_roll_joint
17 left_ankle_roll_joint
18 right_ankle_roll_joint
19 left_shoulder_yaw_joint
20 right_shoulder_yaw_joint
21 left_elbow_joint
22 right_elbow_joint
23 left_wrist_roll_joint
24 right_wrist_roll_joint
25 left_wrist_pitch_joint
26 right_wrist_pitch_joint
27 left_wrist_yaw_joint
28 right_wrist_yaw_joint
29 lh_index_mcp_pitch
30 lh_middle_mcp_pitch
31 lh_pinky_mcp_pitch
32 lh_ring_mcp_pitch
33 lh_thumb_cmc_yaw
34 rh_index_mcp_pitch
35 rh_middle_mcp_pitch
36 rh_pinky_mcp_pitch
37 rh_ring_mcp_pitch
38 rh_thumb_cmc_yaw
39 lh_index_dip
40 lh_middle_dip
41 lh_pinky_dip
42 lh_ring_dip
43 lh_thumb_cmc_pitch
44 rh_index_dip
45 rh_middle_dip
46 rh_pinky_dip
47 rh_ring_dip
48 rh_thumb_cmc_pitch
49 lh_thumb_ip
50 rh_thumb_ip
```

粗略分组：12 个下肢 + 3 个腰 + 14 个双臂 + 22 个 O6 双手关节。由于 articulation 原始顺序是交错的，读取局部子集时建议按名字查索引。

### 2.2 action 维度到 state 51 维的对应关系

`full36` action 每一维对应的 `joint_position_51` 索引：

```text
[11,15,19,21,23,25,27, 12,16,20,22,24,26,28,
 29,39,30,40,31,41,32,42,33,43,49,
 34,44,35,45,36,46,37,47,38,48,50]
```

`arm26` action 每一维对应的 `joint_position_51` 索引：

```text
[11,15,19,21,23,25,27, 12,16,20,22,24,26,28,
 29,30,31,32,33,43, 34,35,36,37,38,48]
```

其中 `arm26` 丢掉的 10 个 `full36` action 槽位是 `[15,17,19,21,24, 26,28,30,32,35]`，对应 joint 名：

```text
lh_index_dip, lh_middle_dip, lh_pinky_dip, lh_ring_dip, lh_thumb_ip,
rh_index_dip, rh_middle_dip, rh_pinky_dip, rh_ring_dip, rh_thumb_ip
```

运行时仍推荐从 status 动态获取并校验：

```python
await ws.send(json.dumps({"type": "status"}))
status = json.loads(await ws.recv())
assert len(status["full_body_joint_names"]) == status["step_result_state_dim"] == 51
assert len(status["action_joint_names"]) == status["action_dim"]
action_state_indices = [
    status["full_body_joint_names"].index(name)
    for name in status["action_joint_names"]
]
```

---

## 3. server 端怎么开 arm26

### 3.1 单 worker（直接调 `scene_ws_action_server_g1o6.py`）

```bash
isaaclab.sh -p source/easim/scripts/scene_ws_action_server_g1o6.py \
    --action_layout arm26 \
    --enable_cameras --headless --host 0.0.0.0 --port 8090 \
    --actions_cfg cfg_list ...
```

启动 banner 应该看到：

```
Scene WS Action Server G1+O6 (26-D bimanual jointpos replay, layout=arm26)
client_action_dim=26 env_action_dim=26
action_layout=arm26 unactuated_joint_names=['lh_index_dip', ..., 'rh_thumb_ip']
[layout] action_layout=arm26 applied to terms=['joint_pos']: 26 joints, dropped=[...]
```

### 3.2 通过 wrapper（`run_g1o6_server_loop.sh`）

```bash
ACTION_LAYOUT=arm26 PORT=8090 SCENE_STATE_FILE=/tmp/g1o6_test_8090 \
  setsid /root/code/easim/run_g1o6_server_loop.sh \
  >/tmp/test_arm26.log 2>&1 </dev/null &
```

如果 8090 专门用来调试 `pick_fruit` 泛化/负样例，建议给 scene/action/pick_fruit layout 都用独立 state file，避免误读 8080-8083 正式池状态：

```bash
PORT=8090 INITIAL_SCENE=pick_fruit ACTION_LAYOUT=arm26 \
SCENE_STATE_FILE=/tmp/g1o6_debug_scene_id_8090 \
ACTION_LAYOUT_STATE_FILE=/tmp/g1o6_debug_action_layout_8090 \
PICK_FRUIT_LAYOUT_STATE_FILE=/tmp/g1o6_debug_pick_fruit_layout_8090 \
  setsid /root/code/easim/run_g1o6_server_loop.sh \
  >/root/code/easim/server_logs/debug_worker_8090.log 2>&1 </dev/null &
```

### 3.3 整池（`run_g1o6_pool.sh`）

整池统一一种 layout：

```bash
ACTION_LAYOUT=arm26 setsid /root/code/easim/run_g1o6_pool.sh >/tmp/pool.log 2>&1 </dev/null &
```

per-worker 异构 layout（罕用）：

```bash
ACTION_LAYOUT=full36 \
ACTION_LAYOUT_8083=arm26 \
  setsid /root/code/easim/run_g1o6_pool.sh >/tmp/pool.log 2>&1 </dev/null &
# 8080,8081,8082 跑 full36；8083 跑 arm26
```

参考 [`RESTART_GUIDE.md`](RESTART_GUIDE.md) §1.1 的整池重启 one-liner。

### 3.4 运行时通过 client 切换 layout（无需手动重启 pool）

如果池已经在跑、不想下整池，可以让 client 直接发 WS 指令 `switch_action_layout` 让某个 worker 自我重启加载新 layout。机制完全对称已有的 `switch_scene`：

1. client 发 `{"type": "switch_action_layout", "action_layout": "arm26"}` 到该 worker
2. server 同步回 `switch_action_layout_response.ok=true`，并向所有订阅者推送 `action_layout_switching`，然后 `os._exit(50)`
3. wrapper `run_g1o6_server_loop.sh` 检测到 `/tmp/g1o6_action_layout_<port>` 内容变化，重启 server 加载新 layout（冷启动 ~60–90s）
4. client 轮询 `status` 直到 `action_layout` 变成目标值

可以用两种方式触发：

方式 A：直接发 WS 消息（无额外 helper 依赖）。

```python
import asyncio
import json

import websockets

URI = "ws://172.31.0.226:8083"
TARGET_LAYOUT = "arm26"


async def main():
    async with websockets.connect(URI, open_timeout=10, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "switch_action_layout",
            "action_layout": TARGET_LAYOUT,
        }))
        ack = json.loads(await ws.recv())
        print(ack)


asyncio.run(main())
```

方式 B：调用当前工作区 helper，自动等待 worker 重启并轮询到目标 layout。

```bash
python3 - <<'PY'
import asyncio
import sys

sys.path.insert(0, 'source/easim/scripts')
from g1o6_action_layout import request_action_layout_switch

asyncio.run(request_action_layout_switch(
    'ws://172.31.0.226:8083', 'arm26',
    restart_timeout=240, poll_interval=5,
))
PY
```

主模型 client 如果在客户端仓库支持 `--switch_action_layout`，也可以启动时自动切；本 server 工作区只保证上述 WS 协议和 helper 可用。

要点：

- layout 切换是 **worker 独占操作**：触发时若有别的 client 正在订阅 / submit_actions，那条连接会被 server exit 直接断掉。一个 worker 同一时刻只服务一个评测任务
- `/tmp/g1o6_action_layout_<port>` 状态文件**跨整池重启持久化**——客户端切过的 layout 会被下一次 pool 启动尊重，而非被 `ACTION_LAYOUT` env 抹掉
- `switch_action_layout` 不改变 scene_id（`/tmp/g1o6_server_scene_id_<port>` 独立）
- `switch_action_layout` 也不改变 pick_fruit 桌面布局或负样例配置：启动期三果 layout 由 `/tmp/g1o6_pick_fruit_layout_<port>` 独立保存，rollout 级负样例只通过 `reset_env` / `subscribe_step_result` 生效，详见 [`SCENE_SWITCH_CLIENT_GUIDE.md`](SCENE_SWITCH_CLIENT_GUIDE.md) §8
- 目标 layout 与当前一致时 server 视为 no-op：返回 ack，但不退出、不重启

---

## 4. client 端怎么对齐

### 4.1 主 model client（`client_demo_wam_policy_G1O6.py`）

新增 CLI：

```bash
--action_layout {auto, arm26, full36}   # 默认 auto，从 status_response 读取
```

启动后 client 在 `subscribe_step_result` 之前会先发 `status` 做 preflight：

- 解析 `status_response.action_layout` / `action_dim`
- 用户传了 `--action_layout` 非 auto → 与 server 比对，不一致 fail-fast
- `cfg.action_dim` / `len(reindex_map.action)` 推出的 `submit_action_dim` 与 server `action_dim` 不一致 → fail-fast

例：

```bash
python client_demo_wam_policy_G1O6.py \
    --uri ws://172.31.0.226:8090 \
    --config <26-D model cfg> \
    --action_layout arm26 \
    --task_name pull_bowl
```

### 4.2 模型 cfg 怎么写 26-D

cfg 里 `ActionStateReIndexTransform.reindex_map.action` 的长度必须 == `cfg.action_dim` 必须 == server `action_dim`：

| layout | cfg.action_dim | reindex_map.action 长度 | 模型 backbone 输出 |
| --- | --- | --- | --- |
| full36 | 36 (或 37 enable_episode_time) | 36 | (T, 36) 或 (T, 37) |
| arm26 | 26 (或 27 enable_episode_time) | 26 | (T, 26) 或 (T, 27) |

`reindex_map.action` 的具体值由训练时如何 reorder 模型输出决定，与 layout 解耦——只要长度对齐 server 期望维度即可。

### 4.3 H5 老数据跨 layout replay

HDF5 里历史的 `pred_actions_denormed_no_GTCorrection_smooth` 等数据集都是 36-D。要把它们 replay 到 arm26 worker：

- 主 client / replay 脚本：如果要把 36-D H5 replay 到 arm26 worker，可调用 `source/easim/scripts/g1o6_action_layout.py` 里的 `compact_full36_to_arm26` 做 36→26 gather。日志通常会有：

  ```
  [G1O6Policy] preflight layout: server='arm26' action_dim=26 unactuated=[...]
  ```

- smoke / replay 工具：每个都新增 `--action_layout` CLI（默认 auto）；H5 36-D → arm26 server 时也是自动 downcast。详见各工具的 `--help`。

---

## 5. 协议变化（向后兼容）

### 5.1 `status_response` 新增 2 个字段

```jsonc
{
  // ...原字段...
  "action_layout": "full36" | "arm26",
  "unactuated_joint_names": ["lh_index_dip", ..., "rh_thumb_ip"]   // arm26 时 10 个，full36 时 []
}
```

老 client 不读这两个字段，行为不变。

### 5.2 `subscribe_step_result_response` 新增 1 个字段

```jsonc
{
  "action_layout": "full36" | "arm26"
}
```

### 5.3 `step_result` 新增 1 个 top-level 字段

```jsonc
{
  "action_layout": "full36" | "arm26"
}
```

每个 frame 也带：

```jsonc
{
  "external_action_36": [...],          // 历史字段名保留；arm26 时实际长度 26
  "external_action_layout": "full36" | "arm26"
}
```

> 注：`external_action_36` 是历史包袱字段名。下一版协议会改名为 `external_action`，届时同时移除 `_36` 后缀。当前所有读这个字段的 client 都应改成读 `len(external_action_36)` 而不是 hardcode 36。

### 5.4 `submit_actions` 错误信息

dim 不对时 server 返回更明确的错误：

```jsonc
{
  "ok": false,
  "error": "actions[0] dim=36 expected=26 (action_layout='arm26')"
}
```

### 5.5 新增 `switch_action_layout` 协议（运行时切换）

client → server：

```jsonc
{"type": "switch_action_layout", "action_layout": "arm26"}
```

server → client（同步 ack）：

```jsonc
{
  "type": "switch_action_layout_response",
  "ok": true,
  "action_layout": "arm26",
  "from_action_layout": "full36",
  "note": "action_layout switch queued; server will exit and wrapper will restart with the new layout. Reconnect after ~60-90s and check status.action_layout.",
  "timestamp": ...
}
```

server → 全体订阅者（push，紧接着 `os._exit(50)`）：

```jsonc
{
  "type": "action_layout_switching",
  "ok": true,
  "action_layout": "arm26",
  "from_action_layout": "full36",
  "note": "server is restarting; reconnect after ~60-90s",
  "timestamp": ...
}
```

老 client 不发该指令、不解析 `action_layout_switching` 时行为不变。

---

## 6. 常见错误 / 排查

### 6.1 client 启动报 "client submit_action_dim=36 but server action_dim=26"

模型 cfg 是 36-D，但 server 是 arm26。两个修法：

- 改 cfg 让 `action_dim=26 / reindex_map.action` 长度 = 26；
- 或者重启 server 加 `ACTION_LAYOUT=full36`。

### 6.2 launch 报 "FATAL: ACTION_LAYOUT='xxx' is invalid"

`run_g1o6_pool.sh` / `run_g1o6_server_loop.sh` 校验 ACTION_LAYOUT 必须是 `full36 | arm26`，其他值（包括空格、大小写错误）拒绝启动。

### 6.3 server 日志报 "could not find a `joint_names` attribute on ... to narrow"

vendored 的 `G1O6BimanualJointPosAbsoluteReplayActionsCfg` 的字段命名变了。看 server 脚本里的 `_maybe_apply_arm26_layout` helper 如何遍历 dataclass 字段；通常加一个反射兜底就行。

### 6.4 跑了 arm26 但发现手指完全不动 / 抓不到东西

预期行为。arm26 模式下 dip/ip 不被 action 控制，由 PD 撑在初始姿态附近，所以"手势"是固定的展开/微弯。如果策略需要主动合拢，请用 full36。

---

## 7. 相关文档

- [`source/easim/scripts/g1o6_action_layout.py`](source/easim/scripts/g1o6_action_layout.py)：joint 名字表 + 26↔36 helper（**唯一可信源**）
- [`CLIENT_USAGE_GUIDE.md`](CLIENT_USAGE_GUIDE.md)：client 协议速查 + pick_fruit rollout 泛化/负样例示例
- [`SCENE_SWITCH_CLIENT_GUIDE.md`](SCENE_SWITCH_CLIENT_GUIDE.md)：scene / pick_fruit 启动期 layout 切换协议，以及 rollout 级负样例边界
- [`RESTART_GUIDE.md`](RESTART_GUIDE.md)：池重启手册（含 `ACTION_LAYOUT` 配置示例）
