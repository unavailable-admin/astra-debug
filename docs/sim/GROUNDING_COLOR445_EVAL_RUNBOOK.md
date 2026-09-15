# Scene11 Grounding 闭环评测运行说明（color445 + ref seed10）

本文档说明如何在远程仿真上跑 **LetterPick grounding** 四实验（arm_a / arm_b / arm_s / arm_s_run1）的标准闭环评测，以及带 **机器人靠近桌子（pos_y+0.10）** 的变体。

协议细节以 [`CLIENT_USAGE_GUIDE.md`](./CLIENT_USAGE_GUIDE.md) 为准；本文只覆盖本仓库已落地的批跑脚本与踩坑项。

---

## 1. 评测协议摘要

| 项 | 值 |
|---|---|
| 场景 | `showroom_scene_11` |
| 布局 | `thirteen_custom`，字母 A–M，四项 randomize 全开（color445） |
| 固定 seed | `subscribe_seed=10`（各字母串行，layout 一致） |
| Prompt | `Target letter: {B\|C\|E\|M}` |
| Warmup | `jointproxy_qnext_median_nearest_state0.npy`，repeat=10 |
| 每字母 | 1 rollout × `max_loops=10` × `actions_per_submit=32` |
| 闭环脚本 | `closedloop_test_fastwam.py` |
| Python 环境 | `kairos_3_1_pre` |
| 仿真主机 | `172.31.4.140`，WebSocket 端口 `8080–8083` |

**color445 正确表现**（新版 sim）：subscribe ack 里 `color_source=tmp_layout`，`letter_color=[0,0,0]`（黑字），桌面有 `table_surface_color`。

**机器人位姿**（可选）：subscribe **顶层**字段 `showroom_scene_11_robot_pose`（与 `showroom_scene_11_generalization` 并列，不在 layout JSON 内）。示例 pos_y+0.10：

```json
{
  "showroom_scene_11_robot_pose": {
    "pos": [0.003253, 1.495587, 0.76]
  }
}
```

默认 Y≈`1.3956`；Y 增大 = 机器人 root 往桌子靠近。

---

## 2. 端口与 GPU 分配（重要）

| 实验 | 默认端口 | GPU | 说明 |
|---|---|---|---|
| arm_a | 8080 | 0 | |
| arm_b | 8083 | 7 | 参考 layout，与 color445 sweep 一致 |
| arm_s | 8082 | 1 | **不要用 8081** |
| arm_s_run1 | 8083 | 2 | 与 arm_b 同端口时需错开时间 |

### ⚠️ 8081 是旧版 sim，禁止用于本评测

实测 `172.31.4.140:8081`：

- 不支持 `showroom_scene_11_robot_pose`（ack 无 `robot_pose`，距离不生效）
- `thirteen_custom` 走 **random 配色**（字母也是彩色，难以辨认）

可用端口：**8080 / 8082 / 8083**。

批跑脚本 `run_serial_one_exp.py` 启动前会自动 preflight：检查 `color_source=tmp_layout`，若传了 robot_pose 则检查 `actual_pos Y≈1.496`。

---

## 3. 路径速查

```
评测批跑根目录（脚本 + layout）:
  .../result/vis/robot/Demo/BuildingBlocks/running/
  20260826_grounding_4exp_color445_refseed10_warmup10/

关键文件:
  run_serial_one_exp.py              # 单实验串行 B→C→E→M
  run_parallel_4exp.py               # 多实验并行（各占一端口）
  scene11_thirteen_custom_color445_am.json
  scene11_robot_pose_yplus10.json    # pos_y+0.10

Warmup:
  .../result/vis/robot/Demo/BuildingBlocks/archive/_warmup/
  jointproxy_qnext_median_nearest_state0.npy

闭环入口:
  .../code/FastWam/tmp/qworld/tools/closedloop_test_fastwam.py

Python:
  /kairos_vepfs_volc/embodied/fuzuoyi/anaconda3/envs/kairos_3_1_pre/bin/python
```

---

## 4. 单实验运行

```bash
RUN_ROOT=/kairos_vepfs_volc/action/zhangruixuan/result/vis/robot/Demo/BuildingBlocks/running/20260826_grounding_4exp_color445_refseed10_warmup10
PY=/kairos_vepfs_volc/embodied/fuzuoyi/anaconda3/envs/kairos_3_1_pre/bin/python

cd "$RUN_ROOT"

# 示例：arm_b @ 7.5k（默认端口/GPU 见 EXP_SLOTS）
"$PY" run_serial_one_exp.py arm_b

# 示例：arm_s @ 10k，pos_y+0.10，显式指定与 arm_b 同端口 8083
"$PY" run_serial_one_exp.py arm_s \
  --ckpt_step 10000 \
  --out_suffix _robotpos_yplus10 \
  --port 8083 \
  --gpu 1 \
  --robot_pose_json scene11_robot_pose_yplus10.json

# 示例：arm_a @ 20k，pos_y+0.10（默认端口 8080 / GPU 0，可与 arm_s 并行）
"$PY" run_serial_one_exp.py arm_a \
  --ckpt_step 20000 \
  --out_suffix _robotpos_yplus10 \
  --port 8080 \
  --gpu 0 \
  --robot_pose_json scene11_robot_pose_yplus10.json
```

### CLI 参数

| 参数 | 含义 |
|---|---|
| `exp` | `arm_a` / `arm_b` / `arm_s` / `arm_s_run1` |
| `--ckpt_step` | 强制 checkpoint step（如 `10000`） |
| `--out_suffix` | 输出目录后缀（如 `_robotpos_yplus10`） |
| `--port` | 覆盖默认 sim 端口 |
| `--gpu` | 覆盖默认 GPU |
| `--robot_pose_json` | Scene11 机器人位姿 JSON（绝对路径写入 closedloop） |

### 输出目录命名

- 固定 tag：`arm_b_s7500ema` 等
- arm_a / arm_s 动态：`arm_a_s{step}ema{suffix}`、`arm_s_s{step}ema{suffix}`
  例如 `arm_a_s20000ema_robotpos_yplus10`、`arm_s_s10000ema_robotpos_yplus10`

每个字母：

```
{out_root}/letter_{B|C|E|M}/rollout_01/preview.mp4
{out_root}/letter_{B|C|E|M}/closedloop_run.log
```

批日志：`{out_root}/batch.log`、根目录 `chain.log`。

---

## 5. 多实验并行

```bash
cd "$RUN_ROOT"

# 默认跑全部四实验；也可只列子集
"$PY" run_parallel_4exp.py arm_a arm_s arm_s_run1
```

- 每个实验独占 `(port, gpu)`，实验内 **B→C→E→M 串行**（保证 seed=10 下 layout 一致）。
- 并行前确认端口不冲突；**不要**把任一实验分到 8081。
- 当前 `arm_s_run1` 与 `arm_b` 共用 8083，不能同时跑。

---

## 6. 后台 nohup 示例

```bash
cd "$RUN_ROOT"
nohup "$PY" run_serial_one_exp.py arm_s \
  --ckpt_step 10000 \
  --out_suffix _robotpos_yplus10 \
  --port 8083 \
  --gpu 1 \
  --robot_pose_json scene11_robot_pose_yplus10.json \
  > arm_s_s10000ema_robotpos_yplus10.nohup.log 2>&1 &
```

查看进度：

```bash
tail -f arm_s_s10000ema_robotpos_yplus10/batch.log
```

---

## 7. 手动 preflight / 单帧 inspect

不加载模型，仅验证端口是否支持 color445 + robot_pose：

```bash
QWORLD=/kairos_vepfs_volc/action/zhangruixuan/code/FastWam/tmp/qworld
export PYTHONPATH="$QWORLD${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$QWORLD/tools/closedloop_test_fastwam.py" \
  --inspect_observation_only \
  --uri ws://172.31.4.140:8083 \
  --obs_log_dir /tmp/scene11_inspect \
  --showroom_scene_11_generalization_json "$RUN_ROOT/scene11_thirteen_custom_color445_am.json" \
  --showroom_scene_11_robot_pose_json "$RUN_ROOT/scene11_robot_pose_yplus10.json" \
  --subscribe_seed 10
```

检查 `/tmp/scene11_inspect/inspect_subscribe_ack.json`：

- `showroom_scene_11_layout.letters[0].color_source` → `"tmp_layout"`
- `showroom_scene_11_layout.letters[0].letter_color` → `[0, 0, 0]`
- `showroom_scene_11_robot_pose.actual_pos[1]` → ≈ `1.4956`

首帧图：`inspect_initial.jpg`。

也可用 Python 一行探针（无需 qworld 重 import）：

```bash
"$PY" - <<'PY'
import asyncio, json, websockets
from pathlib import Path

HOST, PORT, SEED = "172.31.4.140", 8083, 10
RUN = Path("/kairos_vepfs_volc/action/zhangruixuan/result/vis/robot/Demo/BuildingBlocks/running/20260826_grounding_4exp_color445_refseed10_warmup10")
gen = json.loads((RUN / "scene11_thirteen_custom_color445_am.json").read_text())
pose = json.loads((RUN / "scene11_robot_pose_yplus10.json").read_text())["showroom_scene_11_robot_pose"]

async def main():
    async with websockets.connect(f"ws://{HOST}:{PORT}", max_size=64*1024*1024) as ws:
        await ws.send(json.dumps({
            "type": "subscribe_step_result",
            "showroom_scene_11_generalization": gen,
            "showroom_scene_11_robot_pose": pose,
            "force_reset": True,
            "seed": SEED,
        }))
        ack = json.loads(await ws.recv())
    L = ack["showroom_scene_11_layout"]["letters"][0]
    print("color_source", L["color_source"], "letter_color", L["letter_color"])
    print("robot_pose Y", ack["showroom_scene_11_robot_pose"]["actual_pos"][1])

asyncio.run(main())
PY
```

---

## 8. 常见问题

### 距离 / 配色都没变

1. 查 `batch.log` 里的 **port**；若是 **8081**，结果无效。
2. 看 subscribe ack：`color_source` 是否为 `random`（旧 sim）而非 `tmp_layout`。
3. robot_pose 必须用顶层 `showroom_scene_11_robot_pose`，**不是** layout JSON 内的 `robot_pose` / `pos_delta`。

### 与 arm_b layout 不一致

- 对齐参考：**同一端口 + `subscribe_seed=10` + 同一 layout JSON**。
- 字母间串行跑在同一端口；换端口或并行占同一端口会导致 layout 漂移。

### preflight 失败

| 报错 | 处理 |
|---|---|
| `port 8081 is legacy` | 换 8080/8082/8083 |
| `color_source='random'` | 该端口 sim 版本过旧，换端口 |
| `robot_pose preflight failed` | 该端口不支持 robot_pose，换端口 |

### 无效结果归档

8081 误跑目录建议改名保留，避免与有效结果混淆，例如：

`arm_s_s10000ema_robotpos_yplus10_INVALID_port8081/`

---

## 9. 相关文档

- [`CLIENT_USAGE_GUIDE.md`](./CLIENT_USAGE_GUIDE.md) — Scene11 泛化、robot_pose 协议、subscribe 字段
- 距离 sweep 首帧参考：`.../running/20260826_robot_table_distance_sweep_v2/pos_y+0.10/`
