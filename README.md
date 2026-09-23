# AstraBot

G1 / O6 双目字母积木抓放：视觉 API 识别、双目定位、IK 与碰撞规划、反馈执行。
支持 Scene11 仿真和独立真机后端；默认 `run` 仍进入仿真。

真机日常操作请看 **[操作指南](docs/pick_a_steps.md)**；本轮实现与验证见
[变更及验收记录](docs/real_robot_automation_status.md)。最近一次 `ACE` 已完成全流程，
并由操作者目视确认成功。单次成功不等于已测得稳定成功率。

## 安装与配置

Python 3.10+。使用自己的环境安装：

```bash
python3 -m pip install -e .              # 仿真、视觉与离线工具
python3 -m pip install -e '.[robot]'      # 真机模型与可选规划加速
```

真机还需要 Linux V4L2、g++、SocketCAN、`unitree_sdk2py`、`linkerbot` 和设备权限。
机器人专用依赖不会在默认仿真入口启动硬件。详细边界见 [真机架构与标定](docs/real_robot.md)。

视觉客户端从 `.secrets/openai_api_key` 读取密钥，或使用 `OPENAI_API_KEY_FILE` 指定文件。
密钥、本机配置与实验输出不进 Git。当前代码不读取 `OPENAI_API_KEY`。

| 配置 | 用途 |
| --- | --- |
| `OPENAI_BASE_URL` | HTTPS API 基础地址，默认 `https://api.openai.com/v1` |
| `OPENAI_MODEL` | 账号可用模型，代码默认 `gpt-6-astra` |
| `OPENAI_PROXY` | 显式 HTTP 代理；默认直连 |
| `OPENAI_CA_BUNDLE` | 可选 CA 证书文件 |

也可在 `.secrets/openai_config.json` 配置 `base_url`、`model`，环境变量优先。
视觉请求使用 **Responses API**，支持完整 JSON 和 SSE `response.completed`；服务须支持图像输入。
断流、拒答和不完整结果不会用于建模。旧 `chat()` 仅为兼容方法。

## 仿真

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m astrabot reset --uri ws://10.19.4.253:8081
python3 -m astrabot run --backend sim --word ACE --speed 1.0 --max-skills 4
python3 -m astrabot run --help
```

`--word` 支持 1–4 个不重复字母，当前使用左手。`--inspect-only` 仍会调整观察姿态。
`reset` 是独立操作，运行拼字不会隐式重置。`scripts/astrabot.sh` 可用
`ASTRABOT_PYTHON=/path/to/python` 覆盖开发机解释器路径。

仿真保持原 WebSocket 协议、运动执行和放置后重新观察逻辑。
本轮共享变化包括 Responses API、畸变反算检查、共享夹指参数/IK 和字形深度分散门限
从 6 mm 调整为 8 mm，因此不能称为“零影响”。真机的 120 mm 抬高、人工清空流程、
阶段到达容差和假定放置成功策略不会应用到仿真执行器。本轮仅做离线回归，未重跑在线 Scene11。

[标定约定](docs/calibration.md) · [WebSocket 协议](docs/protocol.md) · [倍速基准](docs/benchmark.md)

## 真机

在 Thor 宿主机的不同终端中运行（保持执行器和 console 开启）：

```bash
bash scripts/robot_operator.sh start
bash scripts/robot_operator.sh console
bash scripts/robot_operator.sh prepare
# 手举到预备位后，放回桌子和积木，再在 console 按空格。
bash scripts/robot_operator.sh spell ACE
# 空手，移走桌子和障碍物后：
bash scripts/robot_operator.sh shutdown
```

`prepare` / `recover` / `shutdown` 输入 `CLEARED` 后会实际执行已检查的动作，
不拍桌子或积木、不调用视觉 API。`spell WORD` 每次抓取前识别当前字母，
复用首块相对桌面的夹持高度，将积木摆到靠机器人一侧的前排，从机器人左到右排列。
已放置积木按计划位置保留为障碍，不重新识别，也不要求逐块输入成功确认。
控制台空格中止当前任务；任务取消后不会自动续跑。

当前现场配置使用位置反馈夹持、120 mm 抬高、4× 抓放时间缩放，关闭 `GRIP_OK`；
这些是现场配置，**不是新生成配置的默认值**。手指位置反馈不等于触觉确认。
`motion_completed=true` 仅表示动作流程完成；自动视觉验收未启用时 `success_verified=false` 是预期结果。

设置 `ASTRA_CONFIG` 可指定配置；现有机器默认配置仍位于
`outputs/operator-startup-20260917T080408Z/config.json`。
新机器先 `python3 -m astrabot.robot init-config > robot-config.local.json`，再完成设备及标定配置。
通用 `run --backend real` 是另外的标定门控入口，不能当作现场 `spell` 的等价替代。

## 代码与数据

| 路径 | 职责 |
| --- | --- |
| `astrabot/controller.py`、`simulation.py`、`motion.py` | 仿真闭环与运动 |
| `astrabot/api.py`、`vision.py`、`stereo.py`、`geometry.py` | API、字形、双目定位 |
| `astrabot/kinematics.py`、`pinch.py` | 共用 IK 与夹指约定 |
| `astrabot/robot/` | 独立执行器、设备通信、规划、标定、拼字及模型资产 |
| `scripts/robot_*.sh` | 宿主机操作和专业调试入口 |
| `tests/` | 离线回归及精简实测样本 |
| `outputs/` | 本地报告、图片、审计及当前机器标定/配置，Git 忽略 |

本次清理仅保留 `outputs/spelling-1790077682041773690/` 成功实验及其必需依赖，
具体清单在本机 `outputs/retained-manifest.json`。不要整目录删除 outputs：当前启动配置和相机标定仍在里面。
历史排障笔记备份在本机 `.local_archive/`，不提交。回归样本和运行时模型必须提交，不能当临时产物删除。

## 检查与诊断

```bash
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -q
python3 -m pre_commit run --files <本次修改文件>
```

离线测试不连接真实 API、仿真服务器或机器人。开发检查依赖可用 `pip install -e '.[dev]'` 安装。
提交检查使用 Black（120 列）、Ruff 和 Bash 语法检查。
这是独立 Python 包，无 ROS/ament 构建目标。仿真 `benchmark` 会连接仿真并执行多轮冷重置，
不是离线测试。

仿真报告含阶段计时 `timing.stages`；嵌套计时不能相加，WebSocket 等待也不是纯网络耗时。
真机报告含规划耗时、阶段事件、终点反馈及执行器审计路径。
`control_deadline_missed`、反馈过期、温度、任务心跳保护仍启用；
优化日志、规划和 GC 不代表保证硬实时。参数与检查时机见 [运动检查](docs/motion_checks.md)。
