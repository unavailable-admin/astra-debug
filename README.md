# AstraBot

G1 Scene11 双目拼字闭环：**API识别字母 → 双目定位 → 抓放 → 再观测**。
仅保留最新版双目实现。旧单目、HI演示、OCR尝试和一次性探针已移除，可在 Git 历史查看。

## 目录

```text
astrabot/
  __main__.py       # 唯一命令入口
  controller.py     # 拼字闭环与验收
  vision.py         # 字形候选、API识别
  stereo.py         # 双目定位
  geometry.py       # 相机几何与服务端标定检查
  decisions.py      # 重试、动作与落点校验
  motion.py         # 抓放和连续IK轨迹
  simulation.py     # WebSocket、反馈与动作执行
  kinematics.py     # G1运动学
  api.py            # 官方API连接
  reset.py          # 冷重置
  benchmark.py      # 50次倍速测试
  config/           # 相机参数和关节模型
tests/             # 离线测试与最小回归数据
docs/              # 当前用法、标定和协议
scripts/           # 使用当前机器已验证环境的启动脚本
```

## 安装与配置

Python≥3.10。运行时不录像，也不需要 `ffmpeg`。仓库目录下运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

需要代理时，联网安装前检查 SSH 隧道：

```bash
curl -I --connect-timeout 10 --proxy http://127.0.0.1:18888 https://github.com
export http_proxy=http://127.0.0.1:18888
export https_proxy="$http_proxy"
```

检查失败需恢复隧道。API默认读取当前工作目录的 `.secrets/openai_api_key`；也可设置
`OPENAI_API_KEY_FILE` 为自己的密钥文件路径。该目录不进 Git。当前不读取 `OPENAI_API_KEY`。

- `OPENAI_MODEL`：默认 `gpt-6-astra`，应按账号权限配置。
- `OPENAI_PROXY`：默认 `http://127.0.0.1:18888`，直连时设为空字符串。
- `OPENAI_CA_BUNDLE`：可选 CA 证书路径。

客户端仅连接官方 `https://api.openai.com/v1`。

## 当前机器与 VS Code

当前已验证环境为 Conda `kairos_3_1_pre`，Python 3.10.12：

```text
/kairos_vepfs_volc/embodied/fuzuoyi/anaconda3/envs/kairos_3_1_pre/bin/python
```

`.vscode/settings.json` 已配置默认解释器。使用 Remote SSH 打开本目录，并在远端安装
Python 和 Pylance 扩展。如果工作区之前选过其他环境，执行
`Python: Select Interpreter` → `Enter interpreter path`，填入上面的路径。
该配置不会覆盖 VS Code 已保存的解释器选择。

## 运行

默认 `ws://10.19.4.253:8081`，使用前确认该端口空闲；所有子命令都支持 `--uri`。

在当前机器无需重新安装依赖，可直接使用启动脚本。先重置，再启动默认 ACE、1×、最多4次抓放：

```bash
./scripts/astrabot.sh reset && ./scripts/astrabot.sh
```

脚本自动进入仓库目录、选用上述 Python 并限制计算线程；默认代理为 `127.0.0.1:18888`，
运行前仍按上文检查隧道。无参数只启动拼字，不隐式重置场景。
也可以传递任意子命令和参数：

```bash
./scripts/astrabot.sh run --word ACE --speed 1.2 --max-skills 4
./scripts/astrabot.sh reset --uri ws://10.19.4.253:8082
./scripts/astrabot.sh run --uri ws://10.19.4.253:8082 --word ACE --max-skills 4
./scripts/astrabot.sh --help
```

其他机器可设置 `ASTRABOT_PYTHON=/path/to/python` 覆盖脚本的解释器路径。
使用自己安装的环境时，也可直接执行下面的 Python 命令：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m astrabot reset
python -m astrabot run --word ACE --speed 1.0 --max-skills 4
```

安装后也可直接用 `astrabot run ...`。`run --help` 查看完整参数。
`--word` 支持1～4个不重复字母，仍需可见且可达；目前只用左手。
`--inspect-only` 只观察和定位，不执行抓放，但会调整机器人观察姿态。

日志和识别所需的观测图片写入当前工作目录的 `outputs/`，自动创建并被 Git 忽略。
运动过程不保存逐帧图片或逐帧状态文件，不生成视频。每轮识别前仅保存当前左右目
`observation_*.jpg`，以及 API 候选图和识别记录。

`report.json` 中的 `wall_seconds` 是 run 阶段墙钟时间（不含单独执行的 reset），
`simulation_seconds` 是实际仿真推进时间。`timing.stages` 包含各项调用次数、
累计秒数、单次最大秒数和失败次数：

| 字段 | 含义 |
|---|---|
| `ik_solve` | 所有 IK 求解，包含失败求解和预检内的求解 |
| `ik_preflight` | 轨迹预检全段，包含候选路径失败重试 |
| `action_round_trip` | 动作提交到完整结果解析完成，含服务端执行及通信 |
| `server_frame_span` | 同批首末帧的服务端 `wall_time` 差；不包含第一帧生成前的时间 |
| `ws_send` / `ws_receive_wait` | WebSocket 发送 / 接收等待；等待中包含服务端执行，不能当作纯网络耗时 |
| `ws_json_encode` / `ws_json_decode` | 客户端消息序列化 / 解析 |
| `feedback_validation` | 本地逐帧反馈校验，不含图片解码或写盘 |
| `api_observation` / `stereo_perception` | API 观测（含候选图、状态检查及重试）/ 双目定位 |
| `observation_write` / `motion_log_write` / `report_write` | 观测图 / 动作日志 / 报告写入 |

这些计时有嵌套，**不可全部相加**：例如预检包含 IK，动作往返包含收发和解析。
`report_write` 不含正在写入的那次报告本身。`motion.jsonl` 每批动作的 `timing`
还记录发送、ACK 等待、结果等待、反馈处理、消息字节数及该批仿真时间。

当前协议没有完整的服务端批次计时，`server_execution_seconds` 和
`network_transfer_seconds` 明确记为 `null`，而不是零或推算值。
服务端帧间隔包含两帧之间的物理步进、渲染等工作，不能等同于完整批次执行耗时。
精确分离还需要服务端增加单调时钟计时：动作执行（含明确的渲染范围）、
结果编码和发送阶段；不能直接相减两台机器的绝对时间戳。
可用 `--output` 指定一个尚不存在的目录。旧 `runs/` 已删除，代码不再依赖它。

```bash
python -m astrabot benchmark --output outputs/speed-test
python -m unittest discover -s tests -v
```

`benchmark` 是五档各10次冷重置测试，不是单次抓取。离线测试不调用真实 API 或仿真。

[标定与控制约定](docs/calibration.md) · [倍速测试](docs/benchmark.md) · [WebSocket协议](docs/protocol.md)

目前仍是实验实现：历史测试中1.2×仅成功2/10，尚未达到稳定抓取成功率。
