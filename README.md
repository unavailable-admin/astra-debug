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
```

## 安装与配置

Python≥3.10，系统需安装 `ffmpeg` 用于录像。仓库目录下运行：

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

## 运行

默认 `ws://10.19.4.253:8081`，使用前确认该端口空闲；所有子命令都支持 `--uri`。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m astrabot reset
python -m astrabot run --word ACE --speed 1.0 --max-skills 4
```

安装后也可直接用 `astrabot run ...`。`run --help` 查看完整参数。
`--word` 支持1～4个不重复字母，仍需可见且可达；目前只用左手。
`--inspect-only` 只观察和定位，不执行抓放，但会调整机器人观察姿态。

新记录和录像写入当前工作目录的 `outputs/`，自动创建并被 Git 忽略。
可用 `--output` 指定一个尚不存在的目录。旧 `runs/` 已删除，代码不再依赖它。

```bash
python -m astrabot benchmark --output outputs/speed-test
python -m unittest discover -s tests -v
```

`benchmark` 是五档各10次冷重置测试，不是单次抓取。离线测试不调用真实 API 或仿真。

[标定与控制约定](docs/calibration.md) · [倍速测试](docs/benchmark.md) · [WebSocket协议](docs/protocol.md)

目前仍是实验实现：历史测试中1.2×仅成功2/10，尚未达到稳定抓取成功率。
