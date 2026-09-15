# Astra robot simulation debug

通过 WebSocket 控制 G1 仿真机器人，用视觉模型识别字母、双目图像定位，执行抓取和拼字。
当前推荐入口为 `closed_loop_stereo.py`。这是实验调试代码，尚未达到稳定抓取成功率。

## 安装

使用 Python 3.10 或更新版本。录像导出需要系统安装 `ffmpeg`，录像核验需要 `ffprobe`。
如需通过 SSH 反向隧道访问公网，安装前先检查代理：

```bash
curl -I --connect-timeout 10 --proxy http://127.0.0.1:18888 https://github.com
export http_proxy=http://127.0.0.1:18888
export https_proxy="$http_proxy"
```

检查失败时需恢复隧道。然后安装 Python 依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

可选工具：`ocr_letters.py` 使用系统 Tesseract；从 USD 重新导出机器人模型才需额外安装 `usd-core`。
主双目流程使用仓库内的机器人模型 JSON。

## API 配置

官方客户端仅连接 `https://api.openai.com/v1`，默认从 `.secrets/openai_api_key`
读取密钥。将自己的密钥写入该文件并设为仅本人可读，此目录不进入 Git。
也可用 `OPENAI_API_KEY_FILE` 指定外部密钥文件；当前客户端不读取
`OPENAI_API_KEY` 环境变量。

- `OPENAI_MODEL`：按账号实际可用模型配置，代码默认 `gpt-6-astra`。
- `OPENAI_PROXY`：默认 `http://127.0.0.1:18888`，直接联网时设为空字符串。
- `OPENAI_CA_BUNDLE`：可选 CA 证书路径。

## 运行双目拼字

先确认 8081 空闲且为 Scene11。重置命令会切换 arm26 再恢复 full36，
等待仿真 worker 重启；固定操作 `ws://10.19.4.253:8081`。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python runs/reset_scene11_worker.py
python closed_loop_stereo.py --word ACE --speed 1.0 --max-skills 4
```

`--word` 暴露字母接口，但字母须可见、定位有效且轨迹可达，不保证任意单词成功。
`--speed` 为全局轨迹倍速，默认 1.0，允许 0.25～3.0。
用 `python closed_loop_stereo.py --help` 查看完整参数。
每次运行在 `runs/` 下保存报告、图像和录像；视频按仿真 30fps 导出。

五档各十次、每次冷重置的批量测试：

```bash
python benchmark_stereo_speed.py --output runs/speed_benchmark_new
```

此命令串行执行 50 次 ACE，固定使用 8081。请使用新的输出目录；断点续跑要求代码与参数未变。
2026-09-14 批次中，1.2× 成功 2/10，成功仿真耗时 111.17～113.03 秒；
1.5×、2×、2.5×、3× 均为 0/10。尚未验证稳定加速档。

## 离线验证

以下 39 项测试不连接仿真，也不调用真实 API：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest test_stereo_letters test_stereo_geometry test_spell_interface test_closed_loop test_openai_connection
```

## 目录

- `closed_loop_stereo.py`、`stereo_*.py`：双目定位与闭环。
- `astra_sim_agent.py`、`grasp_letter_demo.py`、`g1_kinematics.py`：通信、抓放与运动学。
- `closed_loop_spell.py`、`track_letters.py`、`spell_hi_demo.py`：历史单目和 HI 流程。
- `g1_*.json`、`g1_head_camera_calibrations.yaml`：机器人数据与相机标定。
- `fixtures/`：离线回归数据；`runs/README.md` 说明保留的历史参考输入。
- `docs/sim/`：协议、使用说明与实验记录。

双目仿真使用理想针孔内参和 Isaac 渲染外参；真机配置不能直接混用。
详见[双目几何](docs/sim/STEREO_GEOMETRY.md)、
[WebSocket 使用说明](docs/sim/CLIENT_USAGE_GUIDE.md)、
[历史闭环说明](docs/sim/CLOSED_LOOP_SPELL.md)。

本地密钥、依赖环境、参考仓库归档和批量运行产物均不上传。
文档中的历史录像路径仅在保留原始数据的机器上有效。
