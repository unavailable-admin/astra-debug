# 真机架构、标定与专业调试

日常操作以 [操作指南](pick_a_steps.md) 为准；本页说明底层工具与边界。
运行代码和模型资产均位于 astra-debug，不依赖启动 Novus robot_runner/WMP/ROS 相机节点。
Git worktree 隔离不隔离物理 CAN、DDS 或相机设备。

## 架构

`robot.spelling` 调度字母，`pick_a` 与 `trial_plan` 规划抓取，`transfer` 规划完整搬运，
`trial_runtime` 根据反馈执行。`SceneCheckedTask` 绑定场景与当前会话；Unix socket 服务
持有独立执行器，任务心跳及 console 暂停共同管理任务代次。

执行器约 250 Hz；规划放在独立进程，O6 CAN 手部也使用独立进程，Unitree 关节订阅保留最新反馈。
控制循环、日志队列和状态查询分离；冻结长寿命 Python 对象，新增对象仍自动回收。
状态/审计保留控制周期、锁等待、编码发送与 GC 耗时，超时保护仍然存在。

固定基座 FK/IK、手指包围盒、源网格凸包与重力模型随包分发。
Numba 是规划加速选项；无 Numba 时使用 NumPy 参考实现。首次编译不是控制周期内工作。
模型资产来源摘要在 `astrabot/robot/assets/provenance.json`，运行时不需要原参考仓库网格目录。

## 环境与设备

Python 3.10+，安装 `.[robot]`；额外系统依赖为 Linux V4L2、g++、SocketCAN、
`unitree_sdk2py`、`linkerbot`。视觉 API 使用 Responses 与图像输入，配置见 [README](../README.md)。

`robot_operator.sh start` 从宿主机调用设备授权脚本，配置相机 ACL 和 CAN。
默认 can4 左手（0x28）、can5 右手（0x27），以实际查询为准，不能把 USB 编号当永久身份。
编译缓存与 socket 在 `/tmp/astra-robot-<UID>/`。同一机器人不能有多个动作发布者争用控制。

## 标定与验证

```bash
python3 -m astrabot.robot init-config
python3 -m astrabot.robot.manual_capture --help
python3 -m astrabot.robot.manual_grasp_capture --help
python3 -m astrabot.robot.marker_calibration --help
python3 -m astrabot.robot validate-calibration --help
```

`init-config` 输出未标定模板。`manual_capture` 仅订阅反馈并保存双目图和曝光前后状态，
不创建动作接口；手工采样不自动使抓取就绪。
`marker_calibration` 联合拟合相机与手背标记，需保留训练/留出样本、尺度误差和来源摘要。
`torso_link` 外参会按本帧身体关节转换至骨盆，不能当作固定相机到骨盆外参直接复制。

曝光前要求连续 300 ms 稳定反馈；曝光期间关节变化上限 0.005 rad、手指 2 raw。
失败保存原因并标记 `capture_valid=false`，不会导出为可用场景。
畸变反算检查往返像素误差和局部折叠；数值通过不等于实际标定精度已验证。
真机独立油墨颜色范围与已识别字形局部掩膜避免暗色字形丢失，仍检查双目一致性。

现有现场配置使用受监督模型夹点流程；通用 `run --backend real` 对有效标定、TCP 和就绪状态
有自己的要求。不能通过改 `tcp_measured` 等标志把一次现场成功推广为通用标定通过。

## 仿真与实机模型对齐、完整首抓离线规划

```bash
python3 -m astrabot.robot.trial_plan --help
python3 -m astrabot.robot.scene_builder --help
python3 -m astrabot.robot.spelling --help
```

共享夹指约定和 IK 求解核心；物理安装 FK、碰撞凸包、桌面平面及观察工作区由真机模型处理。
规划覆盖接近、下降、闭指、提起、横移、放下、松手和撤回；路径检查通过不证明真实接触成功。
`spelling.spell(word, config=..., execute=False, scene=...)` 可离线规划首块，
`execute=True` 才进入执行流程。函数返回持久化报告，完整字母调度仍需实时观测。

```python
import asyncio
from astrabot.robot.spelling import spell

# 离线首块规划；scene 必须是已保存的场景目录。
report = asyncio.run(spell("ACE", config="robot-config.local.json", scene="/path/to/scene"))
```

## 控制权、暂停与关机

普通 shutdown 通过已检查的返程和站立模式交接结束执行器，不等于整机断电。
空格暂停废弃任务代次，旧请求不能继续控制；错误不会通过自动领取新代次绕过。
调整腿腰可能使接管基准失效，即使已移走桌子，关机返程也仍可能被拒绝。

阻尼 `disarm` 是独立现场交接方式，需要可靠支撑且已确认阻尼。
零 SDK 权重及零增益/力矩帧至少持续 0.5 秒后才记录交还；发送成功不等于机器人端状态回执。
停止进程不证明旧 SDK 目标已清除。不要将 disarm 当作普通 shutdown 的通用兜底。

当前电机内部上限 120°C、壳体/手部 90°C。内部限值按操作者提供的厂家信息配置，
本仓库未独立验证额定温度。达到任一上限仍触发故障，不自动重启或继续运动。

## 诊断与验证边界

```bash
python3 -m astrabot.robot.diagnose_control --help
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -q
```

通信诊断工具可运行包含编码、禁止动作发送的负载测试；连接真实设备的诊断仍应在对应环境进行。
单元测试使用 mock、合成场景和精简记录，不证明硬实时、硬件热额定值或重复成功率。
最近一次 ACE 的现场成功与离线验证分别记录在 [变更及验收记录](real_robot_automation_status.md)。
