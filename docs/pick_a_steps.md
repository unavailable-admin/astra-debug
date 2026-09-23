# 真机操作指南：准备、单块试抓与多字母搬运

本文为当前现场流程。命令在 **Thor 宿主机**、astra-debug 仓库根目录执行。
执行器在容器内工作；默认容器 `ros2humble-wmp`、UID `3100`，分别可由
`ASTRA_CONTAINER`、`ASTRA_OPERATOR_UID` 覆盖。不同终端使用同一配置与容器。

## 配置与首次启动

现有机器沿用 `outputs/operator-startup-20260917T080408Z/config.json`。
可用 `export ASTRA_CONFIG=/absolute/path/config.json` 指定其他文件。
默认配置引用的相机外参也在 outputs，已随成功记录保留，不随代码提交。

新机器可生成未标定模板：

```bash
python3 -m astrabot.robot init-config > robot-config.local.json
```

按真实设备和标定填写，再设置 `ASTRA_CONFIG`。不能直接复制另一台机器的已测标志。
当前现场配置与模板不同：位置夹持、抬高 120 mm、抓放速度缩放 4、免 `GRIP_OK`。
安装和 API 配置见 [README](../README.md)，标定工具见 [真机参考](real_robot.md)。

## 日常完整流程

1. **终端 A** 启动执行器并保持开启：

   ```bash
   bash scripts/robot_operator.sh start
   ```

   脚本设置设备权限后启动；已有执行器时不会自动重启。

2. **终端 B** 打开控制台并保持开启：

   ```bash
   bash scripts/robot_operator.sh console
   ```

   空格暂停并使当前任务失效。`Q` 仅退出控制台，不代表机器人已卸载控制。

3. **终端 C**：确认空手，移走桌子及双臂完整运动范围内的障碍物；保持站姿稳定。
   在 console 按空格，再运行：

   ```bash
   bash scripts/robot_operator.sh prepare
   ```

   按提示输入 `CLEARED`。程序检查路径并**实际抬手到预备位**，到位后调整手型。
   这一步不采集桌面/积木、不调用 API。等待完成，不能只看到“路径已通过”就放回桌子。

4. 放回桌子和积木，清空搬运范围内无关物品；在 console 按空格，然后运行：

   ```bash
   bash scripts/robot_operator.sh spell ACE
   ```

   可替换字母串；输入须为非空 A–Z 字母串；数量受实际桌面和左手可达范围限制，重复字母需要相应数量的实体积木。
   首次观察当前字母和桌面，后续每次抓新字母前只识别当前目标。
   搬运顺序是闭指、抬高、横移、下降、松手、撤回。高度复用首块相对桌面的夹持高度。
   前排靠机器人，按机器人视角左到右排；默认中心距近桌边向内 60 mm，字母间距 120 mm。
   实际坐标由桌面几何计算，仍须通过已有可达性和碰撞检查。

   不逐块要求成功确认；程序按计划放置位置更新障碍物，实际是否成功由操作者观察。
   中途有问题可按空格，不会因暂停而自动续跑旧任务。

5. 结束时确认空手，移走桌子和障碍物，在 console 按空格，再运行：

   ```bash
   bash scripts/robot_operator.sh shutdown
   ```

   输入 `CLEARED` 后执行松手、检查返程及控制权交接。等正常完成与执行器退出。
   这里的 shutdown 是结束 Astra 执行器并交还控制，**不是机器人整机断电**。

## 单块调试与恢复

```bash
bash scripts/robot_operator.sh trial       # A 原地拿放；先完成 prepare 并放回桌子
bash scripts/robot_operator.sh status      # 只读状态
bash scripts/robot_operator.sh recover     # 空手、移走障碍后，回预备位
```

`trial` 和 `spell` 的成功确认行为并不完全相同：现场配置关闭闭指后的 `GRIP_OK`，
单块 trial 仍可能要求整轮结果确认；spell 不逐块询问 `RESULT_OK`。
`recover` 与 prepare/shutdown 一样要求 `CLEARED`，不拍照、不调 API。
console 的 `P/R`、`scene/raise` 是其他受场景约束的入口；日常使用上述命令，不混用旧三步建模流程。

## 等待与检查发生在哪里

| 时机 | 当前检查或工作 |
| --- | --- |
| 启动/任务开始 | 模式、控制权、反馈、console、温度和故障；拍照前等稳定反馈 |
| 每块动作前 | 识别/双目、IK、限位、抓取/携物/撤回全路径碰撞预检；可能有 Numba 首次编译 |
| 连续轨迹中 | 反馈、心跳、跟踪硬门限、身体漂移、温度、控制周期；普通搬运不逐小步停稳 |
| 阶段终点 | 实测到达；搬运位置容差 4 mm。闭指/开指按位置反馈限步推进 |
| 落桌松手 | 距原落点 8 mm 内按当前位置继续开指；不把位置滞后解释为已检测到桌面接触 |
| 抓下一块前 | 稳定拍照并识别下一目标；不重新核验已放好的字母 |

API、完整路径预检、首次编译和真实到达等待不能保证都在 2 秒内。
详细参数见 [运动检查](motion_checks.md)。闭指位置反馈不是触觉，不能据此证明抓牢。

## 报告和失败处理

成功参考：`outputs/spelling-1790077682041773690/report.json`。
`stage=completed`、`motion_completed=true` 是流程完成；`success_verified=false` 表示未做自动实物验收。
报告及图片只保留在本机；操作者已确认这次 ACE 成功。

- 规划或识别失败：程序不会因此证明后续路径安全；先查看当前状态和报告，再处理场景后重新发起任务。
- `task_cancelled`：当前任务已失效。查看原始故障/暂停原因，不复用旧任务代次。
- `base_or_waist_moved`：接管基准改变；prepare/shutdown 也可能被阻止，不能靠改基准或强杀进程绕过。
- `control_deadline_missed` / 温度 / 反馈故障：保留原故障与状态，按现有交接流程处理，不反复重启抢控制。
- `disarm` 只适用于已可靠支撑且已进入阻尼的现场条件；它不是普通 shutdown 的替代命令。

修改执行器代码或配置后，正在运行的旧进程不会自动更新。正常 shutdown 完成后再 start，
随后重新 prepare。无法安全交接时，不以终止进程代替卸载控制。
