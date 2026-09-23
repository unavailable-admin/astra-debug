# 运动检查与当前配置

以 `robot.motion_checks.MotionChecks` 为阈值定义来源，`Config` 保存顶层设备和控制参数。
本表是代码默认值；现场 JSON 可在支持范围内覆盖，执行器 status 会返回实际值。
修改配置需正常交接并重启执行器；不能假定运行中进程已读取新值。

## 检查时机

- 开始前：反馈/模式/温度/控制台就绪，稳定起点，IK、关节限位及整条碰撞路径。
- 连续执行中：心跳、反馈新鲜度、身体漂移、跟踪硬门限、温度、控制周期。
- 阶段终点：实测到达；普通搬运不会在每个离散路点等到低误差后才前进。
- 闭指/开指：以最新规划反馈限制步长；位置反馈不等于触觉接触确认。
- 采集：曝光前 300 ms 稳定窗口（身体跨度 0.0015 rad、手指 1 raw）；
  曝光前后允许 0.005 rad / 2 raw。深度分散检查是有效高度偏差的 90 分位，目标门限 8 mm。

## 现场配置与代码默认值

最近成功的现场配置：`trial_lift_height_m=0.12`、`trial_speed_scale=4`、
`trial_grip_mode=supervised_position`、`trial_require_grip_confirmation=false`，
`scene_task_rad=0.005`、`pinch_substep_m=0.005`、`pinch_arrival_m=0.0015`。
新配置模板分别仍是 20 mm、1×、tactile、需要确认，以及下表的默认值。

完整搬运终点 4 mm，松手阶段距原落点 8 mm 内按实测位置继续开指；两者用途不同。
prepare 手腿网格间隙 −1 mm 表示允许模型最多穿入 1 mm，仍加入扫掠误差界；
它不放宽其他部位的碰撞检查，也不是物理接触检测。

顶层默认反馈超时 0.25 s、任务租约 2 s；控制周期停顿保护 100 ms 仍启用。
电机内部/壳体/手温默认上限分别为 120/90/90°C。内部限值为操作者指定配置，
不代表本项目独立验证了厂家额定值。

## motion_checks 默认值

| 参数 | 默认值 | 单位 | 用途 |
| --- | --- | --- | --- |
| `planning_stable_window_s` | 0.3 | s | 接管完成后规划起点持续稳定的时间 |
| `planning_settle_timeout_s` | 15.0 | s | 等待接管和起点稳定的最长时间 |
| `planning_body_drift_rad` | 0.1 | rad | 规划期间相对请求起点的身体变化硬上限 |
| `planning_recheck_rad` | 0.03 | rad | 普通路径起点变化触发重算 |
| `escape_recheck_rad` | 0.001 | rad | 低位外展起点/腿腰变化触发几何复核 |
| `wrist_planning_drift_rad` | 0.001 | rad | 手腕限位恢复规划起点变化上限 |
| `planning_hand_drift_raw` | 5.0 | raw | 普通规划手部变化上限 |
| `fixed_hand_drift_raw` | 1.0 | raw | 外展/手腕恢复/右手保持的手部变化上限 |
| `planning_refreshes` | 2 | count | 同一次规划起点自动重算次数上限 |
| `escape_refreshes` | 2 | count | 低位外展运动中自动复核次数上限 |
| `motion_body_drift_rad` | 0.05 | rad | 运动期间腿腰变化硬上限，外展重算不重置累计上限 |
| `tracking_wait_ratio` | 0.4 | ratio | 等待反馈触发值占当前跟踪门限的比例 |
| `tracking_resume_ratio` | 0.75 | ratio | 恢复门限占等待门限的比例，严格小于等待门限 |
| `tracking_resume_stable_s` | 0.1 | s | 恢复前误差和速度持续稳定的时间 |
| `tracking_resume_speed_rad_s` | 0.02 | rad/s | 恢复前各关节窗口平均绝对速度上限 |
| `tracking_speed_window_s` | 0.05 | s | 恢复速度判定的反馈采集时间窗口 |
| `tracking_resume_peak_speed_rad_s` | 0.06 | rad/s | 恢复前原始反馈瞬时速度上限 |
| `tracking_resume_ramp_s` | 0.3 | s | 恢复后轨迹进度从零平滑加速的时间 |
| `adaptive_brake_horizon_s` | 0.1 | s | 快速逼近参考时提前释放自适应偏置的预测时间 |
| `adaptive_brake_speed_rad_s` | 0.02 | rad/s | 触发偏置提前释放的相对逼近速度 |
| `tracking_wait_timeout_s` | 5.0 | s | 轨迹进度连续等待反馈的超时 |
| `tracking_transient_s` | 0.25 | s | 外展超过软跟踪门限允许的持续时间 |
| `escape_tracking_soft_rad` | 0.005 | rad | 低位外展软跟踪门限 |
| `escape_tracking_hard_rad` | 0.01 | rad | 低位外展即时停止门限 |
| `wrist_tracking_rad` | 0.01 | rad | 手腕限位恢复跟踪门限 |
| `prepare_arrival_rad` | 0.05 | rad | 最终预备位实测到位容差 |
| `prepare_command_lead_rad` | 0.08 | rad | 准备及抓取指令相对反馈的最大位置差 |
| `prepare_arrival_timeout_s` | 3.0 | s | 准备段发完后等待新鲜到位反馈的时间 |
| `prepare_thigh_clearance_m` | -0.001 | m | 准备路径手腿网格有符号间隙；负值允许最多对应深度的重叠 |
| `escape_arrival_rad` | 0.005 | rad | 低位外展终点到位容差 |
| `wrist_arrival_rad` | 0.01 | rad | 手腕限位恢复终点到位容差 |
| `wrist_reverse_rad` | 0.002 | rad | 手腕恢复允许的反向漂移 |
| `wrist_envelope_rad` | 0.01 | rad | 手腕恢复额外位移、非运动关节及腿腰变化上限 |
| `wrist_entry_overshoot_rad` | 0.002 | rad | 手腕恢复允许的初始超限量 |
| `wrist_trigger_margin_rad` | 0.01 | rad | 手腕靠近限位时触发恢复的余量 |
| `wrist_target_margin_rad` | 0.05 | rad | 手腕恢复目标距模型限位的余量 |
| `escape_speed_rad_s` | 0.03 | rad/s | 低位外展最高关节速度 |
| `wrist_speed_rad_s` | 0.02 | rad/s | 手腕限位恢复最高关节速度 |
| `path_step_rad` | 0.001 | rad | 碰撞路径采样的最大等效角增量 |
| `cartesian_step_m` | 0.005 | m | 笛卡尔路径离散位置步长 |
| `rotation_step_rad` | 0.03 | rad | 笛卡尔路径离散旋转步长 |
| `ik_position_error_m` | 0.003 | m | IK 最终位置残差上限 |
| `ik_rotation_error_rad` | 0.03 | rad | IK 最终朝向残差上限 |
| `ik_branch_jump_rad` | 0.35 | rad | 单次 IK 最大关节跳变 |
| `rest_entry_rad` | 0.5 | rad | 低位准备路径的 REST 姿态分类容差 |
| `raised_entry_rad` | 0.05 | rad | 抬手中间位 RAISE 分类容差 |
| `escape_elbow_entry_rad` | 0.25 | rad | 低位外展肘关节相对 0.98 rad 的允许偏差 |
| `escape_shoulder_roll_rad` | 0.2 | rad | 低位外展单次肩侧摆增量上限 |
| `escape_shoulder_yaw_rad` | 0.04 | rad | 低位外展单次肩偏航增量上限 |
| `scene_pose_rad` | 0.003 | rad | 场景加载、缓存起点和准备后规划的姿态一致性容差 |
| `scene_hand_raw` | 2.0 | raw | 场景加载及准备缓存手部姿态容差 |
| `ready_plan_rad` | 0.01 | rad | 开始执行抓取时相对规划起点的身体姿态容差 |
| `scene_task_rad` | 0.001 | rad | 每段实机任务局部规划起点一致性容差 |
| `scene_task_hand_raw` | 1.0 | raw | 每段实机任务局部规划手部一致性容差 |
| `target_observation_m` | 0.01 | m | 抓取目标实测位置与预期位置差的上限 |
| `trial_joint_step_rad` | 0.15 | rad | 实机抓取单段关节增量硬上限 |
| `trial_joint_substep_rad` | 0.1 | rad | 抓取关节路径主动分段步长 |
| `trial_right_arm_rad` | 0.005 | rad | 抓取期间右臂保持误差上限 |
| `pinch_step_m` | 0.006 | m | 指尖补偿单段位置增量硬上限 |
| `pinch_substep_m` | 0.003 | m | 指尖移动主动分段位置步长 |
| `pinch_arrival_m` | 0.001 | m | 携物末端到位容差，执行时不超过微步长度一半 |
| `transfer_arrival_m` | 0.004 | m | 完整搬运阶段终点到位容差，微步容差独立保留 |
| `release_position_tolerance_m` | 0.008 | m | 松手时停止追原落点的位置容差 |
| `pinch_rotation_rad` | 0.04 | rad | 指尖补偿单段旋转增量上限 |
| `pinch_hand_step_raw` | 2.0 | raw | 抓取闭合每段手指增量上限 |
| `prepare_planning_timeout_s` | 180.0 | s | 客户端等待准备规划的超时 |
| `prepare_motion_timeout_s` | 120.0 | s | 客户端等待准备运动的最小预算 |
| `prepare_motion_extra_s` | 30.0 | s | 准备路径名义时长之外的等待预算 |
| `recovery_timeout_s` | 360.0 | s | recover 等待规划及回位的总预算 |
| `release_timeout_s` | 15.0 | s | recover 等待松手完成的预算 |
