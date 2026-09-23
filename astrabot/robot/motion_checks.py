"""Validated motion thresholds, serialized inside the operator configuration."""

import math
from dataclasses import dataclass, field, fields


def threshold(value, unit, maximum, description):
    """Attach the unit and supported envelope to each configurable threshold."""
    return field(default=value, metadata={"unit": unit, "maximum": maximum, "description": description})


@dataclass(frozen=True)
class MotionChecks:
    """No disable switches; defaults preserve existing checks and geometry bounds."""

    planning_stable_window_s: float = threshold(0.3, "s", 5.0, "接管完成后规划起点持续稳定的时间")
    planning_settle_timeout_s: float = threshold(15.0, "s", 60.0, "等待接管和起点稳定的最长时间")
    planning_body_drift_rad: float = threshold(0.1, "rad", 0.1, "规划期间相对请求起点的身体变化硬上限")
    planning_recheck_rad: float = threshold(0.03, "rad", 0.03, "普通路径起点变化触发重算")
    escape_recheck_rad: float = threshold(0.001, "rad", 0.001, "低位外展起点/腿腰变化触发几何复核")
    wrist_planning_drift_rad: float = threshold(0.001, "rad", 0.001, "手腕限位恢复规划起点变化上限")
    planning_hand_drift_raw: float = threshold(5.0, "raw", 5.0, "普通规划手部变化上限")
    fixed_hand_drift_raw: float = threshold(1.0, "raw", 1.0, "外展/手腕恢复/右手保持的手部变化上限")
    planning_refreshes: int = threshold(2, "count", 5, "同一次规划起点自动重算次数上限")
    escape_refreshes: int = threshold(2, "count", 5, "低位外展运动中自动复核次数上限")
    motion_body_drift_rad: float = threshold(0.05, "rad", 0.05, "运动期间腿腰变化硬上限，外展重算不重置累计上限")
    tracking_wait_ratio: float = threshold(0.4, "ratio", 1.0, "等待反馈触发值占当前跟踪门限的比例")
    tracking_resume_ratio: float = threshold(0.75, "ratio", 0.95, "恢复门限占等待门限的比例，严格小于等待门限")
    tracking_resume_stable_s: float = threshold(0.1, "s", 1.0, "恢复前误差和速度持续稳定的时间")
    tracking_resume_speed_rad_s: float = threshold(0.02, "rad/s", 0.1, "恢复前各关节窗口平均绝对速度上限")
    tracking_speed_window_s: float = threshold(0.05, "s", 0.1, "恢复速度判定的反馈采集时间窗口")
    tracking_resume_peak_speed_rad_s: float = threshold(0.06, "rad/s", 0.1, "恢复前原始反馈瞬时速度上限")
    tracking_resume_ramp_s: float = threshold(0.3, "s", 2.0, "恢复后轨迹进度从零平滑加速的时间")
    adaptive_brake_horizon_s: float = threshold(0.1, "s", 0.2, "快速逼近参考时提前释放自适应偏置的预测时间")
    adaptive_brake_speed_rad_s: float = threshold(0.02, "rad/s", 0.1, "触发偏置提前释放的相对逼近速度")
    tracking_wait_timeout_s: float = threshold(5.0, "s", 30.0, "轨迹进度连续等待反馈的超时")
    tracking_transient_s: float = threshold(0.25, "s", 0.25, "外展超过软跟踪门限允许的持续时间")
    escape_tracking_soft_rad: float = threshold(0.005, "rad", 0.005, "低位外展软跟踪门限")
    escape_tracking_hard_rad: float = threshold(0.01, "rad", 0.01, "低位外展即时停止门限")
    wrist_tracking_rad: float = threshold(0.01, "rad", 0.01, "手腕限位恢复跟踪门限")
    prepare_arrival_rad: float = threshold(0.05, "rad", 0.05, "最终预备位实测到位容差")
    prepare_command_lead_rad: float = threshold(0.08, "rad", 0.08, "准备及抓取指令相对反馈的最大位置差")
    prepare_arrival_timeout_s: float = threshold(3.0, "s", 5.0, "准备段发完后等待新鲜到位反馈的时间")
    prepare_thigh_clearance_m: float = threshold(
        -0.001, "m", 0.004, "准备路径手腿网格有符号间隙；负值允许最多对应深度的重叠"
    )
    escape_arrival_rad: float = threshold(0.005, "rad", 0.005, "低位外展终点到位容差")
    wrist_arrival_rad: float = threshold(0.01, "rad", 0.01, "手腕限位恢复终点到位容差")
    wrist_reverse_rad: float = threshold(0.002, "rad", 0.002, "手腕恢复允许的反向漂移")
    wrist_envelope_rad: float = threshold(0.01, "rad", 0.01, "手腕恢复额外位移、非运动关节及腿腰变化上限")
    wrist_entry_overshoot_rad: float = threshold(0.002, "rad", 0.002, "手腕恢复允许的初始超限量")
    wrist_trigger_margin_rad: float = threshold(0.01, "rad", 0.01, "手腕靠近限位时触发恢复的余量")
    wrist_target_margin_rad: float = threshold(0.05, "rad", 0.1, "手腕恢复目标距模型限位的余量")
    escape_speed_rad_s: float = threshold(0.03, "rad/s", 0.1, "低位外展最高关节速度")
    wrist_speed_rad_s: float = threshold(0.02, "rad/s", 0.06, "手腕限位恢复最高关节速度")
    path_step_rad: float = threshold(0.001, "rad", 0.001, "碰撞路径采样的最大等效角增量")
    cartesian_step_m: float = threshold(0.005, "m", 0.005, "笛卡尔路径离散位置步长")
    rotation_step_rad: float = threshold(0.03, "rad", 0.03, "笛卡尔路径离散旋转步长")
    ik_position_error_m: float = threshold(0.003, "m", 0.003, "IK 最终位置残差上限")
    ik_rotation_error_rad: float = threshold(0.03, "rad", 0.03, "IK 最终朝向残差上限")
    ik_branch_jump_rad: float = threshold(0.35, "rad", 0.35, "单次 IK 最大关节跳变")
    rest_entry_rad: float = threshold(0.5, "rad", 0.5, "低位准备路径的 REST 姿态分类容差")
    raised_entry_rad: float = threshold(0.05, "rad", 0.05, "抬手中间位 RAISE 分类容差")
    escape_elbow_entry_rad: float = threshold(0.25, "rad", 0.25, "低位外展肘关节相对 0.98 rad 的允许偏差")
    escape_shoulder_roll_rad: float = threshold(0.2, "rad", 0.2, "低位外展单次肩侧摆增量上限")
    escape_shoulder_yaw_rad: float = threshold(0.04, "rad", 0.04, "低位外展单次肩偏航增量上限")
    scene_pose_rad: float = threshold(0.003, "rad", 0.003, "场景加载、缓存起点和准备后规划的姿态一致性容差")
    scene_hand_raw: float = threshold(2.0, "raw", 2.0, "场景加载及准备缓存手部姿态容差")
    ready_plan_rad: float = threshold(0.01, "rad", 0.01, "开始执行抓取时相对规划起点的身体姿态容差")
    scene_task_rad: float = threshold(0.001, "rad", 0.005, "每段实机任务局部规划起点一致性容差")
    scene_task_hand_raw: float = threshold(1.0, "raw", 1.0, "每段实机任务局部规划手部一致性容差")
    target_observation_m: float = threshold(0.01, "m", 0.01, "抓取目标实测位置与预期位置差的上限")
    trial_joint_step_rad: float = threshold(0.15, "rad", 0.15, "实机抓取单段关节增量硬上限")
    trial_joint_substep_rad: float = threshold(0.1, "rad", 0.1, "抓取关节路径主动分段步长")
    trial_right_arm_rad: float = threshold(0.005, "rad", 0.005, "抓取期间右臂保持误差上限")
    pinch_step_m: float = threshold(0.006, "m", 0.006, "指尖补偿单段位置增量硬上限")
    pinch_substep_m: float = threshold(0.003, "m", 0.005, "指尖移动主动分段位置步长")
    pinch_arrival_m: float = threshold(0.001, "m", 0.0015, "携物末端到位容差，执行时不超过微步长度一半")
    transfer_arrival_m: float = threshold(0.004, "m", 0.004, "完整搬运阶段终点到位容差，微步容差独立保留")
    release_position_tolerance_m: float = threshold(0.008, "m", 0.008, "松手时停止追原落点的位置容差")
    pinch_rotation_rad: float = threshold(0.04, "rad", 0.04, "指尖补偿单段旋转增量上限")
    pinch_hand_step_raw: float = threshold(2.0, "raw", 2.0, "抓取闭合每段手指增量上限")
    prepare_planning_timeout_s: float = threshold(180.0, "s", 600.0, "客户端等待准备规划的超时")
    prepare_motion_timeout_s: float = threshold(120.0, "s", 600.0, "客户端等待准备运动的最小预算")
    prepare_motion_extra_s: float = threshold(30.0, "s", 120.0, "准备路径名义时长之外的等待预算")
    recovery_timeout_s: float = threshold(360.0, "s", 900.0, "recover 等待规划及回位的总预算")
    release_timeout_s: float = threshold(15.0, "s", 60.0, "recover 等待松手完成的预算")

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            minimum = 0 if item.metadata["unit"] == "count" else 0.00001
            if item.name == "prepare_thigh_clearance_m":
                minimum = -0.001
            valid_type = type(value) is int if item.metadata["unit"] == "count" else type(value) in (int, float)
            if not valid_type or not math.isfinite(value) or not minimum <= value <= item.metadata["maximum"]:
                raise ValueError(f"invalid motion_checks.{item.name}: expected {minimum}..{item.metadata['maximum']}")
        for smaller, larger in (
            ("planning_stable_window_s", "planning_settle_timeout_s"),
            ("tracking_resume_stable_s", "tracking_wait_timeout_s"),
            ("tracking_speed_window_s", "tracking_resume_stable_s"),
            ("tracking_resume_speed_rad_s", "tracking_resume_peak_speed_rad_s"),
            ("planning_recheck_rad", "planning_body_drift_rad"),
            ("escape_recheck_rad", "motion_body_drift_rad"),
            ("escape_recheck_rad", "planning_body_drift_rad"),
            ("escape_tracking_soft_rad", "escape_tracking_hard_rad"),
            ("escape_arrival_rad", "escape_tracking_hard_rad"),
            ("wrist_arrival_rad", "wrist_tracking_rad"),
            ("wrist_trigger_margin_rad", "wrist_target_margin_rad"),
            ("trial_joint_substep_rad", "trial_joint_step_rad"),
            ("pinch_substep_m", "pinch_step_m"),
        ):
            if getattr(self, smaller) > getattr(self, larger):
                raise ValueError(f"motion_checks.{smaller} must not exceed {larger}")
