# Scene11 双目参数与定位边界

`closed_loop_stereo.py` 已接入双目字母定位，并完成冷重置后的1× ACE抓放验证。
`stereo_geometry.py` 提供独立的参数加载和三角测量模块。
它不读取字母世界坐标，也不依赖已知字母位置拟合单应性。

## 倍速与重复测试

单次运行可传 `--speed 1.5`（范围0.25–3，默认1）；倍速缩短轨迹插值和保持帧数，
保留原有最小帧数。仿真dt与视频帧率不变，实际完成时间不会严格按倍速反比缩短。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/kairos_vepfs_volc/embodied/fuzuoyi/anaconda3/envs/kairos_3_1_pre/bin/python \
benchmark_stereo_speed.py --output runs/stereo_speed_benchmark_20260914
```

该脚本独占8081顺序测试，固定ACE、双目定位、最多4次抓放；
1.2/1.5/2/2.5/3倍速各10次，以固定种子的交错顺序执行。
每次切换arm26再full36冷重启，恢复场景。代码摘要和顺序保存在manifest.json，
每次保存reset日志、episode日志、双目视频及result.json；summary.json和trials.csv持续更新。
仅成功运行进入完成时间统计，指标为末帧减首帧的sim_time，不含API等待和重启。
同命令可跳过已完成试次；若上次在某次动作中断，必须先审计该次记录，不能悄悄丢弃失败。
不要同时运行其他控制器或观察者WebSocket。批测前应先检查AGENTS.md中的代理。

## 完整验收结果

成功运行：`runs/stereo_spell_20260914_135906/`，`success_verified=true`。
3次抓放，4次API调用，无API重试或补充观测，3839动作、127.9667秒仿真。
含初始图像共3840帧；双目视频为该目录 `stereo_1x.mp4`。

| 字母 | 最终双目落点偏差 | 最近手部连杆距离 | 技能仿真时间 |
| --- | --- | --- | --- |
| A | 13.891mm | 282mm | 46.5秒 |
| C | 6.555mm | 303mm | 41.0秒 |
| E | 15.154mm | 341mm | 36.7秒 |

观测姿态准备3.7667秒。全部落点小于18mm，最终VLM也确认ACE顺序、朝上和脱手状态。
每个字母首次有效且尚未抓取的测量，与authored包围盒中心对照：
A 4.024mm、C 2.259mm、E 3.844mm，RMSE 3.467mm，详见 `pre_pick_accuracy.json`。
E第一帧深度质量不足，其对照使用抓E前首次有效观测；各样本帧号均已保存。
这是历史原始布局对照，要求采样前被测字母未改变位置，不是实时PhysX精度认证。
移动后的落点偏差仍由双目测量，尚无独立实时物体真值验证绝对误差。

```python
from stereo_geometry import StereoGeometry

geometry = StereoGeometry.load('sim')  # 当前 Scene11 Isaac 图像
points_left_cv_m = geometry.triangulate(
    left_px=[[320, 240]],
    right_px=[[290, 238]],  # 示例数值；实际必须是同一物理点的匹配
    image_size_wh=(640, 480),
)
```

| 参数 | sim | real |
| --- | --- | --- |
| 内参 | 左焦距274.674、右276.060；fx=fy；主点320,240 | 原始K |
| 畸变 | 全零 | 原始dist，通过undistortPoints校正输入点 |
| 左右相对位姿 | 从渲染挂载位姿推导 | 原始stereo R/T，毫米转换为米 |
| 左相机到head | 渲染挂载外参转换到OpenCV轴 | 未提供；需独立确认/测量手眼外参 |

转换采用 `T_head_camera_cv = T_head_camera_gl @ diag(1,-1,-1,1)`。
渲染挂载已经包含共同平移，不再叠加 `common_mount_translation_head_m`。
真机分支完全不读取渲染挂载，也不自动把文件里的临时pelvis外参当作head外参。

输入必须为同步、未校正原图上的同一物理点，输出为左相机OpenCV坐标，单位米。
不能直接输入校正后图像的像素，也不能默认两个字母检测框的中心是对应点。
函数拒绝分辨率不符、退化解、负深度和重投影误差过大的匹配；这并不能排除所有错误对应，调用者还需匹配质量、深度范围和工作空间检查。

新控制器已接入同步左右图、双目匹配、由关节反馈与机器人模型获得的 `T_robot_head`，
以及现有仿真手指TCP几何。真机还需确认其机器人模型与TCP/手眼标定。
双目点云可用于估计桌面和字母高度；相机固定标定可复用，头部位姿随观测更新，
桌面估计在桌面/机器人相对位置变化或质量不足时更新。
旧单目控制器仍保留参考字母单应性；新双目入口不读取物体真值或旧参考图，抓取高度来自当前双目。

离线验证：`python -m unittest test_stereo_geometry`。
覆盖参数分离、坐标转换、相对位姿一致性、无畸变/带畸变投影回算及错误输入拒绝。
这些单元测试验证几何计算；实际抓取验证见上方完整验收结果。

## 新8081在线检查（2026-09-14）

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/kairos_vepfs_volc/embodied/fuzuoyi/anaconda3/envs/kairos_3_1_pre/bin/python \
test_stereo_live.py --uri ws://10.19.4.253:8081 --seconds 3
```

该脚本先检查空闲的 `showroom_scene_11_stereo` 及在线渲染外参，
随后 `force_reset`，以初始关节角为固定目标提交90个动作，保持正常30Hz仿真。
不调用API、不执行抓取、不读取物体布局或物体真值来重建点。
三维点在左相机和机器人pelvis坐标下输出；机器人转换使用关节反馈和FK。
每次生成独立的 `runs/stereo_live_时间戳/`，保存左右PNG、关节反馈、
匹配图、三维特征点、report.json 和双目并排 H.264 视频 `stereo_1x.mp4`。
视频帧来自实际仿真反馈，没有复制静止图片来补时长。

本次结果：`runs/stereo_live_20260914_125137/`，3.0秒仿真、91帧（含初始帧）。
初末帧分别有37/39个去重后通过互相匹配和0.8像素重投影检查的特征点。
这验证了双目采集、参数一致性、三角测量和录制链路，
尚未验证三维绝对精度、字母身份或抓取成功；特征点不是字母中心。
旧拼字入口仍使用旧单目参考图，不能直接用于新双目场景的ACE测试。

## 双目拼字入口

`closed_loop_stereo.py` 已接入 `stereo_letters.StereoTracker`，独立于旧单目入口。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/kairos_vepfs_volc/embodied/fuzuoyi/anaconda3/envs/kairos_3_1_pre/bin/python \
closed_loop_stereo.py --word ACE --max-skills 4
```

固定1×、trajectory运动配置。先检查空闲双目Scene11/full36，reset并设置已验证的机器人站位，
把手移到观测姿态，然后每次让VLM选择当前图像中的字母编号。
`--inspect-only` 仍会reset并移动双臂到观测姿态，但不抓取。
默认最多3次抓放，可用 `--max-skills` 调整；词接口限最多4个不同字母，目标行为预设左臂工作区。

定位使用当前左右RGB重新校正、SGBM视差和左右一致性检查（<0.75px）。
在已识别字母的青色区域内至少保留10个有效深度像素，剔除高度异常，
要求顶面高度的90%偏差不超过6mm。不同目标顶面高度相差超过15mm时停止。
字母包围框中心射线与测得的水平顶面相交，再减20mm得到方块中心。
抓取中心位于顶面下5mm；桌面高度为当前直立方块顶面中位数减40mm。
这是已知尺寸、顶面朝上、水平桌面的原型，不支持未知尺寸或任意倾斜物体。

控制器不读取 `scene_authored_geometry.json` 或旧单目参考图，不调用旧Tracker。
任务目标行坐标、机器人站位、运动学模型和手指TCP几何仍是已配置的先验。
每次放下并退手后重新调用VLM、重新双目定位，最后同时检查字母顺序与目标行位置。
不在提起阶段额外调用VLM（沿用用户先前指定的信任抓取脚本策略）。

每帧左右JPG、反馈JSON、API记录和最终双目并排 `stereo_1x.mp4` 都保存在
`runs/stereo_spell_时间戳/`。视频30fps对应实际仿真时间，不包含API等待。
左右原图共享同一反馈帧；三维计算使用这帧的关节反馈更新头部位姿。

独立误差评估：

```bash
OPENBLAS_NUM_THREADS=1 \
/kairos_vepfs_volc/embodied/fuzuoyi/anaconda3/envs/kairos_3_1_pre/bin/python \
evaluate_stereo_accuracy.py runs/stereo_spell_时间戳
```

评估脚本只读取保存的结果，不参与控制。当前对照为历史authored包围盒中心，
只适用于尚未移动过的初始布局；它不是当前PhysX真值。
原始USD物体position在方块底部，因此不能直接拿position.z与测得中心z比较。
服务端当前默认场景的layout为null，仍需实时位姿接口才能验证移动后的绝对精度。

重复执行抓取测试前，需要冷重置字母布局（普通订阅reset不能保证恢复已移动的字母）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/kairos_vepfs_volc/embodied/fuzuoyi/anaconda3/envs/kairos_3_1_pre/bin/python \
runs/reset_scene11_worker.py
```

该命令只接受空闲的单目/双目Scene11，切换arm26后恢复full36，等待worker重启；
不会切换到别的场景。reset完成后再运行双目拼字命令。

2026-09-14第一轮抓取记录 `runs/stereo_spell_20260914_130326/`：
A、C成功放到前排，E在接触前的转腕预检停止，累计91.267秒仿真。
这是未完成ACE的失败尝试，不能作为ACE完成时间。
初始对authored包围盒中心的3D误差A/C/E为2.320/2.889/2.368mm，RMSE2.539mm。
之后将抓取端和放置端的转腕高度分开。当前源端依次尝试65/55/45/35mm，
目的端空手转腕高度尝试35/30mm；松手后实际抬高35mm的动作也纳入预检。
每个候选都使用原来的3mm/0.03rad阈值检查完整路径，
没有修改测得的抓取XY或放宽容差。该失败状态已保存成离线回归用例。

第二轮 `runs/stereo_spell_20260914_131606/` 在E转腕的在线IK中停止（100.067秒仿真）。
离线复现表明关键点之间存在IK分支跳变，不能只检查离散端点。
当前版本对双目技能增加5mm/0.02rad间隔的路径预检，每步关节变化不超过0.35rad；
在线求解困难时最多8次减半当前笛卡尔步长，保留3mm/0.03rad约束。
当前版本保持居中夹取，不偏移夹持点；只调整非接触路径。
抬起后先经过Y=1.78m的中间位置再横向搬运，随后经过Y=1.71m前方通道。
被拒绝候选和最终预检路径保存在skill报告中。
成功运行的A/C/E源间距为55/65/65mm，目的端均35mm，夹持XY偏移均为零。
曾试验5mm夹持偏移，但运行 `134229` 的C翻倒（87.367秒），该方案已移除。

单独调试E并保持其ACE目标位置：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/kairos_vepfs_volc/embodied/fuzuoyi/anaconda3/envs/kairos_3_1_pre/bin/python \
closed_loop_stereo.py --word E --start-x -.08 --max-skills 1
```

单字母测试不等于完整ACE验收；完整验收仍需冷重置后运行 `--word ACE`。

E单独抓放记录：`runs/stereo_spell_20260914_133311/`，48.9秒仿真。
本次源转腕间距选择65mm，无XY夹持偏移；最终双目位置误差7.262mm，
最近手部连杆距离336mm，保存的最终图像显示E已在前排。
原控制器报告仍为false：VLM把前排E误描述成原字母阵列，选择再次抓取，
程序因E已在目标点而停止。未修改原始报告；另存 `placement_audit.json` 记录几何验收。
当前版本的完成判定使用当前VLM身份/朝上判断、双目坐标距目标≤18mm、
所有目标距最近手部连杆≥100mm；不让含糊的“原阵列”文字推翻测得的目标位置。
VLM原始decision及 `completion_checks` 一并保留在每次observe记录中。
当前闭环对缺失/低质量定位最多保持姿态后重新观测两次；不会把翻倒或位置不达标当成功。
39项离线测试通过，日志为 `runs/stereo_integration_tests.log`。

### 50次全局倍速测试结果

`runs/stereo_speed_benchmark_20260914/` 已完成五档各10次测试，每次冷重置8081，
固定ACE、双目定位和最多4次抓放，仅改变全局倍速。1.2×成功2/10，
成功仿真耗时111.17和113.03秒；1.5×、2×、2.5×、3×均为0/10。
因此1.2×是本批唯一成功过的档位，但20%成功率不足以称为稳定。
成功仍要求当前字母朝上、双目XY偏差≤18mm、距手部≥100mm。
仿真计时包含准备和补抓，不包含重置、API等待或编码；失败退出时间不算完成耗时。
50份录像的帧数、30fps、实际倍速及仿真计时核验均通过，API错误文件为0。
逐次数据、录像和失败原因见[完整结果](../../runs/stereo_speed_benchmark_20260914/RESULTS.md)，
录像入口见[视频索引](../../runs/stereo_speed_benchmark_20260914/index.html)。
